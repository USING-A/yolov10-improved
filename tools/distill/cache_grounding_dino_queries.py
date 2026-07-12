"""Cache GT-reliable Grounding DINO decoder queries for offline YOLO KD.

The cache is intentionally compact and training-only.  Each record is keyed by
the YOLO-relative image path and contains only decoder queries that are both
confident and matched one-to-one to a ground-truth apple.  This prevents a
strong but imperfect teacher from replacing ground-truth box supervision.

The resulting ``.pt`` file contains no model parameters and is never loaded by
YOLO inference or export.  It is consumed later by the distillation-only
one-to-many loss.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint
from mmdet.models.dense_heads.atss_vlfusion_head import convert_grounding_to_cls_scores
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
from scipy.optimize import linear_sum_assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mmdet-root", type=Path, required=True)
    parser.add_argument("--prompt", default="apple")
    parser.add_argument(
        "--scale", default="640,1024",
        help="MMDetection resize tuple used for teacher adaptation, as width,height.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--min-iou", type=float, default=0.50)
    parser.add_argument("--min-score", type=float, default=0.25)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    os.environ["MMDET_ROOT"] = str(args.mmdet_root.resolve())
    os.environ["GDINO_DATA_ROOT"] = str(args.dataset_root.resolve())
    os.environ["GDINO_ANNOTATION_ROOT"] = str(args.annotation_root.resolve())
    os.environ["GDINO_SCALE"] = args.scale
    os.environ.setdefault("PYTHONUTF8", "1")


def build_loader(cfg: Config, args: argparse.Namespace):
    loader_cfg = copy.deepcopy(cfg.val_dataloader)
    loader_cfg.batch_size = args.batch_size
    loader_cfg.num_workers = args.num_workers
    loader_cfg.persistent_workers = args.num_workers > 0
    loader_cfg.dataset.data_root = str(args.dataset_root.resolve())
    loader_cfg.dataset.ann_file = str(args.annotation_root.resolve() / "train.json")
    loader_cfg.dataset.test_mode = True
    if args.max_images > 0:
        loader_cfg.dataset.indices = args.max_images
    return Runner.build_dataloader(loader_cfg, seed=0, diff_rank_seed=False)


def build_model(cfg: Config, checkpoint: Path, device: torch.device):
    init_default_scope(cfg.default_scope)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, str(checkpoint), map_location="cpu")
    model.to(device)
    model.eval()
    return model


def decoder_outputs(model, batch_inputs: torch.Tensor, batch_data_samples, prompt: str):
    """Run the public Grounding DINO modules and retain final decoder queries."""
    positive_map, caption, _, _ = model.get_tokens_positive_and_prompts(
        (prompt,), custom_entities=True)
    text_dict = model.language_model([caption] * len(batch_data_samples))
    if model.text_feat_map is not None:
        text_dict["embedded"] = model.text_feat_map(text_dict["embedded"])
    visual_features = model.extract_feat(batch_inputs)
    head_inputs = model.forward_transformer(visual_features, text_dict, batch_data_samples)
    all_logits, all_boxes = model.bbox_head(
        hidden_states=head_inputs["hidden_states"],
        references=head_inputs["references"],
        memory_text=head_inputs["memory_text"],
        text_token_mask=head_inputs["text_token_mask"],
    )
    query_scores = convert_grounding_to_cls_scores(
        all_logits[-1].sigmoid(), [positive_map] * len(batch_data_samples)
    ).squeeze(-1)
    return head_inputs["hidden_states"][-1], query_scores, all_boxes[-1]


def empty_record(embed_dim: int) -> dict[str, torch.Tensor]:
    return {
        "gt_boxes_xyxy": torch.empty((0, 4), dtype=torch.float16),
        "teacher_boxes_xyxy": torch.empty((0, 4), dtype=torch.float16),
        "query_embeddings": torch.empty((0, embed_dim), dtype=torch.float16),
        "relations": torch.empty((0, 0), dtype=torch.float16),
        "query_indices": torch.empty((0,), dtype=torch.int16),
        "scores": torch.empty((0,), dtype=torch.float16),
        "ious": torch.empty((0,), dtype=torch.float16),
    }


def cache_record(
    query_embeddings: torch.Tensor,
    query_scores: torch.Tensor,
    query_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    image_shape: tuple[int, int],
    original_shape: tuple[int, int],
    scale_factor: tuple[float, ...],
    min_iou: float,
    min_score: float,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """One-to-one assign GT boxes to decoder queries and retain reliable pairs."""
    image_height, image_width = image_shape[:2]
    original_height, original_width = original_shape[:2]
    if len(gt_boxes) == 0:
        return empty_record(query_embeddings.shape[-1]), {"gt": 0, "kept": 0, "assigned_iou": 0.0, "assigned_score": 0.0}

    query_xyxy = bbox_cxcywh_to_xyxy(query_boxes)
    query_xyxy[:, 0::2] *= image_width
    query_xyxy[:, 1::2] *= image_height
    query_xyxy[:, 0::2].clamp_(0, image_width)
    query_xyxy[:, 1::2].clamp_(0, image_height)
    scale = query_xyxy.new_tensor(scale_factor)
    if scale.numel() == 2:
        scale = scale.repeat(2)
    query_xyxy /= scale
    query_xyxy[:, 0::2].clamp_(0, original_width)
    query_xyxy[:, 1::2].clamp_(0, original_height)
    overlaps = bbox_overlaps(gt_boxes, query_xyxy)
    # Pure IoU assignment can select a geometrically close but low-confidence
    # decoder query.  The cache is deliberately reliability-gated, so resolve
    # one-to-one conflicts using joint localization/classification quality.
    all_scores = query_scores.clamp(0, 1)
    match_quality = overlaps * all_scores.unsqueeze(0)
    gt_indices, query_indices = linear_sum_assignment((-match_quality).detach().float().cpu().numpy())
    gt_indices = torch.as_tensor(gt_indices, device=gt_boxes.device, dtype=torch.long)
    query_indices = torch.as_tensor(query_indices, device=gt_boxes.device, dtype=torch.long)
    ious = overlaps[gt_indices, query_indices]
    scores = query_scores[query_indices]
    assigned_iou = float(ious.mean())
    assigned_score = float(scores.mean())
    keep = (ious >= min_iou) & (scores >= min_score)
    gt_indices, query_indices, ious, scores = (
        value[keep] for value in (gt_indices, query_indices, ious, scores)
    )
    if len(query_indices) == 0:
        return empty_record(query_embeddings.shape[-1]), {
            "gt": int(len(gt_boxes)), "kept": 0,
            "assigned_iou": assigned_iou, "assigned_score": assigned_score,
        }

    gt_normalizer = gt_boxes.new_tensor([original_width, original_height, original_width, original_height])
    selected_queries = query_embeddings[query_indices]
    selected_queries = F.normalize(selected_queries, dim=-1)
    relations = selected_queries @ selected_queries.transpose(0, 1)
    record = {
        "gt_boxes_xyxy": (gt_boxes[gt_indices] / gt_normalizer).detach().cpu().to(torch.float16),
        "teacher_boxes_xyxy": (query_xyxy[query_indices] / gt_normalizer).detach().cpu().to(torch.float16),
        "query_embeddings": selected_queries.detach().cpu().to(torch.float16),
        "relations": relations.detach().cpu().to(torch.float16),
        "query_indices": query_indices.detach().cpu().to(torch.int16),
        "scores": scores.detach().cpu().to(torch.float16),
        "ious": ious.detach().cpu().to(torch.float16),
    }
    return record, {
        "gt": int(len(gt_boxes)), "kept": int(len(query_indices)),
        "mean_iou": float(ious.mean()), "mean_score": float(scores.mean()),
        "assigned_iou": assigned_iou, "assigned_score": assigned_score,
    }


def atomic_save(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)


def serializable_meta(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.min_iou <= 1.0 and 0.0 <= args.min_score <= 1.0):
        raise ValueError("min-iou and min-score must be within [0, 1]")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"cache already exists: {args.output}; pass --overwrite to replace it")
    configure_environment(args)
    cfg = Config.fromfile(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, args.checkpoint.resolve(), device)
    loader = build_loader(cfg, args)
    records: dict[str, dict[str, torch.Tensor]] = {}
    summary: dict[str, float] = {
        "images": 0, "gt_instances": 0, "reliable_instances": 0,
        "sum_iou": 0.0, "sum_score": 0.0,
        "sum_assigned_iou": 0.0, "sum_assigned_score": 0.0,
    }

    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader, start=1):
            data_batch = model.data_preprocessor(raw_batch, False)
            batch_inputs = data_batch["inputs"]
            batch_samples = data_batch["data_samples"]
            query_embeddings, query_scores, query_boxes = decoder_outputs(model, batch_inputs, batch_samples, args.prompt)
            for item_index, data_sample in enumerate(batch_samples):
                image_path = Path(data_sample.metainfo["img_path"]).resolve()
                cache_key = image_path.relative_to(args.dataset_root.resolve()).as_posix()
                if cache_key in records:
                    raise RuntimeError(f"duplicate image cache key: {cache_key}")
                record, stats = cache_record(
                    query_embeddings[item_index], query_scores[item_index], query_boxes[item_index],
                    data_sample.gt_instances.bboxes,
                    data_sample.metainfo["img_shape"],
                    data_sample.metainfo["ori_shape"],
                    data_sample.metainfo["scale_factor"],
                    args.min_iou, args.min_score,
                )
                records[cache_key] = record
                summary["images"] += 1
                summary["gt_instances"] += stats["gt"]
                summary["reliable_instances"] += stats["kept"]
                summary["sum_iou"] += stats.get("mean_iou", 0.0) * stats["kept"]
                summary["sum_score"] += stats.get("mean_score", 0.0) * stats["kept"]
                summary["sum_assigned_iou"] += stats["assigned_iou"] * stats["gt"]
                summary["sum_assigned_score"] += stats["assigned_score"] * stats["gt"]
            if batch_index % args.save_every == 0:
                payload = {"meta": serializable_meta(args), "summary": summary, "records": records}
                atomic_save(payload, args.output)
                print(f"cached {summary['images']} images / {summary['reliable_instances']} reliable instances")

    reliable = max(1, int(summary["reliable_instances"]))
    summary["reliability_rate"] = summary["reliable_instances"] / max(1, summary["gt_instances"])
    summary["mean_iou"] = summary.pop("sum_iou") / reliable
    summary["mean_score"] = summary.pop("sum_score") / reliable
    total_gt = max(1, int(summary["gt_instances"]))
    summary["mean_assigned_iou"] = summary.pop("sum_assigned_iou") / total_gt
    summary["mean_assigned_score"] = summary.pop("sum_assigned_score") / total_gt
    payload = {"meta": serializable_meta(args), "summary": summary, "records": records}
    atomic_save(payload, args.output)
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps({"meta": payload["meta"], "summary": summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
