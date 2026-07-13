"""Training-only, cached-query distillation primitives for YOLOv10."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.ops import xywh2xyxy


_CACHE_BY_PATH: dict[str, dict] = {}


def _load_records(cache_path: str) -> tuple[Path, dict]:
    """Keep cache tensors process-global so checkpoints never pickle the cache."""
    cache_path = str(Path(cache_path).resolve())
    if cache_path not in _CACHE_BY_PATH:
        payload = torch.load(cache_path, map_location="cpu")
        _CACHE_BY_PATH[cache_path] = payload
    payload = _CACHE_BY_PATH[cache_path]
    return Path(payload["meta"]["dataset_root"]).resolve(), payload["records"]


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (rb - lt).clamp_min(0).prod(-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0).prod(-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0).prod(-1)
    return intersection / (area1[:, None] + area2[None, :] - intersection).clamp_min(1e-7)


def _greedy_iou_match(cached_boxes: torch.Tensor, current_boxes: torch.Tensor, threshold: float) -> list[tuple[int, int]]:
    if not len(cached_boxes) or not len(current_boxes):
        return []
    ious = _box_iou(cached_boxes, current_boxes)
    pairs = torch.nonzero(ious >= threshold, as_tuple=False)
    if not len(pairs):
        return []
    order = ious[pairs[:, 0], pairs[:, 1]].argsort(descending=True)
    used_cache, used_current, matches = set(), set(), []
    for pair in pairs[order].tolist():
        cached_index, current_index = pair
        if cached_index not in used_cache and current_index not in used_current:
            matches.append((cached_index, current_index))
            used_cache.add(cached_index)
            used_current.add(current_index)
    return matches


class ReliableQueryDistiller(nn.Module):
    """Align cached teacher queries to scale-aware student RoI descriptors.

    The module is registered on the student solely during training.  It stores
    only the cache path; query tensors are loaded into a process-global cache
    on first use and are therefore absent from state_dict/export artifacts.
    """

    def __init__(
        self,
        channels: list[int],
        cache_path: str,
        teacher_dim: int = 256,
        instance_weight: float = 0.25,
        relation_weight: float = 0.10,
        match_iou: float = 0.995,
        small_threshold: float = 32 / 640,
        medium_threshold: float = 96 / 640,
    ) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError(f"expected P3/P4/P5 channels, got {channels}")
        self.cache_path = str(cache_path)
        self.instance_weight = float(instance_weight)
        self.relation_weight = float(relation_weight)
        self.match_iou = float(match_iou)
        self.small_threshold = float(small_threshold)
        self.medium_threshold = float(medium_threshold)
        self.projectors = nn.ModuleList(
            nn.Sequential(nn.Linear(channel, teacher_dim), nn.LayerNorm(teacher_dim)) for channel in channels
        )

    @staticmethod
    def _roi_descriptor(feature: torch.Tensor, image_index: int, box: torch.Tensor) -> torch.Tensor:
        """Sample a 2x2 grid from one normalized RoI without torchvision ops."""
        x1, y1, x2, y2 = box.unbind()
        fractions = feature.new_tensor((0.25, 0.75))
        grid_y, grid_x = torch.meshgrid(
            y1 + (y2 - y1) * fractions,
            x1 + (x2 - x1) * fractions,
            indexing="ij",
        )
        grid = torch.stack((grid_x.mul(2).sub(1), grid_y.mul(2).sub(1)), dim=-1).unsqueeze(0)
        sampled = F.grid_sample(feature[image_index : image_index + 1], grid, align_corners=False)
        return sampled.mean(dim=(-1, -2)).squeeze(0)

    def _level_index(self, box: torch.Tensor) -> int:
        side = torch.sqrt(((box[2] - box[0]) * (box[3] - box[1])).clamp_min(0)).item()
        if side <= self.small_threshold:
            return 0
        if side <= self.medium_threshold:
            return 1
        return 2

    def forward(self, features: tuple[torch.Tensor, ...] | list[torch.Tensor] | None, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        if not features:
            zero = batch["img"].sum() * 0
            return zero, zero
        dataset_root, records = _load_records(self.cache_path)
        device = features[0].device
        current_boxes = xywh2xyxy(batch["bboxes"].to(device))
        batch_indices = batch["batch_idx"].view(-1).to(device)
        instance_losses, relation_losses = [], []

        for image_index, image_path in enumerate(batch["im_file"]):
            try:
                cache_key = Path(image_path).resolve().relative_to(dataset_root).as_posix()
            except ValueError:
                continue
            record = records.get(cache_key)
            if record is None or not len(record["query_embeddings"]):
                continue
            image_gt_indices = torch.where(batch_indices == image_index)[0]
            if not len(image_gt_indices):
                continue
            cached_boxes = record["gt_boxes_xyxy"].to(device=device, dtype=current_boxes.dtype)
            matches = _greedy_iou_match(cached_boxes, current_boxes[image_gt_indices], self.match_iou)
            if not matches:
                continue
            cached_indices = torch.tensor([item[0] for item in matches], device=device)
            current_indices = image_gt_indices[torch.tensor([item[1] for item in matches], device=device)]
            student_vectors = []
            for current_index in current_indices.tolist():
                box = current_boxes[current_index].clamp(0, 1)
                level = self._level_index(box)
                descriptor = self._roi_descriptor(features[level], image_index, box)
                student_vectors.append(self.projectors[level](descriptor))
            student_vectors = F.normalize(torch.stack(student_vectors), dim=-1)
            teacher_vectors = record["query_embeddings"][cached_indices.cpu()].to(device=device, dtype=student_vectors.dtype)
            instance_losses.append(1 - (student_vectors * teacher_vectors).sum(dim=-1).mean())
            if len(cached_indices) > 1:
                teacher_relations = record["relations"][cached_indices.cpu()][:, cached_indices.cpu()].to(
                    device=device, dtype=student_vectors.dtype
                )
                relation_losses.append(F.mse_loss(student_vectors @ student_vectors.T, teacher_relations))

        zero = features[0].sum() * 0
        instance_loss = torch.stack(instance_losses).mean() if instance_losses else zero
        relation_loss = torch.stack(relation_losses).mean() if relation_losses else zero
        return instance_loss, relation_loss
