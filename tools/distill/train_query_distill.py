"""Launch a geometry-safe cached-query YOLOv10 distillation fine-tune."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
# When launched from ``tools/distill``, Python otherwise resolves an installed
# Ultralytics package before this repository's YOLOv10 implementation.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics.models.yolov10.train import YOLOv10DetectionTrainer
from ultralytics.nn.tasks import YOLOv10DetectionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cfg", type=Path, required=True)
    parser.add_argument("--baseline-weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=19)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--val", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # This is a locally produced, trusted project checkpoint.  PyTorch 2.6+
    # defaults to weights_only=True, which cannot restore the YOLOv10 class
    # object stored by earlier project checkpoints.
    checkpoint = torch.load(args.baseline_weights, map_location="cpu", weights_only=False)
    baseline = (checkpoint.get("ema") or checkpoint["model"]).float()
    model = YOLOv10DetectionModel(str(args.model_cfg), nc=1, verbose=True)
    model.load(baseline)
    overrides = {
        "task": "detect",
        "mode": "train",
        "data": str(args.data),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "seed": args.seed,
        "deterministic": True,
        "optimizer": "AdamW",
        "lr0": 0.0012,
        "lrf": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.01,
        "warmup_epochs": 1.0,
        "box": 8.0,
        "cls": 0.45,
        "dfl": 1.7,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        # Cached boxes are in source-image coordinates.  Keep all geometry
        # unchanged during the KD stage; colour augmentation remains allowed.
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "close_mosaic": 0,
        "fraction": args.fraction,
        "val": args.val,
        "project": str(args.project),
        "name": args.name,
        "exist_ok": False,
        "pretrained": False,
        # Skip Ultralytics' unrelated external YOLOv8 AMP self-check.  The
        # project uses PyTorch 2.7, whose strict checkpoint default rejects
        # that legacy check's serialized class object.
        "amp": False,
        "save": True,
        "plots": True,
    }
    trainer = YOLOv10DetectionTrainer(overrides=overrides)
    trainer.model = model
    trainer.train()


if __name__ == "__main__":
    main()
