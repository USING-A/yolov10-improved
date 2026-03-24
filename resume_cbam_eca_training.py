#!/usr/bin/env python
"""Resume cbam_eca training from checkpoint."""

import sys
from ultralytics import YOLOv10

print("=" * 60)
print("Starting cbam_eca training resume...")
print("=" * 60, flush=True)

try:
    print("[1/3] Loading model...", flush=True)
    m = YOLOv10("ultralytics/cfg/models/v10/yolov10n_cbam_eca.yaml")
    
    print("[2/3] Starting training with resume=True...", flush=True)
    m.train(
        data="datasets/apple_dataset2510_plus/apple1.yaml",
        cfg="configs/apple_hyp.yaml",
        epochs=150,
        workers=0,
        batch=-1,
        project="runs/apple",
        name="combo_cbam_eca_e150_auto",
        exist_ok=True,
        resume=True
    )
    print("[3/3] Training completed successfully!", flush=True)
    sys.exit(0)
    
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
