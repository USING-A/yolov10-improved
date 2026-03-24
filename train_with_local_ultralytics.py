#!/usr/bin/env python
"""Training with proper local ultralytics path."""

import sys
import os

# Ensure local ultralytics is used first
sys.path.insert(0, r'd:/Github Code/yolov10-improved')
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

print("=" * 60, flush=True)
print(f"Python path: {sys.path[:3]}", flush=True)
print("=" * 60, flush=True)

RUN_NAME = 'combo_cbam_eca_e150_fresh_direct'
LAST_CKPT = f'runs/apple/{RUN_NAME}/weights/last.pt'

try:
    print("[1/4] Importing YOLOv10...", flush=True)
    from ultralytics import YOLOv10
    print(f"[2/4] YOLOv10 loaded from: {YOLOv10.__module__}", flush=True)
    
    print("[3/4] Creating model...", flush=True)
    if os.path.exists(LAST_CKPT):
        print(f"Found checkpoint, resuming from: {LAST_CKPT}", flush=True)
        m = YOLOv10(LAST_CKPT)
    else:
        print("No checkpoint found, starting fresh model.", flush=True)
        m = YOLOv10('ultralytics/cfg/models/v10/yolov10n_cbam_eca.yaml')

    print("[4/4] Starting training...", flush=True)
    
    m.train(
        data='datasets/apple_dataset2510_plus/apple1.yaml',
        cfg='configs/apple_hyp.yaml',
        epochs=150,
        workers=0,
        batch=-1,
        project='runs/apple',
        name=RUN_NAME,
        exist_ok=True,
        resume=os.path.exists(LAST_CKPT),
        plots=False,
        patience=30
    )
    
    print("Training completed!", flush=True)
    
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {str(e)[:200]}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
