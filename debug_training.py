#!/usr/bin/env python
"""Debug training initialization."""

import sys
import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

try:
    print("Step 1: Importing YOLOv10...", flush=True)
    from ultralytics import YOLOv10
    
    print("Step 2: Loading model...", flush=True)
    m = YOLOv10('ultralytics/cfg/models/v10/yolov10n_cbam_eca.yaml')
    
    print("Step 3: Calling train() with fresh name...", flush=True)
    result = m.train(
        data='datasets/apple_dataset2510_plus/apple1.yaml',
        cfg='configs/apple_hyp.yaml',
        epochs=150,
        workers=0,
        batch=22,
        project='runs/apple',
        name='combo_cbam_eca_debug_fresh',
        exist_ok=False,
        plots=False,
        verbose=False
    )
    
    print("Step 4: Training completed!", flush=True)
    
except KeyboardInterrupt:
    print("Interrupted", flush=True)
except SystemExit as e:
    print(f"SystemExit: {e}", flush=True)
    sys.exit(0)
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}", flush=True)
    sys.exit(1)
