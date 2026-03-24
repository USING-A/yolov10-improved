from ultralytics import YOLOv10
import sys

print("Step 1: YOLOv10 imported", flush=True, file=sys.stderr)

try:
    # Try to load model
    print("Step 2: Loading model...", flush=True, file=sys.stderr)
    m = YOLOv10('ultralytics/cfg/models/v10/yolov10n_cbam_eca.yaml')
    print("Step 3: Model loaded successfully", flush=True, file=sys.stderr)
    
except Exception as e:
    print(f"ERROR at model loading: {e}", flush=True, file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
