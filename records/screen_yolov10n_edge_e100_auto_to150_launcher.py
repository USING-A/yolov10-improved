import os
import sys
sys.path.insert(0, r'd:/Github Code/yolov10-improved')
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
from ultralytics import YOLOv10

m = YOLOv10('ultralytics/cfg/models/v10/yolov10n_edge.yaml')
m.train(
    data='datasets/apple_dataset2510_plus/apple1.yaml',
    cfg='configs/apple_hyp.yaml',
    epochs=150,
    workers=0,
    batch=-1,
    project='runs/apple',
    name='screen_yolov10n_edge_e100_auto',
    exist_ok=True,
    resume=True,
    plots=False,
    iou_type='ciou',
    cls_loss='bce'
)
