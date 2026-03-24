from ultralytics import YOLOv10

# Fresh training with batch=22
m = YOLOv10('ultralytics/cfg/models/v10/yolov10n_cbam_eca.yaml')
m.train(
    data='datasets/apple_dataset2510_plus/apple1.yaml',
    cfg='configs/apple_hyp.yaml',
    epochs=10,  # Just run 10 epochs to test
    workers=0,
    batch=22,
    project='runs/apple_test',
    name='cbam_eca_test',
    exist_ok=False,
    plots=False  # Disable plotting to avoid potential hang
)
