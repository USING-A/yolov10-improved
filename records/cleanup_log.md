# Workspace Cleanup Log

Date: 2026-03-16
Scope: Remove non-essential cloned repository assets unrelated to current apple single-class improvement project.

## Cleanup Principles
- Preserve core training/inference code under ultralytics/.
- Preserve dataset assets under datasets/apple_dataset2510_plus/.
- Preserve test scripts and core dependency files.
- Remove docs/demo/legacy artifacts not required for current development scope.

## Removed Previously (examples)
- examples/heatmaps.ipynb
- examples/hub.ipynb
- examples/object_counting.ipynb
- examples/object_tracking.ipynb
- examples/tutorial.ipynb
- examples/YOLOv8-CPP-Inference/
- examples/YOLOv8-LibTorch-CPP-Inference/
- examples/YOLOv8-ONNXRuntime/
- examples/YOLOv8-ONNXRuntime-CPP/
- examples/YOLOv8-ONNXRuntime-Rust/
- examples/YOLOv8-OpenCV-int8-tflite-Python/
- examples/YOLOv8-OpenCV-ONNX-Python/
- examples/YOLOv8-Region-Counter/
- examples/YOLOv8-SAHI-Inference-Video/
- examples/YOLOv8-Segmentation-ONNXRuntime-Python/

## Removed in Current Cleanup Pass
- CONTRIBUTING.md
- docker/
- docs/
- mkdocs.yml
- logs/
- figures/
- app.py

## Current Status
- examples/ now keeps: README.md
- Newly standardized folders created: configs/, experiments/, records/, reports/
- No baseline training/evaluation executed in this pass.
