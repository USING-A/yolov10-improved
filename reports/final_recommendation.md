# Final Recommendation

## Status
- Frozen mainline (2026-03-31).

## Recommended Mainline
- Model: `ultralytics/cfg/models/v10/yolov10n_cbam_eca.yaml`
- Data: `datasets/apple_dataset2510_plus/apple1.yaml`
- Unified hyperparameters: `configs/apple_hyp.yaml`
- Core optimizer setup:
	- `optimizer=AdamW`
	- `lr0=0.0012`
	- `lrf=0.01`
	- `momentum=0.9`
	- `weight_decay=0.01`

## Evidence
- Baseline (`tune_cbam_eca_t01_e100_auto`):
	- mAP50 = 0.95346
	- mAP50-95 = 0.81395
- AdamW final validation (`tune_cbam_eca_opt_adamw_e100`):
	- mAP50 = 0.95479
	- mAP50-95 = 0.81675
- AdamW reproducibility (`tune_cbam_eca_opt_adamw_e100_seed123`):
	- mAP50 = 0.95292
	- mAP50-95 = 0.81568
- Delta (AdamW - Baseline):
	- mAP50: +0.00133 (about +0.14%)
	- mAP50-95: +0.00280 (about +0.34%)

## Reproducibility Check
- `seed=123` vs baseline:
	- mAP50: -0.00054 (about -0.06%, near tie)
	- mAP50-95: +0.00173 (about +0.21%)
- Conclusion: performance is stable across seeds with acceptable variance.

## Decision
- Promote AdamW setup to default training mainline.
- Keep `ciou + bce` loss path unchanged.
- Reject multi-scale as mainline in current dataset setting.
- Freeze this mainline for downstream use.

## Next Actions
- Use the frozen AdamW profile as the default for subsequent experiments.
- Record future experiments as deltas relative to this frozen mainline.
