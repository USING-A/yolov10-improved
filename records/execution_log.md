# Execution Log

This file tracks command-level execution in implementation phase.

| Time | Step | Action | Result | Notes |
| --- | --- | --- | --- | --- |
| 2026-03-24 09:40 | Step 6 | Checked combo run checkpoints (`last.pt`) and `results.csv` continuity for `combo_cbam_eca_e150_auto` and `combo_bifpn_edge_e150_auto` | Success | Both runs are resumable (56 and 48 logged epochs). |
| 2026-03-24 09:45 | Step 6 | Resumed `combo_cbam_eca_e150_auto` with `resume=True` to target 150 epochs | Running | Restarted from epoch 57, batch remains `-1`. |
| 2026-03-24 09:47 | Step 6 | Resumed `combo_bifpn_edge_e150_auto` with `resume=True` to target 150 epochs | Running | Restarted from epoch 49, batch remains `-1`. |
| 2026-03-24 10:00 | Step 6 | Backfilled experiment ledgers (`baseline.csv`, `module_screening.csv`, `combination_ablation.csv`) | Success | Added current best metrics and in-progress notes for resumed combos. |
| 2026-03-25 09:10 | Step 6 | Verified combo training completion status from `results.csv` | Success | Main four combo runs all reached 150 epochs. |
| 2026-03-25 09:15 | Step 6 | Added final combo metrics and fresh-direct record into `experiments/combination_ablation.csv` | Success | `combo_cbam_eca_e150_fresh_direct` is current top candidate. |
| 2026-03-25 09:20 | Step 6/8 | Updated plan and recovery log to closed state | Success | Step 6 and Step 8 marked completed; next is Step 9 hyperparameter tuning. |
| 2026-03-25 10:00 | Step 9 | Updated `configs/apple_hyp.yaml` to T01 profile and registered run in `experiments/hyperparam_tuning.csv` | Success | Single-YAML tuning profile activated. |
| 2026-03-25 10:03 | Step 9 | Started `tune_cbam_eca_t01_e100_auto` training | Running | Local ultralytics path priority enabled via `PYTHONPATH`. |
| 2026-03-25 10:40 | Step 9B | Implemented configurable loss switches in `ultralytics/utils/loss.py` and registered new keys in default/hyp YAML | Success | Added `iou_type`, `cls_loss`, `focal_gamma`, `focal_alpha`; default behavior unchanged (`ciou+bce`). |
| 2026-03-25 10:48 | Step 9B | Ran 1-epoch smoke with `diou+focal` | Partial | Training entered epoch loop successfully; command timed out in terminal view but no config/key errors observed. |
| 2026-03-25 11:00 | Step 9B | Started `tune_cbam_eca_loss_diou_focal_e60` and `tune_cbam_eca_strategy_cosine_ls_e60` | Running | Non-structure experiments launched in parallel with T01. |
| 2026-03-25 11:03 | Step 9B | Registered `tune_cbam_eca_loss_giou_varifocal_e60` as queued experiment | Success | Will launch when a training slot is freed. |
| 2026-03-25 11:20 | Step 9B | Enforced training concurrency cap | Success | Reduced active tuning runs to 2 (`tune_cbam_eca_t01_e100_auto`, `tune_cbam_eca_loss_diou_focal_e60`) to avoid stall. |
| 2026-03-25 23:26 | Step 9B | Started queue scheduler `records/step9_queue_scheduler.ps1` | Running | Automatically launches `strategy_cosine_ls_e60` then `loss_giou_varifocal_e60` when active count < 2. |
| 2026-03-26 09:20 | Step 9B | Backfilled finished metrics for T01 and DIoU+Focal to `experiments/hyperparam_tuning.csv` | Success | DIoU+Focal run shows severe regression and is marked as negative candidate. |
| 2026-03-26 09:30 | Step 9B | Stopped noisy queue scheduler loop and relaunched remaining two runs manually under concurrency cap | Running | `tune_cbam_eca_strategy_cosine_ls_e60` and `tune_cbam_eca_loss_giou_varifocal_e60` are active. |
| 2026-03-26 20:20 | Step 9B | Detected two runs stalled around epoch 1; killed and restarted both from `weights/last.pt` with `resume=True` | Running | Active runs: `tune_cbam_eca_strategy_cosine_ls_e60`, `tune_cbam_eca_loss_giou_varifocal_e60`. |
| 2026-03-26 20:35 | Step 9B | Verified resumed runs progressed beyond stalled point | Success | `strategy` reached epoch 3/60; `giou+varifocal` reached epoch 2/60. |
| 2026-03-27 09:20 | Step 9B | Collected final metrics and closed non-structure/loss experiments | Success | Both loss-variant runs are strongly negative; strategy run is also below T01; Step9B closed. |
| 2026-03-27 10:30 | Step 9C | Registered three additional non-framework optimization experiments | Success | Added AdamW, augmentation-tuning, and multi-scale strategy runs to hyperparam ledger. |
| 2026-03-27 10:36 | Step 9C | Started `tune_cbam_eca_opt_adamw_e60` and `tune_cbam_eca_aug_mosaic07_e60` | Running | Concurrency kept at 2. |
| 2026-03-27 10:40 | Step 9C | Started single-run watcher `records/step9c_launch_multiscale_when_slot.ps1` | Running | Auto-launch `tune_cbam_eca_strategy_multiscale_e60` when active training count < 2. |
| 2026-03-28 07:00 | Step 9C | Verified AdamW and Augmentation training completed (60 epochs each) | Success | AdamW: mAP50=0.94108 (-1.24% vs T01); Augmentation: mAP50=0.93896 (-1.52% vs T01). Both show minor regression. |
| 2026-03-28 07:15 | Step 9C | Started `tune_cbam_eca_strategy_multiscale_e60` (multi_scale=True, warmup_epochs=5.0) | Running | Watcher triggered auto-launch after first slot freed. Multi-scale training now in progress. |
| 2026-03-29 09:10 | Step 9C | Collected final metrics for `tune_cbam_eca_strategy_multiscale_e60` and closed Step 9C | Success | Multi-scale: mAP50=0.92477. Fair 60v60 comparison (vs T01@60=0.93807): AdamW +0.32%, Aug +0.10%, Multi-scale -1.42%. |
| 2026-03-29 09:20 | Step 9C->Final Validation | Started `tune_cbam_eca_opt_adamw_e100` | Running | Promoted from fair 60v60 winner to final 100v100 validation against `tune_cbam_eca_t01_e100_auto`. |
| 2026-03-30 09:05 | Step 9C->Final Validation | Collected final metrics for `tune_cbam_eca_opt_adamw_e100` and closed validation | Success | AdamW@100: mAP50=0.95479, mAP50-95=0.81675; vs T01@100 (0.95346/0.81395): +0.14% / +0.34%; promote AdamW as new best candidate. |
| 2026-03-30 09:30 | Mainline | Switched `configs/apple_hyp.yaml` default profile to AdamW | Success | Mainline optimizer now fixed to AdamW (`lr0=0.0012`, `weight_decay=0.01`). |
| 2026-03-30 09:31 | Mainline | Registered reproducibility run `tune_cbam_eca_opt_adamw_e100_seed123` | Success | Seed=123, 100 epochs, no optimizer override outside unified YAML profile. |
| 2026-03-31 09:10 | Mainline | Collected final metrics for `tune_cbam_eca_opt_adamw_e100_seed123` | Success | Seed123: mAP50=0.95292, mAP50-95=0.81568; vs T01@100: -0.06% / +0.21%; reproducibility acceptable. |
| 2026-03-31 09:12 | Mainline | Frozen AdamW as default downstream profile | Success | Mainline freeze decision confirmed by 2-seed 100-epoch validation. |
| 2026-03-31 09:20 | Step 11 | Updated ablation and recommendation reports to closed state | Success | `reports/module_ablation.md` and `reports/final_recommendation.md` now reflect frozen AdamW mainline conclusion. |
| 2026-03-31 09:22 | Step 11 | Synced planning status and report index | Success | `planner/plan.md` marks Step 9 and Step 11 complete; waiting gate is negative-sample dataset delivery. |
| 2026-04-05 17:47 | Module Screening | Started 150-epoch alignment queue for baseline + single-module runs | Running | First launch had python `-c` quoting issue; detected by queue log and stderr. |
| 2026-04-05 17:48 | Module Screening | Fixed `records/screening_to150_queue.ps1` launcher and relaunched queue | Running | Baseline and Edge resume confirmed active; scheduler enforces max concurrency=2 and will continue remaining runs. |
