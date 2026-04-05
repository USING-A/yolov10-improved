# Module Ablation Report

## Summary
- Completed (2026-03-31).

本报告汇总模块筛选、组合实验与非结构参数调优的核心结论，用于支持主线冻结决策。

## Records
- Baseline (100 epochs)
	- run: tune_cbam_eca_t01_e100_auto
	- mAP50: 0.95346
	- mAP50-95: 0.81395

- Single-module and combination screening (completed)
	- 当前最优结构候选来自 cbam_eca 路线，已进入后续超参调优阶段。

- Hyperparameter tuning (completed)
	- tune_cbam_eca_opt_adamw_e100: mAP50=0.95479, mAP50-95=0.81675
	- tune_cbam_eca_opt_adamw_e100_seed123: mAP50=0.95292, mAP50-95=0.81568
	- 结论：AdamW 主线在双种子 100 epochs 下稳定，综合优于或接近 baseline。

- Non-structure strategy outcomes
	- multi_scale: 未带来收益（60 epochs 下明显回退），不进入主线。
	- 过度损失改造（diou+focal, giou+varifocal）显著退化，淘汰。

- Negative fine-tuning
	- 状态：待用户提供空标签负样本数据后执行。

## Final Ablation Conclusion
- 主线冻结：AdamW + 当前 cbam_eca 结构配置。
- 默认超参入口：configs/apple_hyp.yaml。
- 后续新增实验应作为该冻结主线的增量对比，不再回到旧 SGD 主线。
