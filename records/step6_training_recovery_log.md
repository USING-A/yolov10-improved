# Step 6.1: 训练中断事件与恢复（2026-03-22 23:15 ~ 23:27）

## 事件描述

`combo_cbam_eca_e150_auto` 和 `combo_bifpn_edge_e150_auto` 两个训练进程在初始化完成后卡死。

### 卡死现象
- 两个训练的 results.csv 在北京时间 17:13:36 ~ 17:13:56 停止更新
- `cbam_eca` 卡在 epoch 56，`bifpn_edge` 卡在 epoch 48
- 从离线到发现（约 6 小时），进程仍占用 GPU 内存（750MB）但无计算活动

## 排查过程

1. **资源检查**  
   - 磁盘空间充足（229GB 剩余）
   - GPU 正常（RTX 4070, 12GB）

2. **模块路径问题发现**  
   - CLI 调用 `yolo detect train` 返回 `KeyError: 'CBAM'`
   - pip 安装的 ultralytics 包（8.1.34）不包含自定义 CBAM/ECA 模块
   - 本地自定义模块在 `ultralytics/nn/modules/conv.py` 中，但 pip 包无此模块

3. **根本原因**  
   YOLOv10 导入路径冲突：
   - 默认 Python 加载 pip 包而非本地库
   - 当模型 YAML 引用 CBAM 时，pip ultralytics 无法解析该模块
   - 错误被隐含在后续初始化阶段，导致卡死而非立即报错

## 恢复措施

1. **进程清理**  
   - 杀死僵尸进程 PID 29756（cbam_eca）和 36392（bifpn_edge）

2. **新训练启动**  
   - 创建 `train_with_local_ultralytics.py` 脚本
   - 显式设置 `sys.path.insert(0, 'd:/Github Code/yolov10-improved')`
   - 确保本地 ultralytics 优先加载

3. **新训练状态**  
   - 启动 `combo_cbam_eca_e150_fresh_direct`（fresh start，不 resume）
   - 时间：2026-03-22 23:27
   - 当前进度：Epoch 2 运行中
   - Loss 收敛正常

## 关键修复点

```python
# 在所有后续训练脚本中必须执行
import sys
sys.path.insert(0, r'd:\\Github Code\\yolov10-improved')  # Windows 路径

# 仅在本地开发时需要，生产环境应改为正确的 pip 包安装或打包方式
```

## 后续防范措施

1. **所有组合训练脚本**都应使用本地 ultralytics 路径修复
2. **避免使用 CLI 直接调用**（yolo 命令），改为 Python API 并指定本地路径
3. **监控机制**：定期检查 results.csv 更新时间戳，≥6 小时无更新即告警

## 进度更新（2026-03-25 收口）

| 运行名称 | 目标 Epoch | 当前状态 | Best mAP50 | 备注 |
|---------|----------|--------|-----------|------|
| combo_bifpn_cbam_e150_auto | 150 | ✅ 完成 | 0.96248 | 已完成 |
| combo_bifpn_eca_e150_auto | 150 | ✅ 完成 | 0.96073 | 已完成 |
| combo_cbam_eca_e150_auto | 150 | ✅ 完成 | 0.96331 | 恢复续跑完成 |
| combo_bifpn_edge_e150_auto | 150 | ✅ 完成 | 0.96193 | 恢复续跑完成 |
| combo_cbam_eca_e150_fresh_direct | 150 | ✅ 完成 | 0.96542 | 本地路径 fresh 直训，当前最优 |

## 收口结论

1. 训练中断根因已定位并规避：后续统一使用本地 ultralytics 路径优先策略进行训练。
2. Step 6 目标已达成：四个主组合均完成 150 epoch，且指标均已写入 `experiments/combination_ablation.csv`。
3. 组合阶段当前最优为 `combo_cbam_eca_e150_fresh_direct`（mAP50=0.96542，mAP50-95=0.83978）。
4. 下一阶段建议：进入 Step 9 单 YAML 超参数调优，并以 `yolov10n_cbam_eca.yaml` 为主候选配置。
