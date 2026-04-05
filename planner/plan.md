# Planning 阶段交付文档

## 0. 前置结果（已优先完成）
- 已完成无关样本与无关仓库资产清理，详见 `records/cleanup_log.md`。
- 当前 `examples/` 仅保留 `README.md`。
- 已建立功能化目录：`configs/`、`experiments/`、`records/`、`reports/`。

## 1. 本次需求的整体实现方案

### 1.1 总体目标
基于 YOLOv10 单类别苹果检测任务，采用模块化改进与分阶段实验，提升复杂果园场景下的检测鲁棒性与定位精度，同时保持实时性可接受。

### 1.2 约束复述（执行硬约束）
- 仅苹果主体单类别检测，不新增分类任务，不引入果梗显式类别。
- 不做 Docker/Jetson 部署链路改造。
- 不随意修改公共接口（`YOLOv10`、CLI、task_map 主调用链保持兼容）。
- 超参数调整统一在单一 YAML 文件中维护。
- 新依赖边界：默认不新增第三方依赖；若模块实现必须引入新依赖，需先提交兼容性评估与替代方案，再经用户确认后执行。
- 代码保持精简、稳定、可读，注释仅用于关键复杂逻辑。
- 负样本微调放在最后阶段，负样本为空标签数据集。
- 已获用户授权：可执行至“负样本末阶段微调”之前，负样本阶段暂停等待用户提供数据。
- 每完成一个步骤，必须在本计划中追加“resume 小结”，包含过程、结果、分析、下一步。
- 训练过程必须支持 TensorBoard 指标可视化（loss、mAP、Precision、Recall、lr 等核心指标）。

### 1.3 分阶段技术路线
- Phase A: 基线复现实测
  - 使用 `datasets/apple_dataset2510_plus/apple1.yaml` 复现 YOLOv10 基线。
  - 记录 mAP、Recall、Precision、误检样例、速度指标（训练/推理时延）。
  - 同步启用 TensorBoard 日志并校验可视化面板可正常显示。
- Phase B: 单模块筛选实验
  - 多尺度融合方向：BiFPN（已实现）与 ASFF（次优先候选）；AFPN 暂列 P4，需论文与复杂度复核后进入实现。
  - 注意力方向：CBAM、ECA（已实现对照）；RFAConv 暂列 P4，需补全文献与复杂度评估。
  - 边缘增强方向：Sobel/Laplacian 轻量边缘分支（优先于复杂 MSEE）。
  - 每类先做“单模块开关对照”，并执行显著性与效率联合判定，避免在差异接近时误选。
- Phase C: 模块组合实验
  - 从 Phase B 各方向选最佳单模块，做两两组合与三模块组合。
  - 对比精度收益与速度变化，筛出 Pareto 最优组合。
- Phase D: 超参数统一调优
  - 在固定最优模块组合后，进行单 YAML 管理的超参调优。
- Phase E: 负样本末阶段微调
  - 引入空标签负样本集进行短程微调，重点抑制误检。
- Phase F: 消融与总结
  - 输出模块贡献、超参贡献、负样本微调贡献的可复现消融报告。

## 2. 技术选型说明

### 2.1 联网调研扩展后的模块池（学术来源）
以下为已通过公开论文摘要核验的候选方向，按“复杂场景收益/落地成本”综合排序：

- 多尺度融合与特征金字塔
  - BiFPN（EfficientDet, arXiv:1911.09070）：加权双向融合，适合尺度跨度大目标。
  - ASFF（Learning Spatial Fusion, arXiv:1911.09516）：自适应空间融合，抑制跨层冲突。
  - CARAFE（arXiv:1905.02188）：内容感知上采样，提升细节恢复与小目标表达。
  - AFPN：保留为候选方向，但需在实现前再做论文与复杂度复核。

- 注意力与特征选择
  - CBAM（arXiv:1807.06521）：通道+空间注意力，兼容性高，优先实现。
  - ECA-Net（arXiv:1910.03151）：极轻量通道注意力，适合实时性敏感场景。
  - Coordinate Attention（arXiv:2103.02907）：引入方向感知位置信息，利于遮挡定位。
  - Dynamic Head（arXiv:2106.08322）：检测头三维注意力统一建模，作为高收益但中等复杂候选。
  - RFAConv：保留为候选方向，需在实现前补充文献与复杂度评估。

- 小目标与几何适配
  - SPD-Conv（arXiv:2208.03641）：替代部分下采样操作，改善小目标信息保留。
  - DCNv2（arXiv:1811.11168）：增强几何变形建模能力，利于非规则遮挡目标。

- 回归与分类损失
  - GIoU（arXiv:1902.09630）：解决无重叠框优化停滞。
  - DIoU/CIoU（arXiv:1911.08287）：引入中心距离与长宽比，改善回归收敛。
  - Varifocal Loss（arXiv:2008.13367）：IoU-aware 排序优化，抑制密集候选错误排序。

- 边缘增强
  - Sobel/Laplacian（经典算子）作为低风险起点。
  - MSEE 作为扩展候选，待进一步论文复核后决定是否纳入实现。

### 2.2 模块优先级重排（基于风险与实现成本）
- P1（先做）：CBAM / ECA（二选一先行） + BiFPN 或 ASFF（二选一） + Sobel/Laplacian。
- P2（进阶）：Coordinate Attention / CARAFE / DIoU-CIoU 切换。
- P3（条件进入）：Dynamic Head / SPD-Conv / DCNv2 / Varifocal Loss。
- P4（待复核）：AFPN / RFAConv / MSEE。

### 2.3 损失函数策略
- 保持现有 YOLOv10 主损失链路兼容。
- 先做低侵入可配置改造（IoU 变体开关、损失权重微调）。
- 避免早期大改损失实现，优先保证实验可控性。

## 3. 模块拆分与职责定义

### 3.1 代码模块职责
- `ultralytics/nn/modules/`:
  - 新增 `cbam.py`（或并入现有模块文件但保持独立类）。
  - 新增可选边缘增强模块（如 `edge_enhance.py`）。
- `ultralytics/nn/tasks.py`:
  - 注册新模块解析映射，确保 YAML 可直接调用。
- `ultralytics/cfg/models/v10/`:
  - 新增改进模型 YAML（如 `yolov10n_cbam.yaml`、`yolov10n_bifpn.yaml`、`yolov10n_asff.yaml`）。
- `configs/`:
  - 统一超参数文件（单文件）`apple_hyp.yaml`。
- `experiments/`:
  - 存放实验记录表与结果索引。
- `reports/`:
  - 存放消融报告与阶段总结。
- `records/`:
  - 存放清理日志、实验执行日志、配置变更日志。

### 3.2 接口稳定性要求
- 保持 `from ultralytics import YOLOv10` 调用不变。
- 保持 `yolo train/val/predict` 参数接口兼容。
- 新模块通过 YAML 开关控制，不改公共 API 语义。

## 4. 接口设计

### 4.1 模块开关接口（YAML）
- 在模型 YAML 中通过层定义引入模块，例如：
  - `[-1, 1, CBAM, [channels, reduction]]`
  - `[-1, 1, EdgeEnhance, [mode]]`
- 通过不同 YAML 文件实现不同组合，不在训练脚本中硬编码分支。

### 4.2 超参接口（单 YAML）
- 统一文件：`configs/apple_hyp.yaml`
- 统一管理字段：
  - 优化器与学习率
  - box/cls/dfl 权重
  - 数据增强参数
  - 训练轮数、batch、imgsz
- 运行时仅通过该文件覆盖默认配置，禁止多处散落修改。

### 4.3 训练可视化接口（TensorBoard）
- 训练命令与配置需显式开启 TensorBoard 记录。
- 指标至少包含：train/val loss、mAP、Precision、Recall、学习率。
- 日志目录与实验记录关联，保证可追溯到模型 YAML 与超参 YAML。

## 5. 数据结构与目录规范变更

### 5.1 数据集
- 主训练集：`datasets/apple_dataset2510_plus`
- 数据配置：`datasets/apple_dataset2510_plus/apple1.yaml`
- 负样本集：待用户提供（空标签）

### 5.2 记录规范（统一）
- 文档：
  - `planner/research.md`
  - `planner/plan.md`
  - `records/cleanup_log.md`
- 实验：
  - `experiments/baseline.csv`
  - `experiments/module_screening.csv`
  - `experiments/combination_ablation.csv`
  - `experiments/hyperparam_tuning.csv`
  - `experiments/neg_finetune.csv`
- 可视化：
  - TensorBoard 日志沿用 `runs/` 目录，并在 `experiments/*.csv` 记录对应 run 路径。
- 报告：
  - `reports/module_ablation.md`
  - `reports/final_recommendation.md`
- 运行产物：沿用 `runs/`，并在 `experiments/` 记录 run 对应关系。

## 6. 风险点与应对方案
- 风险 1：模块堆叠导致速度明显下降
  - 应对：先单模块评估，设置“收益不足即回退”规则。
- 风险 2：改动 tasks.py 影响通用模型构建
  - 应对：最小侵入、仅新增映射；保留基线 YAML 回归测试。
- 风险 3：负样本引入后召回下降
  - 应对：控制微调轮数与学习率，重点监控 FN/FP 变化。
- 风险 4：超参分散导致不可复现
  - 应对：强制单 YAML；每次实验记录配置快照与commit/时间戳。
- 风险 5：单模块指标差异接近导致误选
  - 应对：引入多种子重复、显著性阈值、速度与误检联合评分；若未形成统计优势，则保持 baseline 并跳过该模块。

## 7. Todo List（顺序、交付物、完成标准）

1. 清理无关文件与目录（已完成）
- 交付物：`records/cleanup_log.md`
- 完成标准：无关 docs/demo/旧实验资产清理完成，核心代码与数据保留。

2. 建立统一实验目录与记录模板
- 交付物：`configs/`, `experiments/`, `records/`, `reports/` 下模板文件
- 完成标准：每次实验可追溯到模型 YAML、超参 YAML、run 路径与 TensorBoard 日志。

3. 模块调研扩展与技术筛选（已完成本轮文档）
- 交付物：本 `plan.md` 的“联网调研扩展模块池”与优先级分层
- 完成标准：候选模块不局限于少量方案，且有公开学术来源支撑。

4. 基线复现实测（已完成）
- 交付物：基线训练/验证结果与误检样例记录
- 完成标准：得到可复现 baseline 指标，且 TensorBoard 可查看完整训练曲线。

5. 实现 CBAM 模块化接口（优先）
- 交付物：CBAM 模块代码 + tasks 注册 + 独立 YAML
- 完成标准：不改公共接口，训练与推理可正常运行。

6. 实现多尺度融合候选模块（BiFPN/ASFF）
- 交付物：模块代码 + 对应 YAML
- 完成标准：单模块实验可跑通并完成与基线对比。

7. 实现边缘增强候选模块（Sobel/Laplacian）
- 交付物：模块代码 + 对应 YAML
- 完成标准：在遮挡/重叠样例上有可观测定位改进趋势。

8. 组合实验与初步选优（已完成）
- 交付物：模块组合对比表
- 完成标准：确定 1 套最优组合进入超参调优。

9. 单 YAML 超参数调优（已完成）
- 交付物：`configs/apple_hyp.yaml` 与调优记录
- 完成标准：所有调优实验仅通过该 YAML 驱动。

10. 负样本末阶段微调
- 交付物：微调结果与误检变化分析
- 完成标准：误检下降且主指标无不可接受退化。

11. 消融总结与阶段报告（已完成）
- 交付物：消融报告与最终推荐配置
- 完成标准：结论可复现、可解释、可交接。

## 8. 阶段状态
- 当前阶段：Step11 收口完成，进入负样本微调等待态。
- 当前执行边界：主线已冻结（AdamW）；后续仅执行主线增量实验与负样本末阶段微调。
- 执行闸门：负样本微调步骤需等待用户提供空标签负样本集后再启动。

## 9. 附录：技术选型模块逐项说明

### 9.1 BiFPN（加权双向特征金字塔）
原理说明：
- BiFPN 通过自顶向下与自底向上的双向路径融合多尺度特征。
- 融合节点使用可学习权重进行归一化加权，自动平衡不同层级特征贡献。
- 适合近大远小尺度跨度大的目标检测场景。

具体修改位置：
- 新增模块实现：`ultralytics/nn/modules/`（建议新文件 `bifpn.py`）。
- 注册解析映射：`ultralytics/nn/tasks.py`。
- 调用入口：`ultralytics/cfg/models/v10/yolov10n_bifpn.yaml`。

实验效果（文献与本项目预期）：
- 文献侧：EfficientDet 报告了较好的精度-效率权衡（arXiv:1911.09070）。
- 本项目预期：
  - 小目标召回提升较明显。
  - 复杂背景下误检有概率下降。
  - 计算量有一定上升，需配合轻量通道策略控制时延。

### 9.2 ASFF（自适应空间特征融合）
原理说明：
- ASFF 在空间位置级别学习各尺度特征的融合权重。
- 核心目标是抑制不同尺度特征之间的冲突信息。

具体修改位置：
- 新增模块实现：`ultralytics/nn/modules/`（建议 `asff.py`）。
- 注册解析映射：`ultralytics/nn/tasks.py`。
- 调用入口：`ultralytics/cfg/models/v10/yolov10n_asff.yaml`。

实验效果（文献与本项目预期）：
- 文献侧：YOLOv3 基线下有较好的速度-精度平衡（arXiv:1911.09516）。
- 本项目预期：
  - 遮挡与重叠区域的定位稳定性提升。
  - 对密集果簇场景更友好。
  - 速度影响通常小于重型融合结构。

### 9.3 CARAFE（内容感知上采样）
原理说明：
- CARAFE 用内容感知的动态重组核替代固定上采样。
- 通过更大感受野聚合上下文，改善细节恢复。

具体修改位置：
- 新增模块实现：`ultralytics/nn/modules/`（建议 `carafe.py`）。
- 在 Neck 上采样节点替换：模型 YAML（如 `yolov10n_carafe.yaml`）。
- 解析映射：`ultralytics/nn/tasks.py`。

实验效果（文献与本项目预期）：
- 文献侧：在检测任务中有稳定增益且开销可控（arXiv:1905.02188）。
- 本项目预期：
  - 果实边界贴合度提升。
  - 小目标纹理恢复更好。
  - 模块实现复杂度中等，优先级低于 CBAM/BiFPN。

### 9.4 CBAM（通道-空间注意力）
原理说明：
- 先通道注意力后空间注意力，对特征进行逐步重标定。
- 能增强目标相关响应，抑制背景干扰。

具体修改位置：
- 新增模块实现：`ultralytics/nn/modules/cbam.py`。
- 导出模块入口：`ultralytics/nn/modules/__init__.py`（若存在集中导出）。
- 注册解析映射：`ultralytics/nn/tasks.py`。
- 调用入口：`ultralytics/cfg/models/v10/yolov10n_cbam.yaml`。

实验效果（文献与本项目预期）：
- 文献侧：检测任务存在一致性增益且额外开销较小（arXiv:1807.06521）。
- 本项目预期：
  - 枝叶背景抑制效果更好。
  - 误检率下降趋势明显。
  - 作为首个实现模块，工程风险最低。

### 9.5 ECA-Net（高效通道注意力）
原理说明：
- 用局部 1D 卷积实现通道交互，避免降维信息损失。
- 以极低参数与 FLOPs 提供通道重标定能力。

具体修改位置：
- 新增模块实现：`ultralytics/nn/modules/eca.py`。
- 注册解析映射：`ultralytics/nn/tasks.py`。
- 调用入口：`ultralytics/cfg/models/v10/yolov10n_eca.yaml`。

实验效果（文献与本项目预期）：
- 文献侧：以极小代价带来稳健提升（arXiv:1910.03151）。
- 本项目预期：
  - 作为实时性优先注意力备选优于更重注意力模块。
  - 精度增益通常小于 CBAM，但速度更友好。

### 9.6 Coordinate Attention（坐标注意力）
原理说明：
- 将位置编码融入通道注意力，分别沿 H/W 聚合。
- 同时保留长程依赖与位置信息。

具体修改位置：
- 新增模块实现：`ultralytics/nn/modules/coordatt.py`。
- 注册解析映射：`ultralytics/nn/tasks.py`。
- 调用入口：`ultralytics/cfg/models/v10/yolov10n_coordatt.yaml`。

实验效果（文献与本项目预期）：
- 文献侧：对下游检测任务有效且开销较低（arXiv:2103.02907）。
- 本项目预期：
  - 遮挡条件下定位更稳定。
  - 对重叠粘连果实的空间区分有帮助。

### 9.7 SPD-Conv（小目标信息保留块）
原理说明：
- 通过 Space-to-Depth + 非步长卷积替换部分下采样。
- 减少早期高频细节损失，强化小目标表达。

具体修改位置：
- 新增模块实现：`ultralytics/nn/modules/spdconv.py`。
- 在 Backbone 下采样节点替换：`ultralytics/cfg/models/v10/yolov10n_spd.yaml`。
- 注册解析映射：`ultralytics/nn/tasks.py`。

实验效果（文献与本项目预期）：
- 文献侧：低分辨率与小目标任务收益明显（arXiv:2208.03641）。
- 本项目预期：
  - 远距离小苹果漏检下降。
  - 开销可能上升，需要配合通道裁剪。

### 9.8 DCNv2（可变形卷积）
原理说明：
- 学习可变形采样偏移与调制，适应几何形变。
- 在遮挡、形变和非规则边界任务中更灵活。

具体修改位置：
- 优先局部替换 Neck/Head 关键卷积层（避免全网替换）。
- 若引入额外依赖，需先确认环境兼容性。
- 调用入口：`ultralytics/cfg/models/v10/yolov10n_dcnv2.yaml`。

实验效果（文献与本项目预期）：
- 文献侧：检测性能显著提升但实现复杂度较高（arXiv:1811.11168）。
- 本项目预期：
  - 重叠与非规则边界场景定位改进。
  - 推理速度可能下降，放在 P3 条件进入。

### 9.9 Sobel/Laplacian 边缘增强
原理说明：
- 通过固定算子提取高频边缘并注入主干特征。
- 强化边界响应，改善模糊边缘回归。

具体修改位置：
- 新增模块实现：`ultralytics/nn/modules/edge_enhance.py`。
- 在 Neck 或 Head 前插入轻量边缘增强块。
- 调用入口：`ultralytics/cfg/models/v10/yolov10n_edge.yaml`。

实验效果（文献与本项目预期）：
- 工程侧：开销低、实现快、可解释性高。
- 本项目预期：
  - 重叠果实分离度提升。
  - 定位框边界贴合度提升。

### 9.10 GIoU / DIoU / CIoU（回归损失族）
原理说明：
- GIoU 解决无重叠时梯度问题。
- DIoU/CIoU 进一步引入中心距离与长宽比约束，加速收敛并提升定位质量。

具体修改位置：
- 损失定义与切换：`ultralytics/utils/loss.py`。
- 参数开关：`configs/apple_hyp.yaml`（单 YAML 统一管理）。

实验效果（文献与本项目预期）：
- 文献侧：回归收敛与定位精度均有改善（arXiv:1902.09630, arXiv:1911.08287）。
- 本项目预期：
  - 边界框回归更稳定。
  - 对边缘模糊目标的 IoU 提升更明显。

### 9.11 Varifocal Loss（IoU-aware 分类监督）
原理说明：
- 将分类置信与定位质量耦合，优化候选排序质量。
- 适合密集候选下减少错误高分框。

具体修改位置：
- 分类损失替换/开关：`ultralytics/utils/loss.py`。
- 可选分支控制：`configs/apple_hyp.yaml`。

实验效果（文献与本项目预期）：
- 文献侧：密集检测任务 AP 提升明显（arXiv:2008.13367）。
- 本项目预期：
  - 复杂背景误检下降。
  - 需关注与 YOLOv10 现有损失结构耦合风险。

### 9.12 Dynamic Head（检测头统一注意力）
原理说明：
- 通过尺度、空间、任务三个维度注意力提升检测头表达能力。

具体修改位置：
- 新增检测头模块并替换原 Head 关键层。
- 改动点涉及 `ultralytics/nn/modules/` 与 `ultralytics/nn/tasks.py`。
- 调用入口：`ultralytics/cfg/models/v10/yolov10n_dyhead.yaml`。

实验效果（文献与本项目预期）：
- 文献侧：具备高收益潜力（arXiv:2106.08322）。
- 本项目预期：
  - 有望提升复杂场景鲁棒性。
  - 工程改动较大，放在 P3 之后评估。

### 9.13 模块实施顺序建议（附录总结）
- 第一批：CBAM 或 ECA、BiFPN 或 ASFF、Sobel/Laplacian。
- 第二批：Coordinate Attention、CARAFE、DIoU/CIoU。
- 第三批：SPD-Conv、Varifocal Loss、DCNv2、Dynamic Head（条件进入）。
- 每个模块均需按“单模块对照 -> 组合实验 -> 超参调优 -> 负样本微调”闭环评估。

## 10. 执行记录（resume）

### Step 1: 仓库清理与目录规范化（已完成）
过程：
- 清理了无关目录与文件，保留核心训练代码、模型配置与数据目录。
- 建立 `configs/`、`experiments/`、`records/`、`reports/` 四类规范目录。

结果：
- 清理记录已写入 `records/cleanup_log.md`。
- 规划文档与实验目录结构满足后续可追溯实验要求。

分析：
- 当前仓库更聚焦检测研发主线，减少了无关资产干扰。
- 仍需补齐实验模板与执行日志模板，才能进入基线与模块实验闭环。

下一步：
- 执行 Step 2，建立统一实验记录模板（含 TensorBoard run 路径、模型 YAML、超参 YAML、关键指标字段）。

### Step 2: 统一实验模板与超参模板（已完成）
过程：
- 新建统一超参文件 `configs/apple_hyp.yaml`，集中维护训练和损失相关关键参数。
- 新建实验记录模板：`experiments/baseline.csv`、`experiments/module_screening.csv`、`experiments/combination_ablation.csv`、`experiments/hyperparam_tuning.csv`、`experiments/neg_finetune.csv`。
- 新建执行记录与报告模板：`records/execution_log.md`、`reports/module_ablation.md`、`reports/final_recommendation.md`。

结果：
- 已形成可追溯实验台账结构，字段覆盖模型配置、超参配置、关键指标与 TensorBoard 路径。

分析：
- 该步骤完成后，后续每次训练都可落表并追溯到具体配置与运行目录。
- 负样本记录模板已预留，但不会提前执行。

下一步：
- 执行 Step 3/5 的实现准备：先完成模块现状核查与最小侵入式模块实现（CBAM YAML、BiFPN、边缘增强、ECA）。

### Step 3: 训练稳定性与关键问题修复（已完成）
过程：
- 修复标签类别越界处理逻辑：将“类别越界即断言失败”改为“自动过滤非法类别标签并继续训练”，避免因少量坏标注中断全流程。
- 启动并验证 TensorBoard 服务监听（6006 端口），确认 run 目录事件文件可被看板读取。
- 校验模型入口：使用 `YOLOv10('ultralytics/cfg/models/v10/yolov10n.yaml')` 实例化并打印核心模型类型，确认训练头为 `v10Detect`。
- 针对 WinError 1455（页面文件不足导致 CUDA DLL 加载失败）添加稳态缓解：设置 `CUDA_MODULE_LOADING=LAZY`（持久化 + 当前会话生效）。

结果：
- 1 epoch 冒烟训练通过（`runs/apple/smoke_fix_e1`），流程可完整走完训练与验证。
- 实际训练模型已确认是 YOLOv10n（核心类型 `YOLOv10DetectionModel`，检测头 `v10Detect`）。
- TensorBoard 已可访问（`127.0.0.1:6006` 正在监听）。
- 标签问题当前主要表现为坐标非归一化异常样本被跳过；类别越界将按新逻辑自动过滤。

分析：
- 命令行出现 “Ultralytics YOLOv8.x” 字样属于上游框架统一日志文案，不代表模型退化为 YOLOv8；以模型结构打印与 head 类型为准。
- WinError 1455 属于系统虚拟内存/内存压力问题，已通过降低加载峰值缓解；若后续重现，需同步增大系统页文件并重启。
- 数据标注仍存在较多坐标越界（非类别越界），会导致样本被忽略，后续应安排数据清洗步骤以提升训练有效样本比例。

下一步：
- 进入 Step 4：基线完整复现实验（非冒烟），并建立更高频监控节奏（训练中按固定间隔同步进度与核心指标）。

### Step 4: 基线完整复现实验（已完成）
过程：
- 使用 `datasets/apple_dataset2510_plus/apple1.yaml` 与统一超参 `configs/apple_hyp.yaml` 执行基线训练。
- 全程采用 `batch=-1`（AutoBatch），并保持 TensorBoard 持续记录。

结果：
- `runs/apple/baseline_bauto_nbs128` 完成 100 epoch，mAP50≈0.958。
- 对照基线 `runs/apple/baseline_preneg_stable` 完成 100 epoch，mAP50≈0.951。

结论：
- 基线有效且可复现，后续模块实验统一以 `baseline_bauto_nbs128` 作为主对照。
- AutoBatch 策略稳定，后续阶段继续强制执行 `batch=-1`。

下一步：
- 进入 Step 5，完成单模块 100 epoch 筛选并输出优先组合候选。

### Step 5: 单模块筛选实验（已完成）
过程：
- 按统一约束完成 Edge、CBAM、ECA、BiFPN 四个单模块长程实验（均为 100 epoch，`batch=-1`）。
- 清理了短跑/错误 run，正式 run 保留在 `runs/apple/`。

结果：
- `runs/apple/screen_yolov10n_edge_e100_auto`：mAP50≈0.954。
- `runs/apple/screen_yolov10n_cbam_e100_auto`：mAP50≈0.957。
- `runs/apple/screen_yolov10n_eca_e100_auto`：mAP50≈0.957。
- `runs/apple/screen_yolov10n_bifpn_e100_auto`：mAP50≈0.956。

结论：
- 本轮单模块均未超过主基线 `runs/apple/baseline_bauto_nbs128`（mAP50≈0.958），差异处于近似区间。
- 进入组合实验的原则从“单指标最优”调整为“结构互补性优先 + 低开销优先”，故先测 BiFPN +（CBAM/ECA）以验证协同增益是否可跨越 baseline。
- 既有 100 epoch 单模块结果不重训、不复训，直接进入下一阶段。

复训测试（当前暂停）：
- 暂停原因：复训测试在会话中多次触发工作区读取卡顿，影响连续执行效率。
- 当前问题：
  - `last.pt`/`best.pt` 的状态完整性（optimizer/scheduler）在不同保存路径下不一致。
  - 复训语义与日志连续性（epoch/lr）验证流程成本较高，且与当前主线目标冲突。
- 改进点（后续需要该功能时再启用）：
  - 固化“仅用未 strip 的 last.pt 进行原生 resume”规范。
  - 复训验证改为独立最小工作区与独立日志目录，避免主会话阻塞。
  - 在 `trainer.py` 增加严格断言开关：缺 optimizer/scheduler 状态即 fail-fast。

下一步：
- 进入 Step 6，执行模块组合训练（每组 150 epoch、一次并行两组）。

### Step 5A: 单模块选优纠偏与实验调整（新增）
背景：
- 单模块相对 baseline 的提升幅度接近或未形成正增益，直接按单次 mAP50 排序存在高误判风险。

调整策略：
- 统一判定指标从“单一 mAP50”升级为“四指标联合”：mAP50-95、Recall、speed_ms_per_img、误检率（FP/img）。
- 增加重复实验：每个单模块至少 3 个随机种子（建议 seed=0/1/2），以均值和标准差判定稳定性。
- 设置显著性门槛：仅当模块在 mAP50-95 上相对 baseline 的均值提升 >= 0.3 个百分点且标准差不重叠时，判定为“有效提升”。
- 设置效率门槛：speed_ms_per_img 退化超过 8% 的模块不进入优先组合。
- 若多个模块均未过门槛：保持 baseline 作为主干，仅保留“低开销模块”参与组合探索。

执行落地：
- 在 `experiments/module_screening.csv` 增加 `seed`、`fp_per_img`、`delta_speed_pct_vs_baseline`、`score` 字段。
- 评分规则（用于同档位 tie-break）：`score = 0.5*delta_map50_95 + 0.3*delta_recall - 0.2*delta_speed_pct`。
- 先补录已完成 run 的实际指标，再决定是否补跑 3-seed 重复实验。

下一步：
- Step 6 持续执行当前组合训练。
- 并行补齐 `experiments/baseline.csv` 与 `experiments/module_screening.csv` 的已完成实验记录，消除“有 run 无台账”的追溯断点。

### Step 6: 模块组合实验（已完成）
执行约束（严格）：
- 严格遵循 research/plan 既定边界：单类苹果检测、统一超参文件、`batch=-1`、不改公共 API。
- 本阶段所有新组合训练总轮次统一为 150 epoch。
- 一次同时运行 2 个组合实验。
- 已完成的 baseline/单模块训练不重训。

调度器与进程安全检查：
- 已核查无活动调度器进程（未发现 `module_scheduler*.ps1` 在运行）。
- 仅保留 TensorBoard 进程，避免后台调度器打断手动并行组合训练。

首批并行组合：
- `ultralytics/cfg/models/v10/yolov10n_bifpn_cbam.yaml` -> `runs/apple/combo_bifpn_cbam_e150_auto`
- `ultralytics/cfg/models/v10/yolov10n_bifpn_eca.yaml` -> `runs/apple/combo_bifpn_eca_e150_auto`

首批结果快照（2026-03-22，已完结）：
- `runs/apple/combo_bifpn_cbam_e150_auto`：已完成 150 epoch，当前 best mAP50≈0.96248。
- `runs/apple/combo_bifpn_eca_e150_auto`：已完成 150 epoch，当前 best mAP50≈0.96073。

下一批两组组合（已完结）：
- `ultralytics/cfg/models/v10/yolov10n_cbam_eca.yaml` -> `runs/apple/combo_cbam_eca_e150_auto`（恢复续跑完成，150/150）。
- `ultralytics/cfg/models/v10/yolov10n_bifpn_edge.yaml` -> `runs/apple/combo_bifpn_edge_e150_auto`（恢复续跑完成，150/150）。

2026-03-24 恢复动作（新增）：
- 已核验两组 run 的 `weights/last.pt` 与 `results.csv` 连续性，均满足原生 `resume=True` 条件。
- 已按“并发上限=2”恢复训练：
  - `combo_cbam_eca_e150_auto` 从 epoch 57 继续；
  - `combo_bifpn_edge_e150_auto` 从 epoch 49 继续。
- 已将当前已完成与进行中组合结果写入 `experiments/combination_ablation.csv`，并与 baseline 对照关系保持一致。

Step 6 最终结果（2026-03-25 收口）：
- `combo_bifpn_cbam_e150_auto`：mAP50≈0.96248，mAP50-95≈0.83293。
- `combo_bifpn_eca_e150_auto`：mAP50≈0.96073，mAP50-95≈0.83147。
- `combo_cbam_eca_e150_auto`：mAP50≈0.96331，mAP50-95≈0.83753。
- `combo_bifpn_edge_e150_auto`：mAP50≈0.96193，mAP50-95≈0.83129。
- `combo_cbam_eca_e150_fresh_direct`：mAP50≈0.96542，mAP50-95≈0.83978（当前最优）。

阶段结论：
- 组合策略成立，且“注意力组合（CBAM+ECA）”在当前数据集上优于“融合+注意力/边缘”路线。
- Step 8（组合实验与初步选优）收口，进入 Step 9 单 YAML 超参数调优。

下一步：
- 以 `ultralytics/cfg/models/v10/yolov10n_cbam_eca.yaml` 为主候选，执行 Step 9 超参数调优。
- 调优仍严格使用 `configs/apple_hyp.yaml` 单文件管理，并把每次变更与 run 路径写入 `experiments/hyperparam_tuning.csv`。

### Step 9: 单 YAML 超参数调优（执行中）
启动动作（2026-03-25）：
- 已将 `configs/apple_hyp.yaml` 切换到 Step9-T01 配置：`lr0/lrf=0.008`、`weight_decay=0.00045`、`box=8.0`、`cls=0.45`、`dfl=1.7`。
- 已登记调优任务：`experiments/hyperparam_tuning.csv` 新增 `tune_cbam_eca_t01_e100_auto`。
- 已启动训练：`runs/apple/tune_cbam_eca_t01_e100_auto`（100 epoch，`batch=-1`，单 YAML 驱动）。

当前状态：
- 训练进程已启动并处于初始化/写盘前阶段；持续监控 `results.csv` 生成与 epoch 进度。
- Step 9 阶段并发硬约束：同一时刻最多 2 个训练进程，超出即暂停/排队，防止训练卡死。

下一步：
- 待 T01 训练结束后回填 `precision/recall/map50/map50_95` 至 `experiments/hyperparam_tuning.csv`。
- 基于 T01 结果决定 T02（学习率或损失权重单变量微调方向）。

### Step 9B: 损失函数改进与非结构策略实验（已完成）
目标：
- 按 Research 要求补充“仅损失函数/训练策略”改进验证，不改网络结构。

已完成改造：
- 在 `ultralytics/utils/loss.py` 增加可配置损失开关：
  - `iou_type`：`iou/giou/diou/ciou`；
  - `cls_loss`：`bce/focal/varifocal`；
  - `focal_gamma`、`focal_alpha`。
- 在 `ultralytics/cfg/default.yaml` 与 `configs/apple_hyp.yaml` 注册上述超参数键（默认保持原行为：`ciou+bce`）。

已启动实验（2026-03-25）：
- `runs/apple/tune_cbam_eca_loss_diou_focal_e60`：`diou + focal`（60 epoch）。
- `runs/apple/tune_cbam_eca_strategy_cosine_ls_e60`：`cos_lr + label_smoothing + close_mosaic`（60 epoch）。
- `runs/apple/tune_cbam_eca_t01_e100_auto` 持续运行（Step9-T01）。

执行结果（2026-03-27 汇总）：
- `tune_cbam_eca_t01_e100_auto`：mAP50≈0.95346，mAP50-95≈0.81395。
- `tune_cbam_eca_loss_diou_focal_e60`：mAP50≈0.36709，mAP50-95≈0.30942（强负向）。
- `tune_cbam_eca_loss_giou_varifocal_e60`：mAP50≈0.48914，mAP50-95≈0.40708（强负向）。
- `tune_cbam_eca_strategy_cosine_ls_e60`：mAP50≈0.93472，mAP50-95≈0.77945（相对 T01 负向）。

阶段结论：
- 在当前数据与配置下，损失函数改造（DIoU+Focal、GIoU+Varifocal）均显著劣于基线损失，不进入下一阶段。
- 非结构训练策略（cos_lr + label_smoothing + close_mosaic 调整）未优于 T01，不进入下一阶段。
- Step9B 收口：保留 `T01` 作为 Step9 输出候选，后续若继续优化，建议回到小步单变量调优（学习率/权重衰减/box-cls-dfl 权重）而非更换分类损失形式。

下一步：
- 持续监控上述 run 的 `results.csv` 落盘与最佳指标。
- 完成后回填 `experiments/hyperparam_tuning.csv`，并比较“损失改进 vs 训练策略改进”的收益差异。

2026-03-26 进展快照（新增）：
- `tune_cbam_eca_t01_e100_auto` 已完成：mAP50≈0.95346，mAP50-95≈0.81395。
- `tune_cbam_eca_loss_diou_focal_e60` 已完成：mAP50≈0.36709，mAP50-95≈0.30942（相对基线显著退化，判定为负向方案）。
- 已停止旧排队器的高频重复拉起行为，改为手动受控并发（上限=2）。
- `tune_cbam_eca_strategy_cosine_ls_e60` 与 `tune_cbam_eca_loss_giou_varifocal_e60` 已重新拉起并进入训练迭代。

当前执行策略：
- 保持最多 2 个训练并发。
- Step9B 已完成，进入 Step 11 消融总结与最终推荐配置整理。

### Step 9C: 非框架参数优化补充实验（已完成）
目标：
- 在不改网络结构和不改框架代码前提下，继续尝试可回滚的参数级优化方法。

新增实验设计（2026-03-27）：
- `tune_cbam_eca_opt_adamw_e60`：优化器切换到 AdamW，`lr0=0.0012`，`weight_decay=0.01`。
- `tune_cbam_eca_aug_mosaic07_e60`：增强策略调整，`mosaic=0.7`，`translate=0.05`，`scale=0.4`，`close_mosaic=15`。
- `tune_cbam_eca_strategy_multiscale_e60`：训练策略调整，`multi_scale=True`，`warmup_epochs=5.0`。

执行约束：
- 并发上限始终为 2。
- 先跑前两组，第三组在槽位释放后启动。
- 所有结果回填到 `experiments/hyperparam_tuning.csv`。

当前状态（2026-03-29 更新）：
- ✅ 三组实验均已完成（60 epochs）：
  - `tune_cbam_eca_opt_adamw_e60`：mAP50=0.94108，mAP50-95=0.79186。
  - `tune_cbam_eca_aug_mosaic07_e60`：mAP50=0.93896，mAP50-95=0.78955。
  - `tune_cbam_eca_strategy_multiscale_e60`：mAP50=0.92477，mAP50-95=0.76839。

公平比较口径说明：
- 若与 `T01@100`（mAP50=0.95346）比，三组都低于基线。
- 若做同预算公平比较（60 vs 60），应与 `T01` 的第60轮对比（mAP50=0.93807）：
  - AdamW：+0.32%（0.94108 vs 0.93807），小幅领先。
  - Augmentation：+0.10%（0.93896 vs 0.93807），基本持平略优。
  - Multi-scale：-1.42%（0.92477 vs 0.93807），明显回退。

阶段结论：
- Step 9C 的公平筛选结果为：`AdamW` 最优，`Augmentation` 次优，`Multi-scale` 淘汰。
- 下一步建议：将 `AdamW`（可选加 `Augmentation`）拉齐到 100 epochs 再与 `T01@100` 做最终决策。

推进执行（2026-03-30 更新）：
- `tune_cbam_eca_opt_adamw_e100` 已完成 100 epochs 终局验证。
- 与 `T01@100` 对比结果：
  - mAP50：0.95479 vs 0.95346（+0.14%）
  - mAP50-95：0.81675 vs 0.81395（+0.34%）

Step 9C 终局结论：
- 在公平口径（100v100）下，AdamW 方案已超过 T01 基线，成为当前最佳候选。
- 建议将 AdamW 配置（`optimizer=AdamW, lr0=0.0012, weight_decay=0.01`）固化为下一阶段默认训练配置。

主线固化与继续推进（2026-03-30）：
- 已将统一超参文件 `configs/apple_hyp.yaml` 的活动配置切换为 AdamW 主线。
- 已登记复现实验 `tune_cbam_eca_opt_adamw_e100_seed123`（100 epochs）。
- 推进策略：以 AdamW 为唯一默认训练入口，后续实验若无特别说明不再显式覆盖优化器参数。

复现验证与冻结决策（2026-03-31）：
- `tune_cbam_eca_opt_adamw_e100_seed123` 已完成 100 epochs：
  - mAP50=0.95292，mAP50-95=0.81568。
- 与 `T01@100` 对比：
  - mAP50：-0.06%（0.95292 vs 0.95346，几乎持平）
  - mAP50-95：+0.21%（0.81568 vs 0.81395）
- 与 AdamW seed42（0.95479/0.81675）对比，波动在可接受范围内，结论稳定。

当前决策：
- 正式冻结 AdamW 主线配置，用于后续训练/验证默认入口。
