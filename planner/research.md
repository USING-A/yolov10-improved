# Research 阶段交付文档

## 1. 项目整体架构与技术栈说明

### 1.1 项目定位
本仓库是基于 Ultralytics 框架扩展的 YOLOv10 工程，核心目标是提供从训练、验证、推理到多后端导出与部署的完整闭环。当前代码已具备 YOLOv10 检测任务的一体化能力，可作为面向果园苹果采摘机器人 2D 视觉感知的改造基线。

### 1.2 分层架构（代码视角）
- 配置与入口层
  - CLI 统一入口：yolo 命令（ultralytics/cfg/__init__.py）
  - 默认超参数配置：ultralytics/cfg/default.yaml
  - 模型结构配置：ultralytics/cfg/models/v10/yolov10{n,s,m,b,l,x}.yaml
- 任务编排层
  - 通用模型封装：ultralytics/engine/model.py (Model 类)
  - 通用训练器：ultralytics/engine/trainer.py (BaseTrainer)
  - 通用预测器：ultralytics/engine/predictor.py (BasePredictor)
  - 通用导出器：ultralytics/engine/exporter.py (Exporter)
- 模型与算子层
  - 网络构建与任务映射：ultralytics/nn/tasks.py
  - 多后端推理适配：ultralytics/nn/autobackend.py
  - YOLOv10 专有任务实现：ultralytics/models/yolov10/{model,train,val,predict}.py
- 数据与评估层
  - 数据加载与增强：ultralytics/data/*
  - 验证与指标计算：验证器 + engine/results.py（结果封装）
- 工程支撑层
  - 测试：tests/*
  - 文档：docs/* + mkdocs.yml
  - 示例与扩展：examples/*

### 1.3 技术栈
- 语言与框架
  - Python 3.8+
  - PyTorch 2.x + torchvision
  - Ultralytics engine（本仓库内）
- 推理/部署生态
  - ONNX / ONNXRuntime
  - TensorRT（.engine）
  - OpenVINO / CoreML / TensorFlow / NCNN / Paddle（通过 AutoBackend & Exporter）
- 常用依赖
  - opencv-python, numpy, scipy, matplotlib, pandas, pyyaml, tqdm
  - 训练与评测在本地 Python 环境执行，不以 Docker 作为本次任务前提

### 1.4 本次任务边界（已对齐）
- 本轮研发不考虑 Docker 与 Jetson 部署链路，不做部署侧改造。
- 改造范围聚焦网络结构与损失函数等算法侧模块。
- 改造目标是在提升复杂果园场景检测效果的同时，尽量避免实时性显著下降。
- 对与本项目无关的示例样本执行“先核验后清理”，严禁误删训练/评测必需程序。

## 2. 核心模块功能与业务逻辑拆解

### 2.1 YOLOv10 专有模块
- ultralytics/models/yolov10/model.py
  - 定义 YOLOv10 类，继承通用 Model，并通过 task_map 绑定检测任务的 model/trainer/validator/predictor。
- ultralytics/models/yolov10/train.py
  - YOLOv10DetectionTrainer 继承 DetectionTrainer，重写 get_validator/get_model。
- ultralytics/models/yolov10/val.py
  - YOLOv10DetectionValidator 重写 postprocess，支持 one2one 输出路径与 v10postprocess。
- ultralytics/models/yolov10/predict.py
  - YOLOv10DetectionPredictor 重写 postprocess，将输出统一为 [x1, y1, x2, y2, conf, cls]。

业务意义：
- 通过 YOLOv10 的 one2one/one2many 训练-推理逻辑，减少对传统 NMS 流程依赖，提升端到端实时性潜力。
- 为后续果园场景的困难样本优化提供了可扩展改造点（训练器、验证器、后处理均已模块化）。

### 2.2 通用训练流程（当前基线）
- 入口：yolo train task=detect model=... data=...
- BaseTrainer 关键步骤
  - 解析参数并保存 args.yaml
  - 校验数据集（check_det_dataset）
  - 构建模型与 dataloader
  - 配置 AMP、优化器、学习率调度、EMA、早停
  - 执行训练/验证循环并输出 results.csv、权重文件

### 2.3 通用推理流程（当前基线）
- 入口：yolo predict 或 Python API
- BasePredictor 关键步骤
  - setup_model -> AutoBackend 自动后端识别
  - preprocess（LetterBox、归一化、dtype 处理）
  - inference
  - postprocess（YOLOv10 分支在专有 predictor 中）
  - 输出 Results 对象（含 boxes、速度统计等）

### 2.4 导出与部署流程（当前基线）
- 入口：yolo export format=onnx/engine/...
- Exporter 会执行
  - 设备与参数检查
  - 模型融合 fuse
  - dry run
  - 按格式导出，并记录 metadata
- 本阶段定位
  - 保留该能力作为通用工程能力说明，但不作为本轮开发重点。

### 2.5 与本次需求的模块改造位点
针对你提出的 3 个核心技术需求，当前可直接扩展的关键点如下：
- 鲁棒性提升（复杂非结构化果园场景）
  - 模型结构：ultralytics/cfg/models/v10/*.yaml + ultralytics/nn/modules/*
  - 特征融合/注意力：tasks.py 中 parse_model 可挂接新模块
  - 难样本策略：trainer/损失函数侧（ultralytics/utils/loss.py + 训练流程）
- 高精度边界定位
  - 回归损失：ultralytics/utils/loss.py（IoU 系列可扩展）
  - 后处理与阈值策略：yolov10/predict.py, yolov10/val.py
- 轻量化与实时性
  - 模型规模：yolov10n/s 及结构替换
  - 导出与推理后端：engine/exporter.py + nn/autobackend.py
  - 训练侧/推理侧速度评测脚本与配置优化

### 2.6 可行改进方法池（学术已有验证，供 Planning 阶段选型）
- 网络结构改进
  - 在 Neck/Backbone 增加轻量注意力：如 EMA、ECA、CBAM、Coordinate Attention。
  - 跨尺度融合增强：如 BiFPN/加权特征融合思想（按 YOLO 结构做轻量适配）。
  - 边缘与细节增强分支：引入轻量上下文增强模块，提高遮挡与粘连场景可分性。
  - 轻量卷积替换：局部层尝试 GSConv/Depthwise 变体，平衡精度与速度。
- 损失函数改进
  - 边框回归：CIoU/DIoU/SIoU/EIoU/WIoU 系列对比，优先考虑定位稳健性。
  - 分类损失：Focal/Varifocal 思路用于困难样本重加权，抑制复杂背景误检。
  - 动态损失权重：根据训练阶段调整 box/cls/dfl 权重，改善收敛与鲁棒性。
- 训练策略改进
  - 难样本挖掘与重采样：面向逆光、遮挡、密集重叠样本提高训练占比。
  - 多尺度与分阶段微调：先稳态收敛后强化困难场景，减少训练震荡。
  - 超参数搜索与消融：统一实验模板进行组合对比，控制变量选最优。

### 2.7 模块化接口约束（以 CBAM 为例）
- 不随意改动公共接口：保持现有 `YOLOv10`、`task_map`、训练/验证/推理入口调用方式不变。
- 新增模块采用可插拔方式：
  - 在 `ultralytics/nn/modules/` 中独立实现 `CBAM` 模块；
  - 在 `ultralytics/nn/tasks.py` 注册解析映射；
  - 在模型 YAML 中通过模块名启用/关闭，避免硬编码耦合。
- 接口适配要求：
  - 输入输出张量形状与现有层兼容；
  - 关闭模块时保持基线图结构可回退；
  - 增加必要注释说明模块作用和插入位置。

## 3. 现有接口、数据结构与依赖关系说明

### 3.1 对外接口
- CLI 接口
  - yolo train/val/predict/export/track/benchmark
  - 参数来源 default.yaml，可通过 arg=value 覆盖
- Python API
  - from ultralytics import YOLOv10
  - model.train(), model.val(), model.predict(), model.export(), model.push_to_hub()

### 3.2 关键数据结构
- 配置对象
  - get_cfg() 将 YAML/字典合并为命名空间对象
- 检测预测结构
  - 常用形态为 Nx6: [x1, y1, x2, y2, conf, cls]
  - YOLOv10 在预测/验证中支持从 one2one 分支恢复该标准格式
- 结果对象
  - Results（engine/results.py）承载图像级预测、可视化与速度信息
- 训练产物
  - runs/.../weights/{best.pt,last.pt}
  - runs/.../results.csv
  - runs/.../args.yaml

### 3.3 核心依赖关系
- YOLOv10 类 -> engine.Model（统一生命周期）
- engine.Model -> trainer/validator/predictor/exporter（按 task_map 装配）
- predictor -> AutoBackend（按权重格式自动绑定后端）
- exporter -> 多框架后端工具链（onnx/tensorrt/openvino/tf 等）
- trainer -> data + loss + optim + scheduler + callbacks

### 3.4 当前与“模块化组合实验”需求的匹配度
- 已满足部分能力
  - 模型结构 YAML 可插拔
  - 训练/验证/预测器可按任务替换
  - 导出链路统一
- 尚未形成完整实验平台化能力
  - 缺“模块组合编排配置”与“自动批量实验脚本”
  - 缺“组合对比结果自动汇总/可追溯归档规范”
  - 缺“消融实验模板化目录与命名规则”

### 3.5 数据集与路径约束（用户已确认）
- 目标数据集路径（已核验）：D:/Github Code/yolov10-improved/datasets/apple_dataset2510_plus
- 数据集 YAML：D:/Github Code/yolov10-improved/datasets/apple_dataset2510_plus/apple1.yaml
- 数据划分目录：train/valid/test（注意：当前数据集为 `valid` 命名而非 `val`）
- 任务类别约束：仅检测苹果主体（单类别），不引入果梗显式检测类别，不做多类别分类任务。

### 3.6 无关示例样本核验与清理约束（用户新增）
- 采用白名单优先原则：先确认训练、评测、文档、依赖脚本的必要性，再标记可清理项。
- 清理流程固定为：
  - 第一步：生成待清理清单（含路径、用途判断、风险等级）；
  - 第二步：与你逐项确认；
  - 第三步：执行删除并输出删除记录。
- 禁止直接批量删除 `examples/` 或其他目录，除非已通过清单确认。

## 4. 项目开发规范与运行环境要求

### 4.1 代码规范与工程约束
- 许可协议：AGPL-3.0
- 文档字符串建议：Google 风格（见 CONTRIBUTING.md）
- 配置驱动优先：default.yaml + 命令行覆盖
- 最小改动原则：优先在任务扩展点新增模块，减少对公共底座侵入
- 代码质量约束：
  - 非必要不增加复杂冗余逻辑；
  - 代码保持精简、可读、稳定；
  - 对非直观逻辑添加简洁注释，避免无意义注释。

### 4.2 运行环境
- 基础环境
  - Python 解释器：D:/Anaconda/envs/yolov10/python.exe
  - 使用该解释器执行训练、验证、推理与基线测试
  - pip install -r requirements.txt
  - pip install -e .

### 4.3 超参数与配置管理约束（用户新增）
- 超参数统一集中在单一 YAML 文件维护（后续在 Planning 阶段确定具体文件路径和字段规范）。
- 禁止在多个脚本中散落硬编码超参数，避免实验不可复现。
- 所有实验运行命令均引用同一超参 YAML，并在日志中落档。

### 4.4 测试与验证
- pytest 基础测试覆盖 train/val/predict/export CLI 链路
- 慢测可用 --slow 开关
- 当前测试集偏通用功能回归，尚未覆盖“果园场景专项指标”（遮挡召回、低光波动、误检率、边界像素误差等）
- 本次约束：基线效果需要由我先完成实测并形成可复现实验记录，作为后续改进对照基准。
- 正负样本策略约束（用户确认）：
  - “匹配机制优化”不在当前阶段直接改动标签分配机制；
  - 在完成模块选优与超参数选优后，再引入负样本数据集做末阶段微调；
  - 负样本数据为全空标签样本，用于抑制误检并提升复杂背景鲁棒性。
- 实时性约束口径（用户确认）：
  - 不设置强制量化阈值；
  - 以“改进模块应具备实时性优化预期并体现一定效果”为验收导向。

### 4.5 文档与交付管理
- 文档站点基于 MkDocs
- 本项目记录规范（必须统一落地）：
  - 设计文档：`planner/` 下维护 `research.md`、`plan.md`；
  - 数据与实验记录：统一目录、统一命名、统一字段，不允许散乱存放；
  - 训练/评测结果：保留参数、版本、指标、时间戳与结论摘要；
  - 模块组合与消融结果：可追溯到具体配置与代码版本。

## 5. 完整理解总结与认知对齐说明

### 5.1 我对项目现状的理解
- 这是一个可运行的 YOLOv10 工程基线，具备完整训练-推理-导出能力，并已为多后端部署预留了统一抽象。
- YOLOv10 在此仓库中并非“单文件魔改”，而是通过 task_map + trainer/validator/predictor 子类化实现，与 Ultralytics 主干兼容性较好。
- 从工程可扩展性看，当前非常适合做“模块化改进 + 组合实验 + 消融分析”的系统化研发，但仍需要新增实验编排与归档层。

### 5.2 与你本次任务目标的对齐结论
- 你的目标（鲁棒性、边界精度、训练/推理实时性）与现有代码架构是可对齐的。
- 三类目标可以分别落在以下改造域：
  - 鲁棒性：特征提取/融合与难样本训练机制
  - 定位精度：回归损失与后处理策略
  - 实时性：轻量结构与损失/训练策略取舍，约束精度收益下的速度回退
- 你新增的“模块化组合 + 对比训练 + 超参调优 + 消融 + 文件管理留档”需求，目前需要在工程层补齐专门的实验管理方案。

### 5.3 已知信息缺口（进入 Planning 前需补齐）
为保证下一阶段 plan.md 可直接落地，以下信息仍需你确认：
- 允许引入的新依赖边界（是否可增加额外 attention/lightweight block 依赖）

### 5.4 会话连续性约束（已对齐）
- 长对话过程中采用“阶段性 resume 复述”机制：在每个阶段切换或关键决策点，输出当前结论、未完成项与下一步。
- 在未得到你确认前，保持 Research 阶段，不进入 Planning。

### 5.5 本轮补充背景的最终对齐结论
- 目标对象：仅苹果主体，单类别检测任务。
- 正负样本策略：负样本微调作为最后阶段步骤执行（在模块与超参确定后进行）。
- 实时性：采用软约束，不做硬性数值门槛，但要求改进模块具备效率可接受性。
- 其余已提炼对象与难点经用户确认准确，可作为 Planning 阶段输入。

---

以上为 Research 阶段认知对齐文档。
请你先人工校验并指出偏差；在你确认“认知一致”之前，我不会进入 Planning 阶段。
