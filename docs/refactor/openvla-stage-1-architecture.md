# OpenVLA 第一版重构架构与验收

记录日期：2026-07-23

最终验收：2026-07-24，已完成

## 1. 范围与决策

本轮只重构 OpenVLA 的 LIBERO 鲁棒性评估入口：

`openvla/experiments/robot/libero/attack_openvla.py`

采用以下决策：

1. 论文与代码冲突时，以重构前提交 `b4c2e7c` 的运行行为为准。
2. 不要求兼容旧 Python 内部 interface；命令行实验流程、关键数值语义和
   Attack Artifact 结构需要保持。
3. OpenVLA、OpenVLA-OFT 和 π0 的模型专用实现暂不合并。OpenVLA 先建立可理解、
   可测试的 module 划分，成熟设计再迁移。
4. TAAO、EoT 等论文中缺失的算法不在本轮推测实现，只保留与其语义对应的 seam。
5. 新 module 使用明确类型、中文注释，并记录可确定 tensor/array shape。

## 2. 入口现在负责什么

`attack_openvla.py` 现在只承担实验编排：

- 解析 `GenerateConfig`
- 解析 object asset 与 benchmark task
- 创建模型、processor、renderer 和 run 级依赖
- 决定训练新纹理或加载已有纹理
- 遍历 task/episode
- 统计成功率、写评估文本并调用正式 rollout 视频保存
- 在 `finally` 中关闭 Runtime Asset Transaction、日志和 W&B

模型 token 语义、可微渲染、帧采集、优化、live-test、资源恢复和产物序列化
均不再实现在入口中。

## 3. Module 划分

| Module | 主要 interface | 隐藏的实现知识 |
|---|---|---|
| `assets.py` | `OBJECT_ASSETS`, `parse_mesh_scale` | Object/Suite/task 映射与 XML mesh scale |
| `configuration.py` | `GenerateConfig`、CLI 类型收窄 | 无运行时副作用的 Draccus schema |
| `spectral_geometry.py` | `SpectralBasisData`、谱基生成/校验 | CPU 曲面谱几何与严格拓扑映射 |
| `texture_parameterization.py` | Geometry Vertex / Spectral adapters | 可学习参数到 Surface Delta 的统一 seam |
| `state_selection.py` | `InitialStatePartition` | 训练与 held-out 评估初始状态隔离 |
| `action_codec.py` | `decode_action_from_generated_ids` | OpenVLA action token、bin center 和反归一化 mask |
| `scene.py` | `find_target_body_pose`, `compute_render_mvp` | MuJoCo body 查询、相机矩阵和去目标背景 |
| `renderer.py` | `DifferentiableRenderer` | mesh/UV、顶点扰动、nvdiffrast、光照校准、texture bake/加载 |
| `compositing.py` | `render_and_composite`, `build_single_view_samples` | NHWC/NCHW 转换、mask 合成和单视图构造 |
| `frame_collection.py` | `TrainingFrameCollector.collect` | 环境推进、clean forward、抓取窗口和光照校准 |
| `objective.py` | `get_attack_loss` | action token 选择与当前 untargeted token 目标 |
| `vision_features.py` | `extract_siglip_patch_features` | 按 checkpoint 配置定位共享 SigLIP 分支并校验 feature shape |
| `optimization.py` | `AttackOptimizer.optimize` | frame batch、view loss、反向传播、SignSGD、日志和 callback 调度 |
| `training.py` | `AttackTrainer.train` | 采帧、优化、live rollout 与终态产物的 task 级数据流 |
| `evaluation.py` | `LiberoEpisodeRunner.run` | 单 episode 状态机、MuJoCo 策略图像、动作后处理和环境关闭 |
| `artifacts.py` | `AttackArtifactStore` | run 目录、历史文件名及 PT/NPY/PNG/MP4 序列化 |
| `runtime_assets.py` | `RuntimeAssetTransaction` | Clean Asset 备份、Active Texture 注入、恢复和 signal/atexit |

删除上述任一深 module 后，其知识会重新散落到入口或多个调用点；它们不是单纯的
转发封装。

## 4. 数据流

### 4.1 Attack Training

```text
LIBERO task + candidate initial states
                 │
                 ▼
       TrainingFrameCollector
                 │
                 │ list[TrainingFrame]
                 ▼
          AttackOptimizer
          ├─ FrameBatchSampler
          ├─ ViewSampler
          ├─ OpenVLA forward
          ├─ action/feature loss
          └─ SignSGD update adv_noise
                 │
        callback │ every N iterations
                 ▼
       LiberoEpisodeRunner (live)
                 │
                 ▼
       AttackArtifactStore
```

`TrainingFrame` 不是普通图像。主要字段包括：

- `bg_tensor`：float32 NCHW `[1, 3, render_height, render_width]`
- `bg_tensor_no_obj`：同 shape，或 `None`
- `mvp`：float32 `[4, 4]`，目标 body 缺失时为 `None`
- `model_rot`：float32 `[3, 3]`
- `clean_output_ids`：整数 `[1, generated_sequence_length]`
- `clean_hidden`：`[1, generated_sequence_length, hidden_size]`
- SigLIP/DINO mean/std：float32 `[1, 3, 1, 1]`
- clean/executed action：NumPy `[action_dim]`

### 4.2 正式评估

```text
task initial state
       │
       ▼
LiberoEpisodeRunner
  ├─ wait dummy actions
  ├─ MuJoCo camera frame（已加载 Active Texture）
  ├─ 同源 resize → OpenVLA policy image
  ├─ 高分辨率源帧 → rollout video
  ├─ OpenVLA action
  ├─ gripper binarize + invert
  └─ env.step / close
       │
       ▼
RolloutResult → success statistics + rollout video
```

训练期间 live-test 与正式评估共享同一个 episode 状态机，仅分辨率、最大步数
和产物命名不同。可微 renderer 只服务 Attack Training；live-test 和正式评估
均通过新建 LIBERO 环境读取 XML 中已经激活的纹理，不再删除 MuJoCo 物体并用
nvdiffrast 重画。这保证策略、录像和物理场景共享同一个渲染来源。

### 4.3 Active Texture 生命周期

```text
begin run
  → backup Clean Asset
  → restore before each task
  → activate loaded/trained Attack Artifact
  → rollout
  → final/signal/atexit restore
  → delete backup
```

live-test 只修改 XML 引用；新建 MuJoCo 环境会从该引用加载当前纹理。正式
loaded/trained texture 在真实纹理原本存在时仍同步镜像，兼容资产内部可能保留
的直接文件引用。

## 5. 当前数值与行为约束

- action attack loss 只选择 action token，目标 token 为 `255 - clean_bin`。
- 没有 action token 时返回可反向传播的零 loss。
- total loss 保持
  `alpha_action * action_loss + alpha_feature * feature_loss`。
- feature loss 保持最后一层 hidden state 的负 MSE。
- 默认 frame batch 为随机无放回、按抽取 batch 大小等权。
- 默认每个有效 frame 只构造一个 adversarial view。
- 优化更新保持 `adv_noise -= attack_lr * sign(gradient)`。
- OpenVLA rollout 的 gripper action 先二值化，再反转符号。
- renderer 未显式提供 position offset 时使用精确零偏移；特殊资产只能在完成
  MuJoCo/nvdiffrast 轮廓测量后显式传入非零 offset。
- live-test 和正式评估的策略输入均来自 MuJoCo 相机帧，不使用可微 renderer
  合成图；录像保存同一帧的高分辨率版本。
- `mvp=None` 时 compositor 不调用 renderer，直接使用背景。
- `.pt` 加载的是 `[num_vertices, 3]` 无界参数；PNG 会经 UV 采样和 `atanh`
  转为同一参数化。
- Clean Asset 在每个 task 前和任何正常/信号退出路径上恢复。

## 6. 为论文缺失模块保留的 seam

### 6.1 TAAO

`optimization.FrameBatchSampler` 接收完整 `TrainingFrame` 池，并返回
`WeightedTrainingFrame`。默认 adapter 是随机等权采样。

未来实现 TAAO 时应：

1. 扩展 `TrainingFrame`，显式保存 trajectory/state/step 边界；
2. 从连续 visual hidden state 计算 latent velocity/acceleration；
3. 实现新的 FrameBatchSampler adapter，输出 criticality-aware frame 和权重；
4. 用新测试替换“随机等权”假设。

当前没有实现 latent dynamics、criticality score、temperature softmax。

### 6.2 EoT / 多视角

`optimization.ViewSampler` 把一个有效 Training Frame 转换为一个或多个可微
adversarial image。默认 adapter 是 `build_single_view_samples`。

未来 EoT adapter 必须：

- 返回至少一个 float NCHW tensor
- 保持到 renderer 参数的梯度
- 明确采样的相机、光照和几何扰动分布
- 增加多视图 loss 聚合测试

当前没有实现 EoT 或论文中的真实世界迁移增强。

## 7. 测试与验收

### 无 GPU 回归

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 \
NUMBA_DISABLE_JIT=1 \
/home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python \
  -m pytest -q tests
```

测试直接覆盖 module interface；重构初期用于导入单体脚本的 characterization test
已经删除。

### GPU 验收

GPU smoke 命令与通用验收条件见
[openvla-stage-1-baseline.md](openvla-stage-1-baseline.md)。本轮已在
`tex3d-openvla` 环境、Spatial task 0 上验证：

- checkpoint 与 processor 加载
- LIBERO/EGL 环境创建
- nvdiffrast forward/backward
- 一轮纹理优化
- Attack Artifact 落盘
- 正式 rollout
- XML/真实纹理最终恢复

上述结果先后覆盖优化器、Attack Artifact 和 Runtime Asset Transaction 的
里程碑提交。2026-07-24 又在收尾提交 `f6cf29d` 上完成最终 GPU smoke：

- 一轮 Attack Training 正常结束
- `live_test_every_n_iters=1` 成功触发共享的 live episode 状态机
- 一个正式评估 episode 正常结束
- Attack Artifact 正常写入
- 最终 XML/真实纹理恢复正常

因此 OpenVLA 第一版重构的无 GPU 回归、入口导入和真实 GPU 流程均已验收通过。

### 完整 Spatial task 0 回归

2026-07-24 在最终重构代码上又完成了一次非 smoke 的完整流程：

- run ID：
  `spatial_task0_attack_5000_refactor_v1-EVAL-libero_spatial-2026_07_24-00_07_12`
- 优化执行 5000 次，loss history 和 gradient log 均有 5000 条有效记录；
- 所有 loss、gradient norm 和保存的对抗参数均为有限值，没有 NaN/Inf；
- total loss 从 `1.510287` 降至 `-1.394392`；
- action loss 从 `20.459118` 降至 `8.568576`；
- feature loss 从 `-0.535352` 降至 `-2.251562`；
- gradient norm 范围为 `[0.193309, 0.330754]`，没有梯度消失；
- 保存的无界参数 shape 为 `[21932, 3]`，经 `tanh * (128 / 255)`
  后逐通道扰动严格位于 `[-128/255, 128/255]`；
- UV Map 和最终 Attack Texture 均为 `4096 x 4096` RGB 图像，内容一致；
- 正式评估完成 50 个非空 rollout，25 次成功，最终成功率为 `50.00%`；
- XML 与真实纹理均已恢复，且没有遗留本次运行的 clean backup。

该成功率与基线文档记录的一次历史 Spatial task 0 完整运行相同。由于同配置的
历史结果存在随机波动，本次验收的核心结论是：重构后的完整训练、产物落盘、正式
评估和资源恢复数据流均正常，且攻击优化的数值趋势合理。

需要注意：该运行发生在 renderer 对齐审计之前，Attack Training 和正式评估都
使用了历史默认 offset `[0.02, 0.01, 0.025]`，正式策略图像还经过
nvdiffrast 重合成。因此 `50.00%` 只作为重构流程验收结果保留，不作为修正后
“纯纹理影响”的最终科学结果。

### 完整 Object task 0 流程验收

2026-07-24 又完成
`object_task0_alphabet_soup_5000_refactor_v1-EVAL-libero_object-2026_07_24-10_40_51`
完整流程：

- 5000 次优化正常结束，loss 和 gradient 全部有限；
- total loss 从 `2.124564` 降至 `1.093403`；
- 保存参数 shape 为 `[36359, 3]`，有效扰动严格位于
  `[-128/255, 128/255]`；
- 50 个 rollout 中 11 次成功，任务成功率 `22.00%`；
- 正式纹理、梯度日志和视频均正常生成，XML/真实纹理最终恢复。

该运行同样使用了修正前的历史 offset 和 nvdiffrast 正式评估路径，因此只证明
Object suite 的重构流程与 HOPE XML 纹理解析已经连通；`22.00%` 不作为修正后
攻击效果基线。

### Renderer 对齐审计与评估语义修正

2026-07-24 使用
`scripts/diagnose_openvla_renderer_alignment.py` 对固定 LIBERO 状态进行
MuJoCo segmentation 与 nvdiffrast mask 对照：

| 场景 | offset | IoU | MuJoCo 轮廓覆盖率 | 中心偏差（像素） |
|---|---|---:|---:|---:|
| Object task 0 / alphabet soup state 0 | 历史默认 | 0.3368 | 0.5337 | (-14.29, -6.58) |
| Object task 0 / alphabet soup state 0 | zero | 0.8742 | 1.0000 | (-0.11, +3.22) |
| Spatial task 0 / black bowl state 0 | 历史默认 | 0.3852 | 0.5573 | (-13.79, -15.74) |
| Spatial task 0 / black bowl state 0 | zero | 0.9990 | 0.9997 | (-0.01, +0.02) |

alphabet soup 的 zero mask 完整覆盖 MuJoCo 可见轮廓，额外像素集中于物体底部：
nvdiffrast 前景合成没有 MuJoCo depth buffer，因而会画出被桌面遮挡的部分。这
进一步说明可微 renderer 适合训练梯度近似，但不应替代正式环境渲染。

本次修正据此完成：

1. renderer 缺省 position offset 改为精确零，并增加 CPU 回归测试；
2. 非零 offset 保留为显式、经过测量的资产校准接口；
3. live-test 与正式评估直接使用安装 Active Texture 后的 MuJoCo 相机帧；
4. 录像保存模型输入所对应的同源高分辨率帧；
5. 对齐脚本长期保留 IoU、轮廓覆盖率、precision、中心和 bbox 检查。

修正后无 GPU 测试为 `30 passed, 1 skipped`；上述 Object 与 Spatial GPU
对齐回归均通过。

随后使用 run ID
`alignment_fix_smoke-EVAL-libero_spatial-2026_07_24-18_50_39`
完成修正后的端到端 GPU smoke：

- 一次优化 loss 为有限值 `20.071451`，gradient norm 为 `0.304311`；
- 顶点参数 shape 为 `[21932, 3]`，更新后范围为 `[-0.05, 0.05]`；
- UV Map、loss history、参数、gradient log 和最终纹理均正常生成；
- 正式 rollout 使用 512×512 MuJoCo 同源视频，单个 episode 正常成功；
- XML、真实纹理和临时 backup 均在退出时恢复/清理。

该单 episode 的 `100%` 只用于证明修正后完整流程连通，不用于评价攻击效果。

Spatial checkpoint：
`/data/huangsimin/openvla-7b-finetuned-libero-spatial`

Object checkpoint：
`/data/huangsimin/openvla/openvla-7b-finetuned-libero-object`

## 8. 第一版之后

建议按以下顺序继续：

1. 先基于 `FrameBatchSampler` 设计并实现 TAAO，补充 trajectory metadata。
2. 基于 `ViewSampler` 实现 EoT/multi-view adapter。
3. 稳定 OpenVLA 行为后，将领域结构迁移到 OpenVLA-OFT；action head 和模型输入
   仍保持专用实现。
4. 最后迁移到 π0，并为其 action chunk、proprioception 和预处理建立独立 adapter。

迁移时复用的是 module 职责和领域语言，不应直接假设不同 VLA 模型共享 token、
processor 或 action 解码语义。
