# OpenVLA 第一版重构架构与验收

记录日期：2026-07-23

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
| `action_codec.py` | `decode_action_from_generated_ids` | OpenVLA action token、bin center 和反归一化 mask |
| `scene.py` | `find_target_body_pose`, `compute_render_mvp` | MuJoCo body 查询、相机矩阵和去目标背景 |
| `renderer.py` | `DifferentiableRenderer` | mesh/UV、顶点扰动、nvdiffrast、光照校准、texture bake/加载 |
| `compositing.py` | `render_and_composite`, `build_single_view_samples` | NHWC/NCHW 转换、mask 合成和单视图构造 |
| `frame_collection.py` | `TrainingFrameCollector.collect` | 环境推进、clean forward、抓取窗口和光照校准 |
| `objective.py` | `get_attack_loss` | action token 选择与当前 untargeted token 目标 |
| `optimization.py` | `AttackOptimizer.optimize` | frame batch、view loss、反向传播、SignSGD、日志和 callback 调度 |
| `training.py` | `AttackTrainer.train` | 采帧、优化、live rollout 与终态产物的 task 级数据流 |
| `evaluation.py` | `LiberoEpisodeRunner.run` | 单 episode 状态机、策略图像、动作后处理和环境关闭 |
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
  ├─ MuJoCo camera frame
  ├─ optional adversarial foreground composition
  ├─ OpenVLA action
  ├─ gripper binarize + invert
  └─ env.step / close
       │
       ▼
RolloutResult → success statistics + rollout video
```

训练期间 live-test 与正式评估现在共享同一个 episode 状态机，仅分辨率、最大步数
和产物命名不同。

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

live-test 只修改 XML 引用；正式 loaded/trained texture 在真实纹理原本存在时还会
同步镜像，以保持 MuJoCo 录像行为。

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

上述结果覆盖优化器、Attack Artifact 和 Runtime Asset Transaction 的里程碑
提交。收尾提交 `f6cf29d` 进一步统一了 live/正式 rollout 并移动 Attack Training；
第一版最终验收需要在该提交上重新执行同一条一轮 GPU smoke。

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
