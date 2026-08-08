# OpenVLA 谱约束局部纹理：基础设施实施计划

设计基线：`1c2d005`（`docs(openvla): freeze spectral support foundation design`）

## 当前目标

本轮只实现正式 Attack Training 之前的可信基础设施。成功标准是 deployment
输入、可见性、coverage、renderer surrogate 与 bake response 均有可复查证据；
不以 rollout 攻击成功率作为本轮完成条件。

尚需真实审计后冻结的候选值包括 `A_obs_min`、renderer recall 和 Gate 2R 的
更强方向门槛。Gate 2C 数值门槛已由 WSL 与服务器证据冻结。实现不得为了让
某个 state 或 support 通过而调整其余候选值。

## 实施进度（2026-08-07）

- `b7d61cb`：固定 512 Policy Source，将 224 Policy Pre-Crop Canvas 与录像
  分辨率解耦；
- `a2efb7f`：建立共享 `CenterCropSpecification`、TensorFlow exact uint8
  deployment path 和 PyTorch float32 surrogate，并让 `get_vla_action()` 消费
  同一 exact helper；
- `2bb6ce7`：补齐 Gate 2C 的五类 forward 图案、五类 VJP 上游梯度和权威
  JSONL 审计。WSL 参考环境与 commit `f39525a` 的服务器 OpenVLA 环境均为
  10/10 case 通过，Gate 2C 已正式通过；
- `5e67938`：建立 512 Policy Source、224 Pre-Crop Canvas 和 Effective View
  的三个 exact uint8 stage，以及覆盖前两层空间变换的 float32 BPDA；
- `9fc1d48`：冻结 Gate 1D 的逐 state `openvla-gate-1d-v1` 判定 schema。RGB、
  processor tensor、全部 action token 和连续 action 采用严格零误差 Gate，汇总
  层拒绝缺失、重复或额外 state；
- `546440a`：实现 Gate 1D 只读 states 0–9 runner。run manifest 绑定 checkpoint
  配置/权重 inventory、task/object、state fingerprints、framework 版本、完整
  deployment config 和命令；逐 state 保存 exact/candidate PNG、无 pickle NPZ
  与权威 JSONL；
- commit `802733e` 的服务器 Gate 1D 为10/10 states、70/70 token及所有 hard
  L∞=0；WSL 已从50张 PNG 和10个 NPZ 独立复算，Gate 1D 正式通过；
- Gate 2E 尚未开始，也未开始 Visibility/Coverage/Compositor。

Gate 2C 实现过程中发现 TensorFlow 2.15 CPU 与 PyTorch 2.2 CPU 对
`sqrt(float32(0.9))` 的结果相差 1 ULP；稀疏 impulse 会把它放大为约 `2e-5`
的 relative L2。实现没有放宽候选门槛，而是让固定 PyTorch crop box 复用
TensorFlow oracle 产生的两个 float32 常量，并显式复现 pixel coordinate 与
双线性插值顺序。修正后 WSL 最坏 VJP relative L2 为 `5.4239e-8`、最小 cosine
为 `0.9999999999978721`、最大绝对误差为 `4.7684e-7`。

512→224 backward 使用 PyTorch bicubic+antialias 近似 Pillow bicubic，明确属于
BPDA approximation；Gate 1D 只验证其 exact forward，不为该近似新增临时 Gate。
其端到端梯度与真实资产响应分别继续由 Gate 2E 和 Gate 2R 验收。

## 阶段 1：Deployment Path

按行为保持式纵切依次完成：

1. 将固定 512 policy-source 与 224 Policy Pre-Crop Canvas 从录像分辨率中解耦；
2. 建立唯一 `crop_scale=0.9` Deployment Effective View Transform；
3. 让 collector、Seed Audit、Attack Training 与 rollout 复用同一 specification；
4. 建立 Gate 1D 的逐阶段 RGB、processor tensor 与 action-token 对照；
5. 建立 Gate 2C 的 TensorFlow/PyTorch crop forward 与输入 VJP 对照；
6. 运行 Gate 2E 的真实 backward/update/bake/资产恢复 smoke。

每个纵切先增加或迁移 CPU 测试，再移动一个职责。Gate 1D/2E 的 GPU 部分由
用户在服务器运行绑定明确 commit SHA 的命令。

## 阶段 2：Visibility、Coverage 与 Compositor

1. 在同一静止 state transaction 中采集 MuJoCo instance segmentation；
2. 显式解析 object type/id 与 geom→body→instance subtree；
3. 暴露 nvdiffrast triangle ID、barycentric 与严格 seam 映射；
4. 实现非负 premultiplied evidence transform 和所有偏序不变量；
5. 实现多实例加权 coverage、四类 Visibility Evidence Status 与 Primary Gate；
6. 用 Visibility-Masked Renderer Delta Composition 替换新候选的旧前景覆盖路径。

旧 foreground replacement 继续用于历史复现，但不得被新 candidate mode 调用。

## 阶段 3：可信度审计

1. states 0–9 visibility/alignment audit；
2. 根据预注册流程冻结 `A_obs_min` 与 renderer recall；
3. Gate 2R 的 R/G/B `2/255` renderer-to-bake response；
4. 生成权威 JSONL、无 pickle NPZ、派生 CSV 和逐 case 图；
5. 根据审计分布决定 Gate 2R 剩余阈值，不回看 support 成败调门槛。

## 阶段 4：Support 与训练候选

只有 Gate 1D、2C、2E、2R 和 visibility/alignment audit 全部通过后才开始：

1. Action-only Support Seed Gradient Audit；
2. 固定 10% area、最多 3 区域的 support construction；
3. Fixed Support RGB 参数化和谱自然性正则；
4. Spectral Guard Calibration；
5. 第一个 Akita source candidate、held-out Gate，随后才允许 OFT rollout。

## 每个纵切的提交门槛

- 只暂存本纵切涉及的文件，保留用户未跟踪文件；
- `git diff --check` 与相关 CPU 单元测试通过；
- 完整无 GPU 测试在依赖允许时运行，缺失依赖则明确记录；
- 代码、config、输入资产和审计 schema 的版本/hash 可追踪；
- 不把实验日志、NPZ、PNG、rollout 或纹理产物加入 Git。
