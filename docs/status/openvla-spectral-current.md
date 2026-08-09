# OpenVLA 谱纹理当前状态

更新时间：2026-08-09
当前 Visibility/Coverage/Compositor 功能代码基线：
`1884eb7500283eea9f3bcf8793a4410cd1396b87`
服务器 Gate 2E 复核基线：
`de880cee6ddabd827bfcb2c35340f0eec09fa687`
Gate 2R exact-tie 语义修正基线：
`cf4676dded3a0887a57904d21ea9119f06031b85`
Gate 2R 正式30-case证据基线：
`0b0b86ae5d2b831e1cbd8df411404e64cf7e9e51`
Dense Seed Gradient Audit正式证据基线：
`a7196e7b2a53bf36fa554ed46a51e5305529419d`
Seed Score与repeat稳定性正式证据基线：
`870aec9933913908edbc931090a9c8d5346ec5e5`
Support Construction与support-repeat正式证据基线：
`144cd002aa39586f4a69712e3647fe9cb83ce6e9`
Production Fixed Support冻结基线：
`89d3bd8963805198f94f6e37ffe9c7d9b38fa3ef`
谱自然性`rho_nat`校准基线：
`2a684942a79c9ddf1d436f47bc579dc79aaf89d4`
Spectral Guard GPU Calibration runner实现基线：
`839f5ec78f7eaae094989c394ddba01992e4a243`
Spectral Guard GPU Calibration正式证据基线：
`4f9b6bd95384a94719edce117c4ba61e7d493be1`

本文是 OpenVLA 谱纹理研究的**当前状态入口**。新一轮开发应先读本文，再按需
进入专题文档；不要从长篇实验时间线推测当前优先级。历史实验索引见
[`openvla-spectral-ledger.md`](../experiments/openvla-spectral-ledger.md)。

## 研究目标

在保持源 OpenVLA 攻击能力的前提下，寻找更容易迁移到其他 VLA 模型的物理
对抗纹理，最终用这些可复查的攻击证据指导 VLA 鲁棒性提升。

当前研究命题仍是 **source-only transfer**：训练只使用 OpenVLA。OFT 梯度只作
开发期诊断，不得进入 loss、谱基排名或训练配置。方法冻结后，正式无偏迁移结论
还需要未参与开发选择的新任务或第三个 VLA 模型。

## 里程碑状态

| 里程碑 | 状态 | 当前证据 |
|---|---|---|
| OpenVLA 入口重构与资产事务 | 已完成 | 训练、评估、产物和恢复职责已拆分；正式 rollout 只读 MuJoCo Active Texture |
| Geometry/Spectral 公平参数化 | 已完成 | 共享 Surface Delta、L∞ 预算、surface-normalized step 和 UV 保真 bake |
| 第一阶段谱方法可行性 | 已完成（旧预处理） | K=128 仅384个参数，生成的纹理仍能在真实源 rollout 上造成任务失败；需由 BPDA 新基线复核 |
| 第一阶段直接迁移 | 未通过 | Geometry、K=128/K=256 谱纹理在 OFT states 10–19 上均为10/10成功 |
| 共享特征与跨模型梯度诊断 | 已完成 | 共享 Feature 方向存在，Action 方向弱；OFT 腕部 Action 更强且与主视角近似正交 |
| 双视角与动态范数保护 | 机制已实现，源门槛未通过 | 两者均未把 held-out 源攻击恢复到预设的3/10失败 |
| OpenVLA processor 预处理正确性 | Gate 1P 已通过 | states 0–9 pixel MAE/L∞=`0/0`，10/10序列和70/70 token一致 |
| Deployment Effective View 与训练反传 | Gate 1D、2C、2E已通过 | 完整forward零误差；crop VJP对齐；五级梯度、单轮更新、bake/rollout/资产恢复通过 |
| Visibility/Coverage/Compositor | Gate 2R与全部前置Gate已通过 | 20/20对齐证据、10/10零delta states与70/70 action token通过；2R正式30/30 valid且全部cosine为正 |
| Action Objective与Dense Seed Gradient | 已通过 | 新Action hinge语义、10个完整`G_s [21263,3]`及全部artifact/hash由服务器运行并在WSL独立复核 |
| Seed Score/density/smoothing | 已通过 | corrected canonical/repeat artifacts均从raw `G_s`独立复算；连续场与高分区域repeat稳定 |
| Support Construction/coverage | 已通过，Production Support已冻结 | canonical/repeat均以`r=1`、seed 829通过；冻结3514个顶点/10542个RGB标量并绑定全部上游hash |
| 谱自然性uniform Support校准 | 已通过 | 连续K_nat=128+常数频带通过数值审计；`rho_nat=0.0992735862`并由两个输入artifact独立复算 |
| BPDA 下源攻击基线 | 未建立 | Fixed-Support renderer、`rho_nat`与`lambda_spec`校准已通过；下一门槛是新Action+Spectral trainer工程smoke，尚无正式训练或rollout结果 |
| BPDA 下 OFT 迁移信号 | 未开始 | 新源候选未过门槛前不得进入 OFT |
| 无偏跨模型迁移与鲁棒性提升 | 未开始 | 需方法冻结后的新任务/第三模型与后续防御实验 |

## 当前已确认的科学结论

1. 曲面谱参数化工程上可行；旧预处理训练出的 K=128 纹理曾以远少于 Geometry
   Vertex 的参数在真实源 rollout 上造成任务失败，但仍需由 BPDA 新基线复核，
   且尚无证据证明它提高了跨 VLA 模型迁移性。
2. 连续谱维数增加不保证攻击增强。旧预处理下 K=256 强于 K=512，因此不能通过
   继续增加 K 代替机制诊断。
3. Shared-SigLIP 能在 OFT 中产生明显 feature 响应，但动作变化很弱。共享视觉
   特征距离不是充分的跨模型 decision proxy。
4. K=256 renderer VJP 没有消灭跨模型共同方向；主要失配来自 source 单主视角
   目标与 OFT 腕部主导的双视角决策。
5. 双视角 Feature 与源 Action 在零点和旧候选终点都接近正交，而非稳定强负向
   冲突。只处理负内积的 PCGrad 不是当前优先方向。
6. 旧 Action/last-hidden 训练路径存在更基础的预处理错误：DINOv2/SigLIP 顺序
   和 resize 语义均与 checkpoint processor 不一致。旧实验可保留为历史工程
   证据，但不能作为修正后方法的科学基线。

## 当前实现基线

统一入口为
`openvla/experiments/robot/libero/openvla_attack/image_preprocessing.py` 中的
`DifferentiableOpenVLAImageProcessor`：

- 从真实 `model.config.timm_model_ids` 与 processor 读取视觉分支配置；
- 当前 checkpoint 顺序为 DINOv2→SigLIP；
- forward 将合成 RGB 量化为 uint8，并使用 checkpoint 相同的 PIL bicubic、
  ToTensor 和 Normalize；
- backward 使用连续 PyTorch bicubic+antialias 作为显式 BPDA/STE surrogate；
- collector clean label、Action/last-hidden、主/腕部 Shared-SigLIP、源动作响应
  和源像素梯度诊断共享该 interface；
- 这里的“精确”只指给定合成 RGB 的 processor forward，不表示 nvdiffrast 已
  精确复刻 MuJoCo 光照；center crop 由下述独立 Deployment Path module 负责，
  轨迹覆盖仍不属于 processor 的职责。

`b7d61cb` 起，正式 rollout 的 policy 输入不再由录像分辨率隐式决定：固定从
同一 MuJoCo observation 生成 512×512 Policy Source，再用显式 Pillow RGB
bicubic 得到 224×224 Policy Pre-Crop Canvas。`a2efb7f` 起，唯一
`CenterCropSpecification` 同时定义 TensorFlow uint8 exact forward 与 PyTorch
float32 surrogate，`get_vla_action()` 已改为调用 exact helper；`2bb6ce7` 又补齐
Gate 2C 审计和显式 TF 像素坐标/插值顺序。

Gate 1D 通过后，commit `0399c8f` 已让 collector 与 Attack Training 端到端消费
同一完整 Deployment Path；Seed Audit 仍被基础 Gate 阻断。Gate 2E 已由服务器
证明梯度穿过 center-crop 与 processor 两层 BPDA 后能更新 Surface Delta、完成
bake/rollout 并恢复资产，完整证据见下文。当前阻断项已转为真实
Visibility/Alignment audit、零 Surface Delta compositor Gate与Gate 2R均已通过。
旧`696ee68`的states 0–8部分产物和`cf4676d`的state 9仍不作拼接证据；
权威Gate 2R bundle来自同一commit `0b0b86a`的正式30-case。Fixed-Support
Texture Parameterization纯计算契约、Action Objective GPU Audit、完整Dense
Seed Gradient Audit、Seed Score的归一化/跨state聚合/mesh平滑与repeat稳定性，
以及Support Construction/coverage完整artifact契约现均已通过。canonical
candidate已经冻结为独立Production Fixed Support，renderer只允许通过完整
mesh/mapping/provenance hash加载其3514个紧凑参数坐标。uniform Support probe
的`rho_nat`也已通过纯CPU校准与独立复算。新Action-only Spectral Guard runner
已在真实states 0--9上完成GPU calibration，`lambda_spec`与完整状态恢复证据均已
通过独立复核。当前唯一下一门槛是实现并运行Fixed-Support Action+Spectral正式
trainer的工程smoke；在该门槛通过前，正式训练和held-out rollout均不得开始。

真实 Spatial checkpoint 的 CPU 差分记录为 fused pixel values MAE/L∞=`0/0`，
输入梯度有限且非零；文档记录当时全量 CPU 回归为 `113 passed, 1 skipped`。
服务器十状态 forward-only 已进一步通过数值门槛；它仍不能替代真实训练 smoke，
也没有覆盖 rollout 的额外 center-crop。

实验记账在 `25b82b4` 修正：策略、环境或预处理异常现在使 rollout 直接失败，
不能再伪装成攻击成功；最终日志分别报告 policy task success 与 attack success。
服务器定向回归为 `3 passed, 6 warnings in 20.34s`，warning 均来自第三方依赖；
同步日志 SHA-256 为
`8838eaee7227c9548770955fe9982a3d6afa4f9892ccd67ae716cfd624f6716b`。

## 下一实验门槛

### Gate 1P：十状态 processor-level forward equivalence（已通过）

使用训练 states 0–9 和旧 K=256 系数作为 adversarial 探针，只检查 clean
processor 等价性，不更新纹理，不评价旧候选的最终方法效果。

必须同时满足：

- processor fused pixel values MAE/L∞ 均为0；
- processor token Hamming 为0；
- 10/10 action 序列和70/70 action token 一致；
- first-divergence 数量为0；
- clean/processor teacher-first consistency 均为1；
- 所有响应量有限，运行后资产完全恢复。

2026-08-04 的服务器结果使用 states 0–9 和参考系数 SHA-256
`3dedc68cab8841978353eb4ae50ae27c1f6c55b654fef295c4227b19af19df50`：
pixel MAE/L∞=`0/0`，processor 与可微 clean 的 token Hamming=`0`，10/10
序列与70/70 token一致，clean/processor teacher-first consistency均为`1.0`。
adversarial teacher-first为`0.9`，对应预先声明的不纳入等价门槛的近 tie。
同步 bundle 没有 stdout/stderr 或资产 hash，因此资产恢复仍只能依赖运行端确认，
不把这一项写成已独立复核。

命令见
[`openvla-dual-view-siglip.md`](../spectral/openvla-dual-view-siglip.md)
“Processor margin 结果与 BPDA/STE 修正”一节。

### Gate 1D：十状态 deployment-path forward equivalence（已通过）

在 Gate 1P 基础上，把参考 `512 -> 224` Policy Pre-Crop Canvas 和
`crop_scale = 0.9` Deployment Effective View Transform 纳入 exact forward 与
BPDA 路径。使用 states 0–9 对照现有 rollout `get_vla_action()`，必须验证
pre-crop canvas、center-crop 后 uint8 RGB、checkpoint fused pixel values、完整
action sequence 和逐 action token 一致；同时保存各阶段图像、数值误差、token
差异及资产 hash。Gate 1P 的历史结果继续保留，但不得替代本项。

截至 `9fc1d48`，三个 exact uint8 RGB stage、完整空间 BPDA 与权威逐 state
schema 已实现并由 CPU 单元测试保护。schema 对 Pre-Crop RGB、Effective RGB、
processor fused pixel values、全部 action token 和连续 action 使用严格零误差
判定，并显式拒绝 states 0–9 的缺行、重复或额外行。

`546440a` 已实现只读真实采集 runner；它不修改纹理/XML，也不运行攻击优化。
run-level manifest 强制绑定 checkpoint 配置与权重 name/size inventory、task/object、
原始 state fingerprints、Pillow/TensorFlow/PyTorch/NumPy 版本、完整 deployment
configuration 和复现命令；逐 state 另存 exact/candidate stage PNG、无 pickle NPZ
和权威 JSONL。

2026-08-08 服务器在 commit
`802733e30b26c0b1103244cc0ed3d553de99f46a` 上完成 states 0–9 审计：10/10
states 通过，Pre-Crop、Effective View、processor fused pixel values 和连续
action 的 worst L∞ 均为0，70/70 action token一致，无缺失、重复或额外 state。
权威 JSONL SHA-256 为
`bf1a86903ee63cfc3fa0eee0ff533b049f96999a24781d2fe3163fe88d4cd235`，manifest
SHA-256 为 `b8d3856dc3159f3ba575dd6a121b1fe2e24ee38824b7a2cd8c6bcd363344f760`，
日志 SHA-256 为 `5a7ca87bf9cc9791603ccdb6ec380ec01b4d9093a74a6eef432e59586d6c4ec2`。
同步文件位于 `experiments_inbox/deployment_forward_metrics.jsonl`、
`experiments_inbox/deployment_forward_manifest.json` 和
`experiments_inbox/gate1d.log`。

### Visibility/Coverage/Compositor 实现检查点（真实审计前）

从 `af5b91e` 到 `590d766` 已按冻结决策建立以下相互独立、可在 CPU 复核的
契约：

- `coverage_evidence.py`：512 source 上逐实例预乘证据，经非负 area
  downsampling 与统一 center crop 后才聚合；输入和每级输出均检查
  `0 <= alpha*w <= alpha`；
- `visibility_segmentation.py` 与 `visibility_capture.py`：只接受显式
  `[object_type, object_id]` segmentation，递归建立 body→geom→instance 映射，
  并用静止事务检查 time/qpos/qvel 与目标 pose；二维裸 ID 不再兼容；
- `visibility_evidence.py`：严格区分 `not_observable`、
  `insufficient_observation`、`invalid_alignment` 和 `valid`，并使可观测小实例
  的 recall 失败不能被 union 指标掩盖；
- `renderer_correspondence.py`：按 nvdiffrast 的 one-based triangle ID 与
  `(u,v,1-u-v)` face-corner 顺序生成 barycentric `w_S`，背景另带 invalid
  mask；三个 corner one-hot、empty/all/subset/background 不变量已有测试；
- `visibility_compositing.py`：实现 MuJoCo front-most alpha 控制的 renderer
  delta composition，零 delta 恒等、多实例求和与实例顺序无关、clean renderer
  分支必须断开梯度；
- renderer 现可显式返回同一次 rasterization 的 adv/clean/mask/raw raster，
  多实例编排层强制公开 mask 与 triangle-ID valid mask 逐值一致。

截至该检查点，相关本地 CPU 定向回归为 `64 passed`；另有 12 项 renderer
interface 测试通过仅用于导入的 nvdiffrast stub。服务器随后在 commit
`803b702fa2c7ee370891752a649c06a8b7ad9b14` 上完成完整无 GPU 回归：
`197 passed, 1 skipped, 6 warnings in 22.15s`。唯一 skip 是预期的无 CUDA
renderer smoke；warning 均来自 robosuite、wandb 与 pkg_resources 第三方依赖。
同步 pytest 日志 SHA-256 为
`5d73952ae6c027f2662d6507ae367d2987c4d7eeda5fd0e58e3f0d4d4d22c1d3`，
commit 日志 SHA-256 为
`a10ee9c23bfae6fb64ec6d2ca26cd237bc589dc537b57370dcf2f7d4e28f5caa`。

该结果验证 CPU/interface 回归，没有验证真实 CUDA rasterization 或 MuJoCo
segmentation backend schema。commits `8b9392b`、`b506362` 与 `24daa8d`
随后补齐 effective-view visibility 编排、权威
`openvla-visibility-alignment-v1` JSONL schema 和真实 audit runner。runner 不加载
VLA 权重，不修改纹理；它在同一静止事务中采集 Primary 与 wrist source-crop
proxy 的 segmentation、全部共享纹理实例 renderer raster，并保存可复算 NPZ、
mask/overlay、拓扑/资产/事务 hash。

当时 `A_obs_min=1e-3` 与 `recall_min=0.95` 仍为预注册候选；正式冻结要求
完整 states 0–9 结构通过，并人工检查 overlay、precision/IoU 和 recall 分布。

服务器随后在 commit `bf941d9d7f5b74ca97b2e04be45cab81a750ad6a`
完成 state 0 GPU smoke。MuJoCo backend 为 mujoco `3.2.3`、robosuite `1.4.1`，
严格 `[512,512,2]` segmentation schema、两个共享纹理实例、真实 triangle ID/
barycentric、全 Support `w=1` 与静止事务均通过。Primary 与 wrist proxy 均为
`valid`：

- Primary：`A_obs=0.0235737465`、recall=`0.9994029801`、
  precision=`0.9979350885`、IoU=`0.9973405310`；
- wrist proxy：`A_obs=0.0325257730`、recall=`0.9998668607`、
  precision近1、IoU=`0.9998668607`；第二个实例在该 wrist view 中不可观测，
  没有被错误判成 alignment failure；
- transaction 前后 fingerprint 完全相同，time/qpos/qvel 与两个实例 pose 的
  差异均为0；overlay 几乎完全为黄色重合。

WSL 重新验证全部 artifact hash，并以 `allow_pickle=False` 加载两个 NPZ；复算
recall/precision/IoU 与 JSONL 误差小于 `2e-7`。权威 JSONL SHA-256 为
`bb9ba80fa40a8e95df968986faa82fafe8a23794fab3907950ab491c64172a3d`。

本次命令最终非零退出的唯一原因不是 alignment，而是 schema 判定器错误地用
64位 SHA-256 长度检查40位 Git commit，产生
`invalid_or_mixed_code_commit`。commit `74d6e17` 已拆分两种 hash 校验并新增
“64位 SHA-256 不得冒充 Git commit”回归测试；相关本地定向回归为
`72 passed`。旧 state 0 manifest 的 false 结论不改写，保留为该 bookkeeping
bug 的原始证据；该阶段的下一步是运行修正版 states 0–9 audit。

修正版随后在 commit `d0a784cb3e0ef2ae35d11464bd60020170186430`
完成 states 0–9 全量 audit。权威 JSONL 20行完整且唯一，Primary 与 wrist
source-crop proxy 均为10/10 `valid`，无 `A_obs` 边界案例、无
`invalid_alignment`，全部 static transaction 前后 fingerprint 完全相同。分布为：

- Primary `A_obs` 范围 `[0.0230380, 0.0242684]`，最低值是门槛的约23倍；
  recall范围 `[0.9992251, 1.0]`，precision最低`0.9957341`，IoU最低
  `0.9957341`；
- wrist proxy `A_obs` 范围 `[0.0291956, 0.0461461]`；recall范围
  `[0.9995750, 1.0]`，precision最低`0.9996968`，IoU最低`0.9994967`；
- 40个逐实例案例中30个可观测、10个不可观测、0个观测不足。10个不可观测
  案例均为 wrist 中不在视野内的第二个 bowl；可观测实例最低
  recall=`0.9986185`、最低 precision/IoU=`0.9910161`。

WSL 重新验证全部20个 NPZ 与100张 PNG 的 SHA-256，以
`allow_pickle=False` 加载并复算全部 union 指标；最大复算误差
`1.39e-7`，有效 barycentric sum 最大误差`5.96e-8`，背景 barycentric 与
corner index 分别严格为0和-1。人工检查覆盖 union 与逐实例最差指标案例的
state 2、7 Primary 及 state 3 wrist union overlay，差异仅位于抗锯齿边缘，
没有位置、轮廓、尺度或相机方向异常。

权威 JSONL/manifest/log SHA-256 分别为
`4d19dd9e1761266357321d5696e6690ee6cdbeba59aadbb62fb1a2c21e8c54c8`、
`f48a8262e0c8460547f223ec88cd0a3cec6233967962e9fe1b9dc50ad8cd6ab9`、
`c992fcd0e866840851467662e8a5ff8f43b2bc0cd66c46621150b1a3bf5b4a1f`。
因此从 commit `3b51f27` 起正式冻结 `A_obs_min=1e-3` 与
`recall_min=0.95`；类名 `VisibilityThresholdCandidates` 仅为历史 schema
兼容保留，默认值已是正式门槛。零 Surface Delta compositor Gate 随后也已完成
并通过，证据见下文；该阶段当时的下一步为Gate 2R，现也已通过。

WSL 复核没有只读取 manifest 判定：已重新验证 50 张 stage PNG 的数组 hash、
10 个 `allow_pickle=False` NPZ 的 processor/token/action 数组以及全部零误差条件。
checkpoint 与 LIBERO 资产原文件只存在服务器，不能在 WSL 重算内容 hash；其
配置 hash、权重 name/size inventory、物体资产 hash、task/object 和10个 state
fingerprint 已完整绑定。日志中的 wandb/robosuite/Gym warning 均发生在正式
采集前，未改变 Gate 数据或判定。因此 Gate 1D 正式通过；当时的下一门槛为
Gate 2E，现已由下述独立证据通过。

### Gate 2C：center-crop surrogate VJP equivalence（已通过）

在进入 GPU smoke 前，使用 TensorFlow float `crop_and_resize` 作为 oracle，验证
PyTorch crop surrogate 的 forward 空间坐标与输入 RGB VJP。固定 crop box 不参与
优化，因此不验证 box 梯度。

`2bb6ce7` 已实现 schema `openvla-gate-2c-v1` 的确定性 CPU 审计，覆盖五类
forward 图案和五类 VJP 上游梯度。WSL 参考运行的 10/10 case 通过；其权威
JSONL 位于 Git 忽略的
`experiments_inbox/20260807-gate2c-2bb6ce7/center_crop_metrics.jsonl`，SHA-256
为 `8b045d34b7af9d2bf72c57c9e1574966ded40285d38ac4d909448571641e6e53`。

诊断同时发现 TF/PyTorch 的 float32 `sqrt(0.9)` 相差 1 ULP；直接各自开方会使
稀疏 impulse 的 relative L2 约为 `2e-5`。当前 surrogate 改为复用 TensorFlow
计算出的固定 float32 box，并显式复现 TF pixel coordinate 与 x→y 双线性插值
顺序；没有放宽预注册候选值。

2026-08-07 服务器 OpenVLA 环境在 commit
`f39525a8436bd65f5162727f5e610f7f5de5a60e` 上完成独立复核。同步 JSONL 共
10 行，case 名唯一且完整，全部 commit/schema/shape/dtype/crop 字段一致；日志
与权威表分别同步为 `experiments_inbox/gate2c.log` 和
`experiments_inbox/center_crop_metrics.jsonl`。日志
声明的 SHA-256 与文件实算值均为
`492b7f1d486ce9375e46341a7dd6983e77b51238f90b5f05ab7ae3a99f79dc46`，日志本身
SHA-256 为 `72697458783c262120b11d7b131b8089e2b4efd50a3e934e43496d0b981eccf2`。
服务器 TensorFlow/PyTorch/NumPy 版本为 `2.15.0/2.2.0+cu121/1.26.4`；五类
forward 的 relative L2 与 max-absolute error 均为 0，五类 VJP 最坏 relative
L2=`5.425248973361878e-08`、最小 cosine=`0.9999999999978721`、最大 max-absolute
error=`4.76837158203125e-07`。

因此正式冻结 Gate 2C 门槛为 `relative_L2 <= 1e-5` 且
`cosine >= 0.99999`，`max_abs` 继续只作诊断；本次 5/5 VJP case 均以明显余量
通过。framework build、crop specification 或 surrogate 实现变化时必须重跑；
Gate 1D 与2E后来分别用独立证据通过，不能由本 Gate 单独替代。

### Gate 2E：deployment-path end-to-end backward/update/bake smoke

Gate 1D 与 Gate 2C 通过后，验证梯度能穿过 center crop 与 checkpoint processor
的两层 BPDA surrogate，一直回到 Surface Delta，并检查：

- `[256,3]` 系数真实更新；
- center crop 前后的 RGB 梯度均有限且非零；
- Surface Delta 梯度有限且非零；
- Actual Surface Step 不超过 `2/255`；
- Max Surface Delta 不超过 `128/255`；
- UV PNG、Active Texture、rollout 和 Runtime Asset Transaction 正常。

Commits `0399c8f88abcf0688463f33a45b2933c3fea77ce` 与
`de880cee6ddabd827bfcb2c35340f0eec09fa687` 已实现机器可重算的
`openvla-gate-2e-v1` JSON、只在诊断范围启用的 Surface Delta 中间梯度捕获，以及
固定 source Action-only、K=256、train state 0 单 update、rollout state 10 的
真实 runner。判定要求 Policy Source、Pre-Crop、Effective View、Surface Delta
和谱参数五级梯度均有限非零；`rollout_success` 只记录工程 smoke 的策略结果，
不参与 Gate，也不得解释为攻击效果。

服务器在 `de880cee6ddabd827bfcb2c35340f0eec09fa687` 上先完成无 GPU 回归：
`135 passed, 1 skipped`；skip 是预期的无 CUDA renderer smoke，6项 warning 均为
robosuite/wandb 依赖弃用提示。随后真实 Gate 2E 得到：

- Action loss=`34.693824768066406`；Policy Source、Pre-Crop、Effective View、
  Surface Delta 与 `[256,3]` 参数梯度全部有限非零；
- 768/768 个谱系数更新，参数 L∞ change=`9.178405889542773e-05`；
- Actual Surface Step 与 Max Surface Delta 均为
  `0.007843135856091976`，低于 `2/255` 且远低于 `128/255`；
- bake PNG 与 Active Texture SHA-256 同为
  `8095d0e875e9e3ea3e455ffb467e2ec81825f62095bc7775fb20e91c81ddb744`；
- state 10 rollout 正常完成；其 success=true 只作记录；XML 与真实纹理恢复前后
  hash 分别完全一致，backup 已删除。

WSL 重新加载 schema 后独立得到 `gate_pass=true`，并以 `weights_only=True` 检查
谱参数产物为有限 `[256,3]`、768项非零；同步的4096×4096 RGB PNG hash与证据
一致，K=512 basis 的本地 SHA-256 也与 provenance 一致。权威 evidence、日志和
pytest 日志 SHA-256 分别为
`b561b573b46ad79b705bc15644cffd706d72b3c08cc1965ebcc778934c2044e3`、
`aa425f619a55752404f12102fa014e8a71b530108d60779365c1718793da7f0a`、
`7ba5261056f0e5d850fb6a444666fb4c0f663567f82a9fdd79c1343c7c140675`。
因此Gate 2E正式通过；该阶段随后进入的Visibility/Coverage/Compositor与
Gate 2R现也均已通过。

### 零 Surface Delta Compositor Gate（已通过）

commit `5dcec04` 建立纯 CPU 判定层 `openvla-compositor-zero-delta-v1`；commit
`1884eb7` 增加真实 states 0–9 runner。它只使用 source OpenVLA Primary 视角，
在同一静止 transaction 中采集 MuJoCo front-most instance alpha 和全部共享纹理
实例的同次 renderer evidence，不读取 Action 梯度、不构造 Support、不更新或
bake 纹理，也不运行 rollout。

Gate 对每个原始 state 严格要求：

- raw 512 Policy Source、224 Pre-Crop Canvas、center-crop Effective View、
  checkpoint fused pixel values 与连续 action 的 `MAE=0`、`L_inf=0`；
- clean/compositor action token 完全一致，十状态共70个 token；
- `F_adv-F_clean`、compositor total delta、逐实例 delta 和 clamp saturation 在
  零 Surface Delta 时严格为0；
- clean renderer 不连接 autograd；经过 `MuJoCo alpha * renderer mask` 的实际
  compositor 对 trainable Surface 参数的梯度有限非零，且至少有一个 joint-valid
  pixel；
- static transaction 前后 fingerprint 完全一致，十个 state 无缺失、重复或
  混合 commit。

每个 state 保存一个 `allow_pickle=False` 可加载的 NPZ、clean/composited RGB、
MuJoCo alpha 图和完整 SHA-256 inventory；唯一权威长表为
`compositor_zero_delta_metrics.jsonl`，manifest 必须绑定其 SHA-256。WSL 定向
回归为 `22 passed`；服务器定向回归同样为 `22 passed in 1.06s`。第一次 GPU
运行虽然数值通过，但 `commit.log=7e2ff16` 与命令参数/JSONL 中的 `1884eb7`
冲突，因此明确拒绝，没有事后修改产物。

服务器随后在实际 checkout
`7e2ff161451eb57e515be350ba69019284c4e879` 上重新运行。`commit.log`、manifest、
全部10行 JSONL 与保存命令一致；10/10 states、70/70 action token 通过，raw、
Pre-Crop、Effective View、processor pixels、连续 action、renderer delta、逐实例
delta、total delta 与 saturation 的最坏 L∞ 均为0。clean renderer 全部断图，
joint-visible pixel 范围为 `[5416,5734]`；Surface 参数梯度 L2 范围为
`[82.4309581,87.0276452]`、L∞ 范围为 `[2.1930928,2.5591207]`，每个 state
有 `[25173,26013]` 个非零项。全部 static transaction 前后 fingerprint 相同。

WSL 又独立验证60个 artifact SHA-256，使用 `allow_pickle=False` 加载10个 NPZ，
逐值复算所有等价数组、joint-visible counts 与梯度有限/非零统计，失败项为空。
权威 commit/pytest/GPU log/JSONL/manifest SHA-256 分别为：

- `5691e28bdcd72a250df2af08d89db647c95e4386c22bfb9dd3027dd57507b404`；
- `76614c2c6f128aca9516c283424a670890936f188aed56f3064e8f5fb73fd572`；
- `5c59b8edd7d1adc439e99ff779b37ba1a637c68be2e208fb040e969d0fbe06d3`；
- `2a9c330e80df0381ebc88f922da6400438cf05d9949703356c15f14424dd87f6`；
- `6a2429f941071ceee8c91a2a6abc414e062bc3db5bdb264e0c1291798eafb579`。

因此零 Surface Delta Compositor Gate 正式通过。该阶段当时的下一步只允许实现
和运行Gate 2R；其后Gate 2R也已按下文证据通过。

### Gate 2R：renderer-to-bake response（已通过）

Gate 2E 通过后、Support Seed Gradient Audit 前，使用第五十项冻结的 Action-free
小幅颜色探针，比较 Renderer Delta Composition 与真实 bake→MuJoCo observation
在同一 effective view 目标可见区域内的 RGB 响应方向。

实现分为两个职责明确的模块：

- `renderer_bake_response_audit.py` 是不导入模型、LIBERO、renderer 或 CUDA 的
  纯数值层，固定 weighted RMS/cosine/relative L2、sign denominator、Action
  margin、30行完整性、NPZ/JSONL/CSV/manifest schema 与逐 probe 汇总；
- `diagnose_renderer_bake_response.py` 是真实 runner。它对每个 state 只在 clean
  环境采集一次 MuJoCo alpha 和三条 renderer surrogate，再对 R/G/B bake 分别
  激活资产、重建同一初始环境、采集真实 response，随后关闭环境并恢复资产。

每行额外要求 clean/bake 静止仿真 fingerprint 完全相同；响应充分但 cosine
非正时，证据 `status` 仍为 `valid`、行级 Gate 失败，避免把机制反证误写成
采集证据损坏。只有事务、hash、状态或五类可视化不完整时才标记
`invalid_evidence`。sign consistency 的分母冻结为 effective-view 内 alpha 正权重
且两条响应至少一个严格非零的通道分量；双零分量不提供方向证据，一侧为零则
计入分母但不计入分子。runner 还会在加载模型前验证 `--code_commit` 等于真实
Git HEAD 且 tracked worktree clean，避免再次产生“命令写一个 hash、实际运行
另一个版本”的伪 provenance。

WSL 已通过 Gate 2R 纯数值层 `9 passed`，并在排除9个明确依赖
`nvdiffrast`/`libero.libero` 的文件后通过其余 OpenVLA Attack 单元测试
`160 passed`。完整本地 suite 因上述缺失依赖在 collection 阶段停止，未伪记为
代码失败或全量通过。服务器应
先运行下面的无 GPU 定向回归；命令从服务器当前 HEAD 动态取得40位 SHA，避免
手填旧 hash：

```bash
# 在服务器已同步到目标 commit 的仓库根目录执行。
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 \
  NUMBA_CACHE_DIR=/tmp/tex3d-numba-cache \
  /home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python -m pytest -q \
  tests/unit/openvla_attack/test_renderer_bake_response_audit.py \
  tests/unit/openvla_attack/test_diagnose_renderer_bake_response.py
```

定向回归通过后才运行真实30-case命令；`--output_dir` 必须是新的空目录：

```bash
# 在服务器已同步到目标 commit 的仓库根目录执行。
set -o pipefail
CODE_COMMIT="$(git rev-parse HEAD)"
RUN_ID="gate2r-${CODE_COMMIT:0:7}-states0-9"
/home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python \
  openvla/experiments/robot/libero/openvla_attack/diagnose_renderer_bake_response.py \
  --pretrained_checkpoint /data/huangsimin/openvla-7b-finetuned-libero-spatial \
  --output_dir "experiments_inbox/${RUN_ID}" \
  --code_commit "${CODE_COMMIT}" \
  --task_suite_name libero_spatial \
  --task_id 0 \
  --object_name akita_black_bowl \
  --state_ids 0-9 \
  --num_steps_wait 10 \
  --seed 7 \
  --unnorm_key libero_spatial_no_noops \
  2>&1 | tee "experiments_inbox/${RUN_ID}.log"
```

首轮只按预注册必要条件自动判定：30/30 两条 RMS 均不低于 `1e-6`、全部
`cos_alpha > 0`、行集合/事务/hash/状态/产物完整。relative L2、sign consistency
和 Action margin 不进入 Gate；审计后才能讨论更强门槛。

2026-08-08 在 commit `c7f14f347cc7f82c9130a80ce0567db9c2f883ed` 上完成
state 0 三通道工程 smoke。R/G/B 的 surrogate RMS 分别为
`0.00317794/0.00319634/0.00317729`，bake RMS 为
`0.00475871/0.00439715/0.00425495`，cosine 为
`0.590808/0.565539/0.656018`，因此三行均通过预注册最低门槛。relative L2
仍为 `0.810479/0.840361/0.760177`，sign consistency 为
`0.357544/0.347229/0.419539`：现有 surrogate 能提供同向信号，但不能解释为
真实 bake 的高保真颜色或幅值模型。逐通道统计进一步表明 surrogate 常量 probe
只改变目标 RGB 通道，而 MuJoCo bake 在非目标通道也有响应；这是当前较大
relative L2 的直接来源之一，不进入第一版硬 Gate。

WSL 已独立使用 `allow_pickle=False` 加载3个 NPZ，逐值复算全部指标与 Action
margin，核对21个 artifact、config/evidence hash、clean/bake simulation state、
Active Texture 和 XML/纹理恢复，失败项为空。log/manifest/JSONL/CSV SHA-256
分别为：

- `4dab46b976934c432e5d6ad9535c051c288dc6a7cf7393daf2038851a6cb3315`；
- `9a913357dadf98a297de095dfa5f1df2630563df68eae96498ec1dbfe032cc34`；
- `471eacaffaba4f22c7e53ead016e74f3a3dbae5e2afe652286bd378d632f0610`；
- `e690dfe60e6a067e9d2626b651a8885a9cdfac735df64d5314042cb37d829c42`。

smoke 也暴露出非权威 weighted-scatter PNG 的点过小且 alpha 色过浅；正式
30-case 前已改为半径2的高对比度点、增加 `y=x` 参考线和 RGB 图例，并与
sign-consistency 一样排除双零通道分量。该修正只改善人工抽查，不改变 NPZ、
任何数值指标或 Gate 判定。

2026-08-09 在 commit `696ee68b29990ea95e54432065b1b5f46305a0d8` 上的正式
states 0–9 运行完成了states 0–8的27个probe，但state 9在任何真实
bake前触发“clean token与teacher-forced argmax不一致”严格断言。独立的
fresh state 9 运行在同一代码位置再次失败，因此已排除state顺序和
旧输出目录干扰；该fresh log SHA-256为
`1689a326056b643dab7b72262692116786dbb9271e8ecc5cba5e6c7f4b69a55d`。此次不是
Gate 2R 数值失败：未生成权威30行、manifest或state 9 response，部分
artifact 不得作为通过证据。

历史source action-response证据曾在其他state的后续action位置观察到同类
generation/teacher-forward分叉，而首token保持一致；因此当前不将故障
直接归因为causal错位，也不放宽冻结断言。runner已增加失败保持式
诊断：捕获cache generation每步score，验证输入tensor未被修改，并在同一
断言中报告分叉索引、两条路径argmax、固定clean-class margin和逐token
logit MAE/L∞。服务器诊断已定位state 9只在action index 1分叉：
generation选class 125且margin为`0.125`，teacher forward在class 125与120之间
精确并列，clean-class margin为`0.0`；其余6/7位置均一致。输入变异保护
未触发，因此排除系统性causal错位和输入被`generate()`修改，根因是
BF16/cache generation与全序列teacher forward的数值路径差异，加上标量
`argmax`的“首个最大值”tie-break语义。
定向回归为`13 passed, 3 warnings`，pytest/诊断log SHA-256分别为
`05dd8b1028fa80e1567aa31b2fb5842ebdfb73d1594c5749accf291c5027986e`和
`7c945fe8bcf753d61a59acc8e4bbdae894780b0b64957f867d255f565cd3c896`。

严格一致性检查因此修正为集合语义：clean generated class必须属于
teacher-forward argmax集合，等价于冻结clean-class margin `m_a >= 0`。
`m_a == 0`只接受精确并列并显式记录state/位置，不引入人为容差；
任何`m_a < 0`仍作为输入/对齐失败严格中止。随后以fresh state 9验证
这一修正能越过clean阶段并完成三个probe。

commit `cf4676dded3a0887a57904d21ea9119f06031b85` 的fresh state 9已完整
通过：R/G/B surrogate RMS为`0.00315460/0.00317653/0.00318386`，bake RMS
为`0.00454027/0.00440261/0.00432564`，cosine为
`0.623784/0.581268/0.651004`，3/3均为`valid`且通过预注册最低门槛。
relative L2为`0.784816/0.825709/0.763823`，sign consistency为
`0.354368/0.354156/0.413715`，继续支持“方向同向但非高保真幅值模型”的
state0判断。exact tie在manifest中保留为state 9/action index 1，负margin
位置为空。

WSL已独立核对24个文件，在`allow_pickle=False`下加载3个NPZ，逐值复算
全部response指标与Action margins，并验证21个artifact hash，失败项为空。
pytest/log/manifest/JSONL/CSV SHA-256分别为：

- `ea8cfbad58e22a948555e15c5785f92d88b73eddaf533ee059eab6946a2f343b`；
- `ea2e0d6b5adeb8b9ad2aee23da75fc09094d6e9883d0f6454d86114ae1074d8a`；
- `08c6243d8084e003181cf6f7e3b3996caa5af1b246ebc7ee56d6b2abb87b0c33`；
- `59d8975b9fb452e4457cdc4ab26d334d1f1158494f8b627ea464ed854ab3f2fc`；
- `fbfa65c32516c1cd22984eaffd21fca671a5c507d710b8c5eed07cee49637b57`。

该state 9 bundle只证明修正和单点Gate成立，不替代同一code/config hash下的
权威30行；因此随后在当前代码上正式重跑了states 0–9。

commit `0b0b86ae5d2b831e1cbd8df411404e64cf7e9e51` 的同commit正式
states 0–9已完整通过：30/30行均为`valid`，两条response RMS全部
高于`1e-6`，全部`cos_alpha > 0`，无缺失、重复、额外key或结构失败。
逐probe的worst/median cosine为：

- R：`0.590808` (state 0) / `0.626496`；
- G：`0.565539` (state 0) / `0.578714`；
- B：`0.586726` (state 6) / `0.652017`。

全局最小surrogate/bake RMS为`0.00313060` (B, state 8) /
`0.00407509` (B, state 7)。R/G/B最大relative L2为
`0.815153/0.840361/0.824974`，最小sign consistency为
`0.350509/0.340486/0.382972`。因此正式分布稳定复现state0/state9的
结论：surrogate在有效视野内提供稳定同向的小信号，但不是真实bake的
高保真逐像素或幅值模型。Action consistency仅state 9/action index 1有精确
teacher argmax tie，10个state均无负teacher margin位置。

WSL已独立在`allow_pickle=False`下加载30个NPZ，重算所有RMS、cosine、
relative L2、sign consistency及Action margins，重建整个summary，并验证
183个artifact hash与186个文件的精确inventory，失败项为空。人工抽查
R/G/B最低cosine的scatter也均非空且显示同向结构；格点来自真实bake的
8-bit量化。log/manifest/JSONL/CSV SHA-256分别为：

- `c692899eab7bf7408f1c5761edbbd1f8bd75981ba31736bcc928e2e91ce808a1`；
- `231e6eb67182f4caecf956c93908624ec7472874b53dff7ca10877ebc7143067`；
- `e1f3375650e0a5077b66cb7ed5f3ecef8a64b297563f2111f2755f3c8b99b8b5`；
- `79c857e478c973d982bc3c015adff7d9cea54b062dc6a71190a6e6b3be45b056`。

因此Gate 2R按预注册的“response充分且全部cosine为正”最低机制门槛正式
通过。不根据已观测的分布追加一个为本批数据量身定制的更强事后硬门槛；
worst/median cosine、relative L2和sign consistency保留为机制表征。

### Gate 3：建立 BPDA 下的新源候选

Gate 1D、Gate 2C、Gate 2E与Gate 2R均已通过，下文也已冻结“谱自然性约束 +
选定顶点全维优化”的最小参数化、预算与顶点选择规则。当前先实现并测试
新Texture Parameterization契约，再运行Support Seed Audit与train states 0–9的单一
候选。held-out states 10–19至少3/10失败才进入OFT开发期rollout。纯K=256、
`rho=1.0`不再是自动下一候选。

2026-08-09 已完成 Action Objective Contract 的第一段本地实现：历史
`255-clean_bin` CE 已显式隔离为 legacy 函数，新路径新增可微的 Untargeted
Clean-Action Margin hinge，并共同复用唯一的尾部对齐、causal shift 和 action
token 提取 helper。7个纯 CPU 测试覆盖 margin 数值、梯度方向、负 margin
停止施压、精确 tie 与无 action token 失败。该结果只表示数学和 interface
契约在本地通过；尚未运行真实 OpenVLA Objective GPU Audit，也尚未运行
Dense Seed Audit。现有旧优化器仍显式调用 legacy objective，不得将其结果
记为新候选。

同日已完成 Fixed-Support 参数化机制的纯计算契约：它只接受外部提供的唯一
几何顶点索引，以紧凑 `[|S|,3]` 参数表示实际可学习的 `delta_S`，再散射到
完整 `[N_v,3]` Surface Delta；Support 外严格为零，UV seam、Surface-L∞预算
和 surface-normalized step 继续复用公共语义。14个参数化 CPU 测试全部通过，
其中8项直接覆盖紧凑参数、mask、梯度与非法 Support。当前 renderer/CLI 尚未
接入该类，也不存在生产 Fixed Support；Objective GPU Audit 与后续 Dense Seed
Audit 仍必须使用全顶点 `GeometryVertexTextureParameterization`。

Objective GPU Audit runner 与纯 CPU evidence contract 已实现并通过服务器
states 0--9正式验收。正式 runner 固定 Action hinge 为唯一 objective，逐 state
复用全部共享纹理实例、MuJoCo front-most alpha、visibility-masked compositor、
精确 center-crop 与 checkpoint BPDA；参考参数为全几何顶点 `[N_v,3]` 且严格
为零。
每行证据保存完整 clean token/classes、margin/hinge、五级梯度统计及 dense
gradient hash；Feature/wrist/OFT、参数更新、Support Construction 和完整 seed
gradient payload 均不进入该命令。CPU evidence/objective/parameterization 相关
测试共28项通过；服务器绑定commit
`20646568cdbec4affa16ffd5e89ab266b48a1e04`的结果为10/10 state、70/70 token
通过，Action loss均值`12.2232146`，margin mean/min/max为
`12.2232143/0.0/42.125`。只有state 9/index 1为精确tie，无负margin；每个
state的Policy Source、Pre-Crop、Effective View、render Surface Delta和
`[21263,3]` dense geometry梯度均有限非零，dense gradient L2范围
`[0.2435126,0.5750862]`，10个state gradient hash均唯一。metrics SHA-256为
`721e9ee455cebfacd03727ad9847bfceb23029924276c2f1e50f126cf9536895`，manifest
SHA-256为`505e0fcae64ace8c76903d5690e836ff88af40c83540079f5520835256a006b1`；WSL
严格加载10行并独立重算summary与全部逐行decision，结果与manifest一致。

同步的stdout log来自成功后约49秒的同目录重跑：runner按设计因已有metrics/
manifest抛出`FileExistsError`，同时第二次`tee`覆盖了首次成功stdout。该异常
不否定时间上更早、hash自洽且可独立重算的权威文件，但正式记录不把被覆盖的
log当作成功证据。后续重跑必须使用新的run目录和日志名。Objective GPU Audit
已通过，允许进入Dense Seed Audit实现；其现有梯度摘要/hash仍不得冒充正式
Seed Audit的完整`G_s`产物。

Dense Seed Gradient Audit 的实现与完整 artifact/evidence contract 已通过
服务器正式验收。它不重写模型链路，而是把已通过 Objective GPU Audit 的单
state capture作为唯一公共seam；Objective runner继续只消费gradient统计/hash，
Dense Seed runner则保存拥有数据的CPU float32 `G_s [N_v,3]`。正式run
`dense_seed_audit_a7196e7_20260809_101235`绑定commit
`a7196e7b2a53bf36fa554ed46a51e5305529419d`，states 0--9全部通过，共10个
`[21263,3]` NPZ、637890个梯度值和10个唯一gradient hash；L2范围为
`[0.2437936,0.5747361]`。每个NPZ同时绑定clean token/classes、全部margin/
hinge、OBJ mesh、render-to-geometry mapping、Policy Source/Effective View、
MuJoCo instance alpha、renderer visibility及全部共享纹理body ID/name。

WSL已逐文件使用`allow_pickle=False`重新加载全部NPZ，并重算artifact/数组hash、
梯度统计、Objective Gate与states 0--9 summary，10/10 decision均通过；metrics、
manifest和log SHA-256分别为
`bc0bc00865dc4eb83bde4b1b362c09b789a5f04be3b7bf0a2672378c9cadede7`、
`1c9e9f10d370522e62c4360d7e4d9c788982cd797ef8768c80b0b614f894abbb`、
`03c592760fe9e31cf59aa1e6a8825078e20c34b6eeede4decacfb23eadf35178`。
与前序Objective Audit相比，每个state的clean tokens、margins与loss完全一致；
GPU数值非确定性使raw gradient hash不同，但L2相对差异仅
`0.037%--0.182%`。manifest确认legacy optimizer未加载，Feature/wrist/OFT、
score、density、coverage和production support均未计算。Dense Seed Gradient
Gate据此正式通过；下一步只允许实现Seed Score及其evidence contract，不能直接
从artifact生成Fixed Vertex Support。

Seed Score/density/smoothing纯CPU契约已实现并通过服务器正式artifact验收。
它逐state保存`max(abs(G_s))`归一化scale、absolute component p99与
`max/p99`，并完整保存`[S,N_v]`归一化梯度范数、`[N_v,3]`均值梯度、mean
sensitivity、RGB direction consistency、`q_i`、barycentric lumped mass、
`d_i`、`d_tilde_i`、全顶点排名和只读局部峰诊断。平滑固定`alpha=0.05`，
使用原始OBJ cotangent stiffness与`tau=alpha^2*A_total`；linear residual、
mass integral守恒和负density均显式检查，禁止静默clamp。

artifact内嵌float64 vertices/int64 faces并绑定正式Dense metrics、10个NPZ、
10个raw gradient和OBJ file/array SHA-256，因此同步后WSL可脱离LIBERO资产从
raw `G_s`重算全部score。repeat比较接口报告raw score/density/smoothed density
cosine、smoothed Spearman、top `0.1%/0.5%/1%/5%` Jaccard及前100个局部峰
Jaccard；按本轮建议只报告分布，不预注册通过阈值，也不生成候选Support。
相关Objective/Dense/geometry/score纵向CPU测试共47项通过。

正式canonical run为
`seed_score_original_corrected_870aec9_20260809_123241`，直接绑定score commit
`870aec9933913908edbc931090a9c8d5346ec5e5`、原始Dense bundle与服务器Akita
OBJ file/array hash；repeat run使用同一score commit和第二套同Dense实现的raw
`G_s`。两套score artifact均由WSL从各自10个Dense NPZ、内嵌`21263/42522`
vertices/faces完整重算，decision均为10/10通过，无负density且未构造Support。
canonical manifest/artifact/log SHA-256分别为
`c589f3c3a2ac7eadf032d5960186b5e122e72594baeeefbf44a00c45c5367d39`、
`9067331ef000e0351e66d4d3b3673847dd00ff0945b72f58516197175980160b`、
`70b94615565a5514a420b07cc1f59003c066e8618802d7bd38aaf85fa1f5aac7`；
repeat对应为
`4e311db4a5f72dae0cdc6ff7c63192728f2ced760eed09cce8a079329f870cfd`、
`0fed4dc1f2c6010adb673dfbc91da70d6b918323023d70670fcfa27a21206657`、
`92803b4e96f6513a102c415d00b17fdf628a95b05533edb55ccd6276341c5273`。

canonical/repeat的normalization `max/p99`范围分别为`3.08--7.24`与
`3.09--7.27`；raw score、raw density、smoothed density的`max/p99`分别为
`2.709/6.510/1.0511`与`2.714/6.476/1.0513`。repeat raw score/raw density/
smoothed density cosine为`0.9999849/0.9999833/0.9999996`，smoothed Spearman
为`0.9999994`；top `0.1%/0.5%/1%/5%` Jaccard为
`1.000/0.9815/0.9814/0.9925`。local peak总数为42/40，前40峰Jaccard
`0.8605`，但最高15个峰ID及顺序完全一致；差异只出现在更低优先级局部峰。
comparison JSON/log SHA-256为
`f4894f555333ffa3b9c794d24699d76756fd155b488b9ade282e345e71c6f97e`与
`21febbda5d446b7f834008bf0029c4506e12e49c956cc55ccd6276341c5273`。
这些结果没有显示max-normalization导致高分区域不稳定，因此不追加repeat、
不修改冻结公式，也不为迎合本次结果追补阈值。

首次score运行误把score code provenance写成`a7196e7`，随后重复运行同目录又
按设计触发防覆盖`FileExistsError`；首次comparison还引用了错误时间戳目录。
这些产物只保留为诊断历史，不进入正式证据。上述三个`corrected`目录是唯一
权威score/stability记录。

Support Construction/coverage 的 CPU-only 双层 evidence contract 已在commit
`144cd002aa39586f4a69712e3647fe9cb83ce6e9`完成并用上述corrected score运行。
它从已验收Visibility NPZ重放逐`(state, view, instance, vertex)`的source/effective
投影分子：先对冻结的512→224 area downsample与center crop求source-space伴随
权重，再把`alpha * beta`按严格geometry corner mapping散射到原始OBJ顶点。
完整contribution与candidate NPZ保存逐实例分母、全顶点分子、support mask、
region owner、seed、区域质量/周长/compactness和逐state coverage；加载时禁用
pickle，并从score与contribution独立重算全部候选。20个真实view的全Support
effective coverage为`0.998986--1.000000`，伴随分母最大绝对重放误差
`2.454e-5`。这与既有alignment recall一致，没有把裁剪外或renderer未对齐像素
误记为可控投影。

实现对文档未逐字规定的“多区域如何交替取frontier”采用按seed顺序的确定性
round-robin，每轮每个未满区域各取一次自身best-first顶点。当前正式结果在
`r=1`即通过，因此该调度没有影响本轮seed、mask或Gate；若后续方法变体确实
进入`r>=2`，必须在比较调度变体前先把这一低层语义显式复核，不能用结果反调。

正式canonical run为`support_construction_canonical_144cd00_20260809`：第一个
候选`r=1`即通过，seed为原始OBJ顶点829，选择3514个顶点/10542个RGB标量，
目标/实际mass为`0.0071906672/0.0071906824`，实际面积比例
`0.1000002119`；唯一region完整且离散overshoot小于最后一个边界顶点mass。
Primary states 0--9全部为`valid`，effective coverage min/mean/median为
`0.3592615/0.3626509/0.3624516`，最差原始state ID为5，显著高于预先冻结的
`0.20`硬Gate。wrist proxy继续只读，其min为`0.1556022`，没有参与seed、增长、
区域数或Gate。coverage/candidate/manifest SHA-256分别为
`bc589979281807f9d2706f7c79380fd43b42eb96c4ce77628c6b63b13570fbdc`、
`def9dfdfd9466c74f10dbc8bc5f85b9a023e9b95478566a4ac0c361150ebc254`、
`abbb1bc22ee41fef6bbe38c1e1f420d1e47711262f659d4c737a0190e38e768e`。
最差state的MuJoCo mask、renderer mask和`alpha*w_S` effective投影均已保存并
人工抽查；两张可见mask对齐，Support投影是其真子区域，未见裁剪/坐标翻转。

按预先约定又用repeat score运行完全相同的构造，目录为
`support_construction_repeat_144cd00_20260809`。它同样在`r=1`通过，seed仍为
829，最差state仍为5；选择3517个顶点，面积比例`0.1000245587`，Primary min
coverage为`0.3592754`。两套selected Support交集/并集为3511/3520，Jaccard
`0.9974432`；最大逐Primary effective coverage差`2.699e-4`，面积比例差
`2.435e-5`。support-repeat比较没有预注册或事后追加通过阈值，只完成
“gradient稳定→score稳定→最终support稳定”的只读证据闭环，不据结果修改任何
算法或门槛。repeat candidate/manifest SHA-256为
`1da6c4fdef9b762d4b80d085f43d3d583afa89486bb7fab55a9faa5f95c17508`与
`cc2747cb04ac326a86343822272c35d485911e7ab2c58812a83738689fbb67ca`。

上述结果客观支持canonical candidate在当前规则下稳定且通过Gate，但两个audit
仍同时声明`production_support_constructed=false`与
`fixed_support_frozen=false`。因此它们不能被训练入口直接消费；下一步必须先
形成单独、不可变、绑定canonical candidate hash的生产Support冻结artifact，
再把该artifact接到已有Fixed-Support Texture Parameterization。首次因NumPy
`bool_`无法写JSON而中止的`support_construction_canonical_20732ee_...`目录已
改名保留为失败诊断，不进入正式证据。

commit `89d3bd8963805198f94f6e37ffe9c7d9b38fa3ef` 随后新增
`openvla-production-fixed-support-v1`冻结契约；它不重新选点，只从上述已接受的
canonical candidate形成升序几何顶点ID的不可变紧凑坐标。正式本地冻结目录为
`production_fixed_support_89d3bd8_20260809`，独立验收`gate_pass=true`：总几何
顶点21263，Support顶点3514，可学习RGB标量10542，seed 829，实际面积比例
`0.1000002119`，Primary/wrist最低effective coverage分别为
`0.3592615/0.1556022`。artifact、manifest和Support mask语义SHA-256分别为
`686a2becc3688b0cb6bdafef840d0920fb51e584dc860a0c95733b935d22fb6d`、
`39cc70f58bf777919ef069d3defb3b1ba324ee072b215919a928f3c0bf422829`和
`719a33c9b91fbeda0ef3bd506b429c157b03bde7a03406cf2b96461167eb4322`。

该artifact同时绑定canonical manifest/candidate/coverage、Seed Score、Visibility、
OBJ mesh数组与文件、renderer faces及render-to-geometry mapping哈希。冻结阶段
强制声明`production_support_constructed=true`、`fixed_support_frozen=true`、
`rho_nat_calibrated=false`、`lambda_spec_calibrated=false`和
`formal_training_allowed=false`；loader对mesh或mapping不匹配直接失败。renderer
接线只把3514个紧凑RGB参数scatter回完整`[21263,3]` Surface Delta，Support外
严格为零。主入口仍显式拒绝`fixed_support`正式训练，避免现有legacy
Action+Feature trainer绕过新Action-only目标与两级谱校准。

因此后续顺序正式固定为：Production Support冻结与参数化接线（当前阶段）→
`rho_nat=r_high(1_S)`校准→最多64轮Action-only Spectral Guard Calibration、
`lambda_spec`冻结及完整状态恢复→正式Action+Spectral训练→held-out source
states 10--19 rollout。单步backward/update/bake可以提前作为工程smoke，但只有
走新Action-only+Spectral链路时才允许执行；未校准纹理不得进入rollout或反向
影响Support与超参数决策。

commit `2a684942a79c9ddf1d436f47bc579dc79aaf89d4` 随后实现纯CPU
`openvla-spectral-naturalness-calibration-v1`契约。它从Production Support构造
float64 `[21263,3]` uniform probe，并从连续K512谱基产物只取mass-normalized
常数模态与前128个非恒定模态；不加载模型、renderer或训练器。正式目录
`rho_nat_calibration_2a68494_20260809`经同一对输入artifact独立重算后
`gate_pass=true`。谱基文件SHA-256为
`0ab987b4bc4a27b175c90382296571219dfe79a04832c6ec8be848eb02825d61`；
M-正交L∞误差为`3.0713e-15`，常数模态L∞误差为`8.7486e-14`。
`lambda_128=22332.0216539`，总面积为`0.07190667182`，无量纲cutoff为
`1605.82135214`。

uniform probe的`E_total/E_low/E_high`分别为
`0.0215720472522/0.0194305127606/0.00214153449160`；使用冻结数值常数
`epsilon=1e-12`得到`rho_nat=0.0992735861583`。校准artifact与manifest
SHA-256分别为
`23d2ee024ac330be98ed63d151cbee4965ad89db62b841ae17442898afda3beb`和
`bb41b0c56b6ff1ddfd9223aaa6d8f7dd4d5c097cc5d94ad6e95a577f0144bd3b`。
artifact记录全部129个特征值及mesh/mass/basis/Support semantic hash，并强制
`rho_nat_calibrated=true`、`lambda_spec_calibrated=false`、
`formal_training_allowed=false`。这只通过自然性阈值的数值/证据门槛，不提供
源攻击或迁移效果证据，也不允许跳过Action-only Spectral Guard Calibration。

commit `839f5ec78f7eaae094989c394ddba01992e4a243` 已实现真实GPU
Spectral Guard runner与独立CPU bundle evaluator。
runner一次性冻结states 0--9的initial/static-scene fingerprint、MuJoCo
front-most instance alpha、全部共享纹理实例变换及clean teacher-forced token；
每轮严格按0--9逐state构图和释放，再对10个Action loss/紧凑梯度作算术平均。
每个iteration同时绑定完整、唯一且各一次的state ID/fingerprint，并另存逐state
margin/hinge、梯度L2/SHA与饱和统计。Feature、wrist、OFT及legacy
Action+Feature optimizer均被进程级护栏排除。

校准更新与后续正式trainer共同持有唯一
`FixedSupportActionTrainerCore`，由它调用既有surface-normalized step和
Surface-L∞ projection；runner不得复制更新公式。逐轮权威长表保存配置
`surface_step`和全部`SurfaceStepStats`。校准事务覆盖紧凑Surface参数、共享更新
核心、all-state provider/sampler、gradient cache及Python/NumPy/Torch CPU/CUDA
RNG；任一恢复验证失败即不生成可接受结果。

服务器随后在commit `4f9b6bd95384a94719edce117c4ba61e7d493be1`完成正式
states 0--9校准。WSL独立evaluator返回`gate_pass=true`且无failure；第0轮10个
state的fingerprint、clean token、Action loss、margin与hinge和已通过的Objective
GPU Audit逐项完全一致。6轮中每轮states 0--9完整、唯一且各参与一次，两个共享
纹理实例均被累计；Feature、wrist、OFT与legacy optimizer均未进入。

第0轮零delta只执行Action update；第1--5轮构成首个连续稳定激活窗口。窗口内
`r_high=0.778966--0.789506`，均明显高于冻结的
`rho_nat=0.0992735862`；`q_t=71.5783--326.0135`，中位数
`q_med=117.5728258`，因此按预注册公式冻结
`lambda_spec=0.000850536673`。对应逐轮加权谱梯度/Action梯度范数比为
`0.2773/0.1488/0.1000/0.0722/0.0609`；10%只在该窗口中位数处成立，不应解释
为逐轮上限。两梯度cosine范围为`[-0.7981, -0.7821]`，说明自然性项在该局部轨迹上
稳定反对Action方向；这是需要在正式训练继续监测的机制证据，不构成攻击效果或
迁移提升证据，也不得据此事后重调权重。

每轮实际Surface Step均约为`2/255`，delta L∞从`2/255`累积到`12/255`，无
像素或通道饱和，projection scale均为1；全部7项module/component/gradient-cache/
Python/NumPy/Torch CPU/CUDA恢复检查通过。权威action frames、iterations、manifest
与日志SHA-256分别为
`9b870b3e03d4a4116e80b476bd92fb276d6220b1ca98d369218cf476f4cdd1e0`、
`2c8e654c389c160093417c5c14a7a8aeb5f25d6d2798eeb0de5c334769dc6c72`、
`aa8eec38f7d7a419970fd9adab7b77216e0eb6214ace93b406d612f6a6e31336`与
`f246807e9c0484c62ac941f25f4983f11744843a13f884bf3a464d77f842679c`。
该门槛只冻结`lambda_spec`；manifest仍正确保留
`formal_training_allowed=false`，下一门槛为
`fixed_support_action_spectral_training_smoke`。

已冻结的第一项设计决定：谱方法在新候选中作为作用于最终 Surface Delta 的软
自然性正则，只惩罚高频谱能量；它不再把扰动硬限制在前 K 个谱基中，也不是与
顶点扰动相加的第二个可学习分量。

已冻结的第二项设计决定：第一版使用训练前确定且整次训练不变的 Fixed Vertex
Support；集合内顶点具有独立 RGB 优化自由度，集合外 Surface Delta 恒为零。
选点规则只允许使用 source OpenVLA 或模型无关的几何信息，不得读取 OFT 响应。

已冻结的第三项设计决定：Fixed Vertex Support 可以由少量固定的 Connected
Support Region 组成。每个区域必须按 OBJ face 邻接在曲面上连通；允许多个区域
是为了覆盖不同训练视角，但不允许直接采用无连通约束的离散 top-N 顶点。

已冻结的第四项设计决定：Fixed Vertex Support 使用 barycentric lumped vertex
mass 占总曲面 mass 的比例作为 Support Area Budget。预算约束全部连通区域的
面积总和；实际顶点数同时记录，但不作为主要规模控制量。

已冻结的第五项设计决定：Connected Support Region 的数量在训练前按 source
训练 states 的 Support Visibility Coverage 贪心确定，覆盖达标或达到较小区域数
上限时停止；随后区域数和顶点集合一起冻结。共享同一 Active Texture 的全部物体
实例必须共同参与可见面积统计。

已冻结的第六项设计决定：修正后的 source OpenVLA 多状态几何顶点梯度用于生成
Support Seed Score；该分数只控制连通区域的种子和扩张优先级，不能直接取离散
top-N。分数需要逐 state 归一化并在 OBJ mesh 图上平滑，且不得使用 OFT 梯度。

已冻结的第七项设计决定：Support Seed Score 只使用 source OpenVLA Action
objective 的梯度。Primary/Wrist Feature 梯度可以另存为诊断产物，但不得参与
种子排名、区域扩张或可见性优先级；Attack Training 的最终 loss 另行决定。

已冻结的第八项设计决定：Action-only Support Seed Score 在零 Surface Delta
参考点上计算。审计不加载旧候选、不更新纹理，也不在 Attack Training 中重算或
修改 Fixed Vertex Support；真实审计必须在 Gate 2C、Gate 2E 与 Gate 2R
通过后运行。

已冻结的第九项设计决定：每个 state 的几何顶点 Action 梯度先按全局最大绝对
分量做 Surface-L∞ 方向归一化；逐顶点原始分数等于跨 state 平均梯度敏感度乘
RGB 方向一致性。审计必须同时保存两项分量，不能只保存最终排名。

已冻结的第十项设计决定：原始稳定分数除以 barycentric vertex mass，形成单位
曲面面积 seed density。该密度只提供种子和扩张优先级证据，不单独决定最终
support，也不得直接 top-N；最终 mask 仍由连通性、总面积、跨 state 可见性和
区域数上限共同约束。产物必须同时保存原始分数、mass 和 density。

已冻结的第十一项设计决定：单位面积 density 通过
`(M + tau * L) * d_smooth = M * d` 做 mass-aware implicit Laplacian smoothing。
这里的平滑只生成 Smoothed Seed Density，不修改 OBJ 几何、Surface Delta 或
纹理；原始与平滑 density 必须同时保存。

已冻结的第十二项设计决定：使用无量纲
`alpha = smoothing_length / sqrt(total_surface_area)` 配置平滑尺度，并令
`tau = alpha^2 * total_surface_area`。第一版与现有谱基一致地使用原始 OBJ
几何；遇到 XML 非均匀缩放时必须显式重新讨论，不能静默沿用。

已冻结的第十三项设计决定：每个 Connected Support Region 采用确定性的
best-first region growing；每轮从 OBJ face-adjacent frontier 取 Smoothed Seed
Density 最高的未占用几何顶点，同分按顶点 ID 决胜。第一版只记录区域周长、面积
和 compactness，不加入额外形状 penalty。

已冻结的第十四项设计决定（第 35 项修订）：Support Visibility Coverage 同时统计
primary 与 wrist，但只有 Primary 是 source OpenVLA support construction 的约束和
Gate。Action Support Seed Score 始终只来自 source OpenVLA 主视角；wrist
coverage 降为只读诊断，不提供梯度/loss，也不参与任何 support 决策。

已冻结的第十五项设计决定（第 35 项修订）：第一版 wrist coverage 不是
词典序软目标，也不参与 support 是否通过的 Gate。它必须保存逐 state 的
raw-camera/effective-view coverage、Visibility Evidence Status 和 Primary 对照；min/mean
只在 `valid` wrist states 上计算，并同时报告分母及各状态计数。

已冻结的第十六项设计决定：Support Area Budget 是训练前给定的固定目标面积
比例，不是可以为满足 coverage 自动扩大的上限。region growing 达到目标面积后
停止；若 Primary coverage 不合格，则保存失败诊断，不允许静默增加面积。

已冻结的第十七项设计决定：Support Visibility Coverage 使用投影重心
权重，而不是把一个物理三角形整体判为“在 support/不在 support”。像素
`p` 的 support 归属为 `w_S(p) = sum_j beta_pj * 1[v_pj in S]`；内在
vertex mass 只控制固定面积预算，投影权重只控制视角覆盖证据。

已冻结的第十八项设计决定：coverage 按目标表面可见权重计算：
`C_{s,v}(S) = sum_p alpha_p * w_S(p) / (epsilon + sum_p alpha_p)`，其中完全
可见像素接近 1，抗锯齿边缘介于 0 与 1，背景和被遮挡部分为 0。
Primary coverage 必须对应 source OpenVLA 部署时的实际有效输入视野，
包含精确 center-crop/resize 语义；不得在原始相机图上直接统计。实现时
对 `alpha` 和预乘的 `alpha * w_S` 分别执行第 46 项定义的同一非负 evidence
transform 后再求比值；它与 RGB 共享视野几何，但不共享 bicubic 颜色插值核。

已冻结的第十九项设计决定（第 41 项细化）：nvdiffrast 从第一版起只生成投影
重心归属 `w_S`，正式 coverage 的可见权重 `alpha` 始终由经过有效视野变换的
MuJoCo target segmentation 生成。二者逐 `(state, view)` 的关键对齐指标是软
visible recall；IoU 和 renderer precision 仅作诊断，因为缺少场景深度的额外
nvdiffrast 投影会被正式 `alpha` 排除。但高 recall 不是完整对齐的充分条件，
precision、IoU 与 overlay 异常仍必须暂停验收。
不允许因可见性或对齐问题放宽固定 Support Area Budget。

已冻结的第二十项设计决定：第一版 wrist coverage 使用与 Primary 相同的
source-side center-crop/resize 几何，产物显式标记为 `wrist_source_crop_proxy`。
它只是排除裁剪外虚假 coverage 的保守 source-only 代理，不读取 OFT 专用
预处理，也不得把该统计描述为 OFT 真实有效视野。

已冻结的第二十一项设计决定（第 35 项修订）：每个 `(state, view)` 必须保存明确的
Visibility Evidence Status。`not_observable` 表示在有效视野内 MuJoCo segmentation
没有目标像素；其 coverage 为空并可显式排除于汇总。`invalid_alignment`
表示目标可见，但 nvdiffrast visible recall 低于冻结门槛；其 coverage 也为空，
且不得当作零。Primary `invalid_alignment` 必须使 support construction 验收失败；
wrist `invalid_alignment` 只使该只读诊断项无效，必须计入状态统计但不得否决
Primary support。

已冻结的第二十二项设计决定：alignment recall 门槛在 source 训练 states
0–9 对齐审计之后、正式生成 Support 之前冻结，并记入审计与 support 产物。
该门槛只判断 renderer 对齐证据是否可靠，不得根据某个 support 的 coverage
成败、区域数或攻击结果调整。

已冻结的第二十三项设计决定：Primary coverage 硬 Gate 使用全部 `valid`
Primary states 的最小值，即
`min_s C_{s,primary}(S) >= C_primary_min`。均值只用于报告和通过硬 Gate 后的
次级候选比较，不得用高 coverage state 掩盖单个 state 的盲区。固定面积
预算下无法满足时，support construction 明确失败，不得扩大面积。

已冻结的第二十四项设计决定：Visibility Evidence Status 扩展为四个互斥值。
在有效输入上计算 `A_obs = sum_p alpha_p / (H * W)` 并记录等效像素数：
`A_obs = 0` 为 `not_observable`；`0 < A_obs < A_obs_min` 为
`insufficient_observation`；只有 `A_obs >= A_obs_min` 时才检查 renderer recall，
未达门槛为 `invalid_alignment`，否则为 `valid`。前两者保存诊断但不进入
正式 coverage 汇总和硬 Gate；`invalid_alignment` 仍必须使验收失败。
`A_obs_min` 必须与 alignment recall 门槛一样，在 states 0–9 可见性审计后、
正式生成 support 前冻结，不得根据 support 或攻击结果调整。

已冻结的第二十五项设计决定：在固定总 Support Area Budget `B` 下，对候选
区域数 `r = 1, ..., R_max` 每次从头构造 support，并给每个区域分配相同的
`B/r` mass 份额（允许一个边界顶点的离散容差）。选择第一个通过 Primary
coverage 硬 Gate 的最小 `r`；若到 `R_max` 仍不通过则构造失败。新区域
不得追加总面积，第一版也不做区域间自适应面积分配。

已冻结的第二十六项设计决定：第一个 seed 是至少在一个 `valid` Primary
state 中可见的最高 Smoothed Seed Density 顶点。若 `r-1` 区域候选未通过，
先取 coverage 最低的 `valid` Primary state `s*`，再对未占用顶点计算
`Delta C_{s*}(i | S) = C_{s*}(S union {i}) - C_{s*}(S)`。只有超过纯浮点
容差的正增益顶点可作候选，其中 Smoothed Seed Density 最高者成为新 seed。
不得用简单顶点可见性代替投影 coverage 增益；同分按 state ID 再按顶点 ID
决胜。选完 `r` 个 seeds 后，使用新的 `B/r` 份额从头重建全部区域。

已冻结的第二十七项设计决定：新 seed 到 `S_{r-1}` 的原始 OBJ edge-length
图测地距离必须至少为已冻结的 `smoothing_length`。这把“独立区域”与 seed
证据的曲面相关尺度绑定，不增加另一个任意长度超参数。region growing
不得占用其他区域顶点，也不得加入与其他区域 face-adjacent 的顶点；
若因此无法填满 `B/r` mass 份额，该候选构造失败，不得静默缩短分离距离。

已冻结的第二十八项设计决定：谱自然性使用“高频能量比例上限”的
hinge penalty，而不是无条件最小化高频比例。对几何顶点域最终 Surface Delta
`delta [N, 3]` 定义：

```text
E_total = Tr(delta^T M delta)
E_low   = ||Phi_low^T M delta||_F^2
E_high  = max(E_total - E_low, 0)
r_high  = E_high / (stopgrad(E_total) + epsilon)
R_spec  = max(0, r_high - rho_nat)^2
```

`Phi_low` 必须在 `M` 内积下正交归一，并包含 mass-normalized 常数模态和前
`K_nat` 个非恒定低频模态。能量按 RGB 整体计算，不逐通道分别求比。
`stopgrad` 只改变 loss 的梯度，用于阻止正则主动增加低频能量稀释比例；无
`stopgrad` 的同值比例仍必须作为诊断数值记录。

第一版不另外引入 soft boundary mask。Fixed Vertex Support 外的顶点仍严格为零，
边界三角形内已通过现有重心插值在选中与未选顶点之间连续过渡。额外 taper
会引入新长度尺度、改变局部 Surface L∞ 预算，且要求 coverage 改为加权
控制能力；只有后续证据表明谱 penalty 主要被 support 边界主导时才重新讨论。

已冻结的第二十九项设计决定：在 Fixed Vertex Support `S` 冻结后、攻击训练前，
构造“support 内均匀、外部为零”的几何探针 `delta_probe = 1_S`，并使用不带
`stopgrad` 的真实能量比例冻结 `rho_nat = r_high(delta_probe)`。该探针校准
局部 support 形状和边界导致的固有谱泄漏，不读取 rollout、Action loss
或 OFT 结果。第一版不添加 margin，而是使用较弱且受监测的谱正则强度；
只有证据表明该阈值持续过紧时，才将确定性多探针校准作为后续改进。

已冻结的第三十项设计决定：谱正则训练产物必须至少逐 iteration 保存
`r_high`、`rho_nat`、hinge 是否激活、未加权/加权的谱正则梯度范数、
Action 梯度范数、加权谱梯度与 Action 梯度的范数比及余弦。汇总必须报告
hinge 激活 iteration 比例，用于区分“上限过紧”、“正则过强”和“梯度与攻击
目标相容”。若探针高频比例接近 1，正确结论是当前 Support 形状与
`K_nat` 的组合不兼容；原因不得只归于 `K_nat` 太小。

已冻结的第三十一项设计决定：第一版 Attack Training 的唯一攻击目标是
source OpenVLA Primary Action objective，总训练目标为
`L_train = L_action + lambda_spec * R_spec`，其中 `R_spec` 是第二十八项定义的无权重
谱 hinge，`L_action` 必须对当轮全部有效训练帧取均值，禁止因 frame batch size
改变谱正则的相对强度。Source Feature objective、wrist 视角和 OFT 响应都不得进入
loss；Feature 只可作为不影响更新的独立诊断产物，wrist 只保留已冻结的
coverage 语义。这一设置用于单独检验“局部全维优化 + 谱自然性约束”，
不得为追求首个候选的迁移指标而恢复 Feature loss 或目标模型梯度。

已冻结的第三十二项设计决定：第一版不再使用旧的 `255 - clean_bin`
对称 target CE，而使用零置信度 Untargeted Clean-Action Margin hinge。对每个
action token 位置 `a`，在固定 clean action sequence 的 teacher-forced prefix 下定义：

```text
m_a = z[a, y_a] - max_{j != y_a} z[a, j]
L_action = mean_a max(0, m_a)
```

`y_a` 必须来自同一精确 processor/BPDA forward 生成的 clean action sequence；
不得用前一个 adversarial token 改写后续位置的输入上下文。`m_a > 0` 时继续
压低 clean token 优势，`m_a <= 0` 时该位置停止施压；第一版不添加 confidence
margin。零 Surface Delta 的 Support Seed Score 与正式 Attack Training 必须共用该
objective 及同一 causal alignment helper。

审计与训练产物必须逐 state/iteration 保存全部 action 位置的 `m_a`、
`m_a <= 0` 的数量及 margin 的 mean/min/max。零 Surface Delta 时，clean token
必须属于同一 teacher-forced forward 的 argmax 集合，即 `m_a >= 0`。精确
`m_a == 0` 的并列最大只作显式诊断记录，不依赖标量 `argmax` 的任意
tie-break索引，也不引入浮点容差；任何 `m_a < 0` 必须作为输入/对齐失败
显式报告。新 objective 缺少 action token 时必须失败，不得继承 legacy 路径的
可反传零损失。

已冻结的第三十三项设计决定：`lambda_spec` 通过正式训练前的一次性
Spectral Guard Calibration 确定，不直接指定任意小数，也不在正式训练中动态
改权。校准从零 Surface Delta 开始，每轮对全部有效训练帧的 Untargeted
Clean-Action Margin 取均值，只执行 Action-only surface-normalized update；同时
计算但不施加无权重 `R_spec` 梯度。

两个梯度必须在 Fixed Support 上实际可学习的 RGB 参数空间 `delta_S` 中比较：

```text
q_t = ||grad_{delta_S} R_spec||_2
      / (||grad_{delta_S} L_action||_2 + epsilon)
```

第一个连续 5 轮 `R_spec > 0` 且两个梯度均有限、非零的窗口定义为
首个稳定激活窗口。取该窗口 `q_t` 的中位数 `q_med`，冻结：

```text
lambda_spec = min(1, 0.1 / q_med)
```

因此“10%”只是校准窗口附近的目标上限，不是正式训练全程硬保证。
正式训练仍必须逐 iteration 记录
`||lambda_spec * grad R_spec||_2 / (||grad L_action||_2 + epsilon)`，但不得据此
在运行中修改 `lambda_spec`。

校准最多执行
`T_cal = ceil(attack_epsilon / attack_surface_step)` 轮；当前为
`ceil((128/255) / (2/255)) = 64`。它表示等于一次从零点到 Surface-L∞ 边界的
最大累计 step 路径预算，不表示校准已遍历整个 L∞ 可达集。64轮内没有
稳定激活时使用 `lambda_spec = 1`，并强制标记
`uncalibrated_no_stable_activation`；它只表示该确定性 Action-only 校准轨迹未持续触发
护栏，不能声称后续梯度比仍小于 10%。

校准必须保存逐轮 hinge、两项原始梯度范数、`q_t`、被选窗口、`q_med`、
最终权重和校准状态。校准结束后，必须把 Surface Delta、更新/优化器状态、
梯度缓存、frame sampler 与所有相关 RNG 状态恢复到校准前快照，再开始正式训练。
任一恢复验证失败都必须阻止正式候选。

已冻结的第三十四项设计决定：第一版 Spectral Naturalness Band 固定为
mass-normalized 常数模态加原始 OBJ 上前 `K_nat = 128` 个非恒定、按特征值
升序的连续低频模态，即 `Phi_low = [phi_0, ..., phi_128]`。该子空间只由
mesh 几何和 mass matrix 决定；禁止使用 Action-selected、非连续或重排后的谱基
定义自然性，因为那会把 source policy 偏好混入自然性度量。

`K_nat = 128` 是 Akita 第一版的固定工程/历史参照，不声称为跨物体通用的
物理自然性尺度。谱基产物必须记录几何和 basis SHA-256、所有纳入模态的
特征值、cutoff `lambda_128`、总曲面面积 `A_total` 以及无量纲 cutoff
`lambda_128 * A_total`。若 `r_high(1_S)` 接近 1，只判定当前 Support 形状与
`K_nat = 128` 的组合不兼容；同一候选不得自动改为 256。只有后续独立证据
显示该频带系统性过宽或过窄时，才把 `K_nat` 作为独立消融变量。

已冻结的第三十五项设计决定：第一版 wrist coverage 是纯只读迁移性诊断，
不参与 seed selection、region growing、区域数、候选排序、support Gate 或 Attack
Training。Support 的因果链固定为“Primary Action gradient 决定 seed、Primary coverage
决定有效性、面积/连通性决定形状”。wrist 仍必须保存逐 state 的 raw-camera 与
`wrist_source_crop_proxy` coverage、四类 Visibility Evidence Status、与 Primary 对照，
并只在 `valid` wrist states 上报告 min/mean 及明确分母。后续 wrist-aware support
必须作为独立方法变体，与 Action-only support 对照，不得追溯改写第一版。

已冻结的第三十六项设计决定：Akita 第一版的 Support Area Budget 固定为
`B_target = 0.10`，即全部 Connected Support Region 共同以原始 OBJ 总内在曲面
mass 的 10% 为目标。它是该对象 MVP 的固定工程预算，不是最优值、跨 mesh
通用值，也不是“最多 10%”的硬上限；`B_actual` 只允许第二十五项规定的一个
边界顶点离散 mass 容差。

面积比例不能用于推断顶点或参数比例，因为局部网格密度可能不均匀。support
产物必须至少记录 `B_target`、`B_actual`、目标/实际 mass、OBJ 总顶点数、
`N_selected`、可训练 RGB 标量数 `3 * N_selected`，以及每个区域的实际 mass
与顶点数；不得预先声称参数量缩小约十倍。

若 10% support 已通过 Primary coverage Gate 但攻击较弱，不得在同一候选中
自动扩大面积。必须先分别检查 Action margin 是否被有效改变、谱正则是否明显
压制 Action 更新，以及 support 是否覆盖高敏感区域。只有后续独立证据明确支持
“面积不足”时，才可把 15% 或 20% 登记为新的、单独编号的方法变体，不能
追溯修改 Akita MVP。

已冻结的第三十七项设计决定：Akita 第一版固定 `R_max = 3`。在
`B_target = 0.10` 始终不变的前提下，依次从头构造 `r = 1, 2, 3` 的候选，
其每区目标 mass 份额分别为 10%、5% 和约 3.33%；实际份额仍服从一个边界
顶点的离散 mass 容差。选择通过 Primary coverage 硬 Gate 的最小 `r`，不得
在三种候选均失败时自动增加到四个区域。

`r = 1, 2, 3` 全部失败只表示在当前 `B_target = 0.10`、`R_max = 3`、seed/
growing、区域分离和等额 `B/r` 分配规则的联合约束下 Support Construction
失败，不能单独推出“10% 面积不足”。三种候选都必须保存逐 state coverage、
seed、每区目标/实际 mass、增长或分离约束失败原因，以支持区分面积、选种、
增长和等额分配问题。后续若需要改变规则，优先把自适应区域面积分配作为
单独编号的对照变体评估；不得在 Akita MVP 内静默改变面积分配、区域数或总面积。

已冻结的第三十八项设计决定：Akita 第一版固定无量纲平滑尺度
`alpha = 0.05`，因此：

```text
smoothing_length = 0.05 * sqrt(A_total)
tau = alpha^2 * A_total = 0.0025 * A_total
```

在 `B_target = 0.10`、`R_max = 3` 下，最小区域的等效圆半径约为
`0.103 * sqrt(A_total)`，所以该尺度约为其一半；这只是选择量级的启发，
`sqrt(tau)` 是隐式扩散算子的特征平滑尺度，不是严格的测地影响半径。
第一版不扫描 `alpha`。

同一 `smoothing_length` 在 Akita MVP 中继续作为新 seed 到已有 support 的最小
图测地分离距离。这是有意减少自由参数的临时绑定，不代表两种尺度在方法上必须
恒等。审计产物必须保存 `A_total`、`alpha`、`tau`、原始/平滑 density 全向量、
可复算的局部峰值统计规则及结果、候选 seed 的测地距离，以及被分离约束拒绝的
候选数和最终失败标志。峰值统计只作诊断，不参与构造或 Gate。若高价值区域因
距离约束系统性被拒绝，后续应把 seed separation 作为单独参数变体，而不得在
本候选内静默解除绑定。

已冻结的第三十九项设计决定：Akita 第一版固定
`C_primary_min = 0.20`。每个 `valid` Primary state 都必须满足：

```text
C_{s,primary}(S) >= 0.20
```

20% 是预先固定的工程性防盲区门槛，不是理论最优值，也不表示 20% 的像素
必须完全由选中顶点组成；它沿用第十七、十八项的投影重心与可见权重定义，
表示目标可见投影至少有 20% 的 barycentric 加权控制量。不得为使候选通过
而降低到 0.15，也不得因失败自动增加总面积或区域数。

每个 `r = 1, 2, 3` 候选都必须保存逐 state coverage 与状态标签，并在全部
`valid` Primary states 上报告 min、mean、median、决定 min 的原始 state ID，
以及该最差 state 的 MuJoCo 可见 mask、nvdiffrast 投影和最终 support coverage
投影图。若多个 state 同为最小值，按 state ID 决胜并保存完整并列列表。
这些诊断用于区分普遍低 coverage 与单 state 盲区，但不能改变 Gate。

已冻结的第四十项设计决定：最初预注册
`A_obs_min_candidate = 1e-3`，在 states 0–9 visibility audit 完成前不得把它
标记为正式冻结门槛。`A_obs = sum_p alpha_p / (H * W)` 只判断有效视野中的
目标观测是否足以形成稳定 coverage 分母，不衡量 support 好坏；在 `224 x 224`
输入上，候选值约等于 50 个完全可见像素的有效权重。

审计必须对 Primary 与只读 wrist source-crop proxy 的每个 state 同时保存 crop
前后 `A_obs`、`N_equiv = sum_p alpha_p`、输入尺寸、四类状态候选标签及
segmentation 图。正式阈值对两个视角使用同一数值，但 wrist 状态仍不得影响
Primary support Gate。门槛只能依据 crop/segmentation 后观测分母的可靠性冻结，
不得依据排除某个 state 后 support 是否更容易通过而调整。

预注册的边界复核带为 `[0.8e-3, 1.2e-3]`。若任一非零观测落入该区间，必须
暂停正式冻结并检查 crop、resize、segmentation 和抗锯齿权重；不得机械地把
阈值两侧的近邻 state 分成可靠与不可靠，也不得自动移动阈值。若全部观测与
候选值清晰分离且证据无误，才正式冻结 `A_obs_min = 1e-3` 并记录审计产物
SHA-256；任何改值都需要新的明确决策和理由，不能作为 support 调参步骤。

2026-08-08 全量 audit 的20个 union 观测均远离边界带，最小非零
`A_obs=0.0230380`；40个逐实例观测中30个大于门槛、10个精确为0、没有
`0<A_obs<1e-3`。因此现已正式冻结 `A_obs_min=1e-3`。

已冻结的第四十一项设计决定：最初预注册
`recall_min_candidate = 0.95`，但与 `A_obs_min` 一样，必须在 states 0–9
alignment audit 后才能正式冻结。MuJoCo 与 nvdiffrast 的 alpha 必须先进入同一
effective-view/crop 坐标系，再计算：

```text
recall_visible =
    sum_p min(alpha_mujoco(p), alpha_renderer(p))
    / (epsilon + sum_p alpha_mujoco(p))
```

`recall_visible >= 0.95` 是正式 coverage 具备足够 barycentric 对应关系的必要
Gate，不是 renderer 对齐正确的充分条件。审计必须逐 `(state, view)` 保存软
recall、precision、IoU、两张 alpha 图及 MuJoCo/renderer overlay。precision 与
IoU 第一版不设硬门槛，但若其分布存在明显异常值，或 overlay 显示位置、轮廓、
尺度或相机变换错误，必须暂停正式冻结并修复原因，不能因 recall 已通过而继续。

共享同一 Active Texture 的全部命中物体实例必须分别渲染，再以 union alpha
计算正式 recall 与 coverage；同时保存逐实例观测质量和 recall。任一达到正式
`A_obs_min` 的实例若缺失 renderer 对应或 recall 未达候选门槛，即使 union recall
通过也必须暂停验收，防止大实例掩盖小实例错误。只有 states 0–9 的 union、
逐实例指标和 overlay 均无异常且 recall 分布与 0.95 清晰分离后，才正式冻结
`recall_min = 0.95` 并记录审计产物 SHA-256；不得根据 support 成败调低门槛。

2026-08-08 全量 audit 的 Primary/wrist union 最低 recall 分别为
`0.9992251/0.9995750`，可观测逐实例最低 recall=`0.9986185`；最差
precision/IoU 与 overlay 也没有异常。因此现已正式冻结
`recall_min=0.95`。

已冻结的第四十二项设计决定：共享同一 Active Texture 的命中物体实例使用
instance-aware premultiplied aggregation 计算正式 coverage。对同一个
`(state, view)`：

```text
C_{s,v}(S) =
    sum_k sum_p alpha_k(p) * w_{k,S}(p)
    / (epsilon + sum_k sum_p alpha_k(p))
```

`alpha_k` 必须从同一次 MuJoCo instance segmentation 按实例 ID 拆分，使一个
像素只归属于场景中实际最前方的可见实例；`w_{k,S}` 必须来自该实例自己的
nvdiffrast 姿态、投影三角形和 barycentric correspondence。不得先合并 renderer
mask 后套用某一个实例的 `w_S`，也不得先算逐实例 coverage 再做无权平均。

每个实例的 `alpha_k` 与预乘项 `alpha_k * w_{k,S}` 必须分别经过完全相同的
effective-view crop/resize，变换后才跨实例求和并计算比值。这使总 coverage
按实例的实际可见投影质量加权。产物还必须保存每个具有非零观测的实例的
`A_obs^(k)`、状态和 `C_{s,v}^{(k)}(S)`，用于发现总值掩盖的实例异常；逐实例
coverage 第一版不设置额外硬门槛，只有聚合后的 Primary coverage 进入第 39 项
Gate。逐实例 renderer recall 仍按第 41 项执行对齐验收，不能与只读 coverage
诊断混为一谈。

已冻结的第四十三项设计决定：定义唯一的 Deployment Effective View Transform，
即 source OpenVLA 在 checkpoint processor 之前执行的部署空间变换。Akita
第一版固定复现现有 rollout 的 TensorFlow 语义：输入 policy RGB 为 uint8，先
转为 `[0,1]` float，按 `crop_scale = 0.9`（边长比例 `sqrt(0.9)`）做中心
`crop_and_resize` 回 `224 x 224`，clip 后按部署规则恢复 uint8，再交给已验证的
checkpoint processor。Support Seed Gradient Audit、Attack Training 和正式 rollout
必须调用同一变换定义，不允许各模块维护近似 crop。

该变换必须具有显式 BPDA：exact forward 与当前 `get_vla_action()` 部署路径
逐值对齐，backward 使用连续、坐标语义一致的可微 crop/resize surrogate。
Gate 1P 只保留为已经通过的 processor-level equivalence；新增 Gate 1D 验证
center crop 加 checkpoint processor 的 deployment-path forward equivalence，
Gate 2C 验证 crop surrogate 的 VJP，Gate 2E 验证梯度穿过两层 surrogate 回到
Surface Delta 并完成真实 update/bake/资产恢复，Gate 2R 验证 renderer surrogate
与真实 bake 响应至少同向。Gate 1D、Gate 2C、Gate 2E 与 Gate 2R 均通过前，
不得运行 Support Seed Gradient Audit、Spectral Guard Calibration 或正式
Attack Training。

Support Visibility Coverage、`A_obs` 和 renderer alignment 使用同一个
Deployment Effective View Transform 的空间坐标与采样几何。对连续证据只变换
每实例 `alpha_k` 与预乘项 `alpha_k * w_{k,S}`，不得执行 RGB uint8 量化、颜色
Normalize 或把二者先相除；变换后再按第四十二项跨实例聚合。wrist 仍只复用
该 source 变换几何并标记为 `wrist_source_crop_proxy`，不能解释为目标模型预处理。

已冻结的第四十四项设计决定：Gate 2C 必须验证 TensorFlow float
`crop_and_resize` oracle 与 PyTorch center-crop surrogate 不仅“都有梯度”，而且
把梯度传回相同输入空间位置。只比较固定 `crop_scale = 0.9` 下的输入 RGB VJP
`dL/dI`；crop box 不是可学习变量，不测试 box 坐标梯度。逐 case 保存：

```text
relative_L2 = ||g_pt - g_tf||_2 / (||g_tf||_2 + epsilon)
cosine      = cosine_similarity(g_pt, g_tf)
max_abs     = ||g_pt - g_tf||_inf
```

服务器 Gate 2C 证据通过后正式门槛冻结为 `relative_L2 <= 1e-5`、
`cosine >= 0.99999`；`max_abs` 只作诊断。若未来 framework 或实现变化后的结果
卡在门槛附近，必须先检查 dtype、坐标顺序、像素中心、边界、固定 box 常量和
插值运算顺序，不得只为通过测试放宽阈值。

确定性测试集必须同时覆盖 forward 与 VJP。forward 使用横/纵空间 ramp、
checker、中心及 crop 边界附近 impulse、固定种子随机 RGB，以暴露坐标交换、
半像素偏移与边界错误。由于固定 bilinear crop 对输入是线性算子，VJP 不依赖
输入图案，所以 VJP 必须改用多种固定上游梯度：横/纵 ramp、中心/边界/角点
impulse 与固定种子随机梯度。每个 case 均记录 shape、dtype、seed 和三项误差；
本次服务器 10-case 证据满足这一要求；case 集合不得在后续复核中删减。

已冻结的第四十五项设计决定：训练、Support Seed Gradient Audit、coverage 与
rollout 必须共享唯一的 `224 x 224` Policy Pre-Crop Canvas，但第一版不得把它
简化成 MuJoCo/direct renderer 直接输出 224。Akita 参考语义冻结为重构前后正式
rollout 已长期使用的路径：

```text
same MuJoCo observation
  -> get_libero_image(..., 512)
  -> explicit PIL RGB bicubic resize 512 -> 224
  -> Policy Pre-Crop Canvas (uint8, 224 x 224)
  -> Deployment Effective View Transform (crop_scale = 0.9)
  -> checkpoint processor
```

录像帧与 policy 路径必须在 interface 和配置上分离。即使默认录像也为 512，
改变录像分辨率、编码或保存方式不得改变 policy source resolution、pre-crop
canvas 或 action。PIL resize 必须显式指定 bicubic，不依赖 library 默认值；
产物记录 Pillow/TensorFlow 版本、输入输出尺寸和各阶段 SHA-256。若录像与 policy
source 恰为相同 512 图像，可以复用已计算数组，但只能作为不改变语义的缓存。

训练 renderer、MuJoCo instance segmentation 和 nvdiffrast correspondence 第一版
都必须在参考 512 policy-source 坐标系构造，再对 RGB、每实例 `alpha_k` 和
预乘项 `alpha_k * w_{k,S}` 使用各自数值语义下相同的 512->224 空间映射及随后
center crop。现有训练 256 路径标记为历史不一致行为，不能继续充当正式候选。

direct 224 只保留为单独的等价性审计候选。该审计至少逐 states 0–9 比较
`I_direct224` 与 `I_512_to_224` 的像素误差、clean action token/sequence、目标
segmentation、projected geometry 和 overlay；审计通过前不得替代参考路径。
若 action 发生变化，则 direct 224 属于部署输入分布变体，而不是工程重构。

已冻结的第四十六项设计决定：RGB 与连续 coverage evidence 共享 Policy
Effective View 的空间范围和 pixel footprint，但使用符合各自数值语义的重采样核：

```text
RGB:
    explicit PIL bicubic 512 -> 224
    -> TensorFlow deployment center crop

alpha_k, alpha_k * w_{k,S}:
    non-negative area downsampling 512 -> 224
    -> TensorFlow-coordinate-equivalent bilinear center crop
```

area downsampling 表示对源 pixel footprint 做非负面积平均，保持 coverage
evidence 的比例、偏序和预乘语义；不得声称 resize 前后的离散像素值总和逐值
相等。`alpha_k` 与 `alpha_k * w_{k,S}` 必须走同一个 evidence transform，变换后
再跨实例聚合和求 coverage 比值。RGB 必须继续严格复现部署 bicubic，不能因
coverage 选择 area 而改变 policy 输入。

第一版预注册 `coverage_invariant_atol = 1e-6`，并在输入与每个变换阶段执行
fail-fast 检查：所有值有限，且在容差内满足
`0 <= T(alpha_k * w_{k,S}) <= T(alpha_k) <= 1`。确定性测试还必须覆盖：

```text
w = 0  => T(alpha * w) = 0
w = 1  => T(alpha * w) ~= T(alpha)
```

违反不变量表示 kernel、坐标、实例对应或 premultiplication 实现错误；不得通过
事后 clamp 后继续构造 support。若纯浮点误差接近预注册容差，先检查 dtype、
实现和测试证据，再通过新决策决定是否正式冻结或调整容差。

已冻结的第四十七项设计决定：每个实例的原始可见权重 `alpha_k` 定义为同一
静止 MuJoCo state、同一 policy camera 下，第 `k` 个共享 Active Texture 实例的
front-most binary visibility。对每个 `(state, view)` 只执行一次 512 policy-source
分辨率的 MuJoCo segmentation render，再从同一 ID 图拆分实例 hard mask：

```text
segmentation object
  -> validate object_type == GEOM
  -> validate geom_id range
  -> geom_bodyid[geom_id]
  -> unique matching instance-root subtree
  -> alpha_k in {0, 1}
```

实现不得假定 segmentation 返回的裸整数天然是 geom ID，必须按当前 MuJoCo/
mujoco-py backend 的 object-type/object-id schema 显式解析并记录 backend/version。
非 GEOM 像素作为非目标证据计数并记录；越界 geom ID、一个 geom 同时映射多个
instance subtree、目标实例 geom 无法映射或返回 schema 不符合预期都必须
fail-fast。产物保存 object-type/ID histogram、geom→body→instance 映射表、每个
instance 的 body/geom ID 集合及原始 segmentation ID 图 SHA-256。

segmentation 必须复用 policy RGB 的 camera、geom visibility/group 设置和唯一
180°方向变换；不得由 RGB 与 segmentation 各自维护 flip。原始 `alpha_k` 不做
edge feather，soft visibility 只由第四十六项的 area downsampling 与
TF-coordinate-equivalent bilinear center crop 产生。

RGB、segmentation、目标 body pose 和 nvdiffrast correspondence 必须在不推进
环境的同一证据采集事务中生成。事务前后 `simulation time`、`qpos`、`qvel`
必须逐值完全一致；每个命中目标 body 的 `xpos/xquat` 使用预注册
`state_pose_atol = 1e-12` 检查。保存事务前后 state fingerprint 和差异；任一项
超限即使代码未显式调用 `step()` 也必须使审计失败，不能继续计算 coverage。

已冻结的第四十八项设计决定：每个实例 `k` 的投影控制量只能从该实例自己的
nvdiffrast raster 生成。对 valid renderer pixel `p`，把 raster 的 triangle ID
显式解码为 renderer face，并把库定义的三项 barycentric coordinates 与该 face
的三个 corner 一一配对；再通过现有严格 `render_vertex -> geometry_vertex`
映射取得原始 OBJ 顶点 `v_{p,j}`：

```text
w_{k,S}(p) = sum_{j=1..3} beta_{p,j} * 1[v_{p,j} in S]
```

nvdiffrast 的背景 triangle ID 与有效 ID 编码必须按库契约显式处理；当前
`raster[..., 3] == 0` 是背景，有效 ID 解码后必须落在 renderer face 范围内，
不得把背景 0 与内部 face 0 混淆。barycentric 分量顺序与 face-corner 顺序必须
通过三个 corner one-hot raster 单元测试确认，不能仅凭记忆硬编码。禁止从 RGB、
UV 像素、最近邻顶点或三角形中心近似反推 `w_{k,S}`。

预注册 `barycentric_atol = 1e-6`。输入 barycentric 与输出 `w` 必须有限，并在
容差内满足 barycentric sum 为 1 及 `0 <= w <= 1`；超限直接失败，不得 clamp。
确定性不变量测试至少包括：

```text
S = empty                 => w = 0 on all pixels
S = all geometry vertices => w = 1 on valid renderer pixels
S1 subset S2              => w(S1) <= w(S2) on every pixel
background pixel          => invalid evidence, not w = 1
```

每个实例必须保存 MVP、triangle-ID 图、valid-renderer mask、barycentric 统计/
诊断图、renderer faces 与 `render_to_geometry` 映射 SHA-256、mesh/geometry SHA
以及不变量检查结果。triangle ID 越界、face-corner 映射缺失、renderer 顶点未
映射或同一 corner 对应关系不一致均 fail-fast；不得退回最近邻猜测。

已冻结的第四十九项设计决定：新候选的 source 训练与 Support Seed Gradient
Audit 使用 Visibility-Masked Renderer Delta Composition，不再用 nvdiffrast
前景整体替换 MuJoCo 目标外观。对 512 policy-source clean RGB：

```text
I_adv = clamp(
    I_mujoco_clean
    + sum_k alpha_k * renderer_mask_k * (F_adv_k - F_clean_k),
    0, 1,
)
```

`alpha_k` 使用第四十七项同一静止 state 的 raw 512 front-most instance mask；
`F_adv_k`、`F_clean_k` 与 `renderer_mask_k` 必须来自同一实例、camera、pose、
lighting、MVP、512 坐标系和同一次 renderer evidence transaction。`F_clean_k`
固定为零 Surface Delta 的参考渲染，对 trainable texture 参数的梯度必须为零；
`F_adv_k - F_clean_k` 才提供纹理变化的局部可微 surrogate。不得混用跨 state、
跨视角或旧参数缓存。

raw hard `alpha_k` 必须逐像素互斥，并满足 `sum_k alpha_k <= 1`；第 46 项非负
evidence transform 后同样以 `coverage_invariant_atol = 1e-6` 检查该偏序。
多实例贡献使用求和而非顺序 overwrite，因此结果不得依赖实例遍历顺序。
renderer mask 只约束 nvdiffrast 有效投影，不能替代 MuJoCo occlusion；任一
alignment Gate 未通过时不得用零贡献掩盖后继续训练。

零 Surface Delta 是进入新候选的硬 Gate：逐 state 在 raw 512 RGB、Policy
Pre-Crop Canvas、center-crop RGB 和 checkpoint fused pixel values 上都要求
`MAE = 0`、`L_inf = 0`，states 0–9 的 clean action sequence 与全部 action token
必须完全一致；任何差异都说明 compositor 或输入路径错误。还必须验证
`F_clean` 对 trainable Surface Delta 的梯度为零，并在至少一个 valid renderer
pixel 上验证 `F_adv - F_clean` 对 Surface Delta 的梯度有限非零。

该 composition 只属于训练/seed-audit 梯度路径。正式 source/OFT rollout 继续
直接激活 bake PNG，由 MuJoCo 生成 observation；不得把 delta composition 引入
rollout。旧 foreground replacement 与 instance-order compositor 只保留为历史
对照，不得进入 Akita 新候选。产物保存逐实例 RGB delta 范数、可见 mask、合成
前后 RGB、clamp 饱和像素比例及所有 source fingerprint。

已冻结的第五十项设计决定：在正式 Support Seed Gradient Audit 前必须通过
Renderer-to-Bake Response Gate。探针不读取 Action 梯度、support、OFT 或 rollout
结果；第一版固定为原始 OBJ 全部几何顶点上的三个独立正向常数 RGB Surface
Delta，分别为 `(2/255, 0, 0)`、`(0, 2/255, 0)`、`(0, 0, 2/255)`。它们只验证
颜色响应管线，不属于攻击候选，也不改变 `B_target` 或 Fixed Vertex Support。

对 source states 0–9 的每个 `valid` Primary state 和每个 probe，分别生成：

```text
D_sur  = effective_view(I_renderer_delta_probe)
         - effective_view(I_clean)
D_bake = effective_view(I_mujoco_baked_probe)
         - effective_view(I_mujoco_clean)
```

两条路径必须从同一 state、camera、512 Policy Source、224 Pre-Crop Canvas 与
center crop 比较，且用该 state 的 effective-view MuJoCo `alpha` 作为权重；不得
一条在 crop 前、另一条在 crop 后。定义加权 RMS、cosine 和 relative L2：

```text
rms_alpha(D) = sqrt(sum_p,c alpha_p * D[p,c]^2
                    / (3 * sum_p alpha_p + epsilon))
cos_alpha     = <D_sur, D_bake>_alpha
                / (||D_sur||_alpha * ||D_bake||_alpha + epsilon)
rel_L2_alpha  = ||D_sur - D_bake||_alpha
                / (||D_bake||_alpha + epsilon)
```

预注册纯数值响应下限候选 `probe_rms_min_candidate = 1e-6`。任一路径不有限或
低于该值时标记 `insufficient_probe_response` 并暂停 Gate，不计算/接受不稳定
cosine。第一版必要条件是每个 probe、每个 valid state 的两项响应均充分且
`cos_alpha > 0`；它只排除反向 surrogate，是最低门槛，不声称拟合良好。更强
的 worst-state/median cosine 门槛只能在 states 0–9 审计分布后通过新决策冻结。

产物逐 probe/state 保存两张 RGB delta、alpha、weighted scatter/overlay、
`rms_alpha`、`cos_alpha`、`rel_L2_alpha`、逐通道统计、符号一致率及其明确分母。
Untargeted Clean-Action Margin 的逐 token 变化同时记录，但第一版不进入 Gate，
避免把图像 surrogate 正确性与攻击难度混为一谈。每个 probe bake 必须绑定
Runtime Asset Transaction、纹理前后 SHA-256 和 state fingerprint，完成后恢复
clean asset；不得用该诊断结果选择 seed 或调攻击 loss。

已冻结的第五十一项设计决定：Gate 2R 使用
`response_metrics.jsonl` 作为唯一权威长表，schema version 必须固定。按原始
state ID、probe channel 的确定性顺序保存全部 `10 x 3 = 30` 行；每行唯一对应
`(state_id, probe_channel)`，即使 state 不是 `valid` 也必须用明确 status 占位，
不得静默省略。每行至少包含 status、两项 weighted RMS、`cos_alpha`、
`rel_L2_alpha`、sign consistency 及分母、clean/surrogate/bake 的逐 token Action
margins，以及 state、asset、texture、code、config 与 evidence hash。

`response_metrics.csv` 只作为从 JSONL 确定性生成的便读派生文件，不能反向成为
数据源。每个 `(state, probe)` 还必须保存一个不使用 pickle 的压缩 NPZ，至少
包含未量化浮点 `alpha`、`D_sur`、`D_bake` 及 shape/dtype 元数据，使 RMS、
cosine、relative L2 和 sign consistency 可精确复算；NPZ 自身 SHA-256 写回
JSONL。PNG 不得替代这些机器可读数组。

全部 30 个组合都保存 `alpha`、`D_sur`、`D_bake`、两者 difference 和
alpha-weighted scatter 五类可视化。每个 probe 的汇总必须分别报告并保留原始
state ID：最小 cosine、最小 surrogate RMS、最小 bake RMS、median cosine；
relative L2 报告最大值与 median，sign consistency 报告最小值与 median。
并列最差值保存完整 state ID 列表。汇总还必须报告各 status 的分母/计数和最终
Gate 判断，但永远不能替代权威 30 行记录、NPZ 或逐 case 图。

## 当前禁止的捷径

- 不在 Gate 1D、Gate 2C、Gate 2E、Gate 2R 和新参数化契约冻结前启动正式
  训练或进入 OFT；
- 不把 OFT 梯度用于 source-only loss、选基或超参数选择；
- 不把旧预处理候选的成功率当成 BPDA 修正后的基线；
- 不因 total/Feature loss 更优就声称任务攻击或迁移更强；
- 不把单次10-state pilot 描述为统计显著结论；
- 不在目标模型侧执行 PNG→顶点→PNG；迁移必须直接激活 bake PNG。

## 按需阅读

- 第一阶段统一结论：
  [`openvla-spectral-phase-1-results.md`](../spectral/openvla-spectral-phase-1-results.md)
- 谱参数化与公平比较：
  [`openvla-spectral-mvp.md`](../spectral/openvla-spectral-mvp.md)
- source-only 选基审计：
  [`openvla-spectral-gradient-audit.md`](../spectral/openvla-spectral-gradient-audit.md)
- OFT fixed-state 响应：
  [`oft-transfer-response-diagnostic.md`](../spectral/oft-transfer-response-diagnostic.md)
- 跨模型像素梯度与 renderer VJP：
  [`cross-model-pixel-gradient-audit.md`](../spectral/cross-model-pixel-gradient-audit.md)
- 双视角、范数保护、动作响应与 BPDA 时间线：
  [`openvla-dual-view-siglip.md`](../spectral/openvla-dual-view-siglip.md)
