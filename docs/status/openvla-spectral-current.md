# OpenVLA 谱纹理当前状态

更新时间：2026-08-16
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
Fixed-Support Action+Spectral两步smoke实现基线：
`4a061a23365213c0d9a31348db21c2edbc6a8ad4`
Fixed-Support Action+Spectral两步smoke正式证据基线：
`f73b1833fa8f77e15de1579d8b674332bda6980d`
正式5000轮Fixed-Support Action+Spectral训练基线：
`0aca5253caa7525605c3d6ce537468bd917f8d90`
正式paired source Gate恢复评估基线：
`88e5162a9d9c1e222c240bd1a0d78fce596e3384`
Fixed-Support Action-only正式对照实现基线：
`1a2b8e5afb5500a89bdfb1eb7890b4ab30b41888`
Fixed-Support Action-only正式对照证据基线：
`3c085e792c077586065f6b692ba8e59af85a2807`
Gate 6g CPU审计核心实现基线：
`9baeadd75f67a0418828608f0684453106b0936d`
Gate 6g双终态re-bake正式证据基线：
`cfb9f77c16875a826ea4da05b435c8840833ec58`
Gate 6g state 0 C/A/B smoke实现基线：
`efc45fc963da88545bbb76683e10d7a5c69e82dd`
Gate 6g OpenVLA action codec schema修复基线：
`f8943edf9034869e41ef13bd0713c22859ee566f`
Gate 6g numerical inference replay实现基线：
`02f298a71e66ab475a14b55d9f604b5d43b5d937`
Gate 6g deployment-authority v2契约实现基线：
`3e19a7a07fd5f292a75e4f1a23e250cca337640c`
Gate 6g v2结构化响应证据绑定基线：
`575f1ccedc5a1b0a95832ddfba4a1e524ea12fbf`
Gate 6g v2 state 0正式smoke证据基线：
`9abee27ee5876d1bbaac229a2e4801e287bd3762`
Gate 6g states 0--9 formal runner实现基线：
`543d6da527c3ae1f52b5c7164b182aaa50154e65`
Gate 6g states 0--9 formal正式证据基线：
`c1c4361d5c6a5949a192b346234bba04a270ba39`
Gate 6h Scalar-Gain Counterfactual实现基线：
`88bb2d6`
Gate 6h Scalar-Gain Counterfactual正式证据基线：
`3bf8895053753a040073f6763f7a744f221d6c2d`
Gate 6i Terminal Endpoint Action GPU runner实现基线：
`953c21c22b00670e242e19f119b19037e833cf93`
Gate 6i response teacher序列rank修复基线：
`5183ee859c6361bcdc2a004a81de70bd6d28c027`
Gate 6i Terminal Endpoint Action正式证据基线：
`d6147ddee74567ea828f1ae778cf8ece1b63af0d`
Gate 6i跨环境derived归约复核修复基线：
`4fb2b6dbfe52232cbbf23649b7e022db726f47a1`
Gate 6j Radial-vs-Matched-Box实现与smoke基线：
`8cfa791a0b3d2e1f8ec9e270e78045e5ce729617`
Gate 6j独立CPU evaluator脚本入口修复基线：
`c65c7d5a402e542b33a5cfe4ebc93f543d812d37`

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
| 双视角与动态范数保护 | 机制已实现，源门槛未通过 | 两者均未把source development源攻击恢复到预设的3/10失败 |
| OpenVLA processor 预处理正确性 | Gate 1P 已通过 | states 0–9 pixel MAE/L∞=`0/0`，10/10序列和70/70 token一致 |
| Deployment Effective View 与训练反传 | Gate 1D、2C、2E已通过 | 完整forward零误差；crop VJP对齐；五级梯度、单轮更新、bake/rollout/资产恢复通过 |
| Visibility/Coverage/Compositor | Gate 2R与全部前置Gate已通过 | 20/20对齐证据、10/10零delta states与70/70 action token通过；2R正式30/30 valid且全部cosine为正 |
| Action Objective与Dense Seed Gradient | 已通过 | 新Action hinge语义、10个完整`G_s [21263,3]`及全部artifact/hash由服务器运行并在WSL独立复核 |
| Seed Score/density/smoothing | 已通过 | corrected canonical/repeat artifacts均从raw `G_s`独立复算；连续场与高分区域repeat稳定 |
| Support Construction/coverage | 已通过，Production Support已冻结 | canonical/repeat均以`r=1`、seed 829通过；冻结3514个顶点/10542个RGB标量并绑定全部上游hash |
| 谱自然性uniform Support校准 | 已通过 | 连续K_nat=128+常数频带通过数值审计；`rho_nat=0.0992735862`并由两个输入artifact独立复算 |
| BPDA 下源攻击基线 | 主候选与Action-only对照均未过门槛 | 两者paired states 10--19均为Clean 9/10、Adversarial 8/10和2/10新增失败；失败state不同，但都低于预注册3/10门槛 |
| BPDA 下 OFT 迁移信号 | 未开始 | 新源候选未过门槛前不得进入 OFT |
| Gate 6g Terminal Deployment Response | 已完成，20/20工程审计有效 | Action+Spectral有7/10训练响应，部署严格保留3/7；Action-only有6/10训练响应，部署严格保留4/6。两者均存在`lost/altered`，未观察到tie或invalid |
| Gate 6h Scalar-Gain Counterfactual | 已完成，20/20工程审计有效 | 冻结6个主case为4 `gain_sufficient` / 1 `residual_necessary` / 1 `ambiguous`；只有Action+Spectral出现`residual_necessary`，按预注册解释树进入`endpoint_or_trajectory_specific` |
| Gate 6i Terminal Endpoint Action-Gradient/Response | 已完成，20/60/2工程审计有效 | 两终态聚合梯度cosine=`0.9068`；terminal逐state retention均值上升但aggregate retention降至`0.3574/0.3423`；Dense advantage一负一正，不支持共同的终态局部Support瓶颈 |
| Gate 6j Radial-vs-Matched-Box Counterfactual | artifact/evaluator/formal runner已实现，GPU待执行 | 只读Gate 6i双终态与聚合梯度；在相同actual Surface-L∞下比较既有global radial scaling与coordinatewise box projection，不重算梯度、不训练、不rollout |
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
7. Gate 6i显示两个正式终态的聚合source Action梯度仍高度同向，但Fixed Support
   的终态局部作用是endpoint-specific：Action-only放开Support外坐标的一步更好，
   Action+Spectral则是Support一步更好。当前证据不支持把共同的终态局部Support
   瓶颈作为两者均只有2/10 source新增失败的解释，也不排除Support在训练早期
   轨迹或面积预算上的限制。

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
通过独立复核。Fixed-Support Action+Spectral正式trainer的两步工程smoke、5000轮
source training及同一source development states 10--19成对Clean/Adversarial rollout均已
完成并通过artifact完整性复核。主候选与严格匹配的Action-only control均只造成
2/10个paired新增失败，未达到3/10 source门槛；因此没有证据表明Spectral Guard
是source强度不足的主要原因。Gate 6g Terminal Deployment Response Audit的
设计现已冻结，双终态CUDA canonical re-bake preflight也已通过；state 0第二次
C/A/B smoke完成全部采集，但因Action-only训练路径一个近边界token的
generation/teacher严格对齐失败而保持`audit_invalid`。numerical replay已证明该
分叉稳定逐位复现，且不是单纯的KV-cache开关差异；它是一个近决策边界token对
执行shape敏感的测量问题，不能归因于FlashAttention或源攻击瓶颈。commit
`3e19a7a`已将默认cached generation冻结为部署行为权威，把clean-prefix teacher
forward降为训练代理诊断，并把NPZ/smoke/formal bundle升级为v2。全新目录中的
state 0 smoke与states 0--9 formal audit现均已通过工程审计；旧失败bundle仍不得
追认。Gate 6g只读比较两个已有终态在train states 0--9的Renderer Delta
Composition与真实MuJoCo Active Texture响应，不自动启动训练或调参。正式结果
表明两种终态的训练路径响应都只有一部分在部署路径严格保留，且Spectral Guard
没有显示更高保留率；这支持存在决策敏感的训练—部署surrogate gap，但不能证明
它是2/10 rollout结果的唯一或主要原因。Gate 6h进一步表明，6个主case中
4个只用全局scalar gain就能复现部署首次响应，但只有Action+Spectral的
state 7需要non-scalar residual，Action-only state 7则为ambiguous。按预注册
解释树，这是`endpoint_or_trajectory_specific`，不支持共同non-scalar
surrogate fidelity主因。Gate 6i Terminal Endpoint Action-Gradient/Response
Audit现也已完成：两个终态的局部梯度总体相似，但逐state与aggregate retention
方向相反，Dense advantage一负一正，因此没有共同terminal local Support瓶颈
证据。它只检查局部终点几何与Fixed Support的即时单步限制，不观测或归因完整
训练轨迹。Gate 6i同时暴露出两个Support radial step在边界全局投影后actual
L∞明显缩小且descent cosine为负；因此唯一下一门槛已冻结为Gate 6j
Radial-vs-Matched-Box静态反事实。其artifact/evaluator、state 0 GPU smoke和正式
states 0--9三臂runner已经实现，尚无GPU结果；当前不自动启动新训练，OFT仍不得
提前进入。

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
候选。当时预注册为held-out、现归类为source development的states 10–19至少
3/10失败才进入OFT开发期rollout。纯K=256、
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
严格为零。历史上主入口曾显式拒绝`fixed_support`正式训练，避免legacy
Action+Feature trainer绕过新Action-only目标与两级谱校准；该临时阻断已由下文
commit `d646117`的新正式分支替代，旧trainer仍不得进入该进程。

因此后续顺序正式固定为：Production Support冻结与参数化接线（当前阶段）→
`rho_nat=r_high(1_S)`校准→最多64轮Action-only Spectral Guard Calibration、
`lambda_spec`冻结及完整状态恢复→正式Action+Spectral训练→source development
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
`FixedSupportTrainerCore`，由它调用既有surface-normalized step和
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

对该门槛的实现护栏已进一步明确：从零delta只运行一步无法验收谱项，因为此时
`R_spec`及其梯度均为零。commit
`4a061a23365213c0d9a31348db21c2edbc6a8ad4`因此实现严格两步smoke。Step 0从零点
重算states 0--9，要求谱项严格为零，且Action token/loss/margin/hinge逐项复现
已通过calibration的第0轮；正式联合trainer先形成
`g_total=g_action+lambda_spec*g_spec`，再且仅调用一次共享Surface update。Step 1
在第一次更新后的非零delta上重算完整目标，要求谱hinge激活且`g_spec`有限非零，
随后执行第二次联合update。Action loss是否单步下降只记录，不是工程Gate。

smoke保存两个step的四组完整float32紧凑梯度、三个时点的完整几何Surface Delta、
逐state Action证据、两轮`SurfaceStepStats`与bake PNG。WSL evaluator从NPZ逐值
复算`lambda_spec*g_spec`和`g_total`，验证3514×3参数规模、Support外严格为零、
由delta复算step/L∞、Production Support/`rho_nat`/谱基/`lambda_spec` calibration
全部hash绑定，并要求MuJoCo实际建立Active Texture环境及XML/真实纹理完整恢复。
Feature、wrist、OFT与legacy optimizer继续由进程级和证据级双重护栏排除。

本地定向回归为`20 passed`；在显式排除本机缺失nvdiffrast或LIBERO而无法收集的
9个旧测试文件后，OpenVLA Attack CPU回归为`245 passed`。直接运行完整`tests`
在collection阶段因本机未安装nvdiffrast与完整LIBERO出现10个ImportError；这些
均发生在测试执行前，不是本次行为回归。在该实现提交时，真实CUDA import、
两步数值与资产事务仍待服务器GPU smoke验收。

服务器随后在commit `f73b1833fa8f77e15de1579d8b674332bda6980d`完成正式
两步smoke；服务器定向CPU回归为`20 passed`，runner内置独立验收及修正后的WSL
evaluator均返回`gate_pass=true`且无failure。Step 0的Action loss为
`12.2232146`、`||g_A||_2=0.0992591`，全部谱能量与谱梯度严格为零；完整token、
margin、hinge与calibration第0轮逐项一致，`g_total`逐值等于`g_A`。

第一次Action-only等价update后，Step 1的`r_high=0.7888087`、
hinge=`0.6895351`、`||g_S||_2=32.6384430`，证明正式谱分支真实激活。冻结
`lambda_spec=0.000850536673`后，`||lambda*g_S||_2=0.0277602`，相对Action
梯度比为`0.2787987`；`cos(g_A,g_S)=-0.7710031`，而
`cos(g_A,g_total)=0.9753664`，联合梯度L2为`0.0801417`。四组完整梯度artifact
逐值满足`lambda*g_S`及`g_total=g_A+lambda*g_S`。Action loss从
`12.2232146`变为`12.3392862`只作诊断；它不影响工程Gate，也不能解释为方法
效果。

两个实际Surface Step均为`0.00784313772`；完整几何delta L∞依次为
`0/0.00784313772/0.0121839214`，Support外全部三个时点均为0，像素/通道饱和
比例均为0。bake PNG与Active Texture SHA-256同为
`19a6b8ab117b5f1abb4f818615ce2c5ec490cfde88f809bcab3ba222658ea134`；MuJoCo
Active Texture环境成功建立，XML与真实纹理恢复前后hash一致且backup已删除。

权威action frames、arrays、steps、manifest、GPU log与pytest log SHA-256分别为
`57332250beaf30d624eb64a1705bf969d84c75ff22dfab54b2c8b6ddd2e2c94a`、
`0f15d4cc2709d6a506369c4928e73e8e12d88aa99a89ca892429126a750d0b66`、
`aeebe436f481e7d47379cbeb65527bffce80aa0bdd08530d9ffb924703d48258`、
`edc47ff113045825c696b6037c098e72cd132bca1f4537505c5bfdb71a46aae3`、
`2d3c62f04fb5f8004128be8ea1de235710dac68c6f5fe72a747388409c6e1628`与
`9b9376449a836ad3b5e248bdee76af89736335dec2c043ef17cac6f5e5e9d696`。

初次WSL evaluator失败只因manifest记录服务器绝对LIBERO路径且谱基被rsync到
不同相对目录，不是实验失败。commit
`afa7c06aaa158e706ee32cb25155b0fca1bb4d3f`新增路径移动回归：Production
Support、`rho_nat`、谱基和lambda manifest仍必须找到真实同hash文件；未同步的
原始mesh/texture则必须同时匹配Production Support与已接受lambda calibration的
provenance，不能静默跳过。修正后原bundle独立复核通过。smoke manifest的
`formal_training_allowed=true`只表示训练前工程Gate通过；它本身仍不是正式
5000轮训练或rollout效果证据。

commit `d6461179fbb71405c1c02fcc51f0ce585c03a7fe` 已把正式
Fixed-Support Action+Spectral source路径接入`attack_openvla.py`。入口在解析配置
后冻结Spatial task 0、Akita、train states 0--9、source development states
10--19、5000轮、
Surface Step=`2/255`、Surface-L∞=`128/255`、center crop与无Feature配置；还要求
Production Support、`rho_nat`、谱基、lambda calibration及两步smoke五项artifact
齐全，并显式绑定当前40位Git commit。GPU运行前会核验checkout HEAD和tracked
worktree，且正式分支通过延迟导入保证legacy`training`/`optimization`未进入进程。

正式trainer沿已通过的all-state Objective链路一次冻结states 0--9 clean token、
主视角、MuJoCo可见性与全部共享纹理实例。每轮按0--9逐state计算
Untargeted Clean-Action Margin Hinge的算术平均梯度，再计算冻结
Spectral Naturalness梯度，通过同一个`FixedSupportTrainerCore`形成
`g_total=g_action+lambda_spec*g_spec`并且只执行一次surface-normalized update。
逐轮JSONL保存state ID/fingerprint、margin/hinge、两项梯度范数与cosine、加权
谱/Action比、谱能量、联合残差和完整`SurfaceStepStats`；逐state证据增量落盘，
终态保存3514×3紧凑参数、loss history和唯一部署bake，不保存重复的5000轮完整
梯度数组。

独立CPU evaluator会重新复核上游两步smoke与全部SHA-256，要求5000行step、
50000行逐state证据、每轮states 0--9完整唯一、Surface step/L∞不越界、最终紧凑
参数有限且符合Production Support规模，以及loss history逐值绑定。训练结束后入口
先恢复clean asset，在同一source development states 10--19运行Clean Control，再激活manifest
绑定的最终bake运行Adversarial rollout。Gate只统计
`clean_success and adversarial_failure`：至少3/10新增失败才通过；clean原有失败
单独报告，不得计为纹理造成的失败。若Clean=10/10，才可把对抗条件的总失败数
直接解释为新增失败。

若该主候选未达到3/10新增失败，首个预注册诊断是保持同一Frozen Support、Action
objective、states、步长、预算与训练轮数，仅关闭Spectral Guard的Fixed-Support
Action-only control；在该对照之前不调整lambda、K_nat、Support或其他方法设计。
两步smoke中`cos(g_A,g_S)=-0.771`与加权谱/Action比`0.279`继续作为重点风险，
但`cos(g_A,g_total)=0.975`，目前没有依据提前改权。

本次新增定向契约为`39 passed`；排除本机缺失nvdiffrast/完整LIBERO而无法收集的
9个旧测试文件后，OpenVLA Attack CPU回归为`256 passed`。直接运行完整`tests`
仍在collection阶段得到10个相同依赖缺失ImportError。真实CUDA、5000轮训练、
正式evaluator与成对rollout均尚未运行，因此队列6e仍未通过。

服务器随后在commit `81f542facaf415f7fb9f29835b6d89ea3224e616`完成正式
训练前无GPU全量回归：`323 passed, 1 skipped`，6条输出仅来自wandb、setuptools
与robosuite的既有弃用警告，无失败或错误。准备5000轮命令时进一步发现正式长表
虽已保存`cos(g_A,g_S)`和加权谱/Action比，但遗漏已冻结风险监测要求中的
`cos(g_A,g_total)`，且长循环没有stdout心跳。该follow-up只增加联合核心计算后
的只读cosine证据、CPU evaluator范围检查和每10轮进度行，不改变梯度合成、更新、
目标、lambda或任何方法配置；修正后本地定向`18 passed`、可收集回归仍为
`256 passed`。正式GPU运行应使用包含该follow-up的最终commit。

服务器在commit `0aca5253caa7525605c3d6ce537468bd917f8d90`完成了全部
5000轮正式训练，耗时约2小时52分；5000行step、50000行逐state Action证据、
终态3514×3参数、loss history、bake PNG和formal manifest均已完整落盘。训练后
独立evaluator在step 101错误拒绝`cos(g_A,g_total)=1+O(10^-7)`，随后因循环提前
break级联报告step行数和loss history错误，因此尚未进入paired rollout。资产事务
仍在异常路径完整恢复XML与真实纹理。

对完整bundle的只读复核表明这不是训练失败：五个输出文件SHA-256全部与manifest
一致，行数精确为5000/50000，iteration和states 0--9完整唯一，loss history与
JSONL逐值一致，联合梯度残差始终为0。float32 cosine最大上溢仅
`4.79e-7`；Surface Step和L∞最大值相对理论边界也只多约`3e-8`，均属于已有数值
容差范围。修正后的WSL evaluator对原bundle返回`gate_pass=true`且无failure；
同时扩展rsync路径解析到更深的同步目录祖先，仍只接受文件名与SHA-256同时匹配。

训练机制证据初步显示Action loss从`12.2232`降至`8.55536`，全程最小值
`8.37500`；Spectral hinge在4999/5000轮激活。加权谱/Action梯度比从首个非零步
约`0.2829`降至末轮`0.0002785`，中位数`0.0007493`；
`cos(g_A,g_total)`最低`0.974777`，说明冻结guard没有改变Action主方向，但后期
约束相对Action已很弱。Surface Delta在第92次update附近首次触及预算边界，末轮
L∞为`0.501480`。这些只能解释训练机制，尚不能替代source rollout效果。

服务器在commit `88e5162a9d9c1e222c240bd1a0d78fce596e3384`使用恢复入口完成
paired source rollout，耗时约10分31秒。日志明确记录跳过全部梯度计算与5000轮
update，并直接复用原正式bake。Clean在state 15失败，为9/10成功；Adversarial在
states 10、19失败，为8/10成功。按预注册配对语义，真正的
`clean success -> adversarial failure`为states 10、19，共2/10；state 15属于
pre-existing clean failure，且在Adversarial条件下恢复成功，不能计为攻击造成的
失败。因此队列6e的科学Gate为`false`，未达到至少3/10新增失败的门槛。

`paired_source_gate.json`绑定原training manifest、原bake及恢复评估commit，WSL
独立校验器从逐state pairs重算全部字段后返回artifact完整性`gate_pass=true`且无
failure；这里的完整性通过不得与artifact内部的科学Gate `gate_pass=false`混淆。
恢复流程结束后Original XML与真实MuJoCo纹理均已恢复。paired artifact、恢复GPU
日志和rollout文本的SHA-256依次为
`0d8dfd62771dceffedb50c2d27320988864efb67158dcab45d15d2a197bb129f`、
`b53ab498fb9c6ea7ce58f63a844e5c4fc2628feaa278075055e035332563d2cb`和
`306b6f25ff1b1613ed008340b792edfad21dbc46c51e16eeb7ba2b8c68522678`。

该结果证明主候选存在非零source作用，但不足以宣称达到源攻击门槛，也不能单凭
2/10结果归因于Spectral Guard。按预注册顺序，当前唯一下一诊断是保持同一Frozen
Support、Action objective、states、surface step、Surface L∞预算和5000轮训练，
仅关闭Spectral Guard的Fixed-Support Action-only control，并同样执行paired
Clean/Adversarial rollout。若Action-only达到3/10而主候选仅2/10，只能作为
证据开始指向Guard、应优先检查它的go/no-go信号，不能凭10-state pilot和一个
state的差距宣称机制已证实；若达到4--5/10或更高，才形成较有说服力的source
强度差异。若Action-only仍约2/10或更低，则证据更指向Fixed Support或Action
objective本身。该对照完成前不调整lambda、K_nat或Support，且不进入OFT。

2026-08-10 已在commit `1a2b8e5afb5500a89bdfb1eb7890b4ab30b41888`
实现Gate 6f的严格Action-only control。CLI用显式字符串字段
`fixed_support_formal_training_variant=action_only_control`选择该变体；它仍复核
并绑定主候选相同的Production Support、`rho_nat`、谱基、Spectral Guard
calibration与两步smoke artifact，但不会实例化或计算Spectral Naturalness
Regularization。每轮同样消费states 0--9完整Action梯度，并通过同一个
`FixedSupportTrainerCore`执行一次surface-normalized step和Surface-L∞ projection。

Action-only使用独立schema，manifest同时记录同一smoke冻结的
`calibrated_lambda_spec`与本轮`applied_lambda_spec=0`；逐轮谱能量字段为JSON null，
实际谱梯度贡献为零，total gradient必须逐统计等于Action gradient。独立CPU
evaluator同时支持已冻结的旧Action+Spectral v1 bundle和新control bundle；原
`0aca525`正式bundle复核仍为`gate_pass=true`。本地定向契约为`42 passed`；排除
本机缺少nvdiffrast/完整LIBERO的9个既有collection文件后，可收集OpenVLA Attack
回归为`261 passed`。下一步只需服务器无GPU回归与正式5000轮control，不新增方法
变量。

服务器随后在commit `6cf8c12aa1b4f704af50d0ebf1bf97868900a614`完成Gate 6f
训练前无GPU全量回归：`328 passed, 1 skipped, 6 warnings in 27.03s`，无failure
或error；warning仍只来自wandb、setuptools与robosuite的既有弃用接口。同步日志
SHA-256为`02f74e85fc505ef97f2c483d74c3b928b9d0d0b3fc04382e09a319bcfdb44cc8`。
Action-only正式5000轮control现已放行。

服务器随后在commit `3c085e792c077586065f6b692ba8e59af85a2807`完成正式
Action-only control，5000/5000轮update、50000行逐state证据、3514×3终态参数、
loss history、bake与paired artifact均完整。WSL独立训练evaluator及paired
artifact evaluator均返回完整性`gate_pass=true`且无failure；运行结束后Clean
Asset的XML与真实MuJoCo纹理均已恢复。训练/paired artifact分别绑定同一执行
commit、全部上游SHA、states 0--9 fingerprints及eval states 10--19 fingerprints。

Action-only的Clean仍仅state 15失败，为9/10；Adversarial在states 10、13失败，
为8/10，因此paired新增失败仍为2/10，state 15仍属于Adversarial recovery。主候选
同样是2/10，但失败states为10、19。两者count相同而具体state不同，说明Guard改变
了优化终点，却没有改善或损害本轮总体source Gate；Action-only也未达到3/10，
因此Gate 6f完成但未通过，不进入OFT。

训练机制比较进一步支持这一判读：两者Action loss均从`12.2232`开始；Action-only
末轮/最小/末100轮均值为`9.2089/8.9938/9.1875`，主候选对应为
`8.5554/8.3750/8.5314`。二者都在第92次update首次触及Surface预算，Surface step
中位数均约`2/255`。终态紧凑参数cosine为`0.8613`，说明路径确有差异，但主候选
的训练Action目标反而更低，不能把其source失败归因于Guard压制Action优化。

末轮静态Training Frame上，Action-only已有6/10 states、11/70 token margin非正，
主候选已有8/10 states、12/70 token margin非正，但二者source-development
paired失败仍只有2/10。这说明当前teacher-forced proxy能够跨过部分训练帧token
边界，却没有稳定转化为source development任务失败；单凭现有证据仍不能区分
瓶颈主要来自Fixed Support覆盖还是静态Action proxy/轨迹泛化，因此下一轮必须
先设计可证伪诊断。

使用已经冻结且hash校验通过的同一Naturalness Band对两个终态作事后只读复算，
Action-only与主候选的`r_high`分别为`0.806436`与`0.816410`，均远高于
`rho_nat=0.099274`；主候选并未比control得到更低的终态高频比例。软hinge允许
超过rho，所以这不是实现契约违规，但它表明当前固定lambda Guard在本次5000轮中
没有实现预期的相对自然性改善。该比较是单次10-state开发实验的机制证据，不作
统计显著性结论，也不据此直接扫描lambda。

GPU日志、rollout文本、formal manifest与paired artifact的SHA-256依次为
`e5b18f245667c4c33c77a6a64029efb36c71264304bb56447e5bbc7584b1c2ae`、
`ea77ba71a4138c02cdf5dc9648346a90e7b867db018980101218d855277b41ab`、
`80b0df7bb9e9340395c0df57c870559eb0517e68c54601cca7ed0a6ad67b0b70`和
`e35335e36cfc63c6e62d2e11cb8a704c0d47b82413b49ad9548384a70765c1d6`。

原正式training manifest与GPU日志SHA-256分别为
`f7e05cacf1490846d1272bfaea4d719edbac78b8b463cfed676f423c74255e94`和
`b3e518e85d7e4ed3d4bb6c7cae24f8fc4c7276c965a32d19219a4107bd2e022d`；其余
step/action-frame/parameter/bake/loss SHA由该manifest逐项绑定。

### Gate 6g：Terminal Deployment Response Audit（re-bake已通过）

Gate 6g只回答一个问题：两个既有训练终态在Renderer Delta Composition中产生的
静态离散动作响应，是否仍存在于真实MuJoCo Active Texture部署路径。它不重新
训练、不运行rollout，也不修改Support、Action objective、Spectral Guard、
`K_nat`、`lambda_spec`、Feature、wrist或OFT。

正式输入固定为Action+Spectral与Action-only两个formal training bundle及其共同
Production Support，主审计只使用train states 0--9。每个state共享一次Clean
基准`C`；每个终态分别从紧凑Fixed-Support参数重建训练路径`A`，并激活与该参数
逐像素重放校验过的bake PNG构造MuJoCo部署路径`B`。路径A不得读取PNG并反推顶点，
路径B不得用renderer替代MuJoCo成像。

每个`(variant,state)`保存C/A/B的effective-view RGB、hard visibility alpha、最终
BF16 fused `pixel_values` bit pattern、clean-prefix teacher action logits、默认
cached自回归generation logits/token及解码7-D action。task instruction、
prompt/input IDs、action-token slice、generation配置、action codec统计、
checkpoint/processor/policy-view身份、
initial/static-scene fingerprint、全部输入/输出SHA和Runtime Asset Transaction
恢复证据必须同时绑定。GPU runner只采集事实；独立CPU evaluator从权威NPZ复算
全部margin、argmax/tie、首次分歧、动作与RGB响应指标。

主分类只比较默认cached generation相对于Clean的首次因果分歧，不要求A/B完整
自回归序列逐项相同。每步generation token必须属于同一步真实自回归score的精确
argmax集合；固定clean-prefix teacher forward只代表训练代理，其与generation的
关系继续结构化记录，但不决定部署响应分类或case有效性。
冻结分类为`no_training_response`、`deployment_lost`、
`deployment_response_altered`、`deployment_preserved_strict`、
`deployment_preserved_tie_sensitive`与`invalid_response_alignment`；最后一类
只表示generation token与其对应generation score自相矛盾。tie只指保存的
generation logits中多个类与最大值严格相等，不使用数值容差；完整序列equality、
Hamming、teacher/generation关系和top1--top2 gap只作二级诊断。汇总只报告明确
state计数及条件分母，分母为零时写
`null/not_applicable`，禁止用“大量”或“普遍”等未冻结阈值给出总体机制标签。

Gate 6g没有攻击性能pass/fail。只有全部20个唯一`(variant,state)`、parameter到
Surface Delta到重复re-bake再到bound PNG、最终processor tensor、C/B整数
segmentation与hard alpha、generation token/score自对齐、scene fingerprint、
文件inventory及资产恢复全部通过时，独立evaluator才返回`audit_valid`；
teacher/generation只读差异不得作为失败。任一其他硬条件失败均为
`audit_invalid`，不得产生科学结论或事后放宽容差。`iteration=4999`训练Action
证据在第5000次update前采集，而终态parameter/bake在update后保存，因此前者只作
state/static-scene绑定和描述性参考，不能作为终态margin逐值oracle。

实现与验收顺序冻结为：本地CPU schema/evaluator测试；服务器两个终态CUDA
重复re-bake preflight；state 0共享Clean的双终态完整C/A/B smoke；states 0--9
正式20-row采集；rsync回WSL后的独立CPU复算。成功manifest必须最后原子写入，
smoke与formal目录严格隔离；失败记录只作best-effort审计，缺少成功manifest的
目录始终无效。若Gate 6g证明训练state上的首次响应穿过部署链路，下一阶段才允许
单独设计source development states 10--19静态泛化诊断；仍不能据此宣称闭环
机器人发生了恢复。

2026-08-11，commit `cfb9f77`完成双终态CUDA re-bake preflight。Action+Spectral
和Action-only均从formal `.pt`重建完整Geometry Surface Delta `[21263,3]`；各自
连续两次canonical bake与bound PNG的decoded uint8 RGB逐像素相同，repeat/bound
mismatch均为`0`。Production Support SHA为
`686a2becc3688b0cb6bdafef840d0920fb51e584dc860a0c95733b935d22fb6d`，成功
manifest SHA为
`872ab089e96cfccb3b40b023a5d03c5c92eb1cf7d80b3cd6747d0e9469f5301e`。
服务器原子发布与rsync后的WSL按SHA有限重定位、NPZ独立复算均已通过。因此
parameter/bake配对前置条件已关闭，下一步只允许实现state 0 C/A/B smoke；该
结果尚未加载OpenVLA或MuJoCo state，不包含任何动作响应结论。

同日，commit `efc45fc`完成state 0双终态C/A/B smoke runner及其模型边界。
runner对两个variant只采集一次共享Clean，在同一Clean静止场景中从各自紧凑
parameter重建终态Surface Delta并获得Renderer Delta Composition路径A，再逐一
激活hash-bound bake获得真实MuJoCo路径B；不执行训练、backward、rollout，
也不加载Feature、wrist、OFT或legacy optimizer。成功manifest会绑定解析后的
processor/policy-view
specification、最终BF16 bit pattern、完整动作响应数组、静态场景与资产事务证据，
并在原子发布前由纯CPU evaluator复算恰好两个case及跨variant共享Clean。相关定向
无GPU回归为`70 passed`；完整本地收集仅受缺失`nvdiffrast`与LIBERO阻断。当时的
下一步是服务器state 0 smoke，尚无C/A/B动作响应结果。

首次服务器运行使用commit `5ae675a`，在写入Action+Spectral state 0 NPZ前由
CPU schema拒绝：实现错误地要求256个action token class与`bin_centers`等长，
而OpenVLA以256个等距边界形成255个连续action bin center，端点token按正式
codec执行clip。该目录只有`audit_failed.json`，没有成功manifest或权威NPZ，
不得用于动作响应结论；失败路径验证XML与真实纹理均已恢复。commit `f8943ed`
将schema严格修正为`num_centers = num_action_classes - 1`，同时保留token映射和
decoded action逐值复算，最小复现及相关回归为`70 passed`。下一步是在新目录中
重新执行state 0 smoke；这是当时的行动项，后述第二次运行已完成该行动。

第二次服务器运行使用commit `0642188`，两个variant的C/A/B均完成，XML与真实
纹理均恢复；但候选manifest在独立复核时拒绝Action-only训练路径token index 1：
generation唯一argmax为class 128，而固定clean-prefix teacher唯一argmax为
class 138。两者对应logit gap分别为`+0.375`与`-0.125`，只有class 128相差
`0.5`并跨过决策边界。目录没有成功manifest，因此整个Gate 6g smoke仍为
`audit_invalid`；Action+Spectral单NPZ的`deployment_lost`只能作为失败bundle中
的候选诊断观察，不能登记为正式Gate结论。两个原始NPZ均通过schema、processor、
visibility、RGB delta、token mapping与codec独立复算；原始诊断事实精确定位为
`action_only_control/training/token_index=1/generation=128/teacher={138}`。
KV-cache、BF16与FlashAttention只是待验证假设，尚未形成因果解释。

commit `02f298a`实现只读取上述失败NPZ与source OpenVLA checkpoint的numerical
inference replay。Fidelity按原smoke模型调用顺序对共享Clean、两个训练路径A和
两个部署路径B重复三轮；逐输入保存BF16 bit round-trip、默认generation、默认
full-teacher的token及完整`[7,256]` logits。只有全部输入同时满足输入重建、三次
repeat和原NPZ逐位一致，runner才生成attribution artifact；否则保留结构化原始
mismatch与per-input失败分类，run-level `causal_attribution_allowed=false`。
Attribution明确区分generation-prefix no-cache、clean-prefix no-cache、显式
full-teacher no-cache及完整no-cache generation，并为完整generation首次分歧后的
logits标记causal prefix不可比。默认teacher与显式full-teacher no-cache另设对照，
避免把默认`use_cache`配置与sequence-shape差异混在一起。整个命令不导入LIBERO、
renderer，不训练、不反传、不运行rollout，也不修改Gate 6g合同。新增及相邻定向
CPU回归为`30 passed`；WSL全量收集仍被本机缺失`nvdiffrast`和完整LIBERO依赖
阻断，服务器正式执行前仍需运行完整无GPU pytest。

服务器同步到目标commit后，先运行新模块定向回归和完整无GPU回归：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 \
  NUMBA_CACHE_DIR=/tmp/tex3d-numba-cache \
  /home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python -m pytest -q \
  tests/unit/openvla_attack/test_terminal_numerical_inference.py \
  tests/unit/openvla_attack/test_terminal_numerical_inference_audit.py \
  tests/unit/openvla_attack/test_terminal_openvla_response.py \
  tests/unit/openvla_attack/test_terminal_deployment_response_audit.py

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 \
  NUMBA_CACHE_DIR=/tmp/tex3d-numba-cache \
  /home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python -m pytest -q tests
```

两项均通过后再运行GPU replay；源目录必须保持原失败bundle原样，输出目录必须
全新且为空：

```bash
set -o pipefail
CODE_COMMIT="$(git rev-parse HEAD)"
RUN_ID="gate6g-numerical-${CODE_COMMIT:0:7}-20260812"
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 \
  /home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python \
  openvla/experiments/robot/libero/openvla_attack/diagnose_terminal_numerical_inference.py \
  --pretrained_checkpoint \
  /data/huangsimin/openvla-7b-finetuned-libero-spatial \
  --source_bundle_dir \
  experiments_inbox/gate6g_state0_response_0642188_20260811 \
  --output_dir "experiments_inbox/${RUN_ID}" \
  --code_commit "${CODE_COMMIT}" \
  --seed 7 \
  --unnorm_key libero_spatial_no_noops \
  2>&1 | tee "experiments_inbox/${RUN_ID}.log"

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python -m \
  openvla.experiments.robot.libero.openvla_attack.evaluate_terminal_numerical_inference \
  --manifest_path \
  "experiments_inbox/${RUN_ID}/terminal_numerical_inference_manifest.json"
```

CPU evaluator的`bundle_valid=true`只表示诊断产物完整。若`fidelity_pass=false`，
本轮仍是有效的“无法忠实重放”诊断，应直接同步bundle并停止，不得重跑LIBERO或
增加容差；只有`causal_attribution_allowed=true`才允许解释双prefix数值差异。

服务器随后在commit `9bf8f99`完成正式numerical replay。六个逻辑输入各重复三
轮，BF16输入、generation token/完整logits和teacher token/完整logits均与旧失败
NPZ逐位一致；`fidelity_pass=true`、`attribution_valid=true`。唯一原始分叉稳定
保持在`action_only_control/training/token_index=1`：默认cached generation选择
class 128，logits为`27.125`对class 138的`26.75`；默认full teacher选择class
138，二者为`26.75`对`26.625`。完整no-cache generation、generation-prefix
no-cache和clean-prefix no-cache均在128/138上精确并列`26.875/26.875`，而显式
no-cache full teacher与默认teacher逐位相同。六个输入的cached与no-cache生成
序列均相同，只有该位置的argmax集合发生变化。运行时语言模型配置记录
`_attn_implementation=eager`、BF16、`use_cache=True`，因此不能把该现象归因于
加载日志中的FlashAttention字样。上述证据支持“近边界token对推理执行shape稳定
敏感”，不支持“单纯KV-cache故障”“源攻击2/10的主要原因”或“Action objective
已被证伪”。manifest、Fidelity与Attribution SHA-256依次为
`9e49164b2576c7d0753b79af26b84ce4852055fa1dd59ae8f6dcb6b4480d18b7`、
`06159fd4adc6a1678895690e34a0d540d0ba592bd3d85a847931b82d1277ec58`和
`18986e23bc03d8a27fd26ad11796c1fbfbc543167d1661ff30572ddeecd46e0b`。

commit `3e19a7a`据此发布Gate 6g v2契约：默认cached generation是部署行为权威；
teacher forward是训练代理诊断；首次响应与tie均只从generation logits复算；仅
generation token不属于自身score精确argmax时返回
`invalid_response_alignment`。权威NPZ、smoke bundle与formal bundle schema均升
为v2，并将该authority mapping写入成功manifest；follow-up `575f1cc`又要求每个
case把包含teacher/generation诊断的完整response evaluation写入manifest，并由
CPU从NPZ逐值复核，防止诊断状态在产物发布时丢失。numerical工具保留唯一显式的
legacy-v1只读入口用于已完成诊断。旧`0642188`目录仍无成功manifest，其v1 NPZ也
会被v2 evaluator拒绝，不能通过新语义追认。相关本地CPU回归为`33 passed`；全量
WSL收集仍按预期受缺失`nvdiffrast`和LIBERO阻断。下一步只允许从commit
`575f1cc`或包含该commit的新文档基线，在全新目录重跑state 0 smoke；通过后才
实现或运行states 0--9 formal audit。

服务器拉取包含v2实现与本文档的目标commit后，先运行定向与完整无GPU回归。随后
只在前两项均通过时执行state 0；以下路径与既有正式输入逐字绑定，输出目录必须
全新且为空：

```bash
set -o pipefail
CODE_COMMIT="$(git rev-parse HEAD)"
RUN_ID="gate6g-state0-v2-${CODE_COMMIT:0:7}-20260812"

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 \
  NUMBA_CACHE_DIR=/tmp/tex3d-numba-cache \
  /home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python -m pytest -q \
  tests/unit/openvla_attack/test_terminal_deployment_response_audit.py \
  tests/unit/openvla_attack/test_terminal_openvla_response.py \
  tests/unit/openvla_attack/test_terminal_numerical_inference.py \
  tests/unit/openvla_attack/test_terminal_numerical_inference_audit.py

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 \
  NUMBA_CACHE_DIR=/tmp/tex3d-numba-cache \
  /home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python -m pytest -q tests

CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 \
  /home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python \
  openvla/experiments/robot/libero/openvla_attack/diagnose_terminal_deployment_response.py \
  --pretrained_checkpoint \
  /data/huangsimin/openvla-7b-finetuned-libero-spatial \
  --action_spectral_manifest_path \
  experiments_inbox/formal_fixed_support_source_0aca525_20260809/attack_artifacts/fixed-support-action-spectral-source-0aca525-EVAL-libero_spatial-2026_08_09-18_35_55/formal_source_training_manifest.json \
  --action_only_manifest_path \
  experiments_inbox/formal_fixed_support_action_only_6cf8c12_20260810/attack_artifacts/fixed-support-action-only-control-3c085e7-EVAL-libero_spatial-2026_08_10-09_28_56/formal_source_training_manifest.json \
  --production_support_path \
  experiments_inbox/production_fixed_support_89d3bd8_20260809/production_fixed_support.npz \
  --rebake_preflight_manifest_path \
  experiments_inbox/gate6g_rebake_preflight_cfb9f77_20260811/terminal_rebake_preflight_manifest.json \
  --output_dir "experiments_inbox/${RUN_ID}" \
  --code_commit "${CODE_COMMIT}" \
  --seed 7 \
  --unnorm_key libero_spatial_no_noops \
  2>&1 | tee "experiments_inbox/${RUN_ID}.log"
```

验收要求是存在`terminal_response_smoke_manifest.json`且其中
`schema_version=openvla-terminal-deployment-response-smoke-bundle-v2`、
`status=complete`、`response_authority`与代码冻结常量完全一致、恰有两个state 0
case；两个NPZ必须为`openvla-terminal-deployment-response-v2`并由runner发布前
独立复核。任一异常只同步新目录、日志和`audit_failed.json`，不要覆盖或复用旧
目录，也不要提前运行states 0--9。

2026-08-12，服务器在commit `9abee27`和全新目录
`gate6g-state0-v2-9abee27-20260812`完成上述v2 smoke。成功manifest为
`openvla-terminal-deployment-response-smoke-bundle-v2`、`status=complete`，
恰有两个state 0 case；WSL当前代码独立复算返回`audit_valid=true`且无failure。
manifest SHA-256为
`264344905aadcd44f1e5b3b23e30e559805dce53cf5faf01573d31383387759d`；
Action+Spectral与Action-only NPZ SHA-256分别为
`f9ad864f469c2637368b08a0e432a7c42e307549dfdc39dbf52e786ac7a61123`和
`c237d1ca98d3bc21372188262b20d907f1495d1f2e692d3ce6c09b0883cbfd7d`。
同步GPU运行日志SHA-256为
`c76d7fdeeae784430d49568f943662819284a6a1cb76ebef500ff5c716972642`。

两个case的NPZ schema、state fingerprint、共享Clean/static scene、processor
BF16 bits、C/B segmentation与hard alpha、RGB delta、token/codec映射、解码动作、
generation token/score自对齐、结构化response evaluation和资产事务全部通过；
八个上游Support/preflight/formal manifest/parameter/bake SHA也在WSL逐文件重算
匹配。XML与纹理最终恢复，临时backup已删除，两个共享纹理body均进入B路径，
训练合成没有像素或通道饱和。

本state的描述性响应为：Action+Spectral的C/A/B generation class序列分别是
`[143,128,128,127,118,113,0]`、
`[143,145,128,126,118,113,0]`和
`[143,128,128,127,118,113,0]`，因此首次训练响应位于index 1/class 145，部署
路径回到Clean，分类为`deployment_lost`。Action-only的C/A/B generation完全相同，
分类为`no_training_response`；其A路径index 1仍保留teacher class 138与generation
class 128的结构化诊断，但不影响case有效性。该单点没有“保留”响应，不能外推
两个方法的总体部署保留率或攻击效果。同步内容没有单独的pytest输出日志，因此
服务器定向/全量回归只能登记为用户运行完成，不能登记为WSL独立复核。

commit `543d6da`已实现Gate 6g唯一下一门槛所需的states 0--9 formal runner。
`audit_scope=formal`固定十个state，不能由CLI改成任意子集；模型、renderer和单个
Runtime Asset Transaction在run内复用，但每个state独立设定`seed+state_id`、共享
一次Clean并采集两个终态A/B。每行继续使用已通过的v2 NPZ与response evaluation
契约，10个state fingerprint在加载后先整体核对。任一中途异常关闭当前环境、恢复
资产并写formal v2失败记录；只有恰好20个唯一case全部落盘后，CPU evaluator才从
磁盘重载、复算并原子发布`terminal_response_manifest.json`。新增独立CPU CLI可
分别复核smoke/formal；现有v2 smoke已用该CLI重算通过。相关本地定向回归为
`43 passed`；全量WSL仍受缺失`nvdiffrast`和LIBERO阻断。

服务器同步到包含`543d6da`的目标commit后，先运行定向与完整无GPU测试；两项均
通过后才能运行formal GPU审计。输出目录必须全新且为空：

```bash
set -o pipefail
CODE_COMMIT="$(git rev-parse HEAD)"
RUN_ID="gate6g-formal-${CODE_COMMIT:0:7}-20260812"

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 \
  NUMBA_CACHE_DIR=/tmp/tex3d-numba-cache \
  /home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python -m pytest -q \
  tests/unit/openvla_attack/test_terminal_deployment_response_audit.py \
  tests/unit/openvla_attack/test_terminal_deployment_response_inputs.py \
  tests/unit/openvla_attack/test_terminal_openvla_response.py \
  tests/unit/openvla_attack/test_terminal_numerical_inference.py \
  tests/unit/openvla_attack/test_terminal_numerical_inference_audit.py \
  tests/unit/openvla_attack/test_terminal_rebake_preflight.py

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 \
  NUMBA_CACHE_DIR=/tmp/tex3d-numba-cache \
  /home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python -m pytest -q tests

CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 \
  /home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python \
  openvla/experiments/robot/libero/openvla_attack/diagnose_terminal_deployment_response.py \
  --audit_scope formal \
  --pretrained_checkpoint \
  /data/huangsimin/openvla-7b-finetuned-libero-spatial \
  --action_spectral_manifest_path \
  experiments_inbox/formal_fixed_support_source_0aca525_20260809/attack_artifacts/fixed-support-action-spectral-source-0aca525-EVAL-libero_spatial-2026_08_09-18_35_55/formal_source_training_manifest.json \
  --action_only_manifest_path \
  experiments_inbox/formal_fixed_support_action_only_6cf8c12_20260810/attack_artifacts/fixed-support-action-only-control-3c085e7-EVAL-libero_spatial-2026_08_10-09_28_56/formal_source_training_manifest.json \
  --production_support_path \
  experiments_inbox/production_fixed_support_89d3bd8_20260809/production_fixed_support.npz \
  --rebake_preflight_manifest_path \
  experiments_inbox/gate6g_rebake_preflight_cfb9f77_20260811/terminal_rebake_preflight_manifest.json \
  --output_dir "experiments_inbox/${RUN_ID}" \
  --code_commit "${CODE_COMMIT}" \
  --seed 7 \
  --unnorm_key libero_spatial_no_noops \
  2>&1 | tee "experiments_inbox/${RUN_ID}.log"

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python -m \
  openvla.experiments.robot.libero.openvla_attack.evaluate_terminal_deployment_response \
  --audit_scope formal \
  --manifest_path \
  "experiments_inbox/${RUN_ID}/terminal_response_manifest.json"
```

正式验收要求`audit_valid=true`、`case_count=20`、无failure，并逐variant报告10行
分类汇总。若GPU命令失败，只同步日志、该新目录与`audit_failed.json`；不要复用
目录续跑、拼接smoke或跳过失败state。成功后也先同步完整目录和日志，由WSL独立
复核后再解释响应分布；不得仅凭stdout汇总形成科学结论。

2026-08-12，服务器在commit `c1c4361`和全新目录
`gate6g-formal-c1c4361-20260812`完成正式20-case采集。成功manifest schema为
`openvla-terminal-deployment-response-bundle-v2`且`status=complete`；WSL使用
当前CPU evaluator从20个NPZ逐值复算得到`audit_valid=true`、`case_count=20`、
`failures=[]`。states 0--9完整、唯一且各参与两个variant；10个冻结state
fingerprint、每state共享Clean与static-scene hash、processor/visibility/token/
codec、generation token/score自对齐、文件inventory和逐state/final资产恢复均
通过。两个variant所有state的像素与通道饱和比例均为0。正式manifest与GPU日志
SHA-256分别为
`323c37fa492a433f47b71f8469a8f5a9ee50fccdd50527289cd0d641b79d2fca`
和
`6b9f2b3023827aba9f806e3d84c2d38eddabc83fb62a68f3d7eecd845a23ad1a`。
全部八个上游Support、re-bake、双终态formal manifest、parameter与bake SHA也已
在WSL从同步文件重算匹配。state 0的两个正式NPZ与先前v2 smoke对应NPZ SHA完全
相同，提供了跨运行的精确重复证据。

正式响应分布如下：

| Variant | 无训练响应 | 部署丢失 | 部署改变 | 严格保留 | tie-sensitive | invalid | 有训练响应时严格保留 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Action+Spectral | 3 | 2 | 2 | 3 | 0 | 0 | 3/7（42.9%） |
| Action-only | 4 | 1 | 1 | 4 | 0 | 0 | 4/6（66.7%） |

逐state分类为：Action+Spectral在states 1/2/9无训练响应，0/8丢失，4/7改变，
3/5/6严格保留；Action-only在states 0/1/2/9无训练响应，8丢失，7改变，
3/4/5/6严格保留。在两个variant共同产生训练响应的states 3--8上，分类只有
state 4不同：Action-only严格保留，而Action+Spectral改变；其余五个state类别
一致。没有观察到Spectral Guard提高部署保留率的信号；样本量也不足以作统计
显著性主张。`preserved_strict`只表示首次因果响应类别唯一且被保留，不表示后续
完整自回归序列逐项相同。

从同一NPZ作未预注册的描述性像素复算，令`D_A=A-C`、`D_B=B-C`：十状态
`cos(D_A,D_B)`中位数为Action+Spectral `0.9673`、Action-only `0.9622`，
`||D_B||/||D_A||`中位数分别为`0.9694/0.9609`，而
`||D_A-D_B||/||D_A||`中位数分别为`0.2543/0.2769`。因此真实bake并非总体没有
生效；更准确的解释是两条路径像素响应整体方向接近，但约四分之一量级的相对
差异足以在部分近决策边界state中丢失或改变离散动作响应。该补充不是新的Gate
判据。

Gate 6g由此完成。它直接支持“当前Renderer Delta Composition到MuJoCo Active
Texture之间存在决策敏感的surrogate gap”，但train states静态单步证据不能证明
该gap是source-development rollout仅2/10新增失败的唯一或主要原因，也不能替代
闭环rollout。它同样不支持Spectral Guard改善部署保真。下一步先讨论并冻结一个
针对A/B gap的单一机制假设和最小验证；在此之前不新训练、不调Support/K/lambda、
不进入OFT。

### Gate 6h：Scalar-Gain Counterfactual Audit（已完成）

Gate 6h只回答一个更窄的问题：对于Gate 6g中A到B首次响应发生改变的case，单个
全局标量gain能否复现B的首次响应；若不能，移除B中的非标量residual后是否恢复
A的首次响应。它是post-hoc诊断性反事实分解，不是物理renderer，也不证明该gap
是2/10 rollout的原因。

每个`(variant,state)`在完整224×224 Effective RGB和全部通道上独立计算：

```text
D_A = A - C
D_B = B - C
alpha_star = <D_A,D_B> / <D_A,D_A>
I_gain = uint8(round(clip(C + alpha_star * D_A, 0, 255)))
```

`alpha_star`不加人为范围；`D_A`平方范数为零使case无效。`round`固定为与
PyTorch/NumPy相同的ties-to-even。保存连续/量化`I_gain`、连续/量化residual、
裁剪值数量、由Gate 6g已验证的checkpoint-derived training-exact processor生成的
最终BF16 bits、完整generation/teacher logits与token；不再次center crop，也不以
连续BPDA surrogate值作为模型forward。
这里不能加目标mask，因为OpenVLA观察完整Effective View，轮廓和背景合成残差
也可能影响决策。

首次响应签名`g(I)`固定为相对于同一Clean的`None`，或
`(first_divergence_index, generation_class)`；主分类只读默认cached generation，
clean-prefix teacher和完整7-token序列继续只作诊断。20个case全部重放，但正式
机制分母严格固定为Gate 6g的6个`deployment_lost/deployment_response_altered`
case：Action+Spectral states 0/4/7/8与Action-only states 7/8。分类为：

- `gain_sufficient`：`g(I_gain) == g(B)`；不需要非标量residual即可复现B；
- `residual_necessary_for_first_response`：
  `g(I_gain) == g(A) != g(B)`；necessary只针对该case与该构造；
- `ambiguous`：`g(I_gain)`同时不同于`g(A)`和`g(B)`，不强行归因。

Gate 6h不设科学pass/fail或比例门槛。预注册解释分支为：两个variant都至少出现
一个`residual_necessary`，转向共享non-scalar surrogate fidelity机制；只在一个
variant出现，判为endpoint/优化轨迹相关，不能称为共同主因；两者均没有且
`gain_sufficient`总数严格多于`ambiguous`，优先简单scalar gain calibration；
其余情况为counterfactual不足，停止自动扩展，不在看过结果后直接选择gamma、
逐通道或局部模型。即使scalar不足，也不能声称所有photometric calibration无效。

工程有效性要求source Gate 6g manifest先独立复核；20个source NPZ SHA全部绑定；
GPU runner对每个case重放C/A/B且generated classes、完整generation/teacher logits
与Gate 6g逐值相同；gain token属于自身generation score精确argmax；反事实数组由
CPU从source RGB逐值重算；20个唯一case和冻结6-case主分母完整。任一失败使整个
bundle无效，成功manifest最后原子发布。该进程只加载source OpenVLA与已有NPZ，
不加载LIBERO/renderer/纹理参数，不训练、反传、rollout，也不使用Feature、wrist
或OFT。

commits `468ea76`/`88bb2d6`已按上述合同实现纯CPU构造/分类、无pickle NPZ、20-case bundle
evaluator、原子发布、只读OpenVLA GPU runner和可重定位WSL evaluator。实现采用
TDD纵切完成，相关Gate 6g/6h定向回归为`45 passed`；真实Gate 6g 20-case输入的
CPU preflight中`alpha_star`范围为`[0.881867,1.016489]`且裁剪值总数为0。另用
真实20-case source构造的非科学合成bundle仅验证evaluator端到端合同，得到
`audit_valid=true`和严格6-case主分母，该产物不进入实验结论。本机完整测试仍因
缺少`nvdiffrast`与LIBERO在10个既有文件收集失败；显式排除这些环境阻断文件后
其余全仓无GPU回归为`334 passed`。

服务器应同步到同时包含`468ea76`、`88bb2d6`及本节文档提交的最新HEAD，不得停在
中间实现commit；正式运行SHA由下面的`git rev-parse HEAD`动态绑定。在全新空目录
先执行不依赖LIBERO/nvdiffrast的定向无GPU回归，再运行唯一正式GPU audit：

```bash
set -o pipefail
CODE_COMMIT="$(git rev-parse HEAD)"
RUN_ID="gate6h-gain-${CODE_COMMIT:0:7}-20260812"
SOURCE_GATE6G="experiments_inbox/gate6g-formal-c1c4361-20260812/terminal_response_manifest.json"

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 \
  NUMBA_CACHE_DIR=/tmp/tex3d-numba-cache \
  /home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python -m pytest -q \
  tests/unit/openvla_attack/test_terminal_gain_counterfactual_audit.py \
  tests/unit/openvla_attack/test_terminal_deployment_response_audit.py \
  tests/unit/openvla_attack/test_terminal_openvla_response.py \
  tests/unit/openvla_attack/test_terminal_numerical_inference.py \
  tests/unit/openvla_attack/test_terminal_numerical_inference_audit.py

CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 \
  /home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python \
  openvla/experiments/robot/libero/openvla_attack/diagnose_terminal_gain_counterfactual.py \
  --pretrained_checkpoint \
  /data/huangsimin/openvla-7b-finetuned-libero-spatial \
  --source_gate6g_manifest_path "${SOURCE_GATE6G}" \
  --output_dir "experiments_inbox/${RUN_ID}" \
  --code_commit "${CODE_COMMIT}" \
  --seed 7 \
  --unnorm_key libero_spatial_no_noops \
  2>&1 | tee "experiments_inbox/${RUN_ID}.log"

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python -m \
  openvla.experiments.robot.libero.openvla_attack.evaluate_terminal_gain_counterfactual \
  --manifest_path \
  "experiments_inbox/${RUN_ID}/terminal_gain_counterfactual_manifest.json" \
  --source_gate6g_manifest_path "${SOURCE_GATE6G}"
```

正式验收只要求`audit_valid=true`、`case_count=20`、
`primary_case_count=6`、无failure，并报告三类计数与冻结解释分支；没有科学
pass/fail比例门槛。失败时只同步新目录中的`audit_failed.json`与日志，不复用
目录续跑。成功也需同步完整目录、日志和source Gate 6g bundle，由WSL通过显式
本地`--source_gate6g_manifest_path`独立复核；不得仅根据服务器stdout登记结论。

正式运行`gate6h-gain-3bf8895-20260812`绑定commit
`3bf8895053753a040073f6763f7a744f221d6c2d`，成功manifest SHA-256为
`9f9bb6fd0cd811c3e7679dde23946214430f2756b7a6f96509d127e9ea53efee`。
WSL独立evaluator重算得到`audit_valid=true`、`case_count=20`、
`primary_case_count=6`且无failure；20个C/A/B replay的token与完整
generation/teacher logits、gain processor BF16 bits、source NPZ SHA、states
0--9及双variant inventory均逐值通过。进程provenance确认没有加载
legacy optimizer、LIBERO或renderer，也没有训练或反传。

冻结6个主case的结果为：

| Variant | State | Gate 6g分类 | Gate 6h机制分类 | `alpha_star` |
|---|---:|---|---|---:|
| Action+Spectral | 0 | `deployment_lost` | `gain_sufficient` | 0.926905 |
| Action+Spectral | 4 | `deployment_response_altered` | `gain_sufficient` | 0.968170 |
| Action+Spectral | 7 | `deployment_response_altered` | `residual_necessary_for_first_response` | 0.901780 |
| Action-only | 7 | `deployment_response_altered` | `ambiguous` | 0.881867 |
| Action+Spectral | 8 | `deployment_lost` | `gain_sufficient` | 0.919174 |
| Action-only | 8 | `deployment_lost` | `gain_sufficient` | 0.933642 |

因此总计为4 `gain_sufficient` / 1 `residual_necessary` / 1
`ambiguous`。Action+Spectral为3/1/0，Action-only为1/0/1；只有前者出现
`residual_necessary`，故严格按预注册解释树进入
`endpoint_or_trajectory_specific`。这不支持把non-scalar residual称为
两个variant的共同主因，也不允许事后扩展gamma、逐通道或局部
photometric模型。4个`gain_sufficient`说明对多数已发生首次响应变化的
case，全局强度缩放已足以复现B的离散响应；它不等于证明scalar
gain是真实物理原因，也不能证明A/B gap是2/10 rollout的主因。

### Gate 6i：Terminal Endpoint Action-Gradient/Response Audit（数学范围已冻结）

Gate 6i不重跑零点Dense Seed Audit，也不把Gate 6h的
`endpoint_or_trajectory_specific`分支误写为已证明的trajectory机制。它只读正式
Action+Spectral和Action-only两个终态，回答：两个终态的source Action
局部梯度几何是否不同；以及相对Support构造时的零点，当前Fixed
Support是否丢失了更多可执行的Action方向。

对endpoint `e`与state `s`，唯一dense梯度定义为实际进入renderer的
geometry Surface Delta梯度：

```text
g_dense[e,s] = d L_action[e,s] / d Delta_surface
shape = float32 [21263,3]
```

它由共享纹理全部body instance的render-domain梯度求和，再通过严格
`render_to_geometry`映射scatter-add回原OBJ几何顶点；不用Fixed-Support
参数梯度反推，也不使用Feature、Spectral、wrist或OFT梯度。梯度在对原始
compact endpoint artifact执行一次现有functional projection后得到的唯一
float32 `realized_endpoint [21263,3]`上计算；两个反事实arm必须从该
tensor的逐值相同clone开始。

Support retention严格沿用Production Support绑定的零点Dense artifact、相同
geometry RGB flatten、相同Support mask和普通Euclidean gradient-energy定义：

```text
R[e,s] = ||M_S * g_dense[e,s]||_2^2 / ||g_dense[e,s]||_2^2
Delta_R[e,s] = R_terminal[e,s] - R_zero[s]
```

同时报告逐state retention的mean/median/负值数，以及先对十state梯度作
float32算术平均后再计算的aggregate retention；不得用逐state retention的
平均替代后者。零点逐state基线为`0.376108--0.586116`，mean/median为
`0.501823/0.516927`，零点aggregate梯度retention为`0.538463`。本Gate不新增
mass、visible-only或RGB channel weighting。

每个endpoint将十个梯度作与正式训练相同的float32算术平均`g_bar[e]`。
Dense方向为`-g_bar[e]`，Support方向先乘`M_S`再独立归一化。两个arm
不匹配L2 norm，而是在相同`surface_step=2/255`、Surface-Linf预算和
正式surface-space规则下比较放开更多坐标后的可执行能力。唯一更新顺序为：

```text
realize endpoint -> aggregate -> mask -> normalize -> add
-> global Surface-Linf projection -> post-projection step cap -> evaluate
```

`surface_step`是加入方向前的归一化目标；`step_cap_scale`是全局投影后若实际
变化仍超过`2/255`，才从原终点向投影结果作凸插值的二次限制。必须保存
unconstrained normalized step、projected step、executed step、projection residual、
L2/Linf、梯度—实际步cosine、内积与完整`SurfaceStepStats`。

功能主指标使用与训练一致的states 0--9 Action hinge算术平均：

```text
I_D[e] = L_0[e] - L_D[e]
I_S[e] = L_0[e] - L_S[e]
A[e] = I_D[e] - I_S[e] = L_S[e] - L_D[e]
A[e,s] = L_S[e,s] - L_D[e,s]
```

Action hinge、逐token margin与logits是主响应；cached generation只作离散诊断。
`exact_equal`只表示保存的float32 Action loss逐位相同，不自动意味logits、
margin或generation相同。

权威inventory固定为20个唯一gradient case key
`(endpoint,state_id)`，以及60个唯一response key
`(endpoint,state_id,arm)`，其中`arm in {baseline,dense,support}`。每个gradient case必须恰好
对应三个response，不得缺失、重复、混用endpoint或改写原state ID。raw
gradient、realized endpoint、三类步、logits/margins/tokens与projection evidence由GPU
产生并作为authoritative raw evidence逐文件hash绑定。CPU evaluator不重跑OpenVLA
forward/backward；它只从已保存原始数组独立复算retention、Delta_R、cosine、
L2/Linf、projection residual、A[e]/A[e,s]、mean/median/sign counts、array
shape/dtype/SHA和inventory完整性。

Gate 6i只设工程`audit_valid/invalid`，不在evaluator输出
`support_bottleneck=true/false`，也不事后增加“明显”或“科学相等”阈值。任一raw evidence、
hash、shape/dtype、唯一键、基线绑定或派生量复算失败都使整bundle无效；成功
manifest必须在全部CPU复核通过后最后原子发布。

即使Dense单步更好，也只支持“放开当前Fixed Support后终态附近的一步
Action能力增加”，不能区分Support位置选择与10%曲面面积预算，不能外推至重训或
rollout。Dense/Support接近也只表示两个现有终态附近未观察到即时局部限制，
不排除Fixed Support在早期训练阶段已限制轨迹。

2026-08-15完成第一个TDD纵切：新增无pickle gradient/response case NPZ schema、
完整256类Action logits与224×224 Effective View shape/dtype复核、20/60唯一
inventory，以及只消费GPU raw evidence的纯CPU派生核心。它会独立复算Action
hinge、逐state/aggregate retention、`Delta_R`、`A[e]`/`A[e,s]`、sign counts和
双终态梯度cosine，并强制绑定state fingerprint、realized endpoint SHA、每个arm
的Surface Delta SHA与clean Action target；任一失败不返回部分指标。新模块定向
测试为8项，与既有OpenVLA response和Surface Step测试合并为24项通过。该纵切尚未
实现endpoint step artifact、bundle/成功manifest或GPU采集runner，不能据此声称
Gate 6i已经完成。显式排除本机因缺少`nvdiffrast`或完整`libero.libero`而无法
收集的10个既有文件后，其余本地CPU回归为`342 passed`；默认全量命令在上述10个
collection error处停止，未伪记为通过。

第二个TDD纵切随后补齐每个endpoint的完整step NPZ与bundle evaluator。step
artifact同时保存Dense/Support的共同realized endpoint、masked gradient、
unconstrained normalized step、projected/executed step、projection residual及完整
`SurfaceStepStats`；CPU以`1e-6`纯数值容差复算冻结更新顺序，并从raw数组独立复算
L2/Linf、raw gradient--step内积和descent alignment cosine。这里明确保留
“全局投影可使actual step小于`2/255`”的正式语义，不错误要求两arm实际Linf相等。

完整bundle固定为20 gradient、60 response和2 endpoint-step artifact。evaluator
先独立复核原Dense Seed states 0--9与Production Support、共同mesh/mapping、state
fingerprint和clean Action target，再验证两个正式终态manifest/compact parameter的
bundle内逐字节副本及SHA、全部case文件SHA、realized endpoint/三arm Surface SHA和
derived逐值一致性；冻结processor specification也必须hash绑定，最终BF16 fused
input由每条224×224 Effective View在CPU逐位复算。终态的两个轻量输入必须复制到
bundle内部并使用安全相对路径，
以便rsync后在WSL复核；Dense Seed/Production Support允许通过evaluator参数覆盖路径，
但内容SHA不得改变。publisher先验raw candidate，再加入CPU derived后二次验收，最后
才原子发布`status=complete`的成功manifest；失败候选会删除且不能留下成功文件。
新增bundle测试4项，相关Dense Seed/Production Support/endpoint定向测试共23项通过，
全部本地可收集CPU回归更新为`346 passed`。至此CPU artifact/evaluator合同完成。

commit `953c21c`进一步实现正式GPU采集runner与独立CPU CLI。runner先用既有formal
evaluator和SHA复核两个终态，把3514个compact Support坐标scatter到全顶点
Geometry参数化并固化唯一realized endpoint。raw梯度不读取终点参数`.grad`：它
从每个共享纹理实例实际进入renderer的Surface Delta hook取VJP，按实例求和后通过
严格seam mapping `index_add`回`[21263,3]`，从而避开Linf边界处functional
projection Jacobian对参数梯度的改变。每个endpoint的十状态梯度完成后，Dense和
Support分别从同一realized tensor复用正式`surface_normalized_step_`；runner保存
完整step arrays/`SurfaceStepStats`，再采集三arm的60份exact Effective View、BF16
bits、teacher logits和cached generation响应。正式checkout/commit、state
fingerprint、clean Action target、共享实例数、processor exact forward和终态输入
副本SHA均在运行时校验；Feature/wrist/OFT/legacy optimizer/rollout仍被显式排除。
成功manifest仍只能在CPU evaluator复核全部20/60/2 inventory后最后原子发布。
新增4项静态科学边界测试，与Gate 6i已有合同测试合计16项通过。GPU正式audit尚未
运行，因此Gate 6i仍没有科学结果。

第一次正式GPU尝试在commit `3c5a467`完成20/20 gradient和2/2 endpoint step后，
于第一条`action_spectral/state0/baseline` response的runner自检停止；没有写出任何
response或成功manifest。根因不是teacher token改变，而是
`OpenVLAActionResponse.teacher_input_ids`按合同保存为`int64 [L]`，runner却与模型
边界的`int64 [1,L]`直接作shape-sensitive `array_equal`。commit `5183ee8`把该边界
收敛为纯CPU校验函数：只允许单batch二维输入与一维response逐token相等，仍拒绝
额外batch、dtype/长度错误和真实token漂移，并增加对应回归测试。修复后Gate 6i及
OpenVLA response相关19项测试通过。第一次目录没有完整inventory，不得续跑、拼接
或提取科学结论；正式audit必须在新commit和全新目录从头运行。

第二次正式运行`gate6i-terminal-endpoint-d6147dd-20260815`完成20/20 gradient、
2/2 step与60/60 response，成功manifest SHA-256为
`7d0806d5782af139363c05bbd02b62dd141237a0d987c7124f174c390c1f412d`，日志
SHA-256为`3a43ebbd4d4155a586810a05ec12ef5c854ee2480f1aae263c0bd1de1a8ed32f`。
rsync后WSL复核首先只在缓存`derived`的33个float64 dot/norm字段失败；两边177个
叶子键完全相同，最大差仅`5.55e-16`，而`require_derived=False`下全部raw
evidence已经有效。commit `4fb2b6d`因此只对derived有限浮点叶子使用`1e-12`
绝对/相对容差；raw arrays、SHA、float32 Action、结构和离散字段仍严格校验，
`1e-9`派生篡改回归继续失败。修复后原bundle无需GPU重跑，WSL独立复核为
`audit_valid=true`且无failure；Gate 6i相关19项测试通过。

两个终态的聚合梯度cosine为`0.906771`；逐state cosine除states 4/6的
`0.385136/0.458199`外，其余为`0.781119--0.919784`，说明终态梯度几何总体仍
相似而非整体分叉。零点逐state retention mean/median=`0.501823/0.516927`；
Action+Spectral终态为`0.570354/0.548889`且仅3/10 `Delta_R<0`，Action-only为
`0.563780/0.505823`且4/10 `Delta_R<0`。因此逐state层面没有观察到Support内能量
普遍减少；但先平均十状态梯度后的aggregate retention从`0.538463`分别降至
`0.357413/0.342277`。两种统计方向相反，指向跨state方向抵消结构，不能简化为
“梯度整体逃出Support”。

Action+Spectral的Dense/Support mean improvement分别为`0.055357/0.091964`，
主功能量`A=L_support-L_dense=-0.036607`，Support在6/10 states更好；Action-only
分别为`0.083928/0.024107`，`A=+0.059821`，Dense在7/10 states更好。该一负一正
结果不支持双终态共同的即时局部Support瓶颈。还需注意两Support arm在全局Linf
投影后的actual Linf仅`0.002501/0.003050`，小于Dense的`0.007809/0.007802`；其
descent-alignment cosine为`-0.701719/-0.234839`，说明边界投影显著改变了Support
实际步。故该结果是“正式相同更新规则下的可执行一步”证据，不能被改写成相同步长
的无约束坐标消融，更不能外推为完整训练轨迹或rollout因果。

### Gate 6j：Radial-vs-Matched-Box Counterfactual（已实现，GPU待执行）

Gate 6j只读Gate 6i已通过的两个正式终态、聚合Action gradient、Support radial
step和states 0--9 clean target；不得重新backward、重训、rollout或读取
Feature/wrist/OFT。对每个endpoint，先从Gate 6i保存的normalized step构造
coordinatewise box point：

```text
box_surface = clip(realized_endpoint + normalized_step, -epsilon, epsilon)
box_step = box_surface - realized_endpoint
s = ||radial_step||_inf / ||box_step||_inf
matched_step = s * box_step
```

硬合同要求box step有限非零、`0 < s <= 1`、matched endpoint仍在Surface-L∞预算
内，且matched与parent radial的actual Surface-L∞相同。raw box/matched arrays、
SHA、L∞/L2、gradient-step inner product与descent cosine逐endpoint保存。两步即使
actual L∞相同，L2、坐标分布与方向仍可能不同，因此本Gate不是纯方向消融。

正式inventory固定为两个endpoint × states 0--9 ×
`{baseline, radial_support, matched_box_support}`，共60条response。新采集的40条
baseline/radial raw arrays、dtype、shape、SHA、BF16 bits、teacher/generation
logits及离散字段必须严格重放parent Gate 6i；只有从raw float32数组得到的
float64 derived归约允许`1e-12`绝对容差。replay失败只使本次audit invalid，不自动
把原因归为代码错误或backend nondeterminism。

两个endpoint分别报告，不先跨endpoint平均：

```text
D[e,s] = L_radial[e,s] - L_matched[e,s]
I_M[e,s] = L_baseline[e,s] - L_matched[e,s]
```

每项保存mean、median、逐state值与positive/zero/negative counts；evaluator不输出
`projection_harm`科学布尔值，也不暗设`>5/10`门槛。即使两个endpoint都出现
`D>0`且`I_M>0`，也只支持terminal-local box projection优于当前radial算子，不能
证明原2/10 rollout或完整训练轨迹的根因；是否替换正式trainer投影并完整重训必须
另行决策。

commit `057a897`实现无pickle matched-step/三臂response artifact、严格parent replay、
自包含child bundle、CPU evaluator与正式60-response GPU runner；commit `8cfa791`
增加`--smoke_only`，固定只采双endpoint的state 0共6条response并标记
`formal_bundle=false`，不得当作正式结果。commit `c65c7d5`补齐独立CPU evaluator
的仓库脚本路径入口与回归测试。Gate 6i/6j相关定向测试为`25 passed`；
默认全量本地命令仍在10个既有文件的collection阶段因缺少`nvdiffrast`或完整
`libero.libero`停止。下一步必须先在绑定commit的服务器新目录运行state 0 smoke，
通过严格parent replay后才在另一个全新目录运行正式states 0--9 audit。

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

- Gate 6j是当前唯一下一门槛；必须先完成state 0 GPU smoke与严格parent replay，
  再在全新目录运行正式states 0--9三臂audit。两次运行不得拼接；
- Gate 6j不得重新计算梯度、训练、rollout，或修改Support/K/lambda/Action
  objective/Feature/wrist/OFT；不得把matched-box静态结果包装为完整trajectory、
  2/10 rollout根因或纯方向消融；
- 不根据Gate 6h结果事后扩展gamma、逐通道或局部photometric模型，也不得
  把单一Action+Spectral case的`residual_necessary`包装为双variant共同主因；
- 不把 OFT 梯度用于 source-only loss、选基或超参数选择；
- 不把旧预处理候选的成功率当成 BPDA 修正后的基线；
- 不因 total/Feature loss 更优就声称任务攻击或迁移更强；
- 不把单次10-state pilot 描述为统计显著结论；
- 不再把已参与方法选择的states 10--19称为无偏held-out test；
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
