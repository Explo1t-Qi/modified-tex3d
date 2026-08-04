# OpenVLA 谱纹理当前状态

更新时间：2026-08-04
代码基线：`771907c`（文档重组前的最新功能提交）

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
| OpenVLA 训练预处理正确性 | BPDA forward 数值门槛已通过 | states 0–9 pixel MAE/L∞=`0/0`，10/10序列和70/70 token一致；真实 backward smoke 待验 |
| BPDA 下源攻击基线 | 未建立 | 必须在多状态 processor 等价和单轮更新 smoke 后重新训练 |
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
  精确复刻 MuJoCo 光照，也不解决 rollout center-crop 或轨迹覆盖。

真实 Spatial checkpoint 的 CPU 差分记录为 fused pixel values MAE/L∞=`0/0`，
输入梯度有限且非零；文档记录当时全量 CPU 回归为 `113 passed, 1 skipped`。
服务器十状态 forward-only 已进一步通过数值门槛；它仍不能替代真实训练 smoke，
也没有覆盖 rollout 的额外 center-crop。

## 下一实验门槛

### Gate 1：十状态 BPDA forward-only（数值门槛已通过）

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

### Gate 2：真实单轮 backward/update/bake smoke

Gate 1 通过后，验证 BPDA surrogate 能产生有限非零梯度，并检查：

- `[256,3]` 系数真实更新；
- Actual Surface Step 不超过 `2/255`；
- Max Surface Delta 不超过 `128/255`；
- UV PNG、Active Texture、rollout 和 Runtime Asset Transaction 正常。

### Gate 3：建立 BPDA 下的新源候选

前两项通过后，先冻结“谱自然性约束 + 选定顶点全维优化”的最小参数化、预算与
顶点选择规则，再运行 train states 0–9 的单一候选。held-out states 10–19 至少
3/10失败才进入 OFT 开发期 rollout。纯 K=256、`rho=1.0` 不再是自动下一候选；
新设计尚未冻结前不写入正式实验队列。

## 当前禁止的捷径

- 不在 Gate 2 和新参数化契约冻结前启动正式训练或进入 OFT；
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
