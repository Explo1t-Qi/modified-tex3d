# OpenVLA 谱纹理实验账本

本文按证据顺序记录影响研究决策的实验。它是索引和判定账本，不替代专题文档
中的完整命令、逐状态数据和 artifact 路径。当前开发入口见
[`openvla-spectral-current.md`](../status/openvla-spectral-current.md)。

## 口径

- “任务成功率”越低表示攻击越强；只有 clean control 为100%时，
  `1 - task success` 才能直接解释为本组观测到的攻击成功率。
- 默认 Spatial Task 0 / `akita_black_bowl` 使用 train states 0–9、held-out
  states 10–19、5000轮、Surface L∞=`128/255`、每轮最大 Surface Step=`2/255`。
- 10-state pilot 只承担 go/no-go，不代表统计显著性。
- “工程通过”只证明数据流、梯度、预算、产物和资产事务正确，不等价于方法有效。
- 2026-08-03 之前依赖历史六通道 Action/last-hidden 预处理的训练结果统一标记为
  **旧预处理基线**。它们仍能支持参数化和诊断链路的工程结论，但不能替代 BPDA
  修正后的科学基线。

## 决策实验

| 日期 | 实验/变量 | 源 OpenVLA | OFT/诊断结果 | 判定 | 详细记录 |
|---|---|---|---|---|---|
| 2026-07-24 | 第一版重构5000轮 Spatial 回归 | 50%任务成功 | 未测 | 只证明重构流程；使用旧 offset 和 nvdiffrast 正式合成，不作科学基线 | [重构架构](../refactor/openvla-stage-1-architecture.md) |
| 2026-07-24 | renderer 对齐与正式评估修正 | Spatial zero-offset IoU `0.9990` | 正式 rollout 改为 MuJoCo 同源图像 | 训练 renderer 只作梯度近似；评估不得重画物体 | [重构架构](../refactor/openvla-stage-1-architecture.md) |
| 2026-07-25/26 | Geometry Vertex vs Spectral K=128，last-hidden | Geometry 90%；K=128 70%任务成功 | 两者 OFT 均100% | 谱方法以约1/166参数保持源攻击，但未迁移 | [谱 MVP](../spectral/openvla-spectral-mvp.md) |
| 2026-07-26 | K=128 Shared-SigLIP | 80%任务成功，失败 states 10、13 | OFT 100% | 共享 feature 目标未产生迁移 | [谱 MVP](../spectral/openvla-spectral-mvp.md) |
| 2026-07-27 | 连续 K=256 Shared-SigLIP | 70%，失败 states 10、13、15 | OFT 100% | 源攻击增强仍不能单独解决迁移 | [阶段一总结](../spectral/openvla-spectral-phase-1-results.md) |
| 2026-07-27 | 连续 K=512 Shared-SigLIP | 90%，失败 state 13 | 未进 OFT | K 增加不单调；更优 Feature loss 不等价于更强任务攻击 | [选基审计](../spectral/openvla-spectral-gradient-audit.md) |
| 2026-07-27 | Object Task 0 / alphabet soup K=512 | 80%，失败 states 11、17 | 未测 | 证明谱链路可扩展到另一 mesh/task/checkpoint，不用于比较 K | [选基审计](../spectral/openvla-spectral-gradient-audit.md) |
| 2026-07-27 | source-only K=512 零点梯度审计 | K=128/256 Action能量43.61%/69.45%；Feature 37.71%/63.65% | OFT不参与 | 纯 Feature stable-score 会牺牲部分 Action 能量，不立即盲选 | [选基审计](../spectral/openvla-spectral-gradient-audit.md) |
| 2026-08-01 | OFT 五状态 fixed clean/adv 响应 | 使用旧 K=256纹理 | 双视角 SigLIP相对L2 16.28%，Action chunk 2.95%，夹爪0/40翻转 | 信号在共享视觉层存在，进入动作头后明显衰减 | [OFT响应](../spectral/oft-transfer-response-diagnostic.md) |
| 2026-08-01 | 五状态跨模型像素梯度 | Source Feature↔Action cosine `0.339` | 跨模型 Feature/Action `0.529/0.138`；OFT腕部/主视角Action norm `3.88×` | 共同视觉方向存在，Action共同方向弱，视角覆盖是独立瓶颈 | [像素梯度](../spectral/cross-model-pixel-gradient-audit.md) |
| 2026-08-01 | K=256 renderer VJP，多实例修正 | Source方向跨state一致性可用 | 跨模型主视角 Feature/Action `0.662/0.228`；Source→OFT双视角Action `0.133`；腕部 norm `3.31×` | 谱子空间不是主要瓶颈；进入 source-only 双视角机制 | [像素梯度](../spectral/cross-model-pixel-gradient-audit.md) |
| 2026-08-02 | 双视角 Shared-SigLIP K=256 | 90%，仅1/10失败 | 按门槛跳过 OFT | 两视角 Feature 均增强但 Action 变差；先保护源 Action | [双视角](../spectral/openvla-dual-view-siglip.md) |
| 2026-08-02 | 双视角零点与终点梯度审计 | Action/Combined Feature终点 cosine `-0.0106`；加权范数比 `1.7342` | OFT不参与 | 不是强负向冲突；优先动态范数保护而非 PCGrad | [双视角](../spectral/openvla-dual-view-siglip.md) |
| 2026-08-02 | `rho=1.0` 动态范数保护 | 90%，仅1/10失败；保护5000/5000轮触发 | 按门槛跳过 OFT | Action loss改善但源门槛未过；不直接扫描 rho | [双视角](../spectral/openvla-dual-view-siglip.md) |
| 2026-08-02 | 最终纹理回放 train states 0–9 | 仍仅1/10失败，Active Texture hash一致 | 未测 | 状态过拟合不是主要解释，进入 action-response | [双视角](../spectral/openvla-dual-view-siglip.md) |
| 2026-08-03 | 源 action-response 根因诊断 | collector vs 手工 clean 平均4.7/7 token不同 | 未测 | 发现历史 DINO/SigLIP 顺序及 resize 错误；冻结 K/rho/loss/OFT | [双视角](../spectral/openvla-dual-view-siglip.md) |
| 2026-08-03 | processor-equivalent state0 forward smoke | 真实processor与可微输入7/7 token一致 | 未测 | 单状态前向放行 | [双视角](../spectral/openvla-dual-view-siglip.md) |
| 2026-08-03 | processor-equivalent 单轮 update smoke | 768个系数均更新；Surface Step/Delta=`2/255` | state10 rollout成功，仅作工程检查 | backward、bake和资产事务放行 | [双视角](../spectral/openvla-dual-view-siglip.md) |
| 2026-08-03 | processor-equivalent K=256正式源候选 | 100%，0/10攻击成功 | 按门槛跳过 OFT | 训练数值正常但源攻击未建立 | [双视角](../spectral/openvla-dual-view-siglip.md) |
| 2026-08-03 | 正式候选 train-state action-response | clean→adv平均3.6/7 token变化；processor仅6/10序列匹配 | 未测 | 微小 tensor/PIL差异足以翻转近 tie，暂不能归因 target 或轨迹 | [双视角](../spectral/openvla-dual-view-siglip.md) |
| 2026-08-03 | 十状态 margin audit | 首次分叉处 tensor/processor margin均值 `0.4375/0.15625` | 未测 | 支持近 tie 被插值差异翻转，决定使用精确 PIL forward BPDA | [双视角](../spectral/openvla-dual-view-siglip.md) |
| 2026-08-04 | PIL-forward BPDA CPU 差分 | fused/分支 MAE/L∞=`0/0`，输入梯度有限非零；记录为113 passed、1 skipped | 未测 | 代码侧放行；十状态 GPU forward-only 尚未运行 | [当前状态](../status/openvla-spectral-current.md) |
| 2026-08-04 | 十状态 BPDA forward-only | pixel MAE/L∞=`0/0`，10/10序列、70/70 token一致；clean/processor teacher-first均为1 | 未测 | processor等价数值门槛通过；bundle未含stdout/stderr与资产hash | [双视角](../spectral/openvla-dual-view-siglip.md) |
| 2026-08-07 | Gate 2C center-crop forward/VJP | 5类forward逐值相等；5类VJP最坏relative L2=`5.43e-8`、最小cosine=`0.999999999998` | OFT不参与 | WSL与服务器10/10 case通过；正式冻结`relative_L2<=1e-5`、`cosine>=0.99999` | [当前状态](../status/openvla-spectral-current.md) |
| 2026-08-08 | Gate 1D deployment forward | Pre-Crop/Effective/processor/action L∞均为0；10/10 states、70/70 token一致 | OFT不参与 | 完整deployment exact forward通过；允许进入Gate 2E，不代表backward/bake已通过 | [当前状态](../status/openvla-spectral-current.md) |
| 2026-08-08 | Gate 2E deployment backward/update/bake | 五级梯度有限非零；768/768系数更新；Surface Step/Delta=`0.007843135856`；bake/Active hash一致；资产恢复 | OFT不参与；state10 source rollout正常完成且success只作工程记录 | 完整deployment backward与单轮资产链通过；进入Visibility/Coverage/Compositor与Gate 2R | [当前状态](../status/openvla-spectral-current.md) |
| 2026-08-08 | states 0–9 Visibility/Alignment audit | Primary 10/10 valid；最低A_obs=`0.0230380`、recall=`0.9992251`；20行与全部事务完整 | wrist proxy 10/10 valid；最低A_obs=`0.0291956`、recall=`0.9995750`；第二实例10/10不可观测 | 正式冻结`A_obs_min=1e-3`、`recall_min=0.95`；进入零delta Compositor与Gate 2R | [当前状态](../status/openvla-spectral-current.md) |
| 2026-08-08 | 零 Surface Delta Compositor Gate | 10/10 states与70/70 token完全一致；全部阶段最坏L∞=0；Surface梯度L2=`82.43–87.03` | OFT不参与；只验证source OpenVLA Primary训练路径 | 首次运行因commit provenance冲突拒绝；`7e2ff16`重跑与60个artifact独立复算通过 | [当前状态](../status/openvla-spectral-current.md) |
| 2026-08-08 | Gate 2R state0 RGB smoke | 3/3响应充分且cosine=`0.566–0.656`；relative L2=`0.760–0.840` | OFT不参与；Action margin仅记录 | 最低同向门槛通过；surrogate不是高保真幅值模型，完成scatter可读性修正后进入正式30-case | [当前状态](../status/openvla-spectral-current.md) |

## 当前待执行队列

| 顺序 | 候选 | 前置条件 | 通过标准 | 未通过时 |
|---:|---|---|---|---|
| 1（已通过） | Gate 1P processor forward equivalence | states 0–9、服务器环境 | pixel MAE/L∞=0，10/10序列、70/70 token一致 | 已达到数值门槛 |
| 2（已通过） | Gate 2C center-crop forward/VJP | 10个冻结CPU case | 全部case通过正式relative L2/cosine门槛 | 已达到数值门槛 |
| 3（已通过） | Gate 1D deployment-path forward equivalence | 共享Policy Canvas与Effective View实现 | 逐阶段RGB、processor tensor、action sequence/token完全一致 | 已达到零误差门槛 |
| 4（已通过） | Gate 2E backward/update/bake smoke | Gate 1D与2C通过 | 梯度有限非零、参数更新、Surface/asset约束通过 | 已达到全部工程门槛 |
| 5（进行中） | Visibility/Coverage/Compositor与Gate 2R | Gate 2E通过 | Visibility/Alignment、零delta和Gate 2R state0 smoke已通过；待服务器正式30-case验收 | 停止Seed Audit，修复基础证据链 |
| 6 | 谱自然性约束 + Fixed Vertex Support源候选 | 全部基础Gate通过 | held-out states 10–19至少3/10失败 | 先归因参数化/目标/部署输入，不做无依据网格扫描 |
| 7 | OFT开发期 rollout | 队列6通过 | 旧0/10迁移基线上至少出现2/10失败信号 | 记录机制失败，不包装为迁移提升 |
| 8 | 新任务/第三模型无偏验证 | 方法和超参数冻结 | 预注册门槛 | 区分开发期选择偏差与真实迁移 |

## 新实验登记模板

后续每个决策实验至少记录以下字段；大型原始产物继续保留在 Git 忽略目录：

```text
run_id:
date:
code_commit_sha:
hypothesis:
single_primary_change:
non_goals:
full_command_or_config:
checkpoint:
train_state_ids:
eval_state_ids:
seed:
input_artifact_paths_and_sha256:
output_artifact_paths_and_sha256:
clean_control:
pre_registered_gate:
observed_result:
validity_checks:
verdict:
next_allowed_action:
```
