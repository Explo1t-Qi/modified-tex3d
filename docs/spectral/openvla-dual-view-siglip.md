# OpenVLA Source-only 双视角 Shared-SigLIP

## 目标

K=256 系数空间诊断表明，当前 Source 加权目标与 OFT 主视角 Action 尚有
`0.281` cosine，但与 OFT 主视角+腕部 Action 只有 `0.133`。OFT 腕部 Action
梯度范数约为主视角的3.31倍，而且两视角方向近似正交。第一条机制候选因此只
修复训练视角覆盖，不同时修改 K、权重、预算或训练状态：

```text
OpenVLA primary image ──> Action loss
          │
          └─────────────> OpenVLA SigLIP ──> primary Feature loss

LIBERO wrist image ─────> 同一个 OpenVLA SigLIP ──> wrist Feature loss

Feature loss = mean(primary Feature loss, wrist Feature loss)
```

训练仍是严格的 source-only OpenVLA：不加载 OFT，不读取 OFT 梯度，也不使用
OFT 结果选谱基。OFT rollout 只用于开发期判断该机制是否产生迁移信号。

## 实现与数据不变量

CLI 新增 `feature_view_mode`：

- `primary`：默认值，完整保留历史单视角训练行为；
- `primary_wrist`：只允许与 `feature_objective=siglip_patch` 组合。

双视角采帧和优化遵守：

1. 主视角使用 `agentview_image`，腕部使用
   `robot0_eye_in_hand_image`；二者使用相同的180度翻转、JPEG round-trip 和
   Lanczos resize；
2. 两个相机分别计算 MuJoCo MVP，但共同更新同一组谱系数
   `C ∈ R^[256,3]`；
3. Spatial task 0 中两个 Akita Bowl 实例共享一张 PNG，因此每个相机都先隐藏
   两个实例，再将两个可微前景依次合成；只渲染语义目标碗会与最终 PNG 激活
   行为不一致；
4. OpenVLA Action forward 只执行一次且只读取主视角。腕部图像只进入 OpenVLA
   checkpoint 自己的 SigLIP 分支；
5. 主视角和腕部 Feature loss 等权平均，之后仍使用既有
   `alpha_action=0.1, alpha_feature=4.0`；
6. 第一版共用主视角标定出的场景光照参数。两个目标实例当前不相互遮挡；若以后
   扩展到存在遮挡的任务，应接入 MuJoCo depth，而不能依赖实例合成顺序。

运行时首次计算双视角 loss 会输出一行：

```text
[DUAL-VIEW] action_scope=primary_only, instances=primary:2/wrist:2, feature_loss=...
```

这行用于确认两个视角均已进入真实模型前向。独立单元测试另外验证 Action 只
调用一次、SigLIP 调用两次，以及多个实例对共享纹理参数的梯度会累加。
双视角 gradient log 还会在历史字段后追加
`Primary Feature Loss | Wrist Feature Loss`，使正式训练完成后仍能分别分析两个
相机的优化趋势，而不只保留二者均值。

## GPU smoke

在仓库根目录运行；只需把 `<gpu-id>` 替换为可用 GPU：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu-id> \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TF_CPP_MIN_LOG_LEVEL=2 \
TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/openvla" \
/home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python \
openvla/experiments/robot/libero/attack_openvla.py \
  --pretrained_checkpoint /data/huangsimin/openvla-7b-finetuned-libero-spatial \
  --unnorm_key libero_spatial_no_noops \
  --task_suite_name libero_spatial \
  --object_name akita_black_bowl \
  --task_id 0 \
  --num_trials_per_task 1 \
  --enable_attack True \
  --texture_parameterization spectral \
  --spectral_basis_path experiments/spectral_basis/akita_black_bowl_k512.npz \
  --spectral_basis_count 256 \
  --feature_objective siglip_patch \
  --feature_view_mode primary_wrist \
  --alpha_action 0.1 \
  --alpha_feature 4.0 \
  --attack_epsilon 0.5019607843137255 \
  --attack_surface_step 0.00784313725490196 \
  --attack_iters 1 \
  --num_train_init_states 1 \
  --train_init_state_ids 0 \
  --eval_init_state_ids 1 \
  --train_frames_per_state 1 \
  --num_frames_to_attack 1 \
  --photometric_calib_frames 1 \
  --live_test_enabled False \
  --use_wandb False \
  --local_log_dir /tmp/tex3d-openvla-dual-view-smoke \
  --run_id_note spectral-k256-dual-view-smoke
```

验收条件：

- 启动日志包含 `feature_views=primary_wrist`；
- state 0 识别到两个共享纹理实例；
- `[DUAL-VIEW]` 显示 `primary:2/wrist:2`，两个 Feature loss 均为有限值；
- gradient log 中系数梯度非零、`Actual Surface Step <= 2/255`、
  `Max Surface Delta <= 128/255`；
- UV PNG、`Spectral_Coefficients.pt` 和 state 1 rollout 正常生成，运行结束后
  LIBERO XML/原纹理恢复。

## 正式机制实验

Smoke 通过后，运行一个且仅一个 K=256 候选。相较上面的命令替换为：

```bash
  --num_trials_per_task 10 \
  --attack_iters 5000 \
  --num_train_init_states 10 \
  --train_init_state_ids 0-9 \
  --eval_init_state_ids 10-19 \
  --train_frames_per_state 1 \
  --num_frames_to_attack 10 \
  --photometric_calib_frames 5 \
  --local_log_dir experiments/logs/spectral-k256-dual-view-source \
  --run_id_note spectral-k256-dual-view-states0-9
```

其余字段保持 smoke 命令不变。源 OpenVLA 的门槛为 held-out states 10–19 至少
造成3/10失败，即攻击成功率不低于既有 K=256 单视角版本的30%。随后把新生成的
`task_0_adv_texture_*.png` 直接交给 OFT 的
`direct_active_texture_evaluation=True` 路径，在相同 states 10–19 上评估；OFT
至少出现2/10失败才认为得到第一份非零迁移信号。

## 2026-08-01 GPU smoke 结果

真实 OpenVLA 单步流程已经通过：

- total/action/双视角平均 Feature loss 为
  `1.824497 / 21.897314 / -0.091309`，均为有限值；
- K=256 系数梯度 norm 为 `11.49894`；保存参数 shape 为 `[256,3]`，768个
  数值全部非零且有限；
- Actual Surface Step 与 Max Surface Delta 均为 `0.007843138`，等于
  `2/255`，未超出 `128/255` 总预算；
- 最终激活 PNG 与 `Ep0_UV_Map.png` 的 SHA-256 同为
  `2e01de108a6307d591fd0001bf7e3a5b3c0159d9fb71b930ec6f2c62e8ac3eb0`；
- 相对原始 UV 的最大八位通道差为2，符合一次曲面更新的量化预期；
- State 1 rollout 正常结束且视频非空。单次任务成功仅验证工程流程，不用于
  评价攻击效果；
- 运行结束后 Akita Bowl XML 和原始纹理在 LIBERO 仓库中保持 clean。

首次 smoke 的分视角 loss 只输出到交互终端，没有进入原 gradient log。随后仅
补充了上述两个持久化日志列，未改变 loss 聚合或反向传播；对应单元测试已覆盖。
正式5000轮实验尚待运行，当前仍不能提前报告该机制改善了迁移性。

## 2026-08-02 正式源实验结果

已在训练 states 0–9、held-out states 10–19 上完成5000轮 K=256 正式实验。
工程流程正常，但源攻击门槛未通过：

| 方法 | OpenVLA任务成功率 | 攻击成功率 | 失败 states |
|---|---:|---:|---|
| K=256 单主视角 Shared-SigLIP | 70% | 30% | 10、13、15 |
| K=256 主视角+腕部 Shared-SigLIP | 90% | 10% | 10 |

10个 rollout 视频均非空；最终参数 shape 为 `[256,3]`，768个系数全部非零且
有限；5000条 loss/梯度记录也全部有限。UV Map 与最终激活 PNG 的 SHA-256
同为 `d181f85e43981f4a2033d115674c3b85b414f35deeb9614d6051c591171ba409`，
运行结束后 LIBERO XML 和原纹理保持 clean。因此90%任务成功率不是训练中断、
纹理未激活或资产未恢复造成的假结果。

最后100轮的目标对照为：

| 指标 | 单主视角 K=256 | 双视角 K=256 | 优化方向解释 |
|---|---:|---:|---|
| Action loss | 20.3793 | 20.4625 | 越低越接近对称攻击 action；双视角较差 |
| 平均 Feature loss | -0.1546 | -0.1712 | 越负表示 feature 距离越大；双视角更强 |
| 主视角 Feature loss | — | -0.1813 | 主视角 feature 已被明显推远 |
| 腕部 Feature loss | — | -0.1611 | 腕部 feature 也被明显推远 |

双视角纹理仍保持连续谱外观，UV 扰动 MAE/RMSE/TV proxy 为
`2.7450 / 8.3487 / 0.05176`，均低于单主视角的
`2.9336 / 9.7270 / 0.06129`；最大通道差都为128。两次最终谱系数方向 cosine
只有 `0.193`，说明腕部目标把优化带到了明显不同、更加平滑但决策攻击更弱的
解，而不是在旧纹理上做小幅补充。

### 判定

这次结果验证了“腕部 Feature 确实能被优化”，但否定了“等权加入腕部 Feature
即可同时保持源攻击并提高迁移”的最小候选。共享视觉 feature 距离更大没有转化
为更强的 OpenVLA Action 攻击；双视角最后100轮 Action loss 也比旧 K=256 更高。

按照预先固定的 go/no-go，源攻击只有1/10、低于3/10门槛，因此不继续做 OFT
rollout：即使偶然产生目标失败，也不能支持“在近似相同源对抗性下提高迁移性”
这一研究命题。下一步应先量化双视角 Feature 梯度与源 Action 梯度的冲突，再
选择能保护 Action 攻击强度的组合规则，而不是继续增加 K 或直接扫描多个权重。

## 三目标梯度冲突审计

现有 source-only 谱审计已扩展为：在 `feature_view_mode=primary_wrist` 时，除
combined Feature 外，同时保存以下三条未乘 loss weight 的独立梯度：

```text
g_action          = d L_action(primary) / d C
g_primary_feature = d L_feature(primary) / d C
g_wrist_feature   = d L_feature(wrist) / d C

C shape = [256,3]
g_feature = (g_primary_feature + g_wrist_feature) / 2
```

第一轮只在零 Surface Delta、训练 states 0–9 上审计。它直接区分四种原因：

- 腕部 Feature 与 Action 方向是否负相关；
- 腕部方向是否不冲突、但加权后范数过大；
- 主视角多实例 Feature 本身是否已与 Action 冲突；
- 三项目标各自的跨 state 一致性是否不同。

GPU 命令：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu-id> \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TF_CPP_MIN_LOG_LEVEL=2 \
TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/openvla" \
/home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python \
openvla/experiments/robot/libero/attack_openvla.py \
  --pretrained_checkpoint /data/huangsimin/openvla-7b-finetuned-libero-spatial \
  --unnorm_key libero_spatial_no_noops \
  --task_suite_name libero_spatial \
  --object_name akita_black_bowl \
  --task_id 0 \
  --num_trials_per_task 1 \
  --enable_attack True \
  --texture_parameterization spectral \
  --spectral_basis_path experiments/spectral_basis/akita_black_bowl_k512.npz \
  --spectral_basis_count 256 \
  --feature_objective siglip_patch \
  --feature_view_mode primary_wrist \
  --spectral_gradient_audit_enabled True \
  --spectral_gradient_audit_only True \
  --spectral_gradient_audit_top_k 128 \
  --attack_iters 1 \
  --num_train_init_states 10 \
  --train_init_state_ids 0-9 \
  --eval_init_state_ids 10 \
  --train_frames_per_state 1 \
  --num_frames_to_attack 10 \
  --photometric_calib_frames 5 \
  --live_test_enabled False \
  --use_wandb False \
  --local_log_dir experiments/logs/spectral-k256-dual-view-gradient-audit \
  --run_id_note spectral-k256-dual-view-gradient-audit-states0-9
```

产物中的 NPZ 会额外包含 `primary_feature_gradients` 与
`wrist_feature_gradients`。GPU 采集完成后，用
`scripts/analyze_dual_view_spectral_gradients.py` 在 CPU 上生成范数、两两 cosine、
当前 `0.1/4.0` 加权比例和跨状态一致性汇总。若零点不存在冲突，再决定是否值得
扩展到最终 K=256 系数点，避免一开始增加第二套参考点和额外 GPU 工作。

### 零 Surface Delta 审计结果

2026-08-02 已完成 K=256、训练 states 0–9 的审计。10个样本均来自 step 0，
四组梯度 shape 均为 `[10,256,3]`、数值有限；并且
`g_feature=(g_primary_feature+g_wrist_feature)/2` 的最大绝对误差只有
`2.98e-08`。CPU 汇总位于：

```text
experiments/logs/spectral-k256-dual-view-gradient-audit/
  dual_view_gradient_diagnosis_states0-9.json
```

核心结果如下。cosine 比较的是 loss 对谱系数的梯度；由于三项目标都由梯度
下降最小化，符号关系也等价于比较实际更新方向。

| 指标 | 结果 |
| --- | ---: |
| `cos(Action, Primary Feature)` | `0.0166` |
| `cos(Action, Wrist Feature)` | `-0.0093` |
| `cos(Primary Feature, Wrist Feature)` | `0.0355` |
| `cos(Action, Combined Feature)` | `0.0065` |
| `cos(Weighted Total, Action)` | `0.6153` |
| 加权 Feature / Action 梯度范数比 | `1.3291` |
| Wrist / Primary raw Feature 梯度范数比 | `1.6097` |
| Primary / Wrist 跨状态 pairwise cosine | `0.1649 / 0.0712` |

这组证据否定了最直接的假设：腕部 Feature 并没有在零点与 Action 形成强负向
冲突，二者近似正交。Feature 加权后确实比 Action 更强，会把总更新方向旋离
Action；但旧单视角审计的对应范数比约为 `1.4607`，比新双视角的 `1.3291`
还高，因此“Feature 范数过大”不是本次源攻击从3/10退化到1/10的充分解释，
不能据此直接改权重。

方向比较提供了更窄的线索：旧 Feature 与新 Primary/Wrist Feature 的 cosine
分别为 `0.6859 / 0.0443`，旧加权总梯度与新双视角总梯度只有 `0.4994`；同时
旧/新 Action 梯度 cosine 为 `0.6077`。也就是说，腕部 Feature 显著旋转了
feature 更新方向，但多实例 renderer 修正也改变了 Action/Primary 梯度，当前
跨版本结果不能把退化单独归因给腕部视角。

### 最终谱系数参考点审计

零点没有发现直接冲突后，下一步是在已经训练完成的双视角 K=256 系数上重复
完全相同的10状态审计。该路径只读取本地可信的 Tensor `.pt`，每个样本前恢复
同一份 `[256,3]` 系数，不更新参数；JSON 会记录绝对路径与 SHA-256，结束后
renderer 参数仍恢复为零。这样可以判断冲突是否是在优化轨迹后期才出现，无需
再进行一次5000轮训练。

在上一条 GPU 命令中追加：

```bash
  --spectral_gradient_audit_reference_path \
    experiments/logs/spectral-k256-dual-view-source/attack_artifacts/spectral-k256-dual-view-states0-9-EVAL-libero_spatial-2026_08_01-19_39_26/Ep0_Spectral_Coefficients.pt \
  --local_log_dir experiments/logs/spectral-k256-dual-view-final-gradient-audit \
  --run_id_note spectral-k256-dual-view-final-gradient-audit-states0-9
```

其中 `local_log_dir` 和 `run_id_note` 应替换原命令的同名参数，不能重复传入。
若最终点出现明显负 cosine 或总梯度与 Action 对齐进一步下降，再优先验证
Action-protecting 的梯度组合；若最终点仍没有冲突，则应先做“修正后多实例
Primary-only”控制实验，隔离多实例语义变化和腕部视角的影响。

### 最终谱系数参考点结果

2026-08-02 已完成最终 `[256,3]` 谱系数参考点审计。JSON 中记录的参考文件
SHA-256 为 `87fcf983875258b0ce7d33123167f131e6f0cb52536834e3546d98807b890b0d`，
与磁盘上的正式双视角训练系数一致。10个状态、四组 `[10,256,3]` 梯度全部
有限，combined Feature 恒等式最大误差为 `5.96e-08`，因此可以与零点直接做
差分。

| 指标 | 零 Surface Delta | 最终谱系数 | 变化 |
| --- | ---: | ---: | ---: |
| `cos(Action, Primary Feature)` | `0.0166` | `-0.0343` | `-0.0509` |
| `cos(Action, Wrist Feature)` | `-0.0093` | `-0.0055` | `+0.0038` |
| `cos(Action, Combined Feature)` | `0.0065` | `-0.0106` | `-0.0171` |
| `cos(Weighted Total, Action)` | `0.6153` | `0.5058` | `-0.1095` |
| 加权 Feature / Action 范数比 | `1.3291` | `1.7342` | `+30.5%` |
| Action raw norm | `91.3839` | `118.7682` | `+30.0%` |
| Combined Feature raw norm | `2.9805` | `5.1366` | `+72.3%` |
| Action 跨状态 pairwise cosine | `0.0132` | `-0.0300` | `-0.0432` |
| Combined Feature 跨状态 pairwise cosine | `0.1157` | `0.1936` | `+0.0779` |

变化具有逐状态一致性：10个状态中有8个状态的加权 Feature/Action 范数比
上升，也有8个状态的总梯度与 Action 对齐下降。零点与终点的 Action、Combined
Feature、Weighted Total 梯度 cosine 分别只有 `0.3317 / 0.3019 / 0.2725`，
说明优化过程中局部梯度场发生了明显旋转，不能只用零点近似完整训练轨迹。

该结果仍否定“腕部 Feature 与 Action 直接反向冲突”：终点
`cos(Action,Wrist Feature)=-0.0055`，几乎正交。更符合证据的机制是：训练后期
Feature 梯度比 Action 增长得更快，而且 Feature 的跨状态方向更一致；在当前
`0.1/4.0` 权重下，Feature 因而逐渐主导更新，把总方向从 Action 旋开。Primary
Feature 在 states 0、7、8 出现较明显负 cosine，但均值仍只有 `-0.0343`，不能
把整体退化解释为单一视角的符号冲突。

这使两个后续实验承担不同职责：

1. **动态范数保护**直接验证上述机制：每步限制加权 Feature 梯度相对加权
   Action 梯度的范数，避免 Feature 在后期占优；它是最快的效果验证。
2. **修正后多实例 Primary-only**隔离工程语义变化与 Wrist 视角，是更严格的
   因果控制，但不会直接产生新的迁移机制。

当前目标是优先快速验证机制，因此下一纵切应优先设计动态范数保护；在确定
公式、上限和日志验收口径之前，不直接开始实现或扫描超参数。PCGrad 只会处理
负内积，而当前主要问题是近似正交但范数失衡，所以不作为第一候选。

## 动态梯度范数保护

第一版已固定为 batch 级、谱系数空间的 Feature 范数上限。每轮先按原配置和
frame weight 分别累积：

```text
g_action  = sum_frames frame_weight * 0.1 * dL_action/dC
g_feature = sum_frames frame_weight * 4.0 * dL_feature/dC
C shape   = [256,3]

scale   = min(1, rho * ||g_action||_2 / (||g_feature||_2 + epsilon))
g_total = g_action + scale * g_feature
rho     = 1.0
```

保护发生在 batch 累积后、Surface-normalized step 之前。它不改变 Primary/Wrist
等权平均，不放大较弱的 Feature，也不改变已有 Surface Step 与 L∞ 投影。默认
关闭，因此旧命令的反向传播与日志字段保持不变。首版 CLI 只允许
`spectral + siglip_patch + primary_wrist`，并要求两个 alpha 和 rho 均为有限
正数，避免把该机制误用为 Geometry/Legacy 的既有基线。

启用后 gradient log 在原字段后追加：

```text
Weighted Action Grad Norm
Weighted Feature Grad Norm
Feature/Action Grad Norm Ratio
Feature Grad Scale
Action-Feature Grad Cosine
```

模型无关的范数合并规则位于
`openvla/experiments/robot/libero/openvla_attack/gradient_protection.py`；OpenVLA
专用的逐帧求导、batch 累积和 renderer 更新仍由 `optimization.py` 编排。

其中 `Total Loss` 仍记录原始 `0.1*Action + 4.0*Feature` 名义目标，便于与旧实验
对照；真正用于更新的是日志所对应的受保护合并梯度。实现逐帧完成两次 autograd
并只累积 detach 后的 `[256,3]` 梯度，不会把全部训练帧计算图同时保留在显存中。

### 单步 GPU smoke

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu-id> \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TF_CPP_MIN_LOG_LEVEL=2 \
TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/openvla" \
/home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python \
openvla/experiments/robot/libero/attack_openvla.py \
  --pretrained_checkpoint /data/huangsimin/openvla-7b-finetuned-libero-spatial \
  --unnorm_key libero_spatial_no_noops \
  --task_suite_name libero_spatial \
  --object_name akita_black_bowl \
  --task_id 0 \
  --num_trials_per_task 1 \
  --enable_attack True \
  --texture_parameterization spectral \
  --spectral_basis_path experiments/spectral_basis/akita_black_bowl_k512.npz \
  --spectral_basis_count 256 \
  --feature_objective siglip_patch \
  --feature_view_mode primary_wrist \
  --alpha_action 0.1 \
  --alpha_feature 4.0 \
  --gradient_norm_protection_enabled True \
  --feature_gradient_norm_ratio_limit 1.0 \
  --attack_iters 1 \
  --num_train_init_states 1 \
  --train_init_state_ids 0 \
  --eval_init_state_ids 10 \
  --train_frames_per_state 1 \
  --num_frames_to_attack 1 \
  --photometric_calib_frames 5 \
  --live_test_enabled False \
  --use_wandb False \
  --local_log_dir experiments/logs/spectral-k256-gradient-norm-protection-smoke \
  --run_id_note spectral-k256-gradient-norm-protection-smoke
```

smoke 的 go/no-go：

- 启动日志明确显示 ratio limit `1.0`；
- 五个新梯度字段全部有限，`0 <= Feature Grad Scale <= 1`；
- 当原始 ratio 大于 `1` 时，缩放后的比例
  `ratio * scale` 在浮点误差内不超过 `1`；
- `Actual Surface Step <= 2/255`，`Max Surface Delta <= 128/255`；
- loss、谱系数、Active Texture 与 UV Map 正常保存，运行后 XML/纹理资产恢复。

smoke 只验证实现，不用其单次 rollout 成败判断攻击效果。通过后再运行与旧
K=256 双视角完全相同的 states 0–9、5000轮训练；唯一方法变量是开启保护并令
`rho=1.0`。

### 2026-08-02 范数保护 smoke 结果

真实 OpenVLA 单步流程已通过：

- total/action/Combined Feature loss 为
  `1.824497 / 21.897314 / -0.091309`，均有限；
- 加权 Action/Feature 梯度 norm 为 `7.912201 / 8.649742`，原始 ratio
  `1.093216`，Feature scale `0.914733`；`ratio*scale=1.0000003`，仅有
  `3.1e-7` 的日志舍入误差；
- Action–Feature cosine 为 `-0.039141`，与零点审计中的近似正交关系一致；
- 保护后的合并梯度 norm 为 `10.96837`，所有梯度日志字段有限；
- Actual Surface Step 与 Max Surface Delta 均为 `0.007843138`，没有越过
  `2/255` 单步上限或 `128/255` 总预算；
- 谱系数为有限 float32 `[256,3]`，768个参数均发生非零更新；UV Map 与
  Active Texture 的 SHA-256 一致，均为 RGB `4096x4096`；相对原纹理的最大
  PNG 变化为2个 uint8 色阶；
- 运行结束后 XML 已恢复引用 `texture.png`，未遗留 clean backup。

state 10 的单次 rollout 成功率为100%，但 smoke 的样本数和训练轮数都不能
用于判断攻击强度。该结果只证明动态范数保护、双视角求导、曲面更新、产物保存
和运行时资产事务已经形成完整可运行链路。

### 范数保护正式源实验

下一步固定 K=256、`rho=1.0`，使用与未保护双视角实验相同的训练 states 0–9、
held-out states 10–19 和5000轮配置。唯一方法变量是开启动态范数保护。源门槛
仍为至少造成3/10失败；低于门槛则不做 OFT rollout。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu-id> \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TF_CPP_MIN_LOG_LEVEL=2 \
TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/openvla" \
/home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python \
openvla/experiments/robot/libero/attack_openvla.py \
  --pretrained_checkpoint /data/huangsimin/openvla-7b-finetuned-libero-spatial \
  --unnorm_key libero_spatial_no_noops \
  --task_suite_name libero_spatial \
  --object_name akita_black_bowl \
  --task_id 0 \
  --num_trials_per_task 10 \
  --enable_attack True \
  --texture_parameterization spectral \
  --spectral_basis_path experiments/spectral_basis/akita_black_bowl_k512.npz \
  --spectral_basis_count 256 \
  --feature_objective siglip_patch \
  --feature_view_mode primary_wrist \
  --alpha_action 0.1 \
  --alpha_feature 4.0 \
  --gradient_norm_protection_enabled True \
  --feature_gradient_norm_ratio_limit 1.0 \
  --attack_iters 5000 \
  --num_train_init_states 10 \
  --train_init_state_ids 0-9 \
  --eval_init_state_ids 10-19 \
  --train_frames_per_state 1 \
  --num_frames_to_attack 10 \
  --photometric_calib_frames 5 \
  --live_test_enabled False \
  --use_wandb False \
  --local_log_dir experiments/logs/spectral-k256-gradient-norm-protection-source \
  --run_id_note spectral-k256-gradient-norm-protection-states0-9
```

正式检查除源成功率外，还要汇总5000轮 `Feature Grad Scale` 的均值、分位数、
触发比例，以及保护后 `ratio*scale` 是否始终不超过1。这样可以判断保护是仅在
训练后期介入，还是从零点开始就持续改变优化方向。
