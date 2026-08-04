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

### 2026-08-02 范数保护正式源结果

5000轮训练和 held-out states 10–19 均完整结束。只有 state 18 失败，OpenVLA
任务成功率为90%、攻击成功率为10%，没有达到预设的3/10源攻击门槛。因此按
go/no-go 不运行 OFT；单次失败状态从未保护版本的 state 10 变成 state 18，
但攻击数量仍为1/10，不能视为源攻击恢复。

保护机制本身工作正常：

| 指标 | 5000轮结果 |
| --- | ---: |
| Feature/Action 原始 ratio 均值 | `2.9875` |
| ratio 中位数 / 95%分位数 | `2.9635 / 3.3183` |
| Feature scale 均值 | `0.3355` |
| scale 5% / 50% / 95%分位数 | `0.3014 / 0.3374 / 0.3533` |
| 保护触发次数 | `5000/5000` |
| `ratio*scale` 最大值 | `1.00000031` |
| Action–Feature cosine 均值 | `0.2003` |
| cosine 为负的迭代比例 | `66.24%` |

batch 级 ratio 在第0轮已经是 `1.7151`，前100轮平均为 `3.0643`，最后100轮
仍为 `2.9954`。因此保护并非只修正训练后期，而是从第一轮开始持续改变优化
方向。这与逐 state 审计不矛盾：Action 梯度跨状态一致性弱，batch 求和时更易
相互抵消；Feature 梯度更一致，求和后相对范数进一步放大。

保护显著改变了代理目标，但没有改变 held-out 行为：

| 最后100轮均值 | 未保护双视角 | rho=1保护 |
| --- | ---: | ---: |
| Action loss | `20.4625` | `19.5362` |
| Combined Feature loss | `-0.1712` | `-0.1481` |
| Primary Feature loss | `-0.1813` | `-0.1483` |
| Wrist Feature loss | `-0.1611` | `-0.1479` |
| held-out 攻击成功率 | `1/10` | `1/10` |

按当前 loss 的最小化语义，较低 Action loss 表示 Action 代理攻击更强；保护使它
改善约4.5%，同时牺牲了 Feature 距离，但真实 rollout 没有恢复。这反驳了
“只要防止 Feature 梯度压过 Action，就能恢复源攻击”的机制假设，也表明继续
扫描 rho 的证据很弱：`rho=1` 与未保护两个端点的 held-out 攻击都只有1/10。

约束与产物正常。纹理在第70轮首次触及 `128/255`，64%的迭代结束在预算边界；
未保护版本对应第74轮和62.46%，说明触边/振荡不是范数保护独有。最大记录单步
仅比 `2/255` 高 `2.98e-7`，没有超过 `1e-6` 数值容差；总预算从未越界。最终
谱系数为有限 float32 `[256,3]`，UV Map 与 Active Texture 一致，XML 和原纹理
均已恢复。

下一项最小诊断应直接用最终 Active Texture 回放训练 states 0–9，不重新训练：

- 若训练 states 也只有弱攻击，说明平均 Action loss 代理与 rollout 行为不对齐；
- 若训练 states 攻击明显强、held-out 仍弱，则主要问题是状态泛化/过拟合；
- 得到该分支后，再决定是否值得实现修正后多实例 Primary-only 控制。

回放直接加载正式训练保存的 `[256,3]` 系数，避免 PNG 反推谱参数。配置中的
`train_state_ids=20-29` 只用于满足严格不相交的 partition；加载已有纹理时不会
对这些状态采帧或训练。

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
  --load_texture_path experiments/logs/spectral-k256-gradient-norm-protection-source/attack_artifacts/spectral-k256-gradient-norm-protection-states0-9-EVAL-libero_spatial-2026_08_02-09_57_38/Ep0_Spectral_Coefficients.pt \
  --num_train_init_states 10 \
  --train_init_state_ids 20-29 \
  --eval_init_state_ids 0-9 \
  --live_test_enabled False \
  --use_wandb False \
  --local_log_dir experiments/logs/spectral-k256-gradient-norm-protection-train-replay \
  --run_id_note spectral-k256-gradient-norm-protection-train-states0-9
```

### 2026-08-02 训练状态回放结果

加载的 Active Texture SHA-256 为
`f849dd2f2b8d6a64f330a863b6aa3980cb9a3d6a9e054908e34836fab0d3fc1a`，
与正式训练的 UV Map 和 Active Texture 完全一致，因此回放没有发生 PNG 重采样
或纹理加载偏差。states 0–9 中只有 state 3 失败，任务成功率仍为90%、攻击成功率
仍为1/10；XML 和原纹理在退出后正常恢复。

训练状态并没有比 held-out states 10–19 更易受攻击，因此“只是在 held-out
状态过拟合”被否定。即便 state 3 最终被 clean 对照证明是纹理引起，它也只能把
攻击强度确认在1/10；若它是 clean 自然失败，真实攻击只会更弱，不影响当前
分支选择。

当前最强证据链是：动态保护让最后100轮对称 action-bin CE 从 `20.4625` 降到
`19.5362`，但训练与 held-out rollout 都只有1/10失败。也就是说，当前 Action
代理在训练图像上数值改善，却没有可靠改变闭环行为。下一步不继续调 rho、K 或
Feature weight，而应做一次 source OpenVLA action-response 最小诊断，在完全
相同的 states 0–9 初始观测上比较 clean/adv：

- 实际 `predict_action` 向量的 L2/L∞ 变化；
- 生成 action token 的 Hamming distance；
- clean token、当前 argmax 与对称 target token 的 logit/probability margin；
- 当前对称 target CE 与动作是否真正跨过决策边界。

若 CE 下降但 token/action 几乎不变，应把目标改为直接压低 clean/argmax margin
的 untargeted decision loss；若攻击训练输入上的 token/action 已明显改变，还要
先核查 rollout 使用的 center-crop 策略预处理，再判断是否是单步静态帧覆盖不足。
该诊断只需固定状态前向，不再训练5000轮。

### 源 OpenVLA 单步动作响应诊断

诊断已经实现为 forward-only 模式。它先按正式训练配置重新采集 states 0–9 的
初始训练帧，再在 renderer 内加载正式实验保存的 `[256,3]` 谱系数；参考参数
不会写入 MuJoCo XML，也不会启动优化或 held-out rollout。每个状态并列记录：

- collector/processor 的 clean token 与手工6通道 clean 输入重生成 token 的
  Hamming distance，用来先排除预处理数据流不一致；
- clean/adv 贪心 action token 的 Hamming distance，以及同一 codec 解码后的
  连续 action L2/L∞；
- 与真实攻击损失共用 causal 对齐函数的对称 target CE；
- target-clean 和 target-best-other 的 logit/probability margin。其中只有
  `target_minus_best_other > 0` 才表示对称 target 真正跨过 argmax 决策边界。

详细逐动作维数组保存为 NPZ，逐状态汇总保存为 CSV，关键判读保存为 JSON。
命令中的 `attack_iters=1` 只是满足普通入口配置；诊断保存后会直接返回，不会
执行这1轮更新。当前 greedy generation 刻意使用与 Action loss 相同的手工6通道
输入，以首先隔离“代理损失是否改变其直接决策”；它不冒充 rollout 的
center-crop 部署输入。若这一层已经明显改变，下一项最小检查才是部署预处理响应。

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
  --source_action_response_audit_enabled True \
  --source_action_response_reference_path experiments/logs/spectral-k256-gradient-norm-protection-source/attack_artifacts/spectral-k256-gradient-norm-protection-states0-9-EVAL-libero_spatial-2026_08_02-09_57_38/Ep0_Spectral_Coefficients.pt \
  --attack_iters 1 \
  --num_train_init_states 10 \
  --train_init_state_ids 0-9 \
  --eval_init_state_ids 10-19 \
  --train_frames_per_state 1 \
  --photometric_calib_frames 5 \
  --live_test_enabled False \
  --use_wandb False \
  --local_log_dir experiments/logs/source-action-response-diagnostic \
  --run_id_note spectral-k256-protected-source-action-response
```

首轮判读顺序固定为：先检查 clean 重生成是否与 collector token 一致；再检查
adv CE 是否下降以及 target argmax 比例；最后看 token/action 变化。只在这三层
证据一致后决定是实现 untargeted decision-margin loss，还是补一项部署
center-crop 响应；只有部署首步动作也明显变化，才转向轨迹关键帧采样。

### 2026-08-03 源动作响应结果与预处理根因

诊断完整采集 states 0–9、每个状态一帧，参考谱系数 SHA-256 为
`333e67c673a13203b541a46b34dda9b1158c7fe096c37a3439ad78492d3c7fcb`。
NPZ/CSV/JSON 均正常保存，运行没有进入优化或 rollout。

| 指标 | 结果 |
| --- | ---: |
| collector clean vs 手工 clean 平均 token Hamming | 4.7 / 7 |
| 两条 clean 路径完全一致的状态 | 1 / 10 |
| clean→adv 平均 token Hamming | 5.2 / 7 |
| 至少一个 token 改变的状态 | 9 / 10 |
| 连续 action 平均 L2 / L∞ | 0.3764 / 0.2987 |
| clean / adv 对称 target CE | 20.3550 / 19.5073 |
| CE 平均下降 | 0.8478 |
| adv 对称 target 成为 argmax | 0 / 70 |
| adv target-best-other logit margin 均值 | -19.2012 |
| adv token 等于对称 target | 1 / 70 |

CE 确实下降约4.2%，也复现了正式训练最后阶段约19.5的 Action loss，但对称
target 从未成为 action 子词表 argmax；其平均仍落后当前最佳类别19.2 logits。
与此同时，错误训练输入上的 greedy token 和 action 已有显著变化，因此不能把
弱 rollout 简单归因于“纹理没有改变任何单步动作”。

更早且更确定的问题是 clean label 与 Action logits 使用了不同视觉预处理：

1. collector 的 clean token 来自 checkpoint processor；其
   `preprocessor_config.json` 和 `processing_prismatic.py` 按模型配置顺序先处理
   DINOv2、再处理 SigLIP，并用 bicubic + antialias resize；
2. 当前 `frame_collection.py` 与 `optimization.py` 的历史手工路径使用 bilinear
   resize，并按 `SigLIP → DINOv2` 拼接；
3. checkpoint `modeling_prismatic.py` 把前3通道送入 `featurizer`（DINOv2），
   后3通道送入 `fused_featurizer`（SigLIP）。因此手工路径实际上把两种归一化
   图像送反了视觉编码器。

这解释了为什么零纹理下平均已有4.7/7 clean token 不一致，也意味着现有
Action loss 在优化一个正常 OpenVLA 推理不会看到的输入。Shared-SigLIP Feature
分支通过显式 SigLIP encoder 提取，未受到六通道交换影响；受影响的是 Action
loss 以及历史 last-hidden 六通道路径。

下一步不应先改对称 target、扫描 K/rho 或补 OFT。应先建立 processor-equivalent
且保持可微的统一图像预处理，使零 Surface Delta 时满足以下回归门槛：

- 手工/可微 pixel values 与 processor 输出的通道顺序、归一化和 resize 语义一致；
- 同一 clean 图像生成的7个 action token 全部一致；
- teacher-forced 首 action argmax 与 greedy 首 token 一致（若极小 BF16 tie
  仍存在，必须同时保存 top-2 margin 解释）；
- Action loss、正式训练与诊断共同调用同一个预处理 interface，禁止再次手写
  两份六通道拼接。

完成该修正后先做1状态 GPU smoke，再重新训练一个 K=256 源候选。只有修正后的
源攻击仍弱，才继续评估 decision-margin loss 或部署 center-crop/轨迹覆盖问题。

### Processor-equivalent 可微预处理实现

修正已经实现为唯一的 `DifferentiableOpenVLAImageProcessor`：

- 从 `model.config.timm_model_ids` 与真实 `processor.image_processor` 读取分支
  顺序、输出尺寸、bicubic interpolation、antialias 和逐分支 mean/std；
- 当前只接受经过核查的 `resize-naive` 双分支配置，其他 checkpoint 显式失败；
- fused Action/last-hidden、独立 SigLIP、帧采集、动作响应诊断和源像素梯度诊断
  共用该 interface；历史 `SigLIP→DINO` 手写拼接已从正式路径移除；
- collector 的 clean action label 改为由同一可微输入生成，从构造上保证 clean
  label 与优化 logits 属于同一视觉输入；同时额外保留真实 processor pixel
  values，在 smoke 中独立比较，避免循环自证。

CPU 回归为111 passed、1 skipped。使用当前真实 checkpoint 和固定随机256×256
RGB 做无模型差分时，分支顺序解析为 DINOv2→SigLIP、SigLIP index=1；tensor
bicubic 与 PIL processor 的 fused pixel-values MAE 为0.00463。随机高频输入上的
L∞较大，因此不能只用像素误差放行，最终门槛仍是自然 LIBERO frame 的7/7 token。

该修正有意改变 Action/last-hidden 的历史错误行为。此前所有 K/rho/迁移实验仍
作为“旧预处理基线”保留，但其谱系数不能用来代表修正后的优化终点；正式比较
必须重新训练纹理。先运行以下1状态 forward-only smoke，旧 K=256 系数只用于
构造一张 adversarial 对照，不执行更新：

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
  --source_action_response_audit_enabled True \
  --source_action_response_reference_path experiments/logs/spectral-k256-gradient-norm-protection-source/attack_artifacts/spectral-k256-gradient-norm-protection-states0-9-EVAL-libero_spatial-2026_08_02-09_57_38/Ep0_Spectral_Coefficients.pt \
  --attack_iters 1 \
  --num_train_init_states 1 \
  --train_init_state_ids 0 \
  --eval_init_state_ids 10 \
  --train_frames_per_state 1 \
  --photometric_calib_frames 1 \
  --live_test_enabled False \
  --use_wandb False \
  --local_log_dir experiments/logs/processor-equivalent-action-smoke \
  --run_id_note processor-equivalent-state0-smoke
```

放行条件：

1. `collector_vs_differentiable_clean` token Hamming 为0；
2. 真实 processor 与可微预处理的 token Hamming 为0，即7/7一致；
3. clean 和 adversarial 的 teacher-forced 首 token 与 greedy 首 token 均一致；
4. 所有 CE/margin/action 数值有限，诊断结束后没有优化日志和 held-out rollout；
5. XML、原纹理和 renderer 系数正常恢复。

若第2项因 PIL/tensor bicubic 的微小差异失败，不回退通道顺序，而是先检查自然
frame 的 top-2 token margin，再决定是否需要更贴近 PIL kernel 的可微 resize。

### Processor-equivalent GPU smoke 结果

2026-08-03 在 Spatial task 0、state 0 上完成上述 forward-only smoke。结果产物为
`processor-equivalent-action-smoke/.../Ep0_Source_Action_Response.{json,csv,npz}`，
参考的旧 K=256 谱系数 SHA-256 为
`333e67c673a13203b541a46b34dda9b1158c7fe096c37a3439ad78492d3c7fcb`。

五项放行条件全部通过：

- collector 可微 clean 与诊断中的可微 clean 为 7/7 token 一致；
- 真实 processor 与可微预处理为 7/7 token 一致。两者 pixel values 的
  MAE/L∞ 分别为 `0.003101/0.09375`，说明自然帧上的微小插值差异没有跨越
  当前动作决策边界；
- clean 和 adversarial 的 teacher-forced 首 token 都与对应 greedy 首 token
  一致；
- NPZ 中所有 CE、margin、action 与 token 数值有限；目录中没有梯度、loss、
  新纹理或 held-out rollout 产物；
- LIBERO 仓库中的 bowl XML 和原纹理均为零 diff，事务 backup 已删除。

作为旁证，旧 K=256 纹理在修正后的输入上仍改变了 4/7 个 greedy token，连续
action 的 L2/L∞ 为 `0.2246/0.2231`；对称 target CE 从 `22.9691` 降至
`21.7731`，但 target argmax 仍为 0/7，平均 target-vs-best-other margin 为
`-21.4219`。这些数值只表明旧纹理能扰动动作而未命中强对称 target，不能替代
修正后重新优化得到的正式候选。

因此 processor-equivalent 修正已通过真实 checkpoint 的最小 GPU 回归。下一步
先做一次真正执行 backward、谱系数更新和 bake 的单轮训练 smoke；确认梯度、
Surface Step、扰动预算和产物均正常后，再运行 K=256、states 0–9、5000轮的
正式源模型训练。该顺序只增加一次低成本工程门槛，不构成新的超参数扫描。

### Processor-equivalent 单轮更新 smoke 结果

2026-08-03 使用与正式候选一致的 K=256、双视角 Shared-SigLIP 和 `rho=1.0`
配置，在训练 state 0 上执行一次真实 backward/update/bake，并在 state 10 做一次
held-out rollout。完整链路通过：

| 指标 | 单轮结果 |
| --- | ---: |
| Total / Action / Feature loss | `1.914450 / 22.718719 / -0.089355` |
| Primary / Wrist Feature loss | `-0.112305 / -0.066406` |
| Weighted Action / Feature grad norm | `8.543282 / 8.434726` |
| Feature/Action ratio | `0.987293` |
| Feature scale | `1.000000` |
| Action–Feature cosine | `0.035603` |
| Actual Surface Step | `2/255` |
| Max Surface Delta | `2/255` |

ratio 小于 `rho=1.0`，所以本轮不缩放 Feature 梯度是预期行为。保存的 `[256,3]`
谱系数全部768项非零、数值有限，绝对值最大为 `6.0751e-05`；loss history 也全部
有限。`Ep0_UV_Map.png` 与最终 Active Texture 的 SHA-256 完全相同，相对 clean
纹理的像素 L∞ 为2。state 10 rollout 成功，视频可解码为76帧、512×512、30 FPS；
单个 episode 不用于评价攻击效果。退出后 LIBERO XML/原纹理为零 diff，且没有
残留事务 backup。

至此修正后的前向等价性和真实训练链路均已放行。正式候选继续沿用既定的
K=256、`rho=1.0`、states 0–9、5000轮设置，但使用独立目录避免与修正前实验
混合：

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
  --local_log_dir experiments/logs/processor-equivalent-spectral-k256-source \
  --run_id_note processor-equivalent-spectral-k256-states0-9
```

正式验收仍使用既定门槛：先检查5000轮数值、约束与产物，再看 held-out states
10–19 是否至少3/10失败。达到门槛才进入 OFT；未达到则以新预处理下的动作响应
重新诊断 target/margin，而不是回到旧预处理结果上继续调参。

### Processor-equivalent K=256 正式源结果

2026-08-03 的5000轮训练和 held-out states 10–19 均完整结束。10个 episode 全部
成功，OpenVLA 任务成功率为100%、攻击成功率为0%，没有达到至少3/10失败的源
门槛。修正前 rho=1.0 候选曾造成1/10失败；这次结果没有恢复源攻击强度，按既定
规则不进入 OFT。

训练和约束本身正常：

| 指标 | 5000轮结果 |
| --- | ---: |
| Total loss 前100 / 后100轮均值 | `0.152382 / 0.135299` |
| Action loss 前100 / 后100轮均值 | `20.502875 / 19.909253` |
| Feature loss 前100 / 后100轮均值 | `-0.131617 / -0.159484` |
| Feature/Action ratio 均值 | `2.752916` |
| ratio 中位数 / 95%分位数 | `2.618605 / 3.257584` |
| Feature scale 均值 | `0.367256` |
| 保护触发次数 | `5000/5000` |
| `ratio * scale` 最大值 | `1.00000032` |
| Action–Feature cosine 均值 | `0.198215` |
| cosine 为负的迭代比例 | `64.86%` |
| Surface Step 超限次数 | `0/5000` |
| Surface Delta 超限次数 | `0/5000` |

最大 Surface Delta 在第80轮首次到达 `128/255`；共有3167轮处于预算边界附近，
但最终曲面只有58/21263个顶点的任一 RGB 分量达到95%以上预算。因此“全表面
大面积饱和”不是准确描述，更准确的是全局 L∞ 投影很早由少数极值顶点激活。
最终曲面增量 mean-absolute/RMS 为 `0.07025/0.10650`；bake PNG 相对 clean 的
像素 L∞ 为126、MAE为3.98。UV Map 与 Active Texture SHA-256 完全一致。

所有5000行日志、loss history 和 `[256,3]` 谱系数均有限，768个系数均非零；
10个 rollout 视频均可解码。退出后 LIBERO XML/原纹理为零 diff，且没有残留
事务 backup。由此排除训练中断、空梯度、保护公式失效、预算越界、bake 丢失和
运行时纹理未激活等工程解释。

当前最小解释是：修正后的代理目标确实下降，但下降幅度没有把真实动作决策推到
足以破坏任务的区域。下一步固定这份最终系数，在训练 states 0–9 上运行
forward-only action-response；不更新纹理、不做 held-out rollout：

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
  --source_action_response_audit_enabled True \
  --source_action_response_reference_path experiments/logs/processor-equivalent-spectral-k256-source/attack_artifacts/processor-equivalent-spectral-k256-states0-9-EVAL-libero_spatial-2026_08_03-09_49_45/Ep0_Spectral_Coefficients.pt \
  --attack_iters 1 \
  --num_train_init_states 10 \
  --train_init_state_ids 0-9 \
  --eval_init_state_ids 10 \
  --train_frames_per_state 1 \
  --num_frames_to_attack 10 \
  --photometric_calib_frames 5 \
  --live_test_enabled False \
  --use_wandb False \
  --local_log_dir experiments/logs/processor-equivalent-k256-action-response \
  --run_id_note processor-equivalent-k256-states0-9-action-response
```

该诊断只回答“最终纹理在优化所见的固定观测上是否真正改变动作决策”。如果
greedy token/action 变化仍小，下一轮优先改进 decision-margin/target objective；
如果训练观测变化显著但 rollout 仍0/10失败，再转向部署 center-crop 与轨迹覆盖。

### 正式候选 action-response 首轮结果

2026-08-03 对训练 states 0–9 完成首轮 forward-only 诊断。可微 clean→adv 的
结果表面上并不弱：8/10状态至少改变一个 greedy token，平均改变3.6/7个 token，
连续 action L2/L∞ 均值为 `0.1767/0.1508`。但对称 target CE 只从
`21.1260` 降到 `19.8955`，70个动作维中仅1维使 target 成为 argmax，target 与
best-other 的平均 logit margin 仍为 `-19.5450`。因此现有优化主要把动作推向
其他类别，并没有真正达到设定的强对称 target。

不过该结果同时暴露了新的预处理稳定性门槛：真实 processor 与可微 clean 仅
6/10条动作序列完全一致，state 2/5/6/8 分别相差6/3/5/4个 token；平均 pixel
values MAE 只有 `0.003104`、L∞ 为 `0.09375`。collector 与可微 clean 仍是
10/10一致，说明训练内部可复现，但不能据此证明与 rollout processor 一致。
state 6 的可微 adv 序列甚至与真实 processor clean 序列完全相同，进一步说明
微小 resize 数值差异可能与纹理效应处于同一量级。adversarial teacher-forced
首 token 与 greedy 也只有9/10一致，失配发生在 state 9。

因此不能仅凭本轮结果直接选择“改 target loss”或“补轨迹覆盖”。state 0 的单点
7/7 smoke 不足以代表多状态稳定性，后续 processor-equivalent 门槛必须覆盖全部
训练 smoke states。当前代码已扩展诊断但不改变训练行为：为可微 clean、adv、
真实 processor 各保存 `[S,7]` 的 teacher-forced top1−top2 margin，并记录每个
processor 失配状态的第一次 token 分叉位置及两侧 margin。共享前缀位置能直接
区分“接近 tie 被插值误差翻转”与“虽有大 margin 仍被预处理差异显著改变”。

CPU 回归为112 passed、1 skipped。使用相同最终系数重跑上一节命令，仅将输出
目录和 note 改为：

```text
--local_log_dir experiments/logs/processor-equivalent-k256-margin-audit
--run_id_note processor-equivalent-k256-states0-9-margin-audit
```

若第一次分叉处两侧 margin 普遍很小，下一步实现 PIL/uint8 forward 与当前 tensor
gradient 结合的 BPDA/STE 预处理，再做多状态等价 smoke；若 margin 较大，则说明
当前 tensor bicubic 近似本身不可接受，需要优先替换 forward kernel。只有部署
输入门槛重新通过后，才根据 clean→adv 决策 margin 选择 target loss 或轨迹诊断。

### Processor margin 结果与 BPDA/STE 修正

10状态 margin audit 已完整结束。真实 processor 与连续 tensor resize 仍有4/10
状态发生序列分叉；第一次分叉的平均动作维索引为 `1.25`（从0开始），可微路径与
processor 的 top1−top2 margin 均值分别只有 `0.4375/0.15625`。真实 processor
teacher argmax 在所有第一次分叉位置都与其 greedy token 一致。该结果支持预设的
“PIL/tensor 微小插值差异翻转接近决策边界”假设，而不是生成随机性：同一输入路径
的 collector/重生成仍为10/10一致。

因此统一预处理现改为显式 BPDA/straight-through estimator：

1. forward 把 ``[B,3,H,W]`` 合成 RGB clamp 到 ``[0,1]``，乘255后舍入成
   uint8，再逐图调用与 checkpoint 相同的 PIL bicubic、ToTensor 和 Normalize；
2. backward 不对离散 uint8/PIL 运算求导，而沿用连续 PyTorch bicubic+antialias
   resize 的梯度；表达式为 ``exact + (surrogate - surrogate.detach())``；
3. fused Action/last-hidden、主视角 SigLIP、腕部 SigLIP、帧采集、动作响应和源
   像素梯度诊断全部通过同一个 interface；主视角 SigLIP 直接复用 fused 中的
   checkpoint SigLIP 三通道，避免重复 CPU/GPU 同步。

这里“exact”只指给定合成256×256 RGB 的 OpenVLA processor forward；它不表示
可微 renderer 已经精确复刻 MuJoCo 光照，也不解决 rollout 额外 center-crop/轨迹
覆盖问题。BPDA 梯度是有意采用的 surrogate，后续实验和论文描述必须明确标注。

真实 Spatial checkpoint 的 CPU 差分使用3张固定随机 uint8 图像：fused
``[3,6,224,224]`` 与真实 processor 逐值完全一致，整体和两个分支的
MAE/L∞ 都为0；SigLIP slice 也逐值一致。对相同输入执行 backward 后，输入梯度
shape 为 ``[3,3,256,256]``，全部有限且 L1 非零。全量 CPU 回归为
113 passed、1 skipped。

下一步先重跑10状态 forward-only audit。仍读取旧 K=256 系数作为 adversarial
探针，但不更新它；本次唯一门槛是验证 clean processor forward，而不是评价这份
旧系数的最终方法效果：

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
  --source_action_response_audit_enabled True \
  --source_action_response_reference_path experiments/logs/processor-equivalent-spectral-k256-source/attack_artifacts/processor-equivalent-spectral-k256-states0-9-EVAL-libero_spatial-2026_08_03-09_49_45/Ep0_Spectral_Coefficients.pt \
  --attack_iters 1 \
  --num_train_init_states 10 \
  --train_init_state_ids 0-9 \
  --eval_init_state_ids 10 \
  --train_frames_per_state 1 \
  --num_frames_to_attack 10 \
  --photometric_calib_frames 5 \
  --live_test_enabled False \
  --use_wandb False \
  --local_log_dir experiments/logs/bpda-k256-action-response-smoke \
  --run_id_note bpda-k256-states0-9-action-response-smoke
```

放行要求：processor pixel MAE/L∞ 均为0、processor token Hamming 为0、10/10
序列完全一致、first-divergence 数量为0，clean/processor 的 teacher-first
consistency 均为1，所有响应量有限。adversarial teacher/generation 在旧审计的
state 9 因 BF16 路径形状差异出现一次近 tie（teacher margin 仅0.125），因此继续
记录但不把它混入 processor 等价门槛。通过后再做一次真实 backward/update/bake
单轮 smoke；两项都通过后，才重新训练 BPDA K=256 正式候选。
