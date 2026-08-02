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
