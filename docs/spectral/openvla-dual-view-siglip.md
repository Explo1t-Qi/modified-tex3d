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

## 当前状态

截至2026-08-01，CLI、双相机采帧、多实例背景构造、共享 SigLIP loss 和训练
编排已经实现；完整无 GPU 测试为 `93 passed, 1 skipped`。真实 GPU smoke 与
正式5000轮实验尚待运行，当前不能提前报告机制有效。
