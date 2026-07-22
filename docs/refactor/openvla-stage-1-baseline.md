# OpenVLA 阶段一：重构前行为基线

记录日期：2026-07-22

## 基线范围

阶段一只刻画 `openvla/experiments/robot/libero/attack_openvla.py` 的现有行为，不调整生产代码。后续重构以当前代码行为为准；论文中尚未实现的 TAAO 等模块只预留扩展边界，不在此阶段补实现。

代码基线：

- 当前提交：`b4c2e7cef07e4d5473112b5cd9676f8d4637ed6a`
- Conda 环境：`tex3d-openvla`
- Python：3.10.20
- PyTorch：2.2.0+cu121
- Spatial checkpoint：`/data/huangsimin/openvla-7b-finetuned-libero-spatial`
- Object checkpoint：`/data/huangsimin/openvla/openvla-7b-finetuned-libero-object`

## 已固定的无 GPU 行为

`tests/characterization/test_openvla_attack_loss.py` 当前覆盖：

1. 攻击损失仅选择 action token，并以 `255 - clean_bin` 作为目标类别。
2. 没有 action token 时返回可反向传播的零损失。
3. 对抗前景按 mask 与背景合成，并将 RGB 截断到 `[0, 1]`。
4. OpenVLA rollout 的 gripper action 先二值化、再反号。
5. 生成 token 的 action 解码及按统计量 mask 选择性反归一化。
6. action statistics 未提供 mask 时，所有动作维度都执行反归一化。
7. 目标 MVP 缺失时不调用 renderer，原样返回相机背景。
8. 每帧当前只生成一个对抗样本，优先使用移除目标物体后的背景。
9. Spatial bowl 与十个 Object 物体的 suite、task ID 和特殊 mesh 文件映射。
10. MuJoCo XML mesh scale 的三轴、单值扩展和缺省规则。

运行命令：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 \
  /home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python -m pytest -q \
  tests/unit/openvla_attack \
  tests/characterization/test_openvla_attack_loss.py
```

已经提取的纯逻辑由 `tests/unit/openvla_attack/` 直接测试；尚未提取的行为仍通过导入当前单体脚本来刻画。后者在测试侧设置 `NUMBA_DISABLE_JIT=1`，避免 Robosuite 导入时尝试在只读安装目录写 Numba cache；这不改变生产脚本。

## 历史实验参考

历史目录只读使用：`/home/xiaomengqi/src/github/paper_code/tex3d`

该目录当前提交为 `235c5f23f81ff5a2a8bbd1c8f0552d6f2a5eb96f`。其中的 `attack_openvla.py` 与本基线文件不同（SHA-256 分别为 `87f16f...` 和 `162e46...`，diff 为 642 行新增、276 行删除），并包含更晚的实验功能。因此历史结果用于数值量级和产物结构参考，不作为逐位一致的回归 oracle。

选定三个有完整 5000 次优化日志和攻击产物的参考运行：

| Suite / task | 历史运行 ID | 评估 | 优化日志 | 主要产物 |
|---|---|---:|---:|---|
| Spatial task 0 | `spatial_task0_attack_5000-EVAL-libero_spatial-2026_06_28-11_42_32` | 50 trials，成功率 50% | 5000 iters | `Ep0_Vertex_Noise.pt`, `Ep0_UV_Map.png` |
| Spatial task 0 | `spatial_task0_attack_5000-EVAL-libero_spatial-2026_06_28-16_15_31` | 50 trials，成功率 30% | 5000 iters | `Ep0_Vertex_Noise.pt`, `Ep0_UV_Map.png` |
| Object task 0 / alphabet soup | `object_task0_alphabet_soup_5000-EVAL-libero_object-2026_07_03-10_53_00` | 50 trials，成功率 16% | 5000 iters | `Ep0_Vertex_Noise.pt`, `Ep0_UV_Map.png` |

这些运行位于历史目录的 `experiments/logs/`；对应梯度日志和纹理产物位于 `experiments/logs/attack_artifacts/<run-id>/`。同一配置的结果存在明显随机波动，所以未来 GPU 回归只验证流程、产物和合理数值，不把单次成功率设为精确断言。

## GPU 可用后的 smoke

以下命令有意缩成一次训练迭代、一个初始状态、一个训练帧和一个评估 trial。它验证模型加载、LIBERO/EGL、可微渲染、反向传播、产物保存和 rollout 的端到端连通性，不用于判断攻击效果。

在仓库根目录执行：

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
  --attack_iters 1 \
  --num_train_init_states 1 \
  --train_frames_per_state 1 \
  --num_frames_to_attack 1 \
  --photometric_calib_frames 1 \
  --live_test_enabled False \
  --use_wandb False \
  --local_log_dir /tmp/tex3d-openvla-smoke \
  --run_id_note smoke
```

期望至少得到：进程正常结束、一个评估结果文本，以及 `attack_artifacts/<run-id>/` 下的梯度/纹理产物。GPU 空闲前不执行该测试。

## 下一阶段的使用方式

后续拆分 attack objective、action codec、compositor/view sampler 和 asset registry 时，每次只移动一个行为边界并运行上述无 GPU 测试。GPU smoke 在里程碑边界运行，而不是每个小提交都运行。TAAO/EoT 后续应通过 view sampler 接口增加多视图采样；当前的单视图测试届时应由新策略测试明确替换。
