# OpenVLA Source-only 谱基梯度审计

## 目的

第一版 Spectral K=128 能用 384 个参数在 OpenVLA 上造成任务失败，并保持比
Geometry Vertex 更连续的纹理；Shared-SigLIP 版本在源 OpenVLA 上的成功率为
80%，迁移到 OpenVLA-OFT 后仍为 100%。下一阶段不直接盲目增加 K，而是先回答：

1. 前 128 个连续低频模态覆盖了多少 Action/SigLIP 梯度能量；
2. 128–512 的中频候选中是否存在更强且跨状态稳定的方向；
3. 相同 128 个模态预算下，梯度选基能否优于连续最低频选基。

本审计严格使用源 OpenVLA 的训练 states 0–9。OFT 梯度不参与排名，保持迁移
评估的 source-only 语义。

## 数据流

```text
LIBERO train states 0–9
          │
          ▼
TrainingFrameCollector
  每帧记录原始 state ID / collection step
          │
          ▼
零初始化候选谱系数 C = 0，shape [M, 3]
          │
          ├─ Action Loss ── autograd.grad ──> g_action [M, 3]
          └─ SigLIP Loss ── autograd.grad ──> g_feature [M, 3]
                                             │
                                             ▼
                             stack [samples, M, 3]
                                             │
                                             ├─ 强度
                                             ├─ 跨状态方向一致性
                                             ├─ Surface L∞ 预算归一化
                                             └─ 低频累计梯度能量
```

每个训练帧都从零 Surface Delta 开始，审计不执行参数更新。Action 与 Feature
共用一次模型/渲染前向，但分别用 `torch.autograd.grad` 求未乘 `alpha` 的独立
梯度，不向模型参数或 `parameter.grad` 累积梯度。

## 指标定义

候选 basis 为 `Φ ∈ R^[N,M]`，第 k 个模态为 `φ_k`；一个样本关于该模态 RGB
系数的梯度为 `g_{s,k} ∈ R^3`。

### 平均梯度强度

```text
mean_norm_k = mean_s ||g_{s,k}||_2
```

它描述 loss 对该参数坐标的局部敏感度，但没有校正不同模态在曲面上的幅值。

### 跨样本方向一致性

```text
consistency_k =
    ||mean_s g_{s,k}||_2 / mean_s ||g_{s,k}||_2
```

范围为 `[0,1]`。接近 1 表示不同状态/帧希望沿相近的 RGB 方向更新；接近 0
表示梯度互相抵消，容易形成状态特异性方向。

### Surface-budget-normalized score

```text
surface_score_k = mean_norm_k / ||φ_k||_∞
stable_score_k  = surface_score_k * consistency_k
```

在 Surface Delta 使用相同 L∞ 预算时，单个模态允许的系数幅值与
`1 / ||φ_k||_∞` 成正比。因此不能只用 raw coefficient gradient 排名。
`stable_score` 再抑制跨状态不一致的模态。

### 连续低频累计能量

```text
energy_k = sum_s ||g_{s,k}||_2²
cumulative_K = sum_{k < K} energy_k / sum_all_modes energy_k
```

它直接回答“最低 128/256 个连续模态覆盖候选池多少梯度”。该指标按原始特征值
顺序累计，不按梯度排名重新排列。

## 产物

每个 task 在 run 的 `attack_artifacts/<run-id>/` 下生成：

```text
Ep0_Spectral_Gradient_Audit.npz
Ep0_Spectral_Gradient_Audit.csv
Ep0_Spectral_Gradient_Audit.json
```

- NPZ：保留 `[samples,M,3]` 原始 Action/Feature 梯度和全部统计，用于后续选基；
- CSV：每个谱模态一行，包含 eigenvalue、强度、一致性、曲面归一化分数、
  累计能量和两项排名；
- JSON：记录 source-only scope、原始 state IDs、Top-K 索引和低频累计能量，
  便于快速检查。

当前审计只输出排名，不自动生成非连续谱基 artifact。看到真实梯度分布后再固定
选基策略，避免在没有数据时引入频率惩罚或 Action/Feature 混合分数。

## CLI

新增字段：

```text
spectral_gradient_audit_enabled=True
spectral_gradient_audit_only=True
spectral_gradient_audit_top_k=128
```

约束：

- 必须启用 `enable_attack=True`；
- 必须使用 `texture_parameterization=spectral`；
- 必须使用 `feature_objective=siglip_patch`；
- `audit_top_k` 必须位于 `[1, spectral_basis_count]`；
- `audit_only=True` 时，保存审计产物后跳过纹理优化、激活和 held-out rollout。

## 候选 K=512 谱基

候选 basis 仍由同一 Akita OBJ、cotangent Laplacian 和 lumped mass 生成：

```bash
cd /data/xiaomengqi/src/tex3d

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
/home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python \
scripts/generate_openvla_spectral_basis.py \
  --mesh /home/xiaomengqi/src/github/paper_code/LIBERO/libero/libero/assets/stable_scanned_objects/akita_black_bowl/akita_black_bowl.obj \
  --num-basis 512 \
  --output /data/xiaomengqi/src/tex3d/experiments/spectral_basis/akita_black_bowl_k512.npz
```

生成文件属于实验产物，不进入 Git。完成后必须检查 geometry hash、shape、
M-orthogonality error 和 eigen residual，再运行真实 GPU 审计。

2026-07-26 已生成并完成 CPU 数值检查：

- geometry vertices: 21,263；faces: 42,522；
- 512 个非恒定谱基，文件约 162 MiB；
- 最大 M-正交误差：`3.06e-15`；
- 最大特征方程残差：`8.67e-12`；
- geometry hash 与 K=128 文件一致；
- K=512 文件的前 128 个模态与原 K=128 文件逐列一致：最小质量内积绝对
  cosine 大于 `0.999999999999999`，前 129 个特征值最大差为 `2.55e-11`。

因此后续连续 K=128/K=256 与 512 候选池共享同一低频坐标系，可以进行公平
比较。

## GPU 审计命令

先用一个训练状态检查真实 OpenVLA 前向、两项独立梯度和产物写入。该命令只做
审计，不更新纹理，也不运行 held-out rollout：

```bash
cd /data/xiaomengqi/src/tex3d

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
  --spectral_basis_count 512 \
  --feature_objective siglip_patch \
  --spectral_gradient_audit_enabled True \
  --spectral_gradient_audit_only True \
  --spectral_gradient_audit_top_k 128 \
  --attack_iters 1 \
  --num_train_init_states 1 \
  --train_init_state_ids 0 \
  --eval_init_state_ids 1 \
  --train_frames_per_state 1 \
  --num_frames_to_attack 1 \
  --photometric_calib_frames 1 \
  --live_test_enabled False \
  --use_wandb False \
  --local_log_dir /tmp/tex3d-openvla-spectral-audit-smoke \
  --run_id_note spectral-audit-k512-smoke
```

Smoke 通过后，正式 source-only 审计只需把以下字段替换为：

```bash
--num_train_init_states 10 \
--train_init_state_ids 0-9 \
--eval_init_state_ids 10 \
--num_frames_to_attack 10 \
--photometric_calib_frames 5 \
--local_log_dir /data/xiaomengqi/src/tex3d/experiments/logs/spectral-gradient-audit-k512 \
--run_id_note spectral-audit-k512-states0-9
```

2026-07-26 已在真实 OpenVLA checkpoint 上完成单状态 smoke：

- 有效样本为 state 0、step 0，Action/Feature 梯度 shape 均为
  `[1, 512, 3]`，所有数值有限，512 个模态均有非零梯度；
- Action loss 为 `21.916494`，SigLIP feature loss 为 `-0.076172`，
  与此前零扰动 K=128 Shared-SigLIP smoke 一致；
- Action 梯度整体 L2 norm 为 `94.167759`，最大绝对值为 `13.590073`；
- Feature 梯度整体 L2 norm 为 `2.863929`，最大绝对值为 `0.276452`；
- 连续低频 K=128/256/384 的 Action 累计能量分别为
  `49.39% / 71.90% / 88.29%`；
- 连续低频 K=128/256/384 的 Feature 累计能量分别为
  `31.30% / 55.42% / 82.89%`；
- Feature stable-score Top-128 中有 66 个模态来自 `[0,128)`，24 个来自
  `[128,256)`，38 个来自 `[256,512)`；单状态时 consistency 恒为 1，
  该排名只能验证中高频候选确有局部贡献，不能作为最终选基依据；
- NPZ、CSV（512 行）和 JSON 均成功写入。

## States 0–9 正式审计结果

2026-07-27 已完成 source-only 正式审计。10 个有效样本分别来自原始 states
0–9，且均为各状态的 step 0；所有 loss、梯度与统计量均为有限值。

### 连续低频覆盖

| 目标 | K=128 | K=256 | K=384 | 达到 80% 能量所需连续 K |
|---|---:|---:|---:|---:|
| Action | 43.61% | 69.45% | 87.70% | 326 |
| SigLIP Feature | 37.71% | 63.65% | 85.02% | 347 |

连续 K=128 对两项目标都只覆盖不到一半的候选池局部梯度能量；K=256 明显增加
表达能力，但仍不能覆盖全部有效方向。

### Feature stable-score Top-128

- 79 个模态来自 `[0,128)`，30 个来自 `[128,256)`，18 个来自
  `[256,384)`，1 个来自 `[384,512)`；
- 该集合覆盖 41.95% 的 Feature 原始梯度能量，比连续最低频 K=128 的
  37.71% 高 4.24 个百分点；
- 它覆盖全部 Feature stable score 的 51.46%；
- 入选模态 consistency 均值/中位数为 `0.5573 / 0.5276`，连续 K=128
  则为 `0.4630 / 0.4335`；
- 逐状态留一后重新排名，Top-128 与完整10状态排名仍重合 107–119 个模态，
  平均重合 112.5 个，平均 Jaccard 为 0.7853；
- Feature 与 Action 各自 Top-128 重合 74 个，Jaccard 为 0.4066；
- Feature 选出的128个模态覆盖 39.82% 的 Action 原始梯度能量，低于连续
  K=128 的 43.61%。

这说明梯度选基不是简单地偏向高频：约 62% 的入选模态仍来自最低128维，但它
稳定替换了49个低频模态。跨状态排名具有较好稳定性，不过 Feature 原始能量增益
只有4.24个百分点，而且牺牲了一部分 Action 能量。因此当前结果只支持“值得做
训练对照”，尚不能证明梯度选基优于连续低频。

## Feature stable-score K=128 产物

非连续谱基由以下确定性命令生成：

```bash
cd /data/xiaomengqi/src/tex3d

PYTHONDONTWRITEBYTECODE=1 \
/home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python \
scripts/select_openvla_spectral_basis.py \
  --candidate-basis experiments/spectral_basis/akita_black_bowl_k512.npz \
  --audit experiments/logs/spectral-gradient-audit-k512/attack_artifacts/spectral-audit-k512-states0-9-EVAL-libero_spatial-2026_07_27-08_16_39/Ep0_Spectral_Gradient_Audit.npz \
  --objective feature \
  --top-k 128 \
  --expected-state-ids 0-9 \
  --output experiments/spectral_basis/akita_black_bowl_k128_feature_stable_states0-9.npz
```

生成器会验证 audit eigenvalues、basis L∞、梯度 shape、排名 permutation 和
source state IDs，并在输出 metadata 中保存：

- K=512 候选谱基和正式审计 NPZ 的 SHA-256；
- 完整128个源模态索引及对应 stable scores；
- source states 0–9；
- `selection_rank_descending` 列顺序和 source-only 方法名。

2026-07-27 的产物检查结果：

- basis shape 为 `[21263,128]`，所有数值有限；
- 最大 M-正交误差为 `2.38e-15`；
- 最大特征方程残差为 `7.36e-12`；
- 文件约 41 MiB，可由现有 renderer 原样加载；
- eigenvalues 因 basis 按 stable-score 排名排列而不再单调递增，这是显式设计，
  每列的原低频索引由 metadata 恢复。

连续 K=256 不需要生成新文件：使用同一个
`akita_black_bowl_k512.npz` 并设置 `spectral_basis_count=256` 即可，确保其
前128列仍与历史连续 K=128 基线完全一致。

### Shared objective 权重复核

正式审计已经给出零扰动处的独立目标梯度，无需再额外运行标定 forward。在
state 0 上使用既有 `alpha_action=0.1, alpha_feature=4.0` 时：

| 参数化 | Action 梯度 norm | Feature 梯度 norm | 加权 Feature/Action |
|---|---:|---:|---:|
| 连续 K=128 | 66.3740 | 1.6021 | 0.9655 |
| 连续 K=256 | 80.0628 | 2.1321 | 1.0652 |
| Feature stable K=128 | 63.1571 | 1.6941 | 1.0729 |

三种参数化的首状态加权比例都接近1，因此后续训练继续共享 `0.1/4.0`，不为
每种 basis 单独调权。这样实验只改变谱空间，不同时改变优化目标。

### 非连续 basis GPU smoke

2026-07-27 已使用 Feature stable-score K=128 artifact 完成一次真实纹理更新、
bake、Active Texture 激活和 State 1 rollout：

- total/action/feature loss 分别为
  `1.886962 / 21.916494 / -0.076172`，均为有限值；
- 谱系数梯度 norm 为 `10.53578`；
- Actual Surface Step 与 Max Surface Delta 都为
  `0.007843138`，等于 `2/255`；
- 保存系数 shape 为 `[128,3]`，384 个值全部非零且有限；
- `Ep0_UV_Map.png` 与最终 `task_0_adv_texture_*.png` 逐像素相同；
- 相对原始 UV 的最大八位通道差为2，符合单步曲面更新后的量化预期；
- State 1 rollout 成功，完整流程正常结束；单次成功率不用于评价攻击效果。

## 第一轮决策

读取审计结果后只比较两个候选：

1. 连续最低频 K=256：测试增加表达能力是否恢复源攻击；
2. 从前 512 个候选中按 source-only SigLIP stable score 选择 128 个：
   测试相同参数量下“选哪些模态”是否比“连续取最低频”更重要。

两种方法都重新进行 Action/SigLIP 单步梯度标定。只有源 OpenVLA states 10–19
成功率达到 70% 或更低，才进入 OFT 直接迁移。OFT 侧额外记录 feature/action
变化只能作为诊断，不反向参与本轮选基。

## 当前状态

- 纯统计、shape/有限值校验、CSV/NPZ/JSON 序列化：CPU 测试通过；
- Training Frame 原始 state ID 数据流：CPU 测试通过；
- 单帧独立 Action/SigLIP autograd：CPU 测试通过；
- trainer audit-only 编排：CPU 测试通过；
- K=512 候选 basis：已生成并通过几何、正交性、残差与低频一致性校验；
- 真实 OpenVLA state 0 单样本 smoke：通过；
- 真实 OpenVLA states 0–9 梯度审计：完成，数据流与数值检查通过；
- Feature stable-score K=128 artifact：已生成并通过 provenance、正交性和
  特征方程残差校验；
- Feature stable-score K=128 GPU smoke：通过；
- 下一步：分别执行连续 K=256 与 Feature stable-score K=128 的同预算源
  OpenVLA 训练对照。
