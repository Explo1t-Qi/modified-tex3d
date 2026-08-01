# OpenVLA→OFT 跨模型像素梯度审计

## 为什么先审计像素空间

上一轮固定状态响应诊断已经表明，Spectral K=256 纹理能让 OFT 的双视角
SigLIP feature 相对 L2 平均变化16.28%，但 Action chunk 只变化2.95%。下一步
原计划直接比较谱系数梯度，但现有 source-only 审计在零系数处使用可微 renderer，
其 clean reference 与零扰动 render 并非逐像素相等；若直接移植到 OFT，结果会
混入 renderer 和模型预处理误差。另外，OFT 使用主视角+腕部，OpenVLA 只使用
主视角，直接拼成一个系数梯度也会隐藏视角差异。

因此先在已经验证过的真实 policy 输入上比较输入梯度：如果两模型在物体区域的
视觉/决策方向完全不一致，就没有必要先开发谱投影；如果存在共同方向，再使用
同一个 renderer Jacobian 把它投影到 K=256 模态。

## 审计定义

状态固定为10、11、13、15、16，参考点是已经训练完成的 K=256 adversarial
input，而不是 clean/零扰动点。这样负 MSE 在当前点具有非零一阶梯度。

两个模型的 Feature objective 相同：

```text
L_feature = -MSE(SigLIP_adv, SigLIP_clean)
```

Action 表示不同，因此分别使用语义一致的“远离 clean decision”目标：

```text
OpenVLA:     L_action = -MSE(action-token-logits_adv,
                            action-token-logits_clean)
OpenVLA-OFT: L_action = -MSE(action-head_adv, action-head_clean)
```

保存的是 `dL/dRGB_adv`。两项 loss 都通过梯度下降增大 clean/adv 距离，所以两
模型 raw gradient 的 cosine 与实际下降方向 cosine 相同。梯度坐标统一为
center-crop 之前的224×224 RGB；PyTorch 可微 crop 与 TensorFlow
`crop_and_resize` 的随机输入验证误差为 MAE `2.12e-6`、最大 `2.40e-5`。

主要指标只在 `|adv-clean| > 1/255` 的物体可见区域计算。全图 cosine 作为对照，
但背景不是当前纹理可修改变量，不能用它否定物体区域的共同方向。OFT 仍完整
输入 adversarial wrist 与归一化 proprio；腕部梯度单独保存，不与单视角
OpenVLA 强行计算 cosine。

## 运行命令

首先使用
`docs/spectral/oft-transfer-response-diagnostic.md` 中的五状态命令重新生成成对
输入与包含 `robot_state [8]` 的 JSON，然后依次运行：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu-id> \
TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 \
PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 \
PYTHONPATH="$PWD/openvla:$PWD" \
/home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python \
openvla/experiments/robot/libero/openvla_attack/diagnose_pixel_gradients.py \
  --pretrained_checkpoint \
    /data/huangsimin/openvla-7b-finetuned-libero-spatial \
  --input_dir \
    "$PWD/experiments/logs/spectral-k256-oft-response-diagnostic" \
  --output_path \
    "$PWD/experiments/logs/cross-model-pixel-gradient-audit/source-states10-11-13-15-16.npz"
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu-id> \
TOKENIZERS_PARALLELISM=false NUMBA_DISABLE_JIT=1 \
MPLCONFIGDIR=/tmp/tex3d-oft-matplotlib \
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 \
PYTHONPATH="/home/xiaomengqi/src/github/paper_code/openvla-oft:$PWD/openvla-oft:$PWD" \
/home/xiaomengqi/miniconda3/envs/tex3d-oft/bin/python \
openvla-oft/experiments/robot/libero/diagnose_pixel_gradients.py \
  --pretrained_checkpoint \
    /data/xiaomengqi/checkpoints/openvla-7b-oft-finetuned-libero-spatial \
  --input_dir \
    "$PWD/experiments/logs/spectral-k256-oft-response-diagnostic" \
  --output_path \
    "$PWD/experiments/logs/cross-model-pixel-gradient-audit/target-states10-11-13-15-16.npz"
```

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python \
scripts/compare_vla_pixel_gradients.py \
  --source \
    experiments/logs/cross-model-pixel-gradient-audit/source-states10-11-13-15-16.npz \
  --target \
    experiments/logs/cross-model-pixel-gradient-audit/target-states10-11-13-15-16.npz \
  --source_action_weight 0.1 \
  --source_feature_weight 4.0 \
  --output \
    experiments/logs/cross-model-pixel-gradient-audit/summary-states10-11-13-15-16.json
```

NPZ/JSON 是本地实验产物，不进入 Git。

## 2026-08-01 五状态结果

下表均为主视角物体可见 mask 内的 gradient cosine。`Weighted→Target Action`
使用当前训练权重 `0.1*Source Action + 4.0*Source Feature`。

| State | Source↔Target Feature | Source↔Target Action | Source Feature→Target Action | Weighted→Target Action | OFT腕部/主视角 Action norm |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.678 | 0.043 | 0.082 | 0.071 | 7.44× |
| 11 | 0.400 | 0.169 | 0.151 | 0.193 | 4.64× |
| 13 | 0.607 | 0.241 | 0.189 | 0.258 | 0.89× |
| 15 | 0.405 | 0.061 | 0.038 | 0.056 | 3.96× |
| 16 | 0.554 | 0.177 | 0.146 | 0.214 | 2.46× |
| **均值** | **0.529** | **0.138** | **0.121** | **0.158** | **3.88×** |

补充汇总：

- Source Feature↔Action cosine 均值 `0.339`；
- OFT Feature↔Action cosine 均值 `0.403`；
- 全图跨模型 Feature/Action cosine 只有 `0.084 / -0.040`，说明共同方向主要
  集中在纹理真正可改变的物体区域，使用全图会被不可修改背景淹没；
- 每个模型、每个状态的 Feature/Action 梯度均有限且非零；主视角可见 mask 为
  1073–1119个像素。

## 结论与下一步

第一，谱方法寻找“共享视觉空间”并非完全错误：两模型在物体区域的 SigLIP
梯度呈稳定正相关，均值0.529。第二，当前目标仍不足以产生迁移：跨模型 Action
方向只有0.138，Source Feature 对 Target Action 只有0.121，当前加权 source
objective 对 Target Action 也只有0.158。这解释了为什么 feature distance 已经
增大，但 OFT rollout 没有失败。

第三，视角不对称是独立瓶颈。OFT 的腕部 Action 梯度范数平均是主视角的3.88倍，
而 source OpenVLA 优化完全没有腕部目标。物理纹理虽然会同时出现在两台相机，
但现有 loss 不会主动选择对 OFT 腕部决策敏感的方向。

本诊断通过进入下一步：使用同一个 K=256 renderer Jacobian，把保存的
source/target 主视角与 OFT 腕部像素梯度分别做 VJP，得到同一谱系数空间的逐模态
Action/Feature 贡献。该步骤先做诊断，不直接启动新一轮5000步训练。

在形成联合 objective 前还必须明确威胁模型：

- 若目标是 **source-only transfer**，OFT 梯度只能用于分析/评价，最终选基和
  训练不得使用它；
- 若目标是 **multi-model universal texture**，可以用 OpenVLA+OFT 的一致梯度
  选择谱模态并联合优化，但实验中必须把参与训练的模型与真正 held-out 模型分开。

两者对应不同研究命题，不能在实现时默默混用。

## 快速机制验证路线（2026-08-01 决策）

当前选择 source-only 的快速机制验证：新纹理仍只用 OpenVLA 的训练 states 和
梯度优化。OFT 梯度只回答“OpenVLA 的方向经过同一物理纹理后，是否还与 OFT
决策方向相容”，不进入 loss、谱基排名或任何训练配置。因此实现产物也显式记录
`target_gradient_role=diagnostic_only_not_training_or_selection`。

OFT 已经参与开发期诊断，所以后续 OFT rollout 适合验证“机制是否出现迁移信号”，
但不能单独充当最终论文中的完全无偏 held-out 证据。若机制有效，方法与超参数
冻结后再用未参与选择的新任务或第三个 VLA 模型做正式迁移评估。

快速验证的首轮通过标准保持简单：

- 参数量仍为 K=256，即768个可学习标量；
- OpenVLA Spatial task0 held-out 攻击成功率不低于当前30%；
- OFT 从当前0/10至少提升到2/10失败，先证明存在非零迁移信号；
- 纹理继续满足相同 Surface L∞ 预算，且保持谱参数化的连续低频外观。

## K=256 renderer VJP 实现

`scripts/project_vla_pixel_gradients_to_spectral.py` 不再加载 OpenVLA/OFT，只读取
上一节的像素梯度。对每个固定 state 重建 LIBERO 物体姿态及两台相机，计算：

```text
dL/dC = J_renderer(C)^T · dL/dRGB,  C shape = [256, 3]
```

主视角使用 `agentview`，腕部使用 `robot0_eye_in_hand`。两次 rasterization 的
相机 Jacobian 不同，但输入参数都是同一个物理曲面上的 `[256,3]` 系数，因此
OFT 双视角梯度可在系数空间直接相加。renderer 轮廓还会与真实 clean/adv 变化
mask 比较；主要看 `observed_recall`，避免在可微物体错位时误读余弦。

本轮使用已经训练好的 K=256 系数点和生成它的 K=512 候选谱基前256列：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu-id> \
TOKENIZERS_PARALLELISM=false NUMBA_DISABLE_JIT=1 \
MPLCONFIGDIR=/tmp/tex3d-spectral-projection \
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 \
PYTHONPATH="$PWD/openvla:$PWD" \
/home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python \
scripts/project_vla_pixel_gradients_to_spectral.py \
  --source_path \
    experiments/logs/cross-model-pixel-gradient-audit/source-states10-11-13-15-16.npz \
  --target_path \
    experiments/logs/cross-model-pixel-gradient-audit/target-states10-11-13-15-16.npz \
  --response_report \
    experiments/logs/spectral-k256-oft-response-diagnostic/oft_transfer_response.json \
  --spectral_basis_path \
    experiments/spectral_basis/akita_black_bowl_k512.npz \
  --coefficients_path \
    experiments/logs/spectral-source-comparison/attack_artifacts/spectral-k256-siglip-states0-9-EVAL-libero_spatial-2026_07_27-08_39_16/Ep0_Spectral_Coefficients.pt \
  --source_action_weight 0.1 \
  --source_feature_weight 4.0 \
  --output_npz \
    experiments/logs/cross-model-spectral-projection/k256-states10-11-13-15-16.npz \
  --output_json \
    experiments/logs/cross-model-spectral-projection/k256-states10-11-13-15-16.json
```

NPZ 保存六组 `[S=5,K=256,RGB=3]` VJP 及两视角 mask；JSON 汇总主视角和
双视角系数余弦、腕部/主视角范数比，以及 source-only/target 的模态能量分布。
当前服务器 CUDA driver 不可访问，故实现已通过 CPU 测试但该命令尚待 GPU
运行。运行成功后先检查两视角 `observed_recall`，再解释系数梯度指标。
