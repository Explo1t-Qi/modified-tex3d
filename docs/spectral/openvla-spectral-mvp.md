# OpenVLA 曲面谱参数化 MVP

## 目标与判定问题

第一版只回答一个工程与实验问题：

> 在 OpenVLA / LIBERO Spatial task 0 上，K=128 的低维谱参数化能否以接近
> Geometry Vertex 基线的源模型攻击效果，获得更好的 OpenVLA-OFT 迁移效果？

第一轮不实现谱基筛选、SigLIP 专用 loss、跨模型联合训练、EoT、TAAO 或自然性
正则。低频谱 MVP 显示源模型效果值得继续、但迁移 no-go 后，本文后半部分开始
单独验证 Shared-SigLIP objective；其余能力仍不在当前范围。

## 公平比较

四个实验条件为：

| 条件 | Active Texture | 用途 |
|---|---|---|
| Clean | 原始 MuJoCo texture | 原始成功率控制 |
| Rendered Clean | UV 保真路径的零 Surface Delta bake | 检查渲染/序列化偏差 |
| Geometry Vertex | `N × 3` 自由参数 | 高维攻击基线 |
| Spectral K=128 | `K × 3` 非恒定谱系数 | 待验证方法 |

Geometry Vertex 与 Spectral 共享以下条件：

- 原始 UV 直接采样，Surface Delta 在像素/atlas 上叠加；
- `epsilon = 128/255` 的曲面 RGB L∞ 预算；
- `surface_step = 2/255` 的逐轮最大曲面变化；
- 相同训练帧、action + last hidden feature loss 和优化轮数；
- train init states 为 0–9，held-out eval init states 为 10–49。

谱方法排除常数模态，因此 K=128 表示 128 个非恒定低频模态。Akita bowl
原始几何有 21,263 个顶点：

- Geometry Vertex：`21,263 × 3 = 63,789` 个参数；
- Spectral：`128 × 3 = 384` 个参数；
- 谱参数量约为基线的 0.60%。

## 数据流与不变量

```text
coefficients [K, 3]
        │ Phi @ coefficients
        ▼
geometry Surface Delta [N, 3]
        │ exact topology map
        ▼
render Surface Delta [V, 3]
        │ interpolate with geometry faces
        ├──────────────┐
original UV texture    │
        │ sample UV    │
        └────── add ───┘
               ▼
adversarial foreground / baked Active Texture
```

必须保持：

1. 零 Surface Delta 的 float bake 与原 UV 逐元素相同；
2. PNG 序列化使用最近整数舍入，零增量可恢复原 uint8 texel；
3. UV seam 副本映射到同一个 OBJ 几何顶点；
4. 谱基产物与当前 OBJ 的顶点、face 和 SHA-256 完全匹配；
5. 正式评估只使用 held-out states；
6. OpenVLA-OFT 迁移评估直接把 bake PNG 激活为 MuJoCo Active Texture，并读取
   policy 的 MuJoCo observation；不得在目标模型侧重新投影为顶点颜色。

## 生成 K=128 谱基

在仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
/home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python \
scripts/generate_openvla_spectral_basis.py \
  --mesh /home/xiaomengqi/src/github/paper_code/LIBERO/libero/libero/assets/stable_scanned_objects/akita_black_bowl/akita_black_bowl.obj \
  --num-basis 128 \
  --output experiments/spectral_basis/akita_black_bowl_k128.npz
```

2026-07-25 的本机产物检查结果：

- geometry vertices: 21,263；
- faces: 42,522；
- 最大 M-正交误差: `1.37e-15`；
- 最大特征方程残差: `7.45e-12`。

`experiments/` 是本地实验产物目录，不进入 Git。正式运行前若产物不存在，使用
上面的确定性命令重新生成。

## Spectral GPU smoke

该命令验证真实 mesh/basis 加载、UV 保真渲染、谱系数梯度、曲面归一化更新、
bake、Active Texture 激活和一次 held-out rollout。它不用于判断攻击效果。

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
  --spectral_basis_path experiments/spectral_basis/akita_black_bowl_k128.npz \
  --spectral_basis_count 128 \
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
  --local_log_dir /tmp/tex3d-openvla-spectral-smoke \
  --run_id_note spectral-smoke
```

验收至少包括：

- 启动日志显示 `spectral, parameters=384`；
- gradient log 中梯度非零且 `Actual Surface Step <= 2/255`；
- `Max Surface Delta <= 128/255`；
- 生成 `Spectral_Coefficients.pt` 与 UV PNG；
- episode 使用 State 1，流程正常结束。

2026-07-25 已在真实 GPU 流程通过上述 smoke。产物核对结果：

- loss history 为有限值 `[19.760244]`；
- 谱系数 shape 为 `[128, 3]`，384 个系数均发生非零更新且数值有限；
- gradient norm 为 `1.188943e+02`；
- `Actual Surface Step = 7.843138e-03`，等于 `2/255`；
- `Max Surface Delta = 7.843138e-03`，未触及 `128/255` 上界；
- UV PNG、Spectral Coefficients、gradient log 和 State 1 rollout 均正常生成。

本次单次 held-out rollout 成功率为 100%，但 smoke 仅验证工程数据流，不将该
数值解释为谱攻击效果。

## 第一轮正式源模型实验

正式 Geometry Vertex 与 Spectral 命令仅改变参数化相关字段。两者均使用：

```text
attack_iters=5000
num_train_init_states=10
train_init_state_ids=0-9
eval_init_state_ids=10-49
num_trials_per_task=40
epsilon=128/255
surface_step=2/255
```

建议先做 10 个 held-out states（`eval_init_state_ids=10-19`）的 pilot。若谱方法
造成的失败 episode 数至多比 Geometry Vertex 少 1 个，则认为源模型攻击效果
没有明显退化，并在 OpenVLA-OFT 上比较迁移；Spectral 至少比 Geometry Vertex
多造成 2 个目标模型失败 episode，才进入 40-state confirmation。10-state
阈值只是 go/no-go，不是统计显著性结论。

### 10-state 源模型 pilot 结果

2026-07-26 已使用完全相同的训练状态 0–9、held-out 状态 10–19、seed、loss、
5000 次迭代和曲面更新预算完成 Geometry Vertex 与 Spectral K=128 对照：

| 参数化 | 可学习参数 | OpenVLA 任务成功 | 失败 states |
|---|---:|---:|---|
| Geometry Vertex | 63,789 | 9/10（90%） | 10 |
| Spectral K=128 | 384 | 7/10（70%） | 10、14、17 |

这里记录的是机器人任务成功率，数值越低表示攻击影响越强。Spectral 在共同失败
的 state 10 之外额外造成 state 14、17 失败，因此通过源模型 go/no-go：
参数减少约 166 倍（99.4%）的同时，没有牺牲源模型攻击效果。

两组 loss history 和 gradient log 均有 5000 条有限记录，没有 NaN/Inf：

- Geometry Vertex total loss 从 `1.834667` 降至 `1.140203`；
- Spectral total loss 从 `1.834667` 降至 `1.244859`；
- 两组最大 Surface Delta 均为 `0.5019608`，符合 `128/255` 上界；
- Geometry Vertex 最终参数 shape 为 `[21263, 3]`；
- Spectral 最终系数 shape 为 `[128, 3]`，384 个系数均非零。

UV 纹理视觉检查显示，Geometry Vertex 主要出现细碎局部彩色噪点；Spectral
主要形成连续的大尺度色彩变化，符合低频曲面谱参数化预期。该 pilot 只有 10 个
held-out states，只用于推进迁移验证，不作为统计显著性结论。两次运行结束后
LIBERO XML 与原始纹理均已恢复为 Git clean 状态。

## OpenVLA-OFT 直接迁移评估

源模型生成的 `task_0_adv_texture_*.png` 直接作为 OFT 的 Active Texture：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu-id> \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
TOKENIZERS_PARALLELISM=false NUMBA_DISABLE_JIT=1 \
MPLCONFIGDIR=/tmp/tex3d-oft-matplotlib \
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="/home/xiaomengqi/src/github/paper_code/openvla-oft:$PWD/openvla-oft" \
/home/xiaomengqi/miniconda3/envs/tex3d-oft/bin/python \
openvla-oft/experiments/robot/libero/attack_oft.py \
  --pretrained_checkpoint /data/huangsimin/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --object_name akita_black_bowl \
  --task_id 0 \
  --num_trials_per_task 10 \
  --eval_init_state_ids 10-19 \
  --enable_attack True \
  --load_texture_path /absolute/path/to/task_0_adv_texture_*.png \
  --direct_active_texture_evaluation True \
  --use_wandb False \
  --local_log_dir /tmp/tex3d-oft-transfer \
  --run_id_note spectral-transfer
```

direct adapter 会：

- 校验 PNG、记录文件 SHA-256，并把同一路径写入目标物体 XML；
- 同步真实 MuJoCo texture 后，从 `obs` 读取 policy 的 `full_image`；
- 保留 OFT 正常的 proprioception 与 wrist image；
- 不初始化可微 renderer，不做 PNG→顶点颜色→PNG，也不做 nvdiffrast rollout
  composite；
- 在日志中记录原始 State ID。

服务器现有 `/home/xiaomengqi/miniconda3/envs/tex3d-oft` 环境。该环境需要把
本地 OFT fork 放在 `PYTHONPATH` 最前：
`/home/xiaomengqi/src/github/paper_code/openvla-oft`。加入该路径后，
`prismatic.models.action_heads`、LIBERO 和 nvdiffrast 的 CPU import 已通过。
Robosuite 位于只读 site-packages，入口需设置 `NUMBA_DISABLE_JIT=1`，避免 Numba
尝试建立不可用的函数缓存；上述环境组合的真实 `attack_oft.py --help` 已通过。

2026-07-26 使用 Spectral K=128 的 bake PNG 在 OFT State 10 上完成单 episode
GPU smoke：

- direct Active Texture、OFT L1 action head、proprioception 和双相机输入流程
  正常结束；
- State 10 成功，rollout MP4 与评估日志正常生成；
- 退出后 XML 与真实纹理均恢复，相关 LIBERO 资产保持 Git clean。

单个 State 10 的 100% 只表示工程链路通过，不表示谱纹理没有迁移效果。

### 10-state OFT 迁移 pilot 结果

2026-07-26 在与源模型相同的 held-out states 10–19 上依次完成 Clean、
Geometry Vertex bake PNG 和 Spectral K=128 bake PNG 的直接迁移评估：

| OFT 条件 | 任务成功 | 失败 states |
|---|---:|---|
| Clean | 10/10（100%） | 无 |
| Geometry Vertex | 10/10（100%） | 无 |
| Spectral K=128 | 10/10（100%） | 无 |

三组共生成 30 个非空 rollout MP4，运行后 XML 与真实纹理均恢复，LIBERO 资产
保持 Git clean。对 State 10 的首帧进行视觉核查时，Geometry Vertex 的细碎
彩色扰动与 Spectral 的大尺度绿/蓝/橙色变化均清晰出现在 MuJoCo observation
中；direct evaluation 分支的 `renderer=None`，policy 输入直接来自该
observation。因此三组同为 100% 不是纹理未激活或错误走入 nvdiffrast composite
导致的假阴性。

该结果未达到预先约定的迁移 go/no-go（Spectral 至少比 Geometry Vertex 多造成
2 个 OFT 失败 episode），所以不直接进入 40-state confirmation，也不声明迁移
能力得到提升。当前 MVP 只支持以下结论：谱方法以约 1/166 的参数量保持并提升了
源 OpenVLA 上的攻击效果，同时得到更连续的纹理结构；仅使用最低 128 个非恒定
模态和 OpenVLA 专用 action/last-hidden loss，尚未产生可观测的 OFT 迁移优势。

## Shared-SigLIP objective MVP

第一轮迁移 no-go 后，下一版保持 Spectral K=128、状态划分、action loss、优化器
和 Surface Delta 预算不变，只把 feature objective 从 OpenVLA 最后一层 hidden
切换为源/目标模型共享的 SigLIP patch features：

```text
clean RGB                         adversarial RGB
    │                                   │
    ├─ SigLIP normalization             ├─ SigLIP normalization
    │                                   │
    ▼                                   ▼
checkpoint 定位的 SigLIP featurizer（second-to-last patch features）
    │                                   │
    └────────── negative MSE ────────────┘
                         │
                         ▼
            texture / spectral coefficients gradient
```

CLI 使用：

```text
feature_objective=last_hidden  # 默认，完整保留历史行为
feature_objective=siglip_patch # 新共享视觉目标
```

审计 checkpoint 时确认，模型原生 6 通道顺序是 DINOv2→SigLIP；历史攻击代码
手工拼接为 SigLIP→DINOv2。已有 Geometry/Spectral 源实验在相同历史路径下训练，
其相对比较仍然公平。为避免改写已有基线，`last_hidden` 和 action forward 继续
保留历史顺序；`siglip_patch` 不走 6 通道拼接，而是根据
`config.timm_model_ids` 找到唯一 SigLIP 分支，并只接收正确归一化的三通道输入。

干净 SigLIP features 在 Training Frame Collection 阶段以
`[batch, patches, feature_dim]` 保存并 detach；对抗 features 在每轮优化时保持
到渲染纹理的梯度。两者 shape 不一致、模型没有唯一 SigLIP 分支或输入不是三通道
时直接失败，禁止静默回退到 last hidden。

GPU smoke 使用与第一轮相同的最小配置，只新增：

```bash
--feature_objective siglip_patch \
--alpha_action 1.0 \
--alpha_feature 10.0 \
--local_log_dir /tmp/tex3d-openvla-siglip-smoke \
--run_id_note spectral-k128-siglip-smoke
```

该 smoke 首先检查 feature loss 的量级、谱系数梯度和 Surface Step；在看到真实
数值前，不直接沿用 5000 轮正式训练，也不调整 alpha，以免同时改变 feature
来源与 loss 权重。

2026-07-26 已在真实 OpenVLA checkpoint 上完成该 smoke：

- action loss 为 `21.916494`，SigLIP feature loss 为 `-0.076172`，
  `alpha_action=1`、`alpha_feature=10` 时 total loss 为 `21.154776`，三者
  满足预期的加权关系且均为有限值；
- 谱系数梯度范数为 `7.280559e+01`，保存的 coefficients shape 为
  `[128, 3]`，384 个参数全部非零且有限；
- `Actual Surface Step = 7.843137e-03`、`Max Surface Delta =
  7.843137e-03`，均为 `2/255`，没有越过 `128/255` 上界；
- UV 与原始 4096×4096 纹理相比，八位图最大通道差为 2，artifact UV 与最终
  active texture 完全一致；
- held-out State 1 rollout 成功结束。单次任务成功率 100% 只证明流程可运行，
  不用于判断攻击强弱。

同状态的上一版 last-hidden smoke 具有相同 action loss `21.916494`，说明该次
对照没有意外改变采帧或 action 路径。其 feature loss 为 `-0.215820`，绝对值约
为本次 SigLIP feature loss 的 2.83 倍。该比例只能用于 loss 数值尺度的初步
判断，不能代替两个目标关于谱系数的梯度范数和梯度方向对照；正式选择
`alpha_feature` 前先做 action-only 与 SigLIP-only 单步梯度标定。

## 当前验证状态

- CPU 数值与回归测试：59 passed；
- 真实 Akita mesh + K=128 basis 的 CPU 加载：
  `coefficients=(128, 3)`、384 参数、零 Surface Delta；
- GPU spectral smoke：已通过，梯度、曲面归一化更新、bake、产物保存与
  held-out rollout 均正常；
- OpenVLA 10-state pilot：Geometry Vertex 任务成功率 90%，Spectral K=128
  为 70%，通过源模型 go/no-go；
- OFT direct Active Texture adapter：CPU 测试通过；`tex3d-oft` 加本地 OFT
  fork 后 action head import 与单状态 GPU smoke 均已通过；
- OFT 10-state 迁移 pilot：Clean、Geometry Vertex 与 Spectral K=128 均为
  100%，未达到迁移 go/no-go；
- Shared-SigLIP objective：强类型接口、分支定位、三通道校验、Training Frame
  clean feature 与纹理梯度 CPU 测试已通过；真实 GPU smoke 的损失、梯度、
  曲面约束、artifact 与 held-out rollout 均已通过；
- 40-state confirmation：按预先阈值暂不执行。
