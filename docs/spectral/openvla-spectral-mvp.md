# OpenVLA 曲面谱参数化 MVP

## 目标与判定问题

第一版只回答一个工程与实验问题：

> 在 OpenVLA / LIBERO Spatial task 0 上，K=128 的低维谱参数化能否以接近
> Geometry Vertex 基线的源模型攻击效果，获得更好的 OpenVLA-OFT 迁移效果？

本轮不实现谱基筛选、SigLIP 专用 loss、跨模型联合训练、EoT、TAAO 或自然性
正则。它们只有在 MVP 显示谱空间值得继续研究后再加入。

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
在源 OpenVLA 上相对 Geometry Vertex 的攻击失败数差距不超过 1 个 episode，
再在 OpenVLA-OFT 上比较迁移；Spectral 至少比 Geometry Vertex 多造成 2 个
失败 episode，才进入 40-state confirmation。10-state 阈值只是 go/no-go，
不是统计显著性结论。

## OpenVLA-OFT 直接迁移评估

源模型生成的 `task_0_adv_texture_*.png` 直接作为 OFT 的 Active Texture：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu-id> \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
<oft-python> \
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

当前 `tex3d-openvla` conda 环境缺少 OFT 的
`prismatic.models.action_heads`，因此 `<oft-python>` 尚不能填写为该环境。
仓库原有 Dockerfile 会安装 OpenVLA-OFT action/proprio/diffusion 组件；在运行
目标模型 smoke 前，需要确认可用 OFT 容器或 conda 环境。这是运行环境缺口，
不是谱参数化的数据流缺口。

## 当前验证状态

- CPU 数值与回归测试：55 passed、1 skipped；
- 真实 Akita mesh + K=128 basis 的 CPU 加载：
  `coefficients=(128, 3)`、384 参数、零 Surface Delta；
- GPU spectral smoke：尚未执行；
- OFT direct Active Texture adapter：CPU 测试通过；目标运行环境待确认；
- OpenVLA/OFT 正式实验：尚未执行。
