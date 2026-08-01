# OFT 迁移响应最小诊断

## 目的

第一阶段已经确认 Spectral K=256 在源 OpenVLA 的 held-out states 10–19 上造成
3/10 失败，但同一张 bake PNG 在 OpenVLA-OFT 上造成 0/10 失败。继续修改谱基
或增加 K 之前，先用一个不训练、不 rollout 的最小实验定位迁移信号消失在哪一
层。

第一轮选择五个固定状态：

- 源 OpenVLA 失败：10、13、15；
- 源 OpenVLA 成功：11、16。

每个状态分别重建 clean 和 adversarial MuJoCo 环境，执行相同的10步 dummy
action，然后比较 OFT 的真实 policy 输入、SigLIP patch features 和原始连续动作
chunk。攻击 PNG 直接写入物体 XML，不经过 PNG→顶点→PNG，也不使用
nvdiffrast composite。

## 运行命令

该诊断需要 GPU，但一次只加载 OFT 模型，并且只查询10次 action（5个状态 ×
clean/adv），远小于完整 rollout。请在仓库根目录运行：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu-id> \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
TOKENIZERS_PARALLELISM=false NUMBA_DISABLE_JIT=1 \
MPLCONFIGDIR=/tmp/tex3d-oft-matplotlib \
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="/home/xiaomengqi/src/github/paper_code/openvla-oft:$PWD/openvla-oft" \
/home/xiaomengqi/miniconda3/envs/tex3d-oft/bin/python \
openvla-oft/experiments/robot/libero/diagnose_transfer_response.py \
  --pretrained_checkpoint \
    /data/xiaomengqi/checkpoints/openvla-7b-oft-finetuned-libero-spatial \
  --texture_path \
    "$PWD/experiments/logs/spectral-source-comparison/attack_artifacts/spectral-k256-siglip-states0-9-EVAL-libero_spatial-2026_07_27-08_39_16/task_0_adv_texture_2026_07_27-08_39_16.png" \
  --task_suite_name libero_spatial \
  --object_name akita_black_bowl \
  --task_id 0 \
  --state_ids 10,11,13,15,16 \
  --output_dir \
    experiments/logs/spectral-k256-oft-response-diagnostic
```

如果希望先做单状态 GPU smoke，把 `--state_ids` 改为 `10`，同时使用另一个
`--output_dir`，避免覆盖正式结果。

## 输出与约束

入口生成：

- `oft_transfer_response.json`：逐状态与跨状态汇总指标；
- `paired_inputs/`：真正送入 OFT policy 的 clean/adv 主视角和腕部输入图像。

JSON 分三层记录：

1. `observation`：RGB 和机器人 state 的 MAE、MSE、最大差、相对 L2、余弦距离；
2. `siglip`：主视角、腕部与二者合并后的 SigLIP patch feature 差异；
3. `action`：完整8步动作 chunk、第一步动作以及夹爪符号翻转数。

OFT checkpoint 的 `timm_model_ids` 顺序是 `[DINOv2, SigLIP]`。实现根据配置
动态选择 processor 的第4–6通道和 `fused_featurizer`，不使用历史攻击代码中
写死的通道顺序。clean 与 adv 的 `robot_state.max_absolute` 应接近0；否则说明
成对环境没有复现同一物理状态，本次结果无效。

脚本会在同目录备份物体 XML，并在正常结束、Python 异常、Ctrl-C 或 SIGTERM
后恢复。若进程被 SIGKILL 或服务器断电，应先检查并人工恢复：

```text
.<object-name>.xml.tex3d-diagnostic-backup
```

## 如何决定下一步

本轮不预先设定绝对阈值，也不承担完整统计结论；先观察三个层次是否存在清晰的
数量级衰减：

- OFT SigLIP 几乎不变：优先修正共享视觉目标、processor 对齐或选择真正共享
  且敏感的 feature；
- SigLIP 明显变化但动作几乎不变：说明 feature distance 没有形成决策裕量，
  下一版优先改跨模型 action/decision proxy；
- 动作已经明显变化但先前 rollout 仍全部成功：说明单帧响应不是主要瓶颈，
  下一版优先覆盖关键轨迹时刻、抓取窗口和跨状态时序。

只有目标层仍含歧义时，才补同状态 OpenVLA 源响应脚本；源模型已有训练 loss、
feature gradient 和 rollout 成败记录，因此不在第一纵切中重复实现。

## 2026-08-01 五状态结果

使用 Spectral K=256 texture（SHA-256
`853c1b304dfb5ea71c1aa955b3e8e85a2c3400b732b3054340833a88b564d843`）完成
states 10、11、13、15、16。逐状态结果如下；`Source` 是该纹理此前在 OpenVLA
完整 rollout 上的结果，Feature 与 Action 均为 OFT clean→adv 相对 L2：

| State | Source | OFT SigLIP（双视角） | OFT Action chunk | OFT 第一步 Action | 夹爪翻转 |
|---:|---|---:|---:|---:|---:|
| 10 | 失败 | 15.67% | 2.04% | 0.79% | 0/8 |
| 11 | 成功 | 17.61% | 4.17% | 5.26% | 0/8 |
| 13 | 失败 | 13.78% | 1.90% | 1.18% | 0/8 |
| 15 | 失败 | 17.54% | 4.18% | 8.51% | 0/8 |
| 16 | 成功 | 16.78% | 2.43% | 3.99% | 0/8 |
| **均值** | — | **16.28%** | **2.95%** | **3.95%** | **0/40** |

有效性检查：五个 state 的 clean/adv robot state 最大差均为0；20张 policy 输入
图片完整保存；纹理在主视角和腕部视角中均真实可见；运行结束后 XML 事务备份
已删除，LIBERO XML 与原 texture 均保持 clean。

这组结果排除了“OFT 完全看不到该纹理”这一解释。纹理在目标 SigLIP 中造成了
稳定且明显的 feature 响应，但进入动作头后变化缩小到约2%–4%，40个夹爪值没有
一次越过符号边界。更关键的是，源失败 state 10、13 的 OFT Action 变化小于源
成功 state 11；目标响应大小与源攻击成败没有一致关系。因此当前 Shared-SigLIP
MSE 找到的主要是源模型敏感方向，并不是跨模型共同的决策敏感方向。

下一版不优先增加 K，也不先补大规模消融。最小方法迭代应保持 K=256、状态划分
和预算不变，把目标从“无方向地增大 SigLIP patch MSE”改为“寻找经过两个模型
视觉编码与决策投影后仍方向一致的扰动”。在实现新的训练 loss 前，先做一次
source/target 梯度方向审计，至少分别记录纹理系数空间中：

1. SigLIP feature loss 梯度余弦；
2. action deviation / decision-margin proxy 梯度余弦；
3. 每个谱模态的跨模型一致贡献。

如果共同 action-sensitive 梯度确实存在，再把它形成小规模联合 objective 并只跑
Spatial task 0；如果几乎不存在，则说明需要换共享层或训练覆盖，而不是继续调 K。

后续五状态像素梯度审计已完成：共享 SigLIP 方向存在，但跨模型 Action 方向弱，
且 OFT 对腕部视角的 Action 梯度明显更强。完整定义、命令和结果见
`docs/spectral/cross-model-pixel-gradient-audit.md`。
