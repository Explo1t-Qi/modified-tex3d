# OpenVLA 谱纹理方法第一阶段实验总结

## 阶段目标

本阶段建立一条可复现的谱纹理攻击链路，并回答两个最小研究问题：

1. 把原本的 `N × 3` 顶点扰动限制到 `K × 3` 曲面谱空间后，能否保留源
   OpenVLA 上的攻击能力，并减少参数和高频色块；
2. 使用 OpenVLA 与 OpenVLA-OFT 共享的 SigLIP 特征作为目标后，谱纹理能否
   直接迁移到 OpenVLA-OFT。

本阶段是可行性验证，不以证明最终方法有效为完成条件。实现、源攻击、迁移评估
和失败原因均得到可复查结果，即视为阶段完成。

## 统一实验设置

除特别说明外，Spatial 实验使用：

| 项目 | 设置 |
|---|---|
| 任务 | LIBERO-Spatial Task 0 |
| 物体 | `akita_black_bowl` |
| 源模型 | OpenVLA |
| 目标模型 | OpenVLA-OFT |
| 训练 states | 0–9，每个 state 的 step 0 一帧 |
| 评估 states | 10–19 |
| 优化轮数 | 5000 |
| Surface L∞ 预算 | `128/255` |
| 每轮最大 Surface Step | `2/255` |
| 谱基 | cotangent Laplacian 的非恒定特征向量 |

“任务成功率”越低，表示纹理攻击影响越强。只有在对应 clean control 为 100%
时，`1 - 任务成功率` 才能直接解释为该组观测到的攻击失败率。所有正式 pilot
均只有10个 held-out states，适合进行 go/no-go 判断，不代表统计显著性结论。

## 可直接用于汇报的结果

### 1. 低维谱参数化源模型基线

该组使用原 OpenVLA action + last-hidden objective：

| 方法 | 可学习参数 | 参数占 Geometry 比例 | OpenVLA 任务成功率 | 失败 states |
|---|---:|---:|---:|---|
| Geometry Vertex | 63,789 | 100% | 90% | 10 |
| Spectral K=128 | 384 | 0.60% | 70% | 10、14、17 |

结论：K=128 只使用约 `1/166` 的参数量，在该10-state pilot 中没有削弱源
OpenVLA 攻击，反而多造成两个任务失败。Geometry Vertex 产生较细碎的局部
色块，谱纹理主要表现为连续的大尺度颜色变化。

这里的 Geometry Vertex 是 UV 保真重构后的全几何顶点基线：仍优化所有
`N × 3` 曲面颜色扰动，但与谱方法共享原始 UV、Surface Delta、L∞ 预算和曲面
归一化步长。它不是旧版会在零扰动时重采样并模糊 UV 的 Legacy Vertex 路径。

### 2. Shared-SigLIP 目标和连续谱维数

该组固定 `alpha_action=0.1`、`alpha_feature=4.0`，Feature objective 使用源/
目标模型共享的 SigLIP patch features：

| Spatial 方法 | 可学习参数 | OpenVLA 任务成功率 | 观测任务失败率 | 失败 states |
|---|---:|---:|---:|---|
| Spectral K=128 | 384 | 80% | 20% | 10、13 |
| Spectral K=256 | 768 | 70% | 30% | 10、13、15 |
| Spectral K=512 | 1,536 | 90% | 10% | 13 |

结论：攻击效果随 K **不单调**。目前连续低频谱基中 K=256 最强；K=512
虽然表达空间更大，却得到更低的任务失败率，因此不能通过继续盲目增加 K
解决攻击较弱的问题。

K=256 与 K=512 的训练和纹理诊断进一步说明了原因：

| 指标 | K=256 | K=512 |
|---|---:|---:|
| 最终 total loss | 0.1414 | 0.1378 |
| 最后100轮 Action loss 均值 | 20.3793 | 20.4860 |
| 最后100轮 Feature loss 均值 | -0.1546 | -0.1673 |
| UV 扰动 MAE | 2.934 | 2.927 |
| UV 扰动 RMSE | 9.727 | 9.712 |
| UV TV proxy | 0.0613 | 0.0776 |
| 最大通道差 | 128 | 128 |

K=512 的 total/Feature loss 更优，但 Action loss 和 held-out 任务失败率更差；
其 TV proxy 比 K=256 高约 27%。新增256个模态占最终 M-正交曲面扰动能量约
39.4%，并把最终曲面扰动方向改到与 K=256 余弦相似度约 0.25 的不同解。
因此该现象更符合“高维谱空间拟合训练代理目标、但没有形成更强决策攻击”，
而不是训练未收敛。

### 3. OpenVLA 到 OpenVLA-OFT 的直接迁移

源模型生成的 bake PNG 被直接激活为目标模型的 MuJoCo texture；目标侧不重新
渲染或优化纹理：

| OpenVLA-OFT 条件 | 任务成功率 | 失败 states |
|---|---:|---|
| Clean | 100% | 无 |
| Geometry Vertex + last hidden | 100% | 无 |
| Spectral K=128 + last hidden | 100% | 无 |
| Spectral K=128 + Shared SigLIP | 100% | 无 |
| Spectral K=256 + Shared SigLIP | 100% | 无 |

全部纹理均已确认出现在目标 policy 实际读取的 MuJoCo observation 中，因此
100% 不是纹理未激活或评估错误导致的假阴性。第一版谱方法和 Shared-SigLIP
目标均未达到预定迁移门槛，当前不能声称迁移性优于 Geometry Vertex。

其中 K=256 是当前 Shared-SigLIP 连续谱基中源攻击最强的版本：它在源
OpenVLA 上使 states 10、13、15 失败，但同一张纹理迁移到 OFT 后，这三个
states 和其余七个 states 全部成功。10个目标 rollout MP4 均非空，评估后
LIBERO 资产恢复正常。这个对照削弱了“迁移为零只是因为 K=128 源攻击太弱”
的解释，更直接指向源/目标模型之间的代理目标和决策路径差异。

### 4. 跨任务/物体工程验证

LIBERO-Object Task 0 / `alphabet_soup` 的 Spectral K=512 + Shared SigLIP
实验使用同样的训练/评估 state 划分和5000轮优化：

| 任务 | 方法 | OpenVLA 任务成功率 | 失败 states |
|---|---|---:|---|
| LIBERO-Object Task 0 | Spectral K=512 | 80% | 11、17 |

该结果证明谱基生成、拓扑映射、UV bake 和训练链路可以迁移到另一个 mesh、
任务集和 OpenVLA checkpoint。由于没有同任务的 K=128/K=256 和 clean 对照，
它不能用于判断 Object 上哪个 K 更优，也不是跨 VLA 模型迁移证据。

## 梯度审计结果

使用源 OpenVLA 的训练 states 0–9，在 K=512 候选池的零扰动点分别审计 Action
和 Shared-SigLIP 梯度：

| 目标 | 连续 K=128 能量覆盖 | 连续 K=256 | 连续 K=384 |
|---|---:|---:|---:|
| Action | 43.61% | 69.45% | 87.70% |
| SigLIP Feature | 37.71% | 63.65% | 85.02% |

Feature stable-score 选出的128个模态中，79个来自最低128维，49个来自更高
频段。该集合的跨状态一致性优于连续 K=128，但 Feature 原始能量只增加
4.24个百分点，同时 Action 能量覆盖由 43.61% 降到 39.82%。这说明选基值得
研究，但只按 Feature 排序可能继续放大“Feature 变远、Action 不变”的代理
偏差。

## 阶段结论

### 已经得到的正面结果

1. 完成了类型明确、UV 保真、可微且受统一 Surface L∞ 约束的谱参数化实现；
2. 参数由 Geometry Vertex 的63,789个降到 K=128 的384个；
3. 零扰动保持原 UV，避免旧路径的重采样模糊；
4. 在 Spatial Task 0 上，K=128 last-hidden 版本保持了源模型攻击效果，并得到
   更连续的纹理；
5. 谱方法已在另一物体和 Object checkpoint 上完成真实训练验证；
6. OpenVLA→OpenVLA-OFT 的直接迁移评估链路已经建立并验证；
7. source-only 梯度审计、非连续谱基 provenance 和不同 K 对照均已建立。

### 没有得到的研究结论

1. Shared-SigLIP K=128 和当前源攻击最强的 K=256 谱纹理均未在
   OpenVLA-OFT 上造成任务失败；
2. 增大 K 没有单调提高源攻击，K=512 反而弱于 K=256；
3. 当前没有证据证明谱方法提高了跨 VLA 模型迁移性；
4. 尚未在 π0 或更多任务上进行迁移确认。

因此，本阶段的准确表述是：

> 第一版谱参数化及其迁移评估已经完成；谱方法的工程可行性和源模型攻击能力
> 得到验证，但“提高跨模型迁移性”的核心假设尚未通过实验。

## 下一阶段问题

下一阶段应从“继续增加 K”转向解释并解决对抗性与迁移性的脱节，优先回答：

1. **源攻击强度问题**：当前 held-out 源模型最多只有3/10失败，可能仍未形成
   足够大的决策裕量；但 K=256 源失败增加后 OFT 仍为0/10失败，说明它不能
   作为唯一解释；
2. **代理目标问题**：为什么更大的 SigLIP feature distance 没有转化为更差的
   Action 和任务成功率；
3. **优化约束问题**：高频局部峰值是否通过全局 Surface L∞ 步长归一化和边界
   投影，挤占了低频方向的有效更新预算；
4. **数据覆盖问题**：每个训练 state 只使用 step 0 一帧，是否使高维谱空间
   过拟合固定初始视角，而不能覆盖 rollout 中的视角和状态变化；
5. **选基问题**：是否应联合 Action 强度、Shared Feature 一致性和频率/表面
   平滑性进行排序，而不是只使用连续低频或纯 Feature stable score。

建议先建立可量化的诊断：同时记录源/目标模型在 clean 与 adversarial observation
上的 Feature 距离、Action 偏移和 rollout 成功率，再决定优先修改 loss、训练
视角采样、谱基选择还是曲面更新规则。这样可以避免再次只把某个代理损失优化得
更好，却没有提升真实攻击和迁移效果。

## 结果索引

- 实现、基线和迁移流程：
  `docs/spectral/openvla-spectral-mvp.md`
- 梯度审计和非连续选基：
  `docs/spectral/openvla-spectral-gradient-audit.md`
- Spatial K=256：
  `experiments/logs/spectral-source-comparison/`
  `spectral-k256-siglip-states0-9-EVAL-libero_spatial-2026_07_27-08_39_16.txt`
- Spatial K=256 → OFT：
  `experiments/logs/spectral-k256-siglip-oft-transfer/`
  `spectral-k256-siglip-oft-transfer-EVAL-libero_spatial-2026_07_27-16_59_25.txt`
- Spatial K=512：
  `experiments/logs/spectral-source-comparison/`
  `spectral-k512-siglip-states0-9-EVAL-libero_spatial-2026_07_27-12_17_48.txt`
- Object K=512：
  `experiments/logs/spectral-object-k512/`
  `object-alphabet-soup-k512-siglip-states0-9-EVAL-libero_object-2026_07_27-09_02_32.txt`

`experiments/` 下的日志、谱基、纹理、系数和视频均为本地实验产物，不进入 Git。
