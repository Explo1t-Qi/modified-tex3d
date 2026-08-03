# Tex3D 仓库协作约定

## 当前重构目标

- 当前优先重构 OpenVLA 鲁棒性评估代码，入口为
  `openvla/experiments/robot/libero/attack_openvla.py`。
- OpenVLA、OpenVLA-OFT、π0 等 VLA 模型的专用逻辑分别维护；先通过
  OpenVLA 建立清晰的模块划分，再将成熟的设计迁移到其他模型。
- `tex3d.md` 用于理解论文概念。论文描述与现有代码冲突时，以现有代码行为
  为准。TAAO 等未完成能力可以预留 seam，但未经明确任务不要推测性实现。
- 不要求严格保持旧的 Python 内部 interface；实验行为、关键数值语义和可复查
  的命令行流程应通过测试或基线文档保护。

## 当前谱参数化目标

- 第一版谱参数化、Shared-SigLIP 目标、OpenVLA→OpenVLA-OFT 迁移评估、
  source-only 梯度审计及连续 K=128/256/512 对照均已完成；统一结论见
  `docs/spectral/openvla-spectral-phase-1-results.md`。
- 第一阶段验证了工程可行性和源模型攻击能力，但尚未得到迁移提升。后续优先
  诊断源攻击强度、代理目标、训练视角覆盖和曲面更新约束，不盲目继续增加 K。
- 当前第二阶段先运行 OFT clean/adv 固定状态响应诊断，比较真实 policy 输入、
  SigLIP patch feature 和动作 chunk；实现、命令及判读见
  `docs/spectral/oft-transfer-response-diagnostic.md`。在该证据出来前不扩展为完整
  消融框架。
- OpenVLA→OFT 五状态像素梯度审计已完成；模型专用前向分别维护，模型无关的
  crop、artifact 和 cosine 统计位于 `scripts/vla_pixel_gradient_audit.py`。
  结果确认共享 Feature 方向存在但 Action 方向弱，并发现 OFT 腕部 Action 梯度
  更强；完整定义见 `docs/spectral/cross-model-pixel-gradient-audit.md`。
- 当前选择“优先快速验证机制”：训练仍是 source-only OpenVLA，OFT 梯度只作
  诊断，不得进入 loss、谱基排名或训练配置。开发期间可用 OFT rollout 判断机制
  是否出现迁移信号；方法冻结后，正式无偏迁移结论仍需使用未参与开发选择的新
  任务或第三个 VLA 模型。
- 已实现把保存的像素梯度通过同一个 K=256 renderer Jacobian 投影到谱系数空间；
  主视角和腕部使用各自 MuJoCo 相机，但共享同一组物理谱系数。实现与 GPU 命令
  见 `docs/spectral/cross-model-pixel-gradient-audit.md`。
- 多实例修正后的正式 VJP 结果为：主视角跨模型 Feature/Action cosine
  `0.662 / 0.228`，当前 Source目标对 OFT主视角/双视角Action为
  `0.281 / 0.133`；OFT腕部Action范数是主视角3.31倍且两视角方向近似正交。
  source-only dual-view Shared-SigLIP 已完成 CPU 侧实现：Action仍只用
  OpenVLA主视角，Feature覆盖主视角和腕部，不引入OFT梯度；真实 GPU smoke、
  正式命令及验收门槛见 `docs/spectral/openvla-dual-view-siglip.md`。
- 双视角 Shared-SigLIP 的 K=256 单步 GPU smoke 已通过；gradient log 会分别
  持久化主视角与腕部 Feature loss。正式 states 0–9、5000轮源训练已完成：
  OpenVLA只造成1/10失败，低于旧K=256的3/10；尽管两个视角 Feature loss 都
  更强，Action loss 反而更差。因此按预设门槛跳过OFT rollout，下一步先审计
  双视角Feature与源Action的系数梯度冲突。
- 双视角 source-only 谱审计会额外保存主视角/腕部各自的 Feature loss 与
  `[S,K,3]` 梯度；CPU 汇总入口为
  `scripts/analyze_dual_view_spectral_gradients.py`。第一轮固定零 Surface Delta、
  K=256、训练 states 0–9，不更新纹理、不读取 OFT，现已完成：Action 与
  Primary/Wrist Feature cosine 为 `0.0166/-0.0093`，不存在强负向冲突；加权
  Feature/Action 范数比 `1.3291` 也低于旧单视角参考的 `1.4607`，不得仅凭
  零点结果直接改 loss weight。保存的最终 `[256,3]` 谱系数固定参考点审计也
  已完成：Action/Combined Feature 仍近似正交（cosine `-0.0106`），但加权
  Feature/Action 范数比升至 `1.7342`，总梯度与Action的cosine降至`0.5058`；
  10个状态中两项变化均有8个同向。当前优先讨论动态范数保护，不优先采用只
  处理负内积的 PCGrad；完整结果与判读见
  `docs/spectral/openvla-dual-view-siglip.md`。
- 动态梯度范数保护首版已经实现：每轮先在谱系数空间累积包含 alpha 和 frame
  weight 的 Action/Combined Feature 梯度，再把 Feature 范数限制为 Action 的
  `rho` 倍，最后进入既有 Surface-normalized step。功能默认关闭；首轮固定
  `rho=1.0`，仅验证 spectral + siglip_patch + primary_wrist。逐轮日志必须保留
  两组加权范数、原始 ratio、Feature scale 和 cosine；GPU smoke 命令及门槛见
  `docs/spectral/openvla-dual-view-siglip.md`。
- 动态梯度范数保护的单步 OpenVLA GPU smoke 已通过：原始 ratio `1.093216`、
  scale `0.914733`、缩放后 ratio 为 `1.0000003`；Surface Step 与最大扰动均为
  `2/255`，谱系数/PNG/运行时资产均正常。下一步只运行一个 K=256、rho=1.0、
  states 0–9、5000轮正式源候选；held-out states 10–19 至少3/10失败才进入 OFT。
- rho=1.0 正式源候选已完成：保护5000/5000轮触发，ratio/scale均值为
  `2.9875/0.3355`，Action loss 最后100轮从未保护的 `20.4625` 改善到
  `19.5362`，但 held-out 仍只有1/10失败。该候选未过源门槛，不运行 OFT，
  也不直接扫描 rho。下一步用最终 Active Texture 回放训练 states 0–9，区分
  Action loss 代理失配和状态过拟合；完整结果见双视角文档。
- 最终纹理在训练 states 0–9 回放也只有1/10失败，且 Active Texture hash 与
  正式训练完全一致，因此状态过拟合不是主要解释。下一步优先实现 source
  OpenVLA action-response 最小诊断：在相同初始观测比较 clean/adv action 向量、
  token Hamming、clean/argmax/对称 target margin 与 CE；在该证据出来前不再
  扫描 rho、K、Feature weight，也不进入 OFT。
- Source OpenVLA action-response 诊断已实现为 fixed-reference、forward-only
  模式：复用正式采帧和多实例主视角合成，同时报告 collector/manual-clean
  一致性、teacher-forced 对称 target CE/margin、greedy token Hamming 与连续
  action L2/L∞。greedy 路径使用攻击训练的6通道预处理，不代表 rollout 的
  center-crop 输入；若训练输入动作已变化，先补部署预处理响应，再判断轨迹采样。
  详细命令和分支标准见双视角文档；GPU 结果出来前不修改 loss。
- 2026-08-03 动作响应结果定位到更早的历史预处理失配：collector/processor
  clean 与 Action loss 手工输入平均有4.7/7 token 不同。checkpoint 要求
  DINOv2→SigLIP、bicubic+antialias，而历史 Action/last-hidden 路径使用
  SigLIP→DINOv2、bilinear，实际把两种归一化图像送反编码器。当前优先建立
  processor-equivalent 的唯一可微预处理 interface，并以零扰动7/7 token一致为
  门槛；修正前不扫描K/rho、不修改target loss、不进入OFT。
- Processor-equivalent 可微预处理已实现并迁移正式 OpenVLA 路径：配置来自真实
  model/processor，顺序为DINOv2→SigLIP，resize为bicubic+antialias；collector
  clean label、Action/last-hidden、Shared-SigLIP、动作响应和源像素梯度不再各自
  拼接。真实 processor pixel values 会随 TrainingFrame 保留用于独立 smoke。
  CPU全量111 passed、1 skipped。state0 forward-only GPU smoke 已通过：真实
  processor、collector 与可微输入均7/7 token一致，所有诊断量有限，运行时资产
  恢复正常。真实 backward/update/bake 单轮 smoke 也已通过：256×3谱系数全部
  更新，Surface Step/Delta均为2/255，PNG、rollout和资产事务正常。K=256、
  rho=1.0、states 0–9、5000轮 processor-equivalent 正式源候选也已完成，但
  held-out states 10–19 为0/10攻击成功；训练数值、保护、预算、bake和资产事务
  均正常。按门槛不进入OFT。最终系数的训练states 0–9 action-response 首轮显示
  可微clean→adv平均改变3.6/7 token，但真实processor与可微clean只有6/10序列
  完全一致，不能直接归因target或轨迹。诊断已补充三条路径的top1−top2 margin
  和第一次processor分叉信息；下一步用相同系数短重跑，再决定是否采用
  PIL/uint8 forward + tensor-gradient 的BPDA/STE。
- OpenVLA 第一版使用 LIBERO Spatial task 0 / `akita_black_bowl`，从 K=128
  个非恒定低频谱基开始；实现和命令见
  `docs/spectral/openvla-spectral-mvp.md`。
- Geometry Vertex 与 Spectral 必须通过同一个 Texture Parameterization
  interface 生成 Surface Delta，并使用相同 L∞ 预算、曲面归一化步长、原始
  UV 采样路径及训练帧。
- 谱基定义在原始 OBJ 几何顶点上；UV seam 的 renderer 顶点必须用 face
  corner 拓扑严格映射，禁止用最近邻猜测。
- 零 Surface Delta 必须保留原 UV。迁移评估直接激活 bake PNG 并使用目标
  policy 的 MuJoCo observation，禁止在目标模型侧再次做 PNG→顶点→PNG。
- 一个纹理资产可能被场景中的多个物体实例共享。直接激活 PNG 会同时改变所有
  实例，因此物理纹理对应的 renderer Jacobian 必须对第一个命中关键词组中的
  全部 body 分别渲染并累加；不能只对语义目标 body 做 VJP。
- 默认攻击状态划分为 train 0–9、held-out eval 10–49；实验日志必须保留原始
  state ID。谱基、视频、模型权重和攻击纹理等实验产物不进入 Git。
- OpenVLA checkpoint 的视觉分支顺序必须从 `timm_model_ids` 等模型配置读取，
  禁止仅凭 `featurizer` / `fused_featurizer` 属性名猜测 DINOv2 与 SigLIP。
  历史 6 通道顺序只为复现实验保留；新共享特征代码必须走显式模型分支。
- 谱基选择的第一轮审计严格使用源 OpenVLA 训练 states；目标 OFT 梯度只能
  作为迁移诊断，不能进入 source-only 排名。逐模态 raw gradient 必须同时报告
  跨状态方向一致性与 Surface L∞ 幅值归一化，禁止直接按单帧梯度绝对值选基。
- 非连续谱基产物必须记录候选谱基和审计文件 SHA-256、source state IDs、
  选择分数及完整源模态索引；禁止只保存重排后的 basis 而丢失选基 provenance。

## 代码规范

- 新建或重写的 Python 代码尽量完整标注参数、返回值、属性和局部变量类型。
- 结构化字典优先使用 `TypedDict`，可替换对象优先使用 `Protocol`，固定常量
  使用 `Final`。不要只为了类型或 shape 标注新增运行时依赖。
- 当前 Draccus 版本不能解码 `typing.Literal`。CLI dataclass 字段使用 Draccus
  支持的 `str`，并在入口立即校验、收窄为内部 `Literal` 类型。
- 张量和数组在 docstring 或首次出现处用中文标明语义、dtype、device 和 shape，
  例如 `[batch_size, sequence_length, vocab_size]`。
- 新模块使用详细中文 docstring 和中文注释说明数据流、算法约束和不直观的转换；
  名称仍使用清晰的英文领域术语。
- `attack_openvla.py` 逐步收敛为实验编排入口。纯计算和模型专用语义放入
  `openvla/experiments/robot/libero/openvla_attack/`。
- 重构应采用小的行为保持式纵切：先建立或迁移测试，再移动一个清晰职责，
  最后运行相关单元测试和完整无 GPU 测试。

## 环境与验证

- OpenVLA Python 环境：`/home/xiaomengqi/miniconda3/envs/tex3d-openvla`。
- Spatial checkpoint：`/data/huangsimin/openvla-7b-finetuned-libero-spatial`。
- Object checkpoint：
  `/data/huangsimin/openvla/openvla-7b-finetuned-libero-object`。
- OpenVLA-OFT Spatial checkpoint：
  `/data/xiaomengqi/checkpoints/openvla-7b-oft-finetuned-libero-spatial/`。
  OFT 原有加载器会在 checkpoint 目录内备份/同步配置，诊断命令必须使用当前
  用户可写的这份副本，不要使用 `/data/huangsimin/` 下的只读权重。
- 默认验证不得占用 GPU，使用 `CUDA_VISIBLE_DEVICES=''` 和
  `PYTHONDONTWRITEBYTECODE=1`。无 GPU 测试入口为：

  ```bash
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 \
    NUMBA_CACHE_DIR=/tmp/tex3d-numba-cache \
    /home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python -m pytest -q tests
  ```

- GPU smoke 只在 GPU 可用且任务需要时运行。命令和验收标准见
  `docs/refactor/openvla-stage-1-baseline.md`。
- 历史实验目录 `/home/xiaomengqi/src/github/paper_code/tex3d` 只读使用；结果用于
  数值量级和产物结构参考，不作为逐位一致的 oracle。

## 工作区与 Git

- 保留用户已有修改和运行产物，不清理或覆盖与当前任务无关的文件。
- `.pyc`、`__pycache__`、实验日志、rollout 和攻击产物不应进入重构提交。
- 一个已验证、职责完整的小改动可以独立提交。精确暂存本次文件，提交前运行
  `git diff --check`、相关测试并检查 `git status --short`。
