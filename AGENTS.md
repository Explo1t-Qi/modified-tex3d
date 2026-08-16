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

- 当前状态、唯一下一门槛和禁止捷径以
  `docs/status/openvla-spectral-current.md` 为准；决策实验的结果与判定统一登记在
  `docs/experiments/openvla-spectral-ledger.md`。专题文档保留完整命令和历史
  细节，但不得用旧段落覆盖当前状态页。
- 第一阶段已验证谱参数化的工程可行性和源攻击能力，但尚未得到迁移提升。连续
  K=128/256/512 的结果不单调，禁止用盲目增加 K 代替机制诊断。
- 当前训练仍是 source-only OpenVLA。OFT 梯度只作诊断，不得进入 loss、谱基
  排名或训练配置；开发期 OFT rollout 只判断迁移信号，方法冻结后还需新任务或
  第三个 VLA 模型提供无偏证据。
- 历史 Action/last-hidden 六通道路径曾把 DINOv2/SigLIP 顺序与 resize 语义用错。
  当前唯一预处理 interface 使用显式 BPDA/STE：forward 为 checkpoint 精确的
  uint8+PIL 路径，backward 为连续 tensor bicubic surrogate。旧候选只作历史
  工程证据，不能代表修正后的科学基线。
- Fixed-Support Action+Spectral 与严格匹配的 Action-only 对照均只在 source
  development states 10–19 造成2/10 paired新增失败。Gate 6g--6j与最终P/I只读
  分析现已完成；未支持renderer residual、Fixed Support、radial projection或
  BPDA/Action-gradient failure是双终态共同主因，机制诊断阶段已冻结，不自动
  扩展新Gate。Action-only κ margin/drift只读校准显示12个crossed token中11个
  正回弹、8个部署翻转；现已冻结
  `κ=median(crossed positive drift)=4.375`，只作为一次预注册的局部
  source-strength干预，不声称最优，也不解决58个尚未越界token。两步
  GPU工程smoke已完成2/2次更新、20/20行state证据与产物恢复；commit
  `6940700`修正了跨GPU/CPU float32 mean约1 ULP造成的evaluator误拒，
  原bundle经WSL独立复核有效。commit `178fdce`已放行正式5000轮路径，并要求
  显式绑定通过验收的κ-smoke manifest；当前唯一下一门槛是服务器完成正式训练、
  独立bundle复核与paired source-development rollout。不得调Support/K/lambda/
  Feature/wrist或提前进入OFT。
- states 10–19 已反复参与方法决策，必须称为 source development/validation
  states，不能再作为无偏 held-out test。方法冻结后的正式无偏结论仍需新任务或
  第三个 VLA 模型；states 20–49 在审计历史使用前也不得自动宣称为 untouched。

## 谱方法长期不变量

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
- 当前方法开发使用 train states 0–9 与 source development states 10–19；实验
  日志必须保留原始 state ID。states 20–49 的历史使用情况需要在未来指定
  source confirmation split 前审计。谱基、视频、模型权重和攻击纹理等实验
  产物不进入 Git。
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

- WSL 用于代码、文档和可用的 CPU 验证；本地虚拟环境为
  `/home/xmq/.virtualenvs/modified-tex3d/`。本地可能没有完整依赖或 LIBERO 数据，
  缺失时先向用户报告最小需求，不自行安装。
- 服务器 OpenVLA Python 环境：
  `/home/xiaomengqi/miniconda3/envs/tex3d-openvla`。服务器 GPU 命令由用户通过
  SSH 执行，Codex 提供绑定明确 commit SHA 的命令和验收门槛。
- Spatial checkpoint：`/data/huangsimin/openvla-7b-finetuned-libero-spatial`。
- Object checkpoint：
  `/data/huangsimin/openvla/openvla-7b-finetuned-libero-object`。
- OpenVLA-OFT Spatial checkpoint：
  `/data/xiaomengqi/checkpoints/openvla-7b-oft-finetuned-libero-spatial/`。
  OFT 原有加载器会在 checkpoint 目录内备份/同步配置，诊断命令必须使用当前
  用户可写的这份副本，不要使用 `/data/huangsimin/` 下的只读权重。
- 默认验证不得占用 GPU，使用 `CUDA_VISIBLE_DEVICES=''` 和
  `PYTHONDONTWRITEBYTECODE=1`。WSL 依赖齐备时使用：

  ```bash
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 \
    NUMBA_CACHE_DIR=/tmp/tex3d-numba-cache \
    /home/xmq/.virtualenvs/modified-tex3d/bin/python -m pytest -q tests
  ```

  服务器无 GPU 测试入口为：

  ```bash
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 \
    NUMBA_CACHE_DIR=/tmp/tex3d-numba-cache \
    /home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python -m pytest -q tests
  ```

- GPU smoke 只在 GPU 可用且任务需要时运行。命令和验收标准见
  `docs/refactor/openvla-stage-1-baseline.md`。
- 服务器实验结果通过 rsync 放入 Git 忽略的
  `experiments_inbox/<run-id>/`；先检查 commit/config/hash/资产恢复，再更新
  实验账本和正式文档。完整协作流程见
  `docs/development/collaboration-workflow.md`。
- 历史实验目录 `/home/xiaomengqi/src/github/paper_code/tex3d` 只读使用；结果用于
  数值量级和产物结构参考，不作为逐位一致的 oracle。

## 工作区与 Git

- 保留用户已有修改和运行产物，不清理或覆盖与当前任务无关的文件。
- `.pyc`、`__pycache__`、实验日志、rollout 和攻击产物不应进入重构提交。
- 一个已验证、职责完整的小改动可以独立提交。精确暂存本次文件，提交前运行
  `git diff --check`、相关测试并检查 `git status --short`。
- Codex 负责本地代码、文档和 commit；用户负责 push 到 GitHub、服务器 pull
  和服务器实验。未经明确请求，Codex 不 push、不操作服务器。

## Agent skills

### Issue tracker

本仓库使用 GitHub Issues；具体命令约定见 `docs/agents/issue-tracker.md`。

### Triage labels

使用五个标准 triage role 名称；映射见 `docs/agents/triage-labels.md`。

### Domain docs

本仓库采用单一根级 `CONTEXT.md`；消费规则见 `docs/agents/domain.md`。
