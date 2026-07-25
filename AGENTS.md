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
- 默认攻击状态划分为 train 0–9、held-out eval 10–49；实验日志必须保留原始
  state ID。谱基、视频、模型权重和攻击纹理等实验产物不进入 Git。

## 代码规范

- 新建或重写的 Python 代码尽量完整标注参数、返回值、属性和局部变量类型。
- 结构化字典优先使用 `TypedDict`，可替换对象优先使用 `Protocol`，固定常量
  使用 `Final`。不要只为了类型或 shape 标注新增运行时依赖。
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
