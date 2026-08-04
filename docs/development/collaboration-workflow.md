# Tex3D 开发与实验协作流程

本文规定用户与 Codex 在“WSL 开发、服务器实验、GitHub 同步”模式下的默认协作
方式。目标是让每轮工作可理解、可中断、可恢复，并避免聊天上下文成为唯一记忆。

## 环境与职责

### WSL

- 仓库：`/home/xmq/src/modified-tex3d`；
- 本地虚拟环境：`/home/xmq/.virtualenvs/modified-tex3d/`；
- 用于代码阅读、编辑、文档、静态检查和环境允许的 CPU 测试；
- 本地可能缺少完整 Python 依赖、LIBERO 数据和 CUDA。缺失时 Codex 先给出最小
  依赖清单，由用户决定和执行安装；Codex 不自行安装依赖。

### 服务器

- 用户通过 SSH 操作；
- 提供 OpenVLA/OFT Python 环境、LIBERO 数据、checkpoint、MuJoCo/EGL 和 CUDA；
- GPU smoke、真实 backward、rollout 和正式实验由用户运行；
- Codex 提供绑定明确 commit SHA 的完整命令、预期产物和验收门槛。

### Git 与 GitHub

- Codex 负责本地代码、文档、精确暂存、检查和职责完整的小 commit；
- 用户负责 push 到 GitHub、服务器 pull 和服务器实验；
- 未经明确任务，Codex 不 push、不操作服务器，也不提交大型实验产物；
- `.pyc`、cache、模型权重、谱基、完整日志、纹理、rollout 和视频不进入 Git。

## 分层项目记忆

| 层级 | 文件 | 内容 |
|---|---|---|
| 长期规则 | `AGENTS.md` | 代码规范、科学不变量、环境和 Git 边界 |
| 当前状态 | `docs/status/openvla-spectral-current.md` | 当前结论、里程碑、唯一下一门槛和禁止捷径 |
| 实验账本 | `docs/experiments/openvla-spectral-ledger.md` | 决策实验、结果、verdict 和 provenance 索引 |
| 专题档案 | `docs/spectral/`、`docs/refactor/` | 完整命令、推理、逐状态结果和历史细节 |
| 临时产物 | `experiments_inbox/` | rsync 回 WSL 的 Git 忽略 result bundle |

新任务只读取长期规则、当前状态和任务直接相关的专题文档。除非要追溯结论，不应
每轮重新加载全部历史。

## 一轮开发的标准流程

### 1. 变更契约

动手前先明确：

```text
problem:
evidence:
single_primary_change:
non_goals:
acceptance_checks:
server_gate_if_needed:
```

若关键语义仍不确定，先与用户讨论；不得用实现细节替代尚未作出的研究决策。

### 2. 小纵切实现

- 一轮只移动或新增一个清晰职责；
- 优先先建立/迁移测试，再改生产代码；
- 保持模型专用 forward 分离，只复用已确认模型无关的基础设施；
- 保护用户已有修改和实验产物；
- 候选未过门槛时停止，不在失败候选上继续堆叠多个变量。

### 3. 本地验证

按风险依次执行可用的：

1. 精确的相关单元测试；
2. 完整无 GPU 测试；
3. `git diff --check`；
4. `git status --short` 和人工 diff 审查。

本地环境不能运行测试时，交付中必须明确写“未运行”及环境原因，不能把历史通过
记录表述为当前验证结果。

### 4. 开发交付

每次交付固定回答：

1. 解决了什么问题；
2. 根因或设计依据；
3. 新数据流如何工作；
4. 修改文件和 interface；
5. 保持不变的行为；
6. 已运行与未运行的验证；
7. 风险和未知项；
8. 当前里程碑和下一门槛；
9. commit SHA。

复杂变更应给出最小调用链或数据流，方便用户从整体进入代码，而不是只给 diff。

## 服务器实验交接

Codex 提供的实验交接至少包含：

- 要拉取的 commit SHA；
- 完整命令和运行目录；
- GPU/环境前置条件；
- 输入 artifact 路径和期望 hash；
- 输出文件清单；
- 有效性检查；
- 预先约定的通过/失败门槛；
- 失败后应保留的 stdout/stderr 和资产恢复检查。

未经用户回传结果，不把“代码已实现”表述为“真实 GPU 方法已通过”。

## Result bundle 与 rsync

服务器实验不通过 Git 传输原始产物。用户使用 rsync 将精简 bundle 放入：

```text
experiments_inbox/<run-id>/
```

仓库已忽略 `/experiments/` 与 `/experiments_inbox/`。推荐 bundle 包含：

- `run_manifest.json`：commit、命令、配置、checkpoint、states、seed；
- stdout/stderr；
- JSON/CSV 汇总和 gradient/loss 日志；
- 输入/输出 artifact SHA-256；
- 必要的 clean/adv PNG；
- 只有需要视觉审计时才同步关键视频或抽帧。

模型权重、完整 rollout 集、大型谱基和可重建中间 tensor 默认留在服务器。Codex
读取 bundle 后先检查 provenance 和有效性，再更新实验账本、当前状态和专题文档。

更新顺序固定为：先在专题文档保存完整证据，再在实验账本登记 result/verdict；
只有实验改变当前门槛、里程碑或禁止项时，才更新当前状态页。历史条目不覆盖，
需要纠正时新增说明并链接原证据。

## 实验解释规则

- 一次实验只检验预先声明的主要假设；
- 成功门槛在结果出来前固定；
- 同时检查 clean control、输入生效、数值有限、预算、bake 和资产恢复；
- proxy loss、feature distance、action response 和 rollout success 分层解释；
- 未过上游门槛的候选不得进入下游模型；
- 失败实验同样登记，并明确它排除了什么、没有排除什么；
- 发现基础语义错误时保留历史记录，但重新建立科学基线。

## 代码讲解与上下文切换

- 代码讲解应锚定 commit SHA；可在独立对话中进行，不依赖原开发聊天历史；
- 先讲模块地图和数据流，再讲关键 interface、算法和测试；
- 自然里程碑结束或上下文将压缩时，生成 handoff，至少记录当前目标、commit、
  已确认事实、已排除假设、未解决问题、下一条命令和相关文件；
- 新对话从 `AGENTS.md`、当前状态、handoff 和目标文件恢复，不从头重读全部实验史。

## 周期性审计

完成若干纵切或发现结果持续退化时，暂停新增机制并检查：

- last-known-good commit 和基线能否恢复；
- 是否同时改变了多个实验变量；
- 默认配置是否意外漂移；
- 测试是否只验证内部自洽而未对照真实 processor/policy；
- 实验 artifacts 是否来自当前 commit；
- 当前状态页是否仍准确；
- 失败候选是否被错误地继续当作新基线。
