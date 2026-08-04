# Tex3D 文档导航

文档按“当前事实优先、历史细节按需读取”组织。不要通过长篇专题文档末尾的旧
“下一步”推测当前任务。

## 开始开发前

1. 阅读仓库长期规则：[`AGENTS.md`](../AGENTS.md)；
2. 阅读 OpenVLA 谱方法当前状态：
   [`openvla-spectral-current.md`](status/openvla-spectral-current.md)；
3. 只打开本轮任务直接相关的专题文档和代码；
4. 需要追溯实验结论时查阅：
   [`openvla-spectral-ledger.md`](experiments/openvla-spectral-ledger.md)。

## 核心文档

| 文档 | 用途 |
|---|---|
| [`CONTEXT.md`](../CONTEXT.md) | Clean Asset、Active Texture、Attack Artifact、Training Frame、Surface Delta 等领域语言 |
| [`tex3d.md`](../tex3d.md) | 论文概念；与代码冲突时以当前代码行为为准 |
| [`collaboration-workflow.md`](development/collaboration-workflow.md) | WSL/服务器分工、Git、rsync、开发交付和上下文恢复协议 |
| [`openvla-stage-1-architecture.md`](refactor/openvla-stage-1-architecture.md) | OpenVLA 重构后的职责、数据流和行为约束 |
| [`openvla-stage-1-walkthrough.md`](refactor/openvla-stage-1-walkthrough.md) | 第一阶段逐模块历史讲解；部分评估描述早于 renderer 对齐修正 |

## 谱方法专题档案

- [`openvla-spectral-phase-1-results.md`](spectral/openvla-spectral-phase-1-results.md)：
  第一阶段统一结论；
- [`openvla-spectral-mvp.md`](spectral/openvla-spectral-mvp.md)：
  参数化、公平比较和初始迁移链路；
- [`openvla-spectral-gradient-audit.md`](spectral/openvla-spectral-gradient-audit.md)：
  source-only 选基指标与连续 K 对照；
- [`oft-transfer-response-diagnostic.md`](spectral/oft-transfer-response-diagnostic.md)：
  OFT clean/adv fixed-state Feature/Action 响应；
- [`cross-model-pixel-gradient-audit.md`](spectral/cross-model-pixel-gradient-audit.md)：
  跨模型像素梯度与 renderer VJP；
- [`openvla-dual-view-siglip.md`](spectral/openvla-dual-view-siglip.md)：
  双视角、动态范数保护、action-response、预处理根因和 BPDA 完整时间线。

## 维护原则

- `AGENTS.md` 只保存长期规则、当前门槛摘要和不可破坏的不变量；
- `status/` 只保存当前仍生效的结论和下一步；
- `experiments/` 保存决策实验索引与 verdict，不复制大型原始产物；
- `spectral/`、`refactor/` 保存完整历史证据；
- 服务器 result bundle 通过 rsync 进入被 Git 忽略的
  `experiments/result_inbox/<run-id>/`，经审计后再更新正式文档。
