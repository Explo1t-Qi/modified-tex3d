# Domain docs

本仓库是 single-context 布局。工程技能在探索或修改代码前应读取根目录
`CONTEXT.md`，并使用其中定义的领域术语；不得改用 glossary 明确列出的
`Avoid` 同义词。

若存在 `docs/adr/`，还应读取与当前改动相关的 ADR。文件或目录不存在时静默
继续，不要为了满足形式提前创建 ADR。若实现计划与现有 ADR 冲突，必须明确
指出冲突并请求重新决策，不能静默覆盖。

当前布局：

```text
/
├── CONTEXT.md
├── AGENTS.md
└── docs/
    ├── agents/
    └── adr/        # 按需创建
```
