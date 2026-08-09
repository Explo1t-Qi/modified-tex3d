# Issue tracker: GitHub

本仓库的 issue 和 PRD 使用 GitHub Issues 管理。相关技能通过 `gh` CLI 操作，
仓库身份由当前 clone 的 `git remote` 推断；未经用户明确请求，不创建、修改或
关闭 GitHub issue。

## Conventions

- 创建：`gh issue create --title "..." --body "..."`
- 阅读：`gh issue view <number> --comments`
- 列表：`gh issue list --state open --json number,title,body,labels,comments`
- 评论：`gh issue comment <number> --body "..."`
- 标签：`gh issue edit <number> --add-label "..."` 或 `--remove-label "..."`
- 关闭：`gh issue close <number> --comment "..."`

技能要求“publish to the issue tracker”时表示创建 GitHub issue；要求读取 ticket
时使用 `gh issue view <number> --comments` 并同时检查 labels。
