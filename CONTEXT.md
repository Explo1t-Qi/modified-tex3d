# Tex3D VLA 鲁棒性评估

本上下文描述 Tex3D 在 LIBERO 中生成、激活和恢复对抗纹理时使用的领域术语，
用于让不同 VLA 模型的评估实现保持一致含义。

## Language

**Clean Asset**:
一次评估开始时的目标物体 MuJoCo XML 与真实纹理，是 task 间和退出时必须恢复的基线资源。
_Avoid_: original file, backup asset

**Active Texture**:
当前由目标物体 XML 引用、并可能同步覆盖真实纹理文件的 clean 或 adversarial texture。
_Avoid_: current image, injected file

**Attack Artifact**:
攻击运行写入日志目录的可复查输出，包括噪声参数、UV texture、loss、日志和视频。
_Avoid_: runtime asset, model asset

**Runtime Asset Transaction**:
一次评估运行内对 **Clean Asset** 进行临时修改并保证最终恢复的生命周期。
_Avoid_: file helper, XML utility

## Relationships

- 一个 **Runtime Asset Transaction** 保存一组 **Clean Asset**
- 一个 **Runtime Asset Transaction** 在任一时刻激活至多一个 **Active Texture**
- 一个 **Attack Artifact** 可以被选为 **Active Texture**，但不会因此成为 **Clean Asset**
- 每个 task 开始前和评估退出时，**Active Texture** 都恢复为 **Clean Asset**

## Example dialogue

> **Dev:** “训练生成的 UV PNG 是 **Attack Artifact**，什么时候会成为
> **Active Texture**？”
>
> **Domain expert:** “进入 rollout 前由 **Runtime Asset Transaction** 激活；
> 下一个 task 开始前再恢复对应的 **Clean Asset**。”

## Flagged ambiguities

- “texture” 曾同时表示源纹理、当前注入纹理和实验输出；现在分别使用
  **Clean Asset**、**Active Texture** 和 **Attack Artifact**。
