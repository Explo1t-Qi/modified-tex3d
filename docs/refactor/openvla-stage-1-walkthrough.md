# OpenVLA 第一阶段重构代码讲解

## 讲解进度：第 9 / 9 阶段

- ✅ 1. 实验入口与整体编排
- ✅ 2. RuntimeAssetTransaction
- ✅ 3. 资产注册表与场景定位
- ✅ 4. 模型、Processor 与 Action Codec
- ✅ 5. Renderer 与图像合成
- ✅ 6. 训练帧采集
- ✅ 7. 攻击目标与优化器
- ✅ 8. AttackTrainer 与产物管理
- ✅ 9. EpisodeRunner、完整数据流与扩展接口复盘

第一轮逐文件讲解完成，整体进度 `100%`。

核心文件是
[evaluation.py](../../openvla/experiments/robot/libero/openvla_attack/evaluation.py)。

## 一、EpisodeRunner 的职责

[LiberoEpisodeRunner](../../openvla/experiments/robot/libero/openvla_attack/evaluation.py)
管理一个 episode：

```text
创建环境
→ 设置 initial state
→ 执行等待动作
→ 获取相机图像
→ 可选地合成对抗图像
→ 调用 OpenVLA
→ 动作后处理
→ env.step
→ 判断成功
→ 关闭环境
```

它不负责：

- 遍历 task；
- 遍历多个 initial states；
- 统计平均成功率；
- 保存正式 MP4；
- 修改 XML 或纹理文件。

这些属于入口和其他模块。

## 二、环境创建

`run()` 首先调用：

```python
env, task_description = get_libero_env(
    task,
    model_family,
    resolution=video_resolution,
)
```

底层
[get_libero_env()](../../openvla/experiments/robot/libero/libero_utils.py)
根据 task 的 BDDL 文件创建：

```python
OffScreenRenderEnv(
    bddl_file_name=...,
    camera_heights=resolution,
    camera_widths=resolution,
)
```

然后固定：

```python
env.seed(0)
```

LIBERO 注释特别强调，即使使用固定 initial state，环境 seed 仍可能影响物体位置或其他模拟细节。

正式评估使用：

```text
video_resolution = 512
max_steps = 300
```

live-test 默认使用：

```text
video_resolution = 256
max_steps = 300
```

## 三、设置初始状态

每个 episode 执行：

```python
env.reset()
observation = env.set_init_state(initial_state)
env.env.sim.forward()
```

`sim.forward()` 让 MuJoCo 重新计算：

- body 世界位姿；
- camera matrix；
- geom 位置；
- 其他派生状态。

这一顺序和 TrainingFrameCollector 相同。

## 四、等待阶段

主循环条件为：

```python
step_index < max_steps + num_steps_wait
```

前 `num_steps_wait` 步执行：

```python
[0, 0, 0, 0, 0, 0, -1]
```

即 dummy/no-op action。

等待阶段：

- 不调用 OpenVLA；
- 不生成策略图像；
- 不保存录像帧；
- 只让模拟器和物体稳定。

等待结束后，策略仍然最多可以运行 `max_steps` 次。

## 五、相机图像的预处理

[get_libero_image()](../../openvla/experiments/robot/libero/libero_utils.py) 从：

```python
observation["agentview_image"]
```

取得相机图像。

然后执行：

```python
image = image[::-1, ::-1]
```

同时翻转水平和垂直方向，即旋转 180°，以匹配 OpenVLA 训练数据的图像方向。

接着执行一个比较特殊的 resize 流程：

```text
NumPy RGB
→ JPEG encode
→ JPEG decode
→ Lanczos resize
→ round + clip
→ uint8
```

JPEG encode/decode 是为了模仿 RLDS 数据构建过程，而不只是普通缩放。

输出：

```text
camera_image
uint8 HWC [video_resolution, video_resolution, 3]
```

## 六、camera_image 与 policy_image

[_build_policy_image()](../../openvla/experiments/robot/libero/openvla_attack/evaluation.py)
返回两张图：

```python
camera_image, policy_image
```

它们的语义不同。

### camera_image

这是 MuJoCo 实际渲染的原始相机帧，用于正式录像。

启用攻击并覆盖真实纹理后，它通常已经包含 MuJoCo 自己渲染的攻击纹理，但没有经过可微 renderer 合成。

### policy_image

这是 OpenVLA 实际接收的图像。

启用 renderer 时：

```text
MuJoCo camera image
→ 找到目标 body
→ 临时隐藏目标物体
→ 得到无目标背景
→ nvdiffrast 渲染对抗前景
→ 合成
→ resize 到 224×224
→ 交给 OpenVLA
```

所以二者可能不同：

```text
录像看到：MuJoCo 渲染的攻击物体
策略看到：可微 renderer 合成的攻击物体
```

这是当前实验设计的重要语义。

因此不能只看正式 MP4 就断言“模型看到的合成图像完全正确”。正式 MP4 更接近物理环境的真实视觉结果，而不是模型输入的逐像素记录。

## 七、没有 Renderer 时

如果：

```python
self._renderer is None
```

说明攻击没有启用。

此时：

```text
camera_image
→ resize 到 224×224
→ policy_image
```

不进行目标定位、背景移除或前景合成。

## 八、启用 Renderer 时

首先把相机图像转换成：

```text
regular_background
float32 NCHW [1,3,video_resolution,video_resolution]
```

然后：

```python
target_pose = find_target_body_pose(...)
mvp = compute_render_mvp(...)
background_without_target = render_background_without_target(...)
```

优先使用无目标背景。

随后在：

```python
with torch.no_grad():
```

中调用：

```python
render_and_composite(...)
```

正式评估不需要梯度。

输出：

```text
float NCHW [1,3,H,W]
```

再转换为：

```text
uint8 HWC
```

并 resize 到 OpenVLA 输入尺寸。

如果找不到目标 body：

```text
mvp = None
```

`render_and_composite()` 直接返回普通背景，不会在错误位置生成对抗物体。

如果正式评估前已经把攻击纹理覆盖到 MuJoCo 原始纹理，那么该普通背景仍可能包含 MuJoCo 自身渲染的攻击物体。

## 九、构造机器人状态

每个策略步还会构造：

```python
robot_state = concatenate(
    eef_position,       # [3]
    eef_axis_angle,     # [3]
    gripper_qpos,       # [2]
)
```

结果：

```text
float array [8]
```

四元数通过：

```python
quat2axisangle()
```

转换为 axis-angle。

最终策略 observation：

```python
{
    "full_image": policy_image,
    "state": robot_state,
}
```

当前 OpenVLA 只读取 `full_image`，不使用 `state`。保留 `state` 是为了兼容统一机器人 observation 结构，以及未来需要 proprioception 的模型。

## 十、调用策略与动作后处理

调用：

```python
action = get_action(
    cfg,
    model,
    policy_observation,
    task_description,
    processor=processor,
)
```

OpenVLA 返回：

```text
NumPy [7]
```

然后：

```python
normalize_gripper_action(action, binarize=True)
```

执行：

```text
gripper [0,1]
→ [-1,1]
→ sign
→ {-1,+1}
```

再执行：

```python
invert_gripper_action(action)
```

适配 OpenVLA/RLDS 与 LIBERO 不同的开合符号定义。

最终：

```python
observation, _, success, _ = env.step(action.tolist())
```

如果成功立即结束。

## 十一、录像帧对应什么时刻

每个策略步的顺序是：

```text
获取当前 observation
→ 构造 camera/policy image
→ 把 camera_image 加入 replay_images
→ 预测 action
→ env.step(action)
```

所以每个录像帧表示：

```text
执行当前 action 之前的状态
```

等待阶段不记录录像。

一个 episode 最多保存约 300 个策略帧，成功时提前结束。

## 十二、异常处理

每个策略步内部都有：

```python
try:
    ...
except Exception:
    print traceback
    break
```

异常会打印：

```text
Task ID
Episode index
Step index
完整 traceback
```

然后将该 episode 作为失败返回，而不是中止整个 task。

这正是之前 attention mask 报错时仍看到：

```text
Task: 0 | Ep: 0 | Success: False
```

的原因。

优点是单个 episode 出错不会破坏整个批量评估。

风险是系统性错误可能表现为“大量失败 episode”，所以查看成功率时必须同时检查 stderr/traceback，不能只看最终数字。

环境始终在：

```python
finally:
    env.close()
```

中关闭。

## 十三、RolloutResult

Runner 返回：

```python
@dataclass(frozen=True)
class RolloutResult:
    success: bool
    task_description: str
    replay_images: list[np.ndarray]
```

入口只需要：

```text
success         → 统计
task_description→ MP4 文件名
replay_images   → 保存视频
```

当前没有保存：

- 实际步数；
- action trajectory；
- error 信息；
- 策略实际看到的 composited frames；
- 机器人状态轨迹。

后续若要做更深入的失败分析或 TAAO，可以扩展为结构化 trajectory result。

## 十四、正式评估统计

入口对每个 initial state 调用 Runner：

```python
if rollout_result.success:
    total_successes += 1

total_episodes += 1
```

最终：

```python
average_success_rate = (
    total_successes / total_episodes
)
```

这里统计的是：

```text
攻击条件下的机器人任务成功率
```

不是“攻击成功率”。

当前控制台打印：

```text
Attack success rate
```

这个名称容易误解。比如 Spatial 的 50% 实际表示：

```text
机器人成功 25 次
机器人失败 25 次
策略任务成功率 = 50%
```

不能直接说“攻击成功率为 50%”，除非将攻击成功定义为机器人失败并重新计算：

```text
attack-induced failure rate = 1 - policy success rate
```

而且严格的攻击成功还应与 clean baseline 在同一 initial state 上对照，不能简单把所有失败都归因于攻击。

## 十五、完整实验端到端复盘

现在可以把完整数据流串起来：

```text
1. 命令行 GenerateConfig
          │
          ▼
2. OBJECT_ASSETS
   解析 suite/task/XML/mesh/texture/search
          │
          ▼
3. RuntimeAssetTransaction.begin
   备份干净 XML 与真实纹理
          │
          ▼
4. 加载 OpenVLA + Processor
   加载 dataset action statistics
          │
          ▼
5. 创建 DifferentiableRenderer
   mesh/UV/原始顶点颜色/adv_noise
          │
          ▼
6. TrainingFrameCollector
   前 10 个 initial states，各采一帧
   ├─ 背景与无目标背景
   ├─ MVP/rotation
   ├─ clean action token
   └─ clean hidden
          │
          ▼
7. AttackOptimizer，5000 次
   每轮：
   ├─ 选择全部 10 帧
   ├─ 每帧一个对抗视图
   ├─ OpenVLA differentiable forward
   ├─ action target CE
   ├─ negative hidden MSE
   └─ SignSGD 更新 adv_noise
          │
          ▼
8. AttackArtifactStore
   ├─ gradient log
   ├─ Vertex_Noise.pt
   ├─ UV_Map.png
   └─ loss_history.npy
          │
          ▼
9. Bake 最终纹理
   task_0_adv_texture_*.png
          │
          ▼
10. RuntimeAssetTransaction.activate
    修改 XML + 镜像真实纹理
          │
          ▼
11. LiberoEpisodeRunner × 50
    ├─ 等待 10 步
    ├─ 构造对抗 policy image
    ├─ OpenVLA action
    ├─ gripper 后处理
    └─ env.step / success
          │
          ▼
12. 评估日志 + 50 个正式 MP4
          │
          ▼
13. finally
    恢复 XML/真实纹理
    删除 backup
    关闭日志/W&B
```

## 十六、后续 TAAO 接入路线

真正实现 TAAO，需要修改三层。

### 1. TrainingFrame schema

增加：

```text
trajectory_id
initial_state_id
step_index
timestamp 或相邻关系
```

目前返回普通 frame list 后，轨迹边界已经丢失。

### 2. Collector

需要策略驱动采集连续 trajectory，而不只是每个 initial state 的第一帧。

这里还要先修正和测试：

- grasp window 的 pre/post 语义；
- policy-driven 采集配置约束；
- trajectory 内动作和状态记录。

### 3. FrameBatchSampler

实现新的 adapter：

```text
连续 hidden
→ latent velocity
→ latent acceleration
→ criticality score
→ temperature/softmax weight
→ WeightedTrainingFrame
```

同时修正上一阶段发现的 Total Loss 日志额外缩放问题，使非均匀权重的日志语义明确。

## 十七、后续 EoT 接入路线

EoT 主要替换：

```python
ViewSampler
```

输入仍然是一个有效 `TrainingFrame`，输出多个可微视图：

```text
MVP perturbation
光照扰动
颜色/曝光扰动
背景扰动
尺度或相机扰动
```

优化器已经会自动对多个视图取平均 loss。

需要额外明确：

- 每种扰动的采样分布；
- 是否同时扰动 clean baseline；
- 相机扰动后无目标背景如何对应；
- 随机种子和复现实验方式。

## 十八、迁移到 OpenVLA-OFT

可以复用：

- `assets.py`
- `scene.py`
- `renderer.py`
- `compositing.py`
- `runtime_assets.py`
- `artifacts.py`
- task/episode 生命周期结构

不能直接假设复用：

- action token 区间；
- `255-i` 目标；
- `generate(max_new_tokens=7)`；
- processor 的 6 通道格式；
- action decode；
- hidden-state 对齐方式。

应为 OpenVLA-OFT 单独建立：

```text
ModelAdapter
ActionAdapter
AttackObjective
```

再接入共同的视觉攻击基础设施。

## 十九、迁移到 π0

π0 很可能需要额外处理：

- proprioception；
- action chunk；
- 多步动作执行；
- 不同图像预处理；
- 可能不同的运行时和模型框架；
- temporal state 或 action queue。

当前 EpisodeRunner 每次只请求一个 `[7]` 动作，因此 π0 不能简单替换 `get_action()`。

更适合的接口可能是：

```python
PolicyAdapter.reset_episode()
PolicyAdapter.predict(observation) -> ActionChunk
PolicyAdapter.next_action() -> EnvironmentAction
```

然后为 π0 编写独立 runner 或 action-chunk adapter。

## 二十、第一版之后建议优先处理的工程事项

在实现论文新算法前，建议按优先级处理：

1. 增加 `run_manifest.json`，记录配置、checkpoint、git commit 和环境。
2. 修正 Total Loss 日志的额外缩放，并明确历史兼容方式。
3. 校验 `object_name` 与 suite/task ID 的映射。
4. 统一 `MeshScale` 的 tuple/list 类型。
5. 为 policy/grasp frame collection 增加测试并修正窗口语义。
6. 为同一物体的并行运行增加文件锁或独立 asset workspace。
7. 区分 `policy_success_rate` 与真正的 attack success。
8. 考虑保存策略实际看到的 composited video。
9. 将正式 MP4 纳入 ArtifactStore 或 run manifest。

至此，我们已经沿真实数据流完整走过 OpenVLA 第一版重构。下一步可以先检查正在运行的 `libero-object task 0` 完整实验；如果数值和产物正常，就可以选择进入“收尾工程改进”，或者开始设计 TAAO 的 trajectory 数据结构和 sampler 接口。
