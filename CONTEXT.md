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

**Training Frame**:
从一条 LIBERO trajectory 采集的场景姿态、背景、clean action token、hidden state 和视觉归一化参数。
_Avoid_: image, observation

**Attack Training**:
在一个 task 上采集 **Training Frame**、优化对抗纹理并生成 **Attack Artifact** 的完整过程。
_Avoid_: optimizer loop, attack step

**Runtime Asset Transaction**:
一次评估运行内对 **Clean Asset** 进行临时修改并保证最终恢复的生命周期。
_Avoid_: file helper, XML utility

**Surface Delta**:
定义在 OBJ 几何顶点上的 RGB 颜色增量；它与 UV seam 复制出的渲染顶点分离，
并由统一的 L∞ 预算约束。
_Avoid_: vertex noise, texture noise

**Texture Parameterization**:
把可学习参数映射为 **Surface Delta** 的方式。当前包括逐几何顶点的
Geometry Vertex adapter、低维谱系数的 Spectral adapter，以及只消费外部冻结
顶点索引、使用紧凑 `[|S|,3]` 参数的 Fixed-Support adapter。Fixed-Support
机制代码存在不表示生产 **Fixed Vertex Support** 已经生成。
_Avoid_: optimizer, texture format

**Spectral Naturalness Regularization**:
施加在最终 **Surface Delta** 上的软约束。它把高频谱能量占总能量的
比例作为自然性证据，但只惩罚超过预先冻结上限的部分；低于上限时
不干预攻击。它不把扰动硬限制在前 K 个谱基张成的子空间内，也不是
一种独立的 **Texture Parameterization**。第一版用“support 内均匀、外部为零”
的几何探针校准上限，以容忍局部 support 形状和边界本身不可避免的
谱泄漏。
_Avoid_: Spectral adapter, spectral component, hard spectral projection

**Spectral Naturalness Band**:
由 OBJ 几何唯一决定、从常数模态开始的连续低频谱子空间，用于
判定 **Surface Delta** 的低频能量。它描述自然性带宽，不按 source Action
梯度重排，也不把固定模态数解释为所有 mesh 共享的物理频率。
_Avoid_: action-selected naturalness basis, universal physical K

**Untargeted Clean-Action Margin**:
在固定 clean action sequence 的 teacher-forced prefix 下，source OpenVLA 分配给
clean action token 的 logit 与最强其他 action token logit 之差。正值表示
clean token 仍占优，非正值表示最近的 untargeted 决策边界已被跨过。
_Avoid_: symmetric action target, adversarial autoregressive prefix,
continuous-action distance

**Dense Seed Gradient**:
source OpenVLA 在零 **Surface Delta**、全顶点 Geometry Vertex 参数化和固定
clean action prefix 下，由 Untargeted Clean-Action Margin hinge 得到的完整
`G_s [N_v,3]`。它只来自 Primary action objective；Feature、wrist 与 OFT
不得进入该梯度。
_Avoid_: legacy optimizer gradient, feature seed gradient, target-model gradient

**Support Seed Score**:
由多 state **Dense Seed Gradient** 逐 state 做全局 Surface-L∞ 方向归一化后，
将逐顶点平均 RGB sensitivity 与跨 state RGB direction consistency 相乘得到的
`q_i`。它只为连通区域的种子与扩张提供排序证据，不能直接 top-N 或等同于
**Fixed Vertex Support**。
_Avoid_: support mask, attack loss, vertex saliency top-N

**Smoothed Seed Density**:
先用 barycentric lumped vertex mass 将 `q_i` 转成单位曲面面积 density
`d_i=q_i/m_i`，再按 `(M+tau L)d_tilde=Md` 得到的 mass-aware implicit
Laplacian 平滑结果。原始与平滑 density 都是可复算证据，不修改 OBJ 几何、
Surface Delta 或纹理。
_Avoid_: mesh smoothing, texture smoothing, selected support

**Spectral Guard Calibration**:
正式 **Attack Training** 前的一次性 source-only 校准过程，用同一实际可训练
参数空间中的 Action/谱梯度相对强度赋予“弱谱护栏”可复查含义。
它在正式训练前冻结谱权重并完全恢复初始状态，不是正式训练中的动态
梯度裁剪或基于 rollout 的调参。
_Avoid_: online spectral clipping, rollout-tuned weight, calibration warm-up

**Deployment Effective View Transform**:
source policy 在 checkpoint processor 之前对相机 RGB 应用的部署期空间变换，
它定义模型真正能够观察到的视野。Support 审计、攻击训练、coverage 与 rollout
必须共享同一变换语义；coverage 的连续可见权重只复用其空间采样几何，不继承
RGB 量化或视觉分支归一化。
_Avoid_: raw-camera view, training-only crop, module-local crop approximation

**Policy Pre-Crop Canvas**:
提供给 **Deployment Effective View Transform** 的唯一规范 RGB 画布。它与录像
帧具有独立职责和固定构造语义；录像分辨率或编码方式变化不能隐式改变 policy
看到的输入。攻击训练、审计、coverage 与 rollout 必须从同一画布定义出发。
_Avoid_: replay frame as policy contract, direct-render shortcut, training canvas

**Fixed Vertex Support**:
在 **Attack Training** 开始前确定、并在整次训练中保持不变的一组 OBJ 几何顶点
索引。新的局部逐顶点参数化只允许该集合中的顶点产生独立 RGB
**Surface Delta**；集合外顶点保持零增量。
_Avoid_: dynamic vertex selection, per-iteration mask, renderer vertex mask

**Connected Support Region**:
由 OBJ 几何顶点及其 face 邻接关系定义的连通曲面区域。一个 **Fixed Vertex
Support** 可以由少量固定的 Connected Support Region 组成，以兼顾局部空间
连续性和跨视角覆盖；区域连通性不依据 UV atlas 邻接或渲染顶点距离猜测。
_Avoid_: isolated top-N vertices, UV island, single mandatory patch

**Support Area Budget**:
**Fixed Vertex Support** 所覆盖的 barycentric lumped vertex mass 占整个 OBJ
曲面 mass 的比例。它是局部参数化规模的主要预算；顶点数量只作为诊断记录，
不作为跨 mesh 的物理覆盖语义。第一版将预算作为固定目标，不能因可见性不足而
在选择过程中自动扩大。相同面积比例可能因局部网格密度不同而对应不同顶点数，
因此面积预算不蕴含固定的可训练参数缩减比例。
_Avoid_: top-N budget, render-vertex count, UV pixel area

**Support Visibility Coverage**:
在一个 `(state, view)` 的策略有效视野中，**Fixed Vertex Support** 占目标
物体实际可见投影的比例。对可见像素 `p` 及其所属三角形的重心坐标
`beta_pj`，定义 `w_S(p) = sum_j beta_pj * 1[v_pj in S]`；再以目标表面
可见权重 `alpha_p` 计算
`C_{s,v}(S) = sum_p alpha_p * w_S(p) / (epsilon + sum_p alpha_p)`。
`alpha_p` 在完全可见内部接近 1，抗锯齿边缘介于 0 与 1，背景或被其他
场景几何遮挡时为 0。Primary 必须复用 source OpenVLA 部署时的精确空间
处理，包括 center crop；不得在原始 MuJoCo 相机画面上直接统计。区域选择
在训练前按该覆盖量贪心补充 **Connected Support Region**，达到预设门槛
或区域数上限后停止并冻结。该 coverage 表示重心插值下的加权控制比例，
不表示相同比例的像素必须完全由选中顶点组成。
_Avoid_: raw-camera coverage, binary pixel count, whole-triangle visibility,
target-model coverage

**Visibility Evidence Status**:
每个 `(state, view)` 的 coverage 证据状态，按固定优先级判定为
`not_observable`、`insufficient_observation`、`invalid_alignment` 或
`valid`。先在有效输入上计算可见目标质量
`A_obs = sum_p alpha_p / (H * W)`，并记录 `sum_p alpha_p` 的等效像素数。
`not_observable` 表示 `A_obs = 0`；`insufficient_observation` 表示
`0 < A_obs < A_obs_min`，其 coverage 只作诊断，不进入正式汇总或硬 Gate。
只有 `A_obs >= A_obs_min` 时才判定对齐：recall 低于冻结门槛则为
`invalid_alignment`，否则为 `valid`。Primary 中的 `invalid_alignment` 使整个
support construction 验收失败；只读 wrist 诊断中的同一状态只使该诊断项
无效，不得反向否决 Primary support。`A_obs_min` 只判断观测分母是否足以形成
可靠 coverage 证据，不是 **Support Visibility Coverage** 的通过门槛。
_Avoid_: zero coverage for missing evidence, silently skipped alignment failure

**Instance Visibility Evidence**:
同一静止 MuJoCo state 中，一个共享 **Active Texture** 实例在指定 policy camera
前景里实际 front-most 的可见区域。它由 segmentation object 显式解析到 geom、
body 和 instance subtree，遮挡由完整场景决定；不能用 renderer 自身的投影
mask 或裸整数 ID 猜测实例归属。
_Avoid_: renderer visibility, assumed geom ID, unoccluded projection

**Renderer Delta Composition**:
在真实 MuJoCo clean RGB 上，仅叠加可见实例由 **Surface Delta** 引起的可微
renderer RGB 变化量。它是训练与 source 审计使用的局部梯度 surrogate；零
**Surface Delta** 必须严格恢复真实 clean policy 输入，正式 rollout 仍由 bake
后的 **Active Texture** 通过 MuJoCo 成像。
_Avoid_: foreground replacement, instance-order overwrite, rollout compositor

**Renderer-to-Bake Response**:
对同一个确定性、与 Action 无关的小幅 **Surface Delta**，比较 **Renderer Delta
Composition** 预测的有效视野 RGB 变化与 bake **Active Texture** 后 MuJoCo
实际产生的 RGB 变化。它验证训练 surrogate 与部署资产响应是否同向，不评价
攻击强弱，也不参与 support 排名。
_Avoid_: action-success gate, target-model probe, rollout replacement

**Terminal Deployment Response Audit**:
对同一个训练完成且哈希绑定的 Fixed-Support 终态，以 Clean、训练期
**Renderer Delta Composition** 和真实 MuJoCo **Active Texture** 构成静态
`C/A/B` 三元组，并比较 source OpenVLA 实际接收的有效视野、processor tensor、
clean-prefix teacher训练代理、默认cached自回归generation的首次分歧和解码动作。
部署行为分类与tie只以真实generation score为权威；teacher/generation关系只作
结构化诊断。它只判断训练路径产生的终态离散响应是否穿过部署链路，不重新训练、
不评价闭环恢复，也不使用source development rollout选择超参数。
_Avoid_: terminal attack gate, rollout recovery proof, bake-only replay

**Support Seed Score**:
由修正后 source OpenVLA 训练 states 的 Action objective 几何顶点梯度生成、
并在 OBJ mesh 图上平滑的标量场。它只决定 **Connected Support Region** 的
种子与扩张优先级，不能绕过连通性和 **Support Area Budget** 直接生成离散
top-N mask。Feature objective 梯度不参与该分数；第一版固定在零
**Surface Delta** 上计算，不读取旧攻击纹理或在训练中重新计算。逐 state 梯度
先按 Surface L∞ 方向归一化，再以跨 state 平均敏感度乘 RGB 方向一致性聚合；
该稳定分数除以 barycentric vertex mass 后形成单位面积 seed density。
_Avoid_: target-model saliency, final vertex mask, unnormalised single-state gradient

**Smoothed Seed Density**:
对单位面积 **Support Seed Score** 使用 OBJ cotangent Laplacian 与 barycentric
mass 执行隐式曲面平滑后得到的证据场。平滑只改变区域选择证据，不修改 mesh
几何、**Surface Delta** 或 **Support Area Budget**。其 smoothing length 是扩散
算子的特征尺度，不表示具有紧支撑的影响范围或严格测地半径。
_Avoid_: smoothed mesh, UV blur, texture regularization

## Relationships

- 一个 **Runtime Asset Transaction** 保存一组 **Clean Asset**
- 一个 **Runtime Asset Transaction** 在任一时刻激活至多一个 **Active Texture**
- 一个 **Attack Artifact** 可以被选为 **Active Texture**，但不会因此成为 **Clean Asset**
- 一次 **Attack Training** 消费多个 **Training Frame** 并生成一组 **Attack Artifact**
- 每个 task 开始前和评估退出时，**Active Texture** 都恢复为 **Clean Asset**
- 一种 **Texture Parameterization** 生成一个 **Surface Delta**
- Geometry Vertex 与 Spectral **Texture Parameterization** 使用相同的
  **Surface Delta** 预算与更新步长语义
- **Spectral Naturalness Regularization** 评估并约束 **Surface Delta**，但不
  改变生成该增量的 **Texture Parameterization** 所具有的优化自由度
- **Spectral Naturalness Regularization** 使用 **Spectral Naturalness Band** 区分
  低频与高频能量；该频带不得读取 **Support Seed Score** 或攻击梯度
- **Deployment Effective View Transform** 位于相机图像与 checkpoint processor
  之间；二者共同定义 source OpenVLA 的完整部署输入路径
- **Policy Pre-Crop Canvas** 是 **Deployment Effective View Transform** 的输入；
  录像图像可以与其共享底层 observation，但不能定义其分辨率或重采样语义
- **Fixed Vertex Support** 属于 **Texture Parameterization** 的固定定义，不是
  optimizer 在训练过程中更新的状态
- 一个 **Fixed Vertex Support** 可以是少量 **Connected Support Region** 的
  并集，但不能退化为没有曲面连通约束的任意离散 top-N
- **Connected Support Region** 从种子开始，在 OBJ face-adjacent frontier 上
  按 **Smoothed Seed Density** 从高到低确定性扩张；同分时按几何顶点 ID
  决胜，多个区域不得重复占用顶点
- 第一个 seed 是至少在一个 `valid` Primary state 可见的最高
  **Smoothed Seed Density** 顶点。增加第 `r` 个区域时，先在 `r-1`
  候选中确定 coverage 最低的 `valid` Primary state `s*`，再从对该
  state 有严格正投影覆盖增益的候选顶点中选择最高 density
- 新 seed 的覆盖增益使用正式 coverage 直接定义：
  `Delta C_s(i | S) = C_s(S union {i}) - C_s(S)`。实现只可以忽略浮点
  数值容差内的零增益，不得用“顶点在该视角可见”代替该条件
- 新 seed 到 `S_{r-1}` 的原始 OBJ edge-length 图测地距离必须不小于
  `smoothing_length`。区域生长时不得占用其他区域的顶点，也不得加入与
  其他区域 face-adjacent 的顶点；无法在该约束下填满 `B/r` 则候选
  construction 失败，不得放宽分离尺度
- **Support Area Budget** 使用生成谱基时相同的曲面 mass 度量，并约束所有
  **Connected Support Region** 的面积总和
- 对候选区域数 `r = 1, ..., R_max`，每个候选必须从头构造，并把同一
  固定 **Support Area Budget** 等分为 `r` 份；每个区域生长至 `B/r`，
  仅允许一个边界顶点造成的离散 mass 容差
- 选择通过 Primary coverage 硬 Gate 的最小 `r`；增加区域数不得在旧
  support 上追加面积，第一版也不根据 coverage 或 density 自适应改变
  区域间面积份额
- 达到固定 **Support Area Budget** 后若 primary
  **Support Visibility Coverage** 仍不合格，本次 support construction 失败并
  保存诊断；禁止继续增加面积直到通过
- **Support Visibility Coverage** 只决定训练前需要多少个
  **Connected Support Region**；它不允许在 **Attack Training** 中动态修改
  **Fixed Vertex Support**
- coverage 的面积预算由内在曲面 mass 控制，视角覆盖证据由投影重心
  权重控制；二者不得混用“整个面可见”的二值近似
- 投影重心控制量只在实例自身的 valid renderer pixels 上定义；空 support 为零，
  完整 support 为一，并随 **Fixed Vertex Support** 集合包含关系单调不减
- Primary coverage 必须在 source OpenVLA 实际输入的有效视野内计算；
  `alpha` 与预乘的 `alpha * w_S` 必须经过同一个非负 evidence transform
  后再分别求和，避免裁剪外区域或边缘重采样污染 coverage。它与 RGB
  deployment transform 共享有效视野坐标和像素 footprint，但不要求使用会
  破坏非负预乘语义的颜色插值核
- nvdiffrast 投影可用于生成重心权重及候选 coverage，但不能单独证明
  完整场景中的真实可见性。正式 coverage 从第一版起始终以 MuJoCo
  target segmentation 生成 `alpha`，nvdiffrast 只生成 `w_S`
- nvdiffrast 与 MuJoCo 的关键对齐量是逐 `(state, view)` visible recall，
  即二者软 alpha 交集占 MuJoCo 可见目标权重的比例。它是 coverage 可计算的
  必要条件，不足以单独证明投影轮廓和位置完全正确。IoU 和 renderer
  precision 只作诊断，因为 nvdiffrast 缺少场景深度造成的额外投影会被
  正式 `alpha` 排除，但其异常仍要求检查对齐证据
- 对齐 recall 门槛必须在 source 训练 states 0–9 对齐审计后、正式生成
  support 前冻结；它只判断投影证据是否可靠，不得根据某个 support
  的 coverage 成败调整
- `not_observable` 与 `insufficient_observation` 显式排除于正式 coverage
  汇总，但必须保存 state/view、`A_obs` 和诊断指标。Primary 中的
  `invalid_alignment` 不得记为零或排除，而必须使 support construction
  验收失败；wrist 中的 `invalid_alignment` 必须在状态计数中显式报告，
  但不参与 Primary Gate
- `A_obs_min` 与 alignment recall 门槛一样，必须在 states 0–9 的
  support-independent 可见性审计后、正式生成 support 前冻结；不得根据
  support coverage 或攻击结果调整
- **Support Visibility Coverage** 同时记录 primary 和 wrist：primary 是源攻击
  的唯一构造/Gate 视角，wrist 是只读迁移性诊断。wrist 不得改变 seed、
  region growing、区域数、候选排序或 support 验收结果
- 多个物体实例共享同一 **Active Texture** 时，**Support Visibility Coverage**
  先在每个实例内形成可见权重与重心控制量的预乘贡献，再按各实例的实际
  可见投影权重聚合；不得对逐实例 coverage 做无权平均，也不得把一个实例的
  barycentric correspondence 套到另一个实例上
- 每个共享纹理实例的可见权重来自对应的 **Instance Visibility Evidence**；RGB、
  segmentation、实例姿态和 barycentric correspondence 必须描述同一 simulation
  state 与 camera
- **Renderer Delta Composition** 由 **Instance Visibility Evidence** 排除场景
  遮挡，并对所有共享纹理实例聚合；它不能替代 bake PNG 的正式 rollout 语义
- **Renderer-to-Bake Response** 在使用 **Renderer Delta Composition** 生成
  **Support Seed Score** 前验收其方向可信度；Action margin 变化只作诊断
- **Terminal Deployment Response Audit** 的训练路径必须从终态紧凑参数重建
  **Surface Delta**，部署路径必须激活与它逐像素重放校验过的 bake
  **Attack Artifact**；禁止将 PNG 反推回顶点参数
- 第一版 wrist coverage 使用与 Primary 相同的 source-side
  center-crop/resize 几何，并在产物中标记为 `wrist_source_crop_proxy`；
  它是保守的 source-only 视野代理，不得声称为 OFT 真实输入预处理
- 第一版 wrist coverage 不是优化目标。它逐 state 保存原始相机/
  `wrist_source_crop_proxy` 有效视野对照、状态标签、与 Primary 的差异，
  并仅在 `valid` wrist states 上汇总 min/mean；汇总必须同时报告分母与
  各状态数，禁止把无效或缺失项当作零覆盖
- **Support Seed Score** 在 **Attack Training** 前计算并持久化；OFT 梯度不得
  进入该分数，source Feature 梯度也只能作独立诊断
- **Support Seed Score** 与第一版 **Attack Training** 必须使用同一个
  **Untargeted Clean-Action Margin**，禁止按对称 target 选 support 后再换目标训练
- 单位面积 seed density 只是区域选择的证据场，不能单独决定最终
  **Fixed Vertex Support**；最终结果还必须满足连通性、
  **Support Area Budget**、**Support Visibility Coverage** 和区域数上限
- **Smoothed Seed Density** 决定种子和扩张的优先级；原始 density 与平滑结果
  必须同时持久化
- **Smoothed Seed Density** 的平滑尺度使用
  `alpha = smoothing_length / sqrt(total_surface_area)` 表示，并令
  `tau = alpha^2 * total_surface_area`；禁止只记录依赖 OBJ 单位的裸 `tau`

## Example dialogue

> **Dev:** “训练生成的 UV PNG 是 **Attack Artifact**，什么时候会成为
> **Active Texture**？”
>
> **Domain expert:** “进入 rollout 前由 **Runtime Asset Transaction** 激活；
> 下一个 task 开始前再恢复对应的 **Clean Asset**。”

## Flagged ambiguities

- “texture” 曾同时表示源纹理、当前注入纹理和实验输出；现在分别使用
  **Clean Asset**、**Active Texture** 和 **Attack Artifact**。
- “vertex noise” 曾混合表示无界优化参数与真正施加到表面的 RGB 变化；现在
  后者统一称为 **Surface Delta**。
- “使用谱方法”可能表示 Spectral **Texture Parameterization**，也可能表示
  **Spectral Naturalness Regularization**；后续讨论和实验必须明确使用哪个术语。
