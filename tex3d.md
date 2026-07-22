# Tex3D: Objects as Attack Surfaces via Adversarial 3D Textures for Vision-Language-Action Models

## Abstract

Vision-language-action (VLA) models have shown strong performance in robotic manipulation, yet their robustness to physically realizable adversarial attacks remains underexplored. Existing studies reveal vulnerabilities through language perturbations and 2D visual attacks, but these attack surfaces are either less representative of real deployment or limited in physical realism. In contrast, adversarial 3D textures pose a more physically plausible and damaging threat, as they are naturally attached to manipulated objects and are easier to deploy in physical environments. Bringing adversarial 3D textures to VLA systems is nevertheless nontrivial. A central obstacle is that standard 3D simulators do not provide a differentiable optimization path from the VLA objective function back to object appearance, making it difficult to optimize through an end-to-end manner. To address this, we introduce Foreground-Background Decoupling (FBD), which enables differentiable texture optimization through dual-renderer alignment while preserving the original simulation environment. To further ensure that the attack remains effective across long-horizon and diverse viewpoints in the physical world, we propose Trajectory-Aware Adversarial Optimization (TAAO), which prioritizes behaviorally critical frames and stabilizes optimization with a vertex-based parameterization. Built on these designs, we present Tex3D, the first framework for end-to-end optimization of 3D adversarial textures directly within the VLA simulation environment. Experiments in both simulation and real-robot settings show that Tex3D significantly degrades VLA performance across multiple manipulation tasks, achieving task failure rates of up to 96.7%. Our empirical results expose critical vulnerabilities of VLA systems to physically grounded 3D adversarial attacks and highlight the need for robustness-aware training. The project page is available at https://vla-attack.github.io/tex3d

CCS Concepts

• Security and privacy → Social aspects of security and privacy. 

Keywords

VLA Models, 3D Adversarial Textures, Embodied Robustness 

## 1 Introduction

Vision-language-action (VLA) models [2, 3, 17] have achieved remarkable progress in robotic manipulation. By processing visual observations and natural language instructions in an end-to-end manner, these models directly map multimodal inputs to low-level control signals [26, 47], delivering strong performance across a wide range of complex manipulation tasks. While VLA models represent a significant step toward general-purpose robotic intelligence, they also introduce potential safety vulnerabilities, particularly against adversarial attacks [25, 37, 43]. A malicious actor could manipulate a VLA model into executing incorrect or harmful actions, causing it to perform operations unrelated to the intended task or to exhibit unsafe behaviors. Assessing the adversarial robustness of VLA models is therefore critical to ensure that deployed systems can respond reliably and safely across varied real-world conditions. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-08/14fc23a2-8e19-421f-87db-25b6a67475d5/e06f6a19af88e65099de1ef210554806e598aec3b0be3ab67ac340dcf0161505.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-08/14fc23a2-8e19-421f-87db-25b6a67475d5/b91b8b1435a0f00f8cbd46607cbd0e32e92e9de13e488d2a98dd695515cc1eb1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-08/14fc23a2-8e19-421f-87db-25b6a67475d5/f7a05c0165d3affabeec73eef85738642fd1600b35b485779c97c3403f5614d5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-08/14fc23a2-8e19-421f-87db-25b6a67475d5/3d26e790ecc272ace09984bbaed61100b0261e1217b01449184801bca3275784.jpg)



Figure 1: Comparison between Tex3D and existing attack paradigms. Bottom-right: VLA exhibits a certain degree of generalization under color changes and Gaussian noise perturbations, but its task failure rate rises sharply under Tex3D.


Existing adversarial attacks on VLA models generally fall into two categories. One line of work [25, 42] targets the language modality by injecting adversarial perturbations into the text instruction, causing the model to output erroneous actions. While effective, these methods are tightly coupled to the language interface. Another line of work [12, 23, 27, 37, 44] operates at the 2D visual front-end by affixing adversarial patches onto the input image, inducing task failure through perception-level manipulation. Due to weaker coupling with model-specific interfaces, this line of attack has been more widely studied. However, 2D adversarial patches are inherently view-specific: their effectiveness is contingent on precise viewpoint and pose alignment, which is difficult to guarantee in physical deployment and easy to detect due to their visible, non-naturalistic appearance. Motivated by these limitations, this paper aims to develop a more physically grounded and less perceptible attack surface: adversarial 3D textures. Unlike 2D patches, adversarial 3D textures are directly bound to object surfaces, making them inherently robust to viewpoint and object pose variations during embodied interaction, and more naturally integrated into the object’s appearance. This object-centric attack surface is particularly suitable for embodied manipulation and real-world deployment. 

However, achieving this goal requires overcoming two sequential challenges. The first is the lack of differentiability with respect to object appearance. Common embodied simulation platforms such as MuJoCo [35] (the physics engine underlying LIBERO [22]) are nondifferentiable w.r.t. 3D object appearance. Since they expose VLA models only through action-level interactions, gradients cannot be propagated back to object textures, making direct optimization of 3D physical perturbations intractable. As a result, there is currently no straightforward way to optimize adversarial 3D textures in such environments. Once such a differentiable optimization pathway is established, a second challenge arises: maintaining adversarial effectiveness over long temporal sequences. Existing 3D texture attacks are mainly developed for non-embodied settings [10, 33, 40] and do not account for the long sequential nature of VLA inference. Even simple tasks span hundreds of frames, and not all frames contribute equally to the model’s decisions. A perturbation effective at one timestep may have diminished impact at another, making it hard to sustain adversarial influence consistently over the full trajectory. 

To address this, we propose Tex3D, an adversarial texture attack framework for VLA models. We first introduce Foreground-Background Decoupling (FBD), which establishes a differentiable optimization pathway through cross-renderer parameter alignment and scene compositing. Specifically, MuJoCo renders the background while the target object is rendered in Nvdiffrast [19]; the aligned MVP transforms and lighting parameters $( I _ { a } , I _ { d } , \rho )$ ensure both spatial and photometric coherence across the two renderers. This enables gradients to flow back to the texture map without reconstructing the full simulation stack. Building on this differentiable pipeline, we further propose Trajectory-Aware Adversarial Optimization (TAAO) which consists of two coupled components: (i) identifying behaviorally critical frames via the latent velocity and acceleration of the observation sequence (computed using central differences on a pre-trained visual encoder output) and weighting these frames accordingly via a temperature-scaled softmax; and (ii) jointly reparameterizing the adversarial texture as per-vertex color attributes to constrain optimization to a smooth, low-rank manifold, reducing overfitting and improving attack effectiveness. Building on TAAO, we further instantiate two attack strategies, including untargeted and targeted attacks, to effectively simulate a broad range of adversarial scenarios. Together, these components enable fully end-to-end optimization of physically grounded 3D adversarial textures directly within the embodied simulation loop. 

Our main contributions are summarized as follows: ❶ To the best of our knowledge, Tex3D is the first framework enabling end to end optimization of 3D adversarial textures directly within a VLA simulation environment. ❷ We introduce two key techniques, Foreground Background Decoupling (FBD) and Trajectory Aware Adversarial Optimization (TAAO), which enable differentiable and temporally consistent optimization of adversarial 3D textures. ❸ We conduct extensive evaluations in simulation and real robot settings across four manipulation task suites, showing Tex3D achieves task failure rates of up to 96.7%, revealing critical vulnerabilities of current VLA models to physically grounded 3D adversarial attacks. 

## 2 Related Work

### 2.1 Vision-Language-Action Models

Recent advances in large vision-language models (LVLMs) have driven the development of VLA models, which unify perception, language grounding, and action generation for robotic control [3, 17, 34, 47]. Existing methods can be broadly categorized into autoregressive, diffusion-based, and hybrid paradigms [2, 3, 17, 24, 26, 32, 34, 38, 47]. Among them, OpenVLA [14, 17, 28] and $\pi_0$ [2] are two representative open-source VLA models, following token-based autoregressive prediction and flow-matching diffusion policies, respectively. Recent extensions such as OpenVLA-OFT [16], $\pi_{0.5}$ [11], and other enhanced VLA frameworks [21, 29, 46] further improve efficiency, spatial-temporal reasoning, and generalization. Despite strong performance, these models remain highly vulnerable to adversarial attacks [13, 15, 25, 42], as even small visual perturbations can propagate through the tightly coupled perception-languageaction pipeline and trigger potentially unsafe or erroneous actions. 

### 2.2 Adversarial Attacks in VLA Models

Adversarial attacks [4, 5, 18, 31] are widely used to probe vulnerabilities in VLA models, where small perturbations can lead to task failures or unsafe behaviors [15, 25, 36]. Existing VLA attacks can generally be divided into two categories: (1) language-based attacks, which manipulate instructions to subtly influence action generation [15, 25, 42], and (2) patch-based attacks, which apply adversarial perturbations to input images, typically via 2D patches that significantly degrade model performance and can often transfer across settings [12, 23, 27, 37, 44]. Compared to language-based methods, visual attacks are more practical due to their direct impact on perception and physical realizability [9]. However, existing 2D patch attacks are inherently viewpoint-dependent, requiring precise alignment between camera pose and object geometry, and their conspicuous appearance makes them easier to detect. While prior work has explored 3D adversarial perturbations using point clouds, meshes, and neural rendering to improve cross-view robustness [20, 39, 40], these approaches are not directly applicable to VLA systems. To effectively address these limitations, we explore adversarial 3D textures bound to target object surfaces, enabling more robust, physically realizable, and less perceptible attacks. 

## 3 Methodology

### 3.1 Problem Formulation

We consider embodied simulation environments such as MuJoCo, where the visual scene consists of K 3D objects. Each object is modeled as a textured mesh $M = \left( { { \bf { V } } , { \bf { F } } , { \bf { T } } } \right)$ , with vertex coordinates V, triangle face indices F, and a texture map $\mathbf { T } \in \mathbb { R } ^ { H \times W _ { t } \times 3 }$ that encodes its surface appearance. The scene is rendered at each timestep t into an RGB observation ${ \mathbf O } _ { t } \in \mathbb { R } ^ { H \times W \times 3 }$ , which serves as the visual input to the VLA model $\pi$. The model takes $O_t$ and a language instruction l and outputs actions $\mathbf { a } _ { t } \in \mathbb { R } ^ { d } , \mathrm { i . e . , } \pi ( \mathbf { O } _ { t } , l ) = \mathbf { a } _ { t }$ . 

Optimization Objective. The goal of Tex3D is to find an adversarial texture map $\mathrm { T } _ { a d v }$ that encodes a texture capable of disrupting the VLA model’s action outputs, while leaving the object’s geometry and semantic identity intact. Since VLA inference is inherently sequential, even a simple task unfolds over hundreds of timesteps; the adversarial texture must therefore sustain its effect across the full trajectory rather than a single observation. Moreover, to improve robustness against viewpoint changes, we optimize the texture jointly over M sampled views at each timestep. Let $\mathbf { O } _ { t , m } ( M )$ denote the observation of mesh M at timestep t under the m-th sampled view, where $m \in \{ 1 , \ldots , M \}$ . Let $\mathbf { a } _ { t , m } ^ { * } = \pi ( \mathbf { O } _ { t , m } ( \mathcal { M } ) , l )$ be the corresponding reference action under the clean mesh. Tex3D thus seeks $\mathbf { T } _ { a d v }$ that maximizes the action deviation in expectation jointly over both the task trajectory T and sampled views during optimization: 

$$
\mathrm{T} _ {a d v} = \underset {\mathrm{T} _ {a d v}} {\arg \max} \underset {t \sim \mathcal {T}, m \sim \mathcal {U}} {\mathbb {E}} \left[ \mathcal {L} _ {\pi} \left(\pi \left(\mathrm{O} _ {t, m} (\mathcal {M} _ {a d v}), l\right), \mathrm{a} _ {t, m} ^ {*}\right) \right], \tag {1}
$$

$$
\mathrm{where} \mathcal {M} _ {a d v} = (\mathbf {V}, \mathbf {F}, \mathbf {T} _ {a d v}).
$$



![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-08/14fc23a2-8e19-421f-87db-25b6a67475d5/c846a1a62502bf03226413f17cdfa2be702e932ddb1acad4c8d69a1b8e97224b.jpg)



Figure 2: Overview of Tex3D. FBD renders the background in MuJoCo and the target object in Nvdiffrast, with cross-renderer alignment of geometric parameters $( \mathbf { P } _ { t } , \mathcal { V } _ { t } , \mathbf { M } _ { t } )$ and lighting parameters $( I _ { a } , I _ { d } , \rho )$ for photometrically consistent scene composition. The composited observation is fed into the frozen VLA model, and gradients from untargeted or targeted objectives are back-propagated to directly optimize the object texture. TAAO further applies dynamics-guided weighting over critical frames, enabling temporally effective adversarial 3D texture optimization over complex long-horizon manipulation trajectories.


To enable end-to-end optimization of $\mathbf { T } _ { a d v }$ directly within the simulator, we design FBD, a differentiable rendering pipeline that decouples the adversarial object from its static background, allowing gradients to directly flow back to the texture map T; details are given in Sec. 3.2. To realize the trajectory-level expectation $\mathbb { E } _ { t \sim \mathcal { T } }$ in Eq. (1) both effectively and efficiently, we further propose TAAO, which automatically identifies behaviorally critical frames via latent dynamics of the observation sequence and concentrates optimization on them; details are given in Sec. 3.3. Finally, to improve the robustness of our framework under real-world conditions, we incorporate an EoT scheme to bridge the gap between digital-domain optimization and physical deployment, as detailed in Sec. 3.4. 

### 3.2 Differentiable Rendering via Foreground–Background Decoupling

Common embodied simulators such as MuJoCo typically serve primarily as physics engines and do not expose a gradient path through 3D object appearance. We propose Foreground-Background 

Decoupling (FBD): MuJoCo is kept running unmodified, while the target object is additionally rendered in the differentiable renderer Nvdiffrast fully in parallel. At each step, the relevant rendering parameters (object pose, camera viewpoint, and lighting) are relayed from MuJoCo to Nvdiffrast for precisely synchronized rendering; corresponding gradients are then computed through Nvdiffrast to update the adversarial texture, which is subsequently applied back to the object in MuJoCo for the next simulation step. This design combines the strengths of both renderers: Nvdiffrast provides differentiable gradients for texture optimization, while MuJoCo retains full control over the simulation environment and the VLA model interaction interface, which Nvdiffrast alone cannot provide. 

❶ Environmental background rendering. At timestep t, MuJoCo renders the complete scene under physically accurate simulation, capturing the full environmental context (robot, tabletop, and surrounding objects) as the faithful background reference: 

$$
\mathbf {x} _ {t} ^ {\mathrm{bg}} \in [ 0, 1 ] ^ {3 \times H \times W}. \tag {2}
$$

❷ Target foreground rendering. We adopt Nvdiffrast, a highperformance differentiable renderer developed by NVIDIA that provides GPU-accelerated primitive operations based on rasterization, enabling efficient gradient computation over 3D mesh attributes including texture. The target object mesh $\mathcal { M } _ { a d v } = \left( \mathbf { V } , \mathbf { F } , \mathbf { T } _ { a d v } \right)$ is rendered under rendering parameters aligned with MuJoCo (see cross-renderer geometric alignment below). This produces the foreground image $\mathbf { x } _ { t } ^ { \mathrm { f g } } ( \mathbf { T } _ { a d v } )$ with gradients available with respect to $\mathbf { T } _ { a d v } ,$ , and a silhouette mask $\mathbf { m } _ { t } \in \{ 0 , 1 \} ^ { H \times W }$ pre-computed from MuJoCo’s native renderer that identifies which pixels correspond to the target object and which belong to the background. 

❸ Scene compositing. After foreground object rendering, the adversarial observation is explicitly assembled by substituting the object with the adversarial texture seamlessly back into the scene: 

$$
\mathbf {O} _ {t} (\mathcal {M} _ {a d v}) = \mathbf {m} _ {t} \odot \mathbf {x} _ {t} ^ {\mathrm{fg}} (\mathbf {T} _ {a d v}) + (1 - \mathbf {m} _ {t}) \odot \mathbf {x} _ {t} ^ {\mathrm{bg}}, \tag {3}
$$

where ⊙ denotes element-wise multiplication. This is equivalent to applying $\mathbf { T } _ { a d v }$ to the object in MuJoCo, while maintaining a differentiable path from $\mathrm { T } _ { a d v }$ back through the action loss. $\mathbf { O } _ { t } ( \mathcal { M } _ { a d v } )$ is fed to the VLA model, which produces the action $a_t$ , completing the gradient chain $\mathbf { T } _ { a d v } \xrightarrow { \mathrm { f g } } \xrightarrow { } \mathbf { O } _ { t } \xrightarrow { } \pi \xrightarrow { } \mathcal { L } _ { \pi }$ required by Eq. (1). 

Cross-renderer parameter alignment. To ensure that Nvdiffrast renders the target object consistently with its appearance in Mu-JoCo, we synchronize the rendering parameters of both renderers at each timestep. Geometric alignment. Let $\mathbf { M } _ { t } \in \mathbb { R } ^ { 4 \times 4 }$ be the model matrix of the target object, encoding its 3D pose in world coordinates. The view matrix $\mathcal { N } _ { t }$ and projection matrix $\mathbf { P } _ { t }$ are derived from the simulator’s camera pose and intrinsic parameters. Following the widely used standard MVP transform in computer graphics, the clip-space coordinate of each vertex is obtained as: 

$$
\mathbf {v} _ {i, t} ^ {\mathrm{clip}} = \mathbf {C} _ {t} \tilde {\mathbf {v}} _ {i}, \quad \mathbf {C} _ {t} = \mathbf {P} _ {t} \mathcal {V} _ {t} \mathbf {M} _ {t} \in \mathbb {R} ^ {4 \times 4}, \tag {4}
$$

where $\tilde { \mathbf { v } } _ { i } \in \mathbb { R } ^ { 4 }$ is the precisely defined homogeneous form of the i-th vertex $\mathbf { v } _ { i } \in \mathbf { V } ; \mathbf { M } _ { t }$ transforms the vertex from object space to world space; $V_t$ further projects it into camera space; and $\mathbf { P } _ { t }$ maps it to clip space. Nvdiffrast subsequently takes $\{ \mathbf { v } _ { i , t } ^ { \mathrm { c l i p } } \}$ as input for rasterization and texture interpolation. Through this transform, the spatial placement of the target object in Nvdiffrast is precisely aligned with its counterpart in MuJoCo, ensuring geometric consistency across the two renderers. Lighting alignment. Beyond geometry, we explicitly align the lighting conditions of the two renderers. The ambient light intensity $I _ { a } ,$ diffuse light intensity $I _ { d } ,$ and material reflectance $\rho$ of the target object are read from MuJoCo’s scene configuration and applied to the Nvdiffrast shading pipeline. Through this joint alignment, the viewpoint, pose, and surface shading of the target object are kept faithfully consistent between the two renderers, yielding a photometrically coherent composited observation. 

### 3.3 Trajectory-Aware Adversarial Optimization

As clearly established in Sec. 3.1, effective adversarial textures must consistently exert influence across the full manipulation trajectory rather than at isolated frames. We address this by proposing trajectory-aware adversarial optimization (TAAO), which exploits task structure to focus adversarial pressure where it matters most. Latent dynamics-guided frame weighting. Frames at which the robot undergoes behavioral transitions $( \mathrm { e . g . }$ , the onset of grasping or lift-off) are disproportionately influential on task success. We identify such frames through the latent dynamics of the observation sequence, without relying on any simulator-specific state. 

Latent feature extraction. At each timestep t, we first extract a compact latent representation via a pre-trained visual encoder [8] E: 

$$
\mathbf {f} _ {t} = E (\mathbf {O} _ {t}) \in \mathbb {R} ^ {d f}. \tag {5}
$$

Latent velocity and acceleration. We estimate the temporal rate and rate-of-change of feature variation using central differences: 

$$
v _ {t} = \left\| \mathbf {f} _ {t + 1} - \mathbf {f} _ {t - 1} \right\| _ {2} / 2, \quad \alpha_ {t} = \left| v _ {t} - v _ {t - 1} \right|, \tag {6}
$$

and jointly normalize both quantities over the trajectory: 

$$
\hat {v} _ {t} = \frac {v _ {t} - v _ {\min}}{v _ {\max} - v _ {\min}}, \quad \hat {\alpha} _ {t} = \frac {\alpha_ {t} - \alpha_ {\min}}{\alpha_ {\max} - \alpha_ {\min}}. \tag {7}
$$

Criticality scoring and frame weighting. After jointly normalizing the two variables, we formally define the criticality score as: 

$$
s _ {t} = \max (\hat {v} _ {t}, \hat {\alpha} _ {t}), \tag {8}
$$

and the per-frame optimization weight is derived via a temperaturescaled softmax to prioritize critical timesteps during optimization: 

$$
w _ {t} = \exp (s _ {t} / \tau) \bigg / \sum_ {t ^ {\prime} = 1} ^ {T} \exp (s _ {t ^ {\prime}} / \tau), \tag {9}
$$

where $\tau > 0$ controls the concentration. High $\hat{\alpha}_t$ indicates rapid perceptual change; high $\hat { \alpha } _ { t }$ signals an abrupt shift in the rate of change—hallmarks of behaviorally critical moments. This instantiates $\mathbb { E } _ { t \sim \mathcal { T } }$ in Eq. (1) as a dynamics-weighted sum over the episode. 

Vertex-based texture parameterization. Direct optimization in the high-dimensional pixel space of $\mathbf { T } _ { a d v } \in \mathbb { R } ^ { H _ { t } \times W _ { t } \times 3 }$ often tends to produce highly irregular perturbations that overfit to the specific model being attacked, resulting in poor cross-model transferability [10]. To mitigate this, we effectively leverage Nvdiffrast to reparameterize $T_{adv}$ as $N_v$ per-vertex color attributes c ∈ $\mathbb{R}^{N_v ×3}$, $\mathrm { T } _ { a d v }$ $N _ { v }$ $\mathbf { c } \in \mathbb { R } ^ { N _ { v } \times 3 }$ with the texture map recovered through barycentric interpolation over the fixed mesh geometry (V, F) defined in Sec. 3.1: 

$$
\mathbf {T} _ {a d v} = \phi (\mathbf {c}). \tag {10}
$$

Since $N _ { v } \ll H _ { t } \times W _ { t }$ , this parameterization implicitly restricts the perturbation to a smooth, low-rank manifold defined by the mesh geometry, thereby effectively reducing the search space and consistently yielding more transferable adversarial textures. 

Untargeted attack. The untargeted objective degrades overall task success of the VLA model, without prescribing a specific failure mode. Given the reference action $\mathbf { a } _ { t , m } ^ { * } = \pi ( \mathbf { O } _ { t , m } ( \mathcal { M } ) , l )$ produced by the model on the clean object under the m-th sampled view, we maximize the weighted action deviation averaged over views: 

$$
\mathbf {c} ^ {*} = \arg \max _ {\mathbf {c}} \sum_ {t = 1} ^ {T} w _ {t} \cdot \frac {1}{M} \sum_ {m = 1} ^ {M} \left\| \pi (\mathbf {O} _ {t, m} (\mathcal {M} _ {a d v}), l) - \mathbf {a} _ {t, m} ^ {*} \right\| _ {2}. \tag {11}
$$

Targeted attack. Targeted manipulation attacks go further by steering the model toward a prescribed erroneous trajectory, enabling deliberate behavioral hijacking. Let $\mathbf { a } _ { t } ^ { \mathrm { t g t } }$ denote the target action at timestep t. The objective minimizes the deviation of the model’s output from this trajectory across all sampled views: 

$$
\mathbf {c} ^ {*} = \underset {\mathbf {c}} {\arg \min} \sum_ {t = 1} ^ {T} w _ {t} \cdot \frac {1}{M} \sum_ {m = 1} ^ {M} \left\| \pi \left(\mathbf {O} _ {t, m} (\mathcal {M} _ {a d v}), l\right) - \mathbf {a} _ {t} ^ {\mathrm{tgt}} \right\| _ {2}. \tag {12}
$$

In our implementation, $\mathbf { a } _ { t } ^ { \mathrm { t g t } }$ is defined as semantically plausible, time-varying actions that redirect the gripper toward an alternative target at each timestep, inducing controlled task deviation. 

### 3.4 Physical Attack

Adversarial textures optimized purely in the digital domain may degrade in effectiveness when applied to physical objects, due to real-world variations such as viewpoint shifts, lighting changes, and imaging artifacts. To bridge this gap, we incorporate the Expectation over Transformations (EoT) [1] strategy during optimization. Rather than optimizing for a single rendered observation, EoT samples a stochastic transformation $g \sim \mathcal { T }$ at each step, covering both 3D variations (object pose perturbations, viewpoint shifts, distance changes) and 2D image-level augmentations (brightness, contrast, and blur). The rendered observation under EoT is thus: 


Table 1: Task failure rates (%) on four LIBERO task variants under untargeted and targeted settings. “No Attack” denotes clean evaluation, “Gaussian”: random Gaussian noise, “Single-frame”: one-frame perturbation, “Vertex Param.”: our parameterized texture attack without temporal consistency, “Tex+Temp.”: the texture-only temporal variant, and Tex3D: our full method.


<table><tr><td rowspan="2">Model</td><td rowspan="2">Task</td><td colspan="6">Untargeted Attack</td><td colspan="6">Targeted Attack</td></tr><tr><td>No Attack</td><td>Gaussian</td><td>Single-frame</td><td>Vertex Param.</td><td>Tex+Temp.</td><td>Tex3D</td><td>No Attack</td><td>Gaussian</td><td>Single-frame</td><td>Vertex Param.</td><td>Tex+Temp.</td><td>Tex3D</td></tr><tr><td rowspan="5">OpenVLA [17]</td><td>Spatial</td><td>15.6</td><td>24.6 (↑9.0)</td><td>75.1 (↑59.5)</td><td>80.5 (↑64.9)</td><td>90.3 (↑74.7)</td><td>95.8 (↑80.2)</td><td>15.6</td><td>24.6 (↑9.0)</td><td>82.4 (↑66.8)</td><td>86.5 (↑70.9)</td><td>93.2 (↑77.6)</td><td>96.7 (↑81.1)</td></tr><tr><td>Object</td><td>11.8</td><td>18.8 (↑7.0)</td><td>62.1 (↑50.3)</td><td>69.8 (↑58.0)</td><td>74.6 (↑62.8)</td><td>81.1 (↑69.3)</td><td>11.8</td><td>18.8 (↑7.0)</td><td>70.9 (↑59.1)</td><td>74.8 (↑63.0)</td><td>81.1 (↑69.3)</td><td>85.5 (↑73.7)</td></tr><tr><td>Goal</td><td>23.6</td><td>28.8 (↑5.2)</td><td>65.5 (↑41.9)</td><td>70.7 (↑47.1)</td><td>79.1 (↑55.5)</td><td>84.8 (↑61.2)</td><td>23.6</td><td>28.8 (↑5.2)</td><td>71.5 (↑47.9)</td><td>74.2 (↑50.6)</td><td>81.6 (↑58.0)</td><td>86.9 (↑63.3)</td></tr><tr><td>Long</td><td>45.4</td><td>52.2 (↑6.8)</td><td>73.2 (↑27.8)</td><td>80.9 (↑35.5)</td><td>87.8 (↑42.4)</td><td>90.9 (↑45.5)</td><td>45.4</td><td>52.2 (↑6.8)</td><td>79.6 (↑34.2)</td><td>84.3 (↑38.9)</td><td>90.5 (↑45.1)</td><td>92.8 (↑47.4)</td></tr><tr><td>Avg.</td><td>24.1</td><td>31.1 (↑7.0)</td><td>69.0 (↑44.9)</td><td>75.5 (↑51.4)</td><td>82.9 (↑58.8)</td><td>88.1 (↑64.0)</td><td>24.1</td><td>31.1 (↑7.0)</td><td>76.1 (↑52.0)</td><td>79.9 (↑55.8)</td><td>86.6 (↑62.5)</td><td>90.5 (↑66.4)</td></tr><tr><td rowspan="5">OpenVLA-OFT [16]</td><td>Spatial</td><td>3.8</td><td>8.8 (↑5.0)</td><td>69.9 (↑66.1)</td><td>72.2 (↑68.4)</td><td>74.9 (↑71.1)</td><td>78.6 (↑74.8)</td><td>3.8</td><td>8.8 (↑5.0)</td><td>70.8 (↑67.0)</td><td>74.4 (↑70.6)</td><td>76.9 (↑73.1)</td><td>80.9 (↑77.1)</td></tr><tr><td>Object</td><td>1.7</td><td>3.6 (↑1.9)</td><td>57.4 (↑55.7)</td><td>64.9 (↑63.2)</td><td>67.7 (↑66.0)</td><td>70.6 (↑68.9)</td><td>1.7</td><td>3.6 (↑1.9)</td><td>63.7 (↑62.0)</td><td>69.2 (↑67.5)</td><td>71.2 (↑69.5)</td><td>76.4 (↑74.7)</td></tr><tr><td>Goal</td><td>3.8</td><td>5.4 (↑1.6)</td><td>61.3 (↑57.5)</td><td>68.1 (↑64.3)</td><td>72.4 (↑68.6)</td><td>76.8 (↑73.0)</td><td>3.8</td><td>5.4 (↑1.6)</td><td>64.6 (↑60.8)</td><td>70.9 (↑67.1)</td><td>73.9 (↑70.1)</td><td>79.8 (↑76.0)</td></tr><tr><td>Long</td><td>9.3</td><td>10.9 (↑1.6)</td><td>65.5 (↑56.2)</td><td>68.0 (↑58.7)</td><td>73.6 (↑64.3)</td><td>78.2 (↑68.9)</td><td>9.3</td><td>10.9 (↑1.6)</td><td>68.9 (↑59.6)</td><td>71.7 (↑62.4)</td><td>75.8 (↑66.5)</td><td>80.2 (↑70.9)</td></tr><tr><td>Avg.</td><td>4.7</td><td>6.5 (↑1.8)</td><td>63.5 (↑58.8)</td><td>68.3 (↑63.6)</td><td>72.1 (↑67.4)</td><td>76.0 (↑71.3)</td><td>4.7</td><td>6.5 (↑1.8)</td><td>67.0 (↑62.3)</td><td>71.5 (↑66.8)</td><td>74.4 (↑69.7)</td><td>79.3 (↑74.6)</td></tr><tr><td rowspan="5"><eq>\pi_0</eq>[2]</td><td>Spatial</td><td>3.5</td><td>11.8 (↑8.3)</td><td>54.7 (↑51.2)</td><td>59.4 (↑55.9)</td><td>65.2 (↑61.7)</td><td>74.9 (↑71.4)</td><td>3.5</td><td>11.8 (↑8.3)</td><td>58.8 (↑55.3)</td><td>63.2 (↑59.7)</td><td>69.5 (↑66.0)</td><td>75.9 (↑72.4)</td></tr><tr><td>Object</td><td>2.3</td><td>8.6 (↑6.3)</td><td>48.6 (↑46.3)</td><td>55.4 (↑53.1)</td><td>59.3 (↑57.0)</td><td>68.9 (↑66.6)</td><td>2.3</td><td>8.6 (↑6.3)</td><td>52.5 (↑50.2)</td><td>57.2 (↑54.9)</td><td>62.6 (↑60.3)</td><td>70.2 (↑67.9)</td></tr><tr><td>Goal</td><td>5.2</td><td>9.6 (↑4.4)</td><td>53.0 (↑47.8)</td><td>58.3 (↑53.1)</td><td>63.1 (↑57.9)</td><td>70.3 (↑65.1)</td><td>5.2</td><td>9.6 (↑4.4)</td><td>56.2 (↑51.0)</td><td>60.5 (↑55.3)</td><td>68.6 (↑63.4)</td><td>73.4 (↑68.2)</td></tr><tr><td>Long</td><td>7.2</td><td>12.7 (↑5.5)</td><td>54.2 (↑47.0)</td><td>57.7 (↑50.5)</td><td>64.5 (↑57.3)</td><td>72.9 (↑65.7)</td><td>7.2</td><td>12.7 (↑5.5)</td><td>57.2 (↑50.0)</td><td>61.4 (↑54.2)</td><td>68.4 (↑61.2)</td><td>73.7 (↑66.5)</td></tr><tr><td>Avg.</td><td>4.6</td><td>10.7 (↑6.1)</td><td>52.6 (↑48.0)</td><td>57.7 (↑53.1)</td><td>63.0 (↑58.4)</td><td>71.8 (↑67.2)</td><td>4.6</td><td>10.7 (↑6.1)</td><td>56.2 (↑51.6)</td><td>60.6 (↑56.0)</td><td>67.3 (↑62.7)</td><td>73.3 (↑68.7)</td></tr><tr><td rowspan="5"><eq>\pi_{0,5}</eq>[11]</td><td>Spatial</td><td>1.2</td><td>7.7 (↑6.5)</td><td>47.1 (↑45.9)</td><td>52.5 (↑51.3)</td><td>60.1 (↑58.9)</td><td>71.8 (↑70.6)</td><td>1.2</td><td>7.7 (↑6.5)</td><td>49.3 (↑48.1)</td><td>55.2 (↑54.0)</td><td>63.2 (↑62.0)</td><td>72.9 (↑71.7)</td></tr><tr><td>Object</td><td>1.8</td><td>6.9 (↑5.1)</td><td>39.7 (↑37.9)</td><td>46.0 (↑44.2)</td><td>52.4 (↑50.6)</td><td>65.2 (↑63.4)</td><td>1.8</td><td>6.9 (↑5.1)</td><td>41.1 (↑39.3)</td><td>48.2 (↑46.4)</td><td>54.3 (↑52.5)</td><td>68.3 (↑66.5)</td></tr><tr><td>Goal</td><td>2.0</td><td>5.5 (↑3.5)</td><td>44.7 (↑42.7)</td><td>50.9 (↑48.9)</td><td>57.7 (↑55.7)</td><td>69.7 (↑67.7)</td><td>2.0</td><td>5.5 (↑3.5)</td><td>47.5 (↑45.5)</td><td>53.2 (↑51.2)</td><td>60.8 (↑58.8)</td><td>71.3 (↑69.3)</td></tr><tr><td>Long</td><td>6.0</td><td>9.3 (↑3.3)</td><td>50.0 (↑44.0)</td><td>55.2 (↑49.2)</td><td>63.0 (↑57.0)</td><td>70.6 (↑64.6)</td><td>6.0</td><td>9.3 (↑3.3)</td><td>51.8 (↑45.8)</td><td>56.8 (↑50.8)</td><td>65.1 (↑59.1)</td><td>72.1 (↑66.1)</td></tr><tr><td>Avg.</td><td>2.8</td><td>7.4 (↑4.6)</td><td>45.4 (↑42.6)</td><td>51.2 (↑48.4)</td><td>58.3 (↑55.5)</td><td>69.3 (↑66.5)</td><td>2.8</td><td>7.4 (↑4.6)</td><td>47.4 (↑44.6)</td><td>53.4 (↑50.6)</td><td>60.9 (↑58.1)</td><td>71.2 (↑68.4)</td></tr></table>


Table 2: Task failure rates (%). Top: dual-renderer fidelity of OpenVLA; values in parentheses show increases from Mu-JoCo direct rendering. Bottom: cross-model transferability; increases are relative to the target model’s clean performance.


<table><tr><td></td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td></tr><tr><td colspan="5">Dual-Renderer Fidelity</td></tr><tr><td>MuJoCo (direct)</td><td>15.6</td><td>11.8</td><td>23.6</td><td>45.4</td></tr><tr><td>MuJoCo + nvdiffrast</td><td>16.8 (↑1.2)</td><td>12.2 (↑0.4)</td><td>24.4 (↑0.8)</td><td>45.8 (↑0.4)</td></tr><tr><td colspan="5">Cross-Model Transferability</td></tr><tr><td>OpenVLA → OpenVLA-OFT</td><td>70.6 (↑66.8)</td><td>61.5 (↑59.8)</td><td>65.4 (↑61.6)</td><td>69.1 (↑59.8)</td></tr><tr><td>OpenVLA-OFT → OpenVLA</td><td>75.3 (↑59.7)</td><td>70.2 (↑58.4)</td><td>73.4 (↑49.8)</td><td>75.7 (↑30.3)</td></tr><tr><td><eq>\pi_0 \rightarrow \pi_{0.5}</eq></td><td>58.4 (↑57.2)</td><td>49.2 (↑47.4)</td><td>54.1 (↑52.1)</td><td>57.6 (↑51.6)</td></tr><tr><td><eq>\pi_{0.5} \rightarrow \pi_0</eq></td><td>63.7 (↑60.2)</td><td>55.8 (↑53.5)</td><td>60.5 (↑55.3)</td><td>62.3 (↑55.1)</td></tr><tr><td>OpenVLA → <eq>\pi_0</eq></td><td>41.2 (↑37.7)</td><td>34.5 (↑32.2)</td><td>38.9 (↑33.7)</td><td>44.1 (↑36.9)</td></tr><tr><td>OpenVLA → <eq>\pi_{0.5}</eq></td><td>32.6 (↑31.4)</td><td>27.8 (↑26.0)</td><td>31.4 (↑29.4)</td><td>36.5 (↑30.5)</td></tr><tr><td><eq>\pi_0 \rightarrow \text{OpenVLA}</eq></td><td>51.2 (↑35.6)</td><td>44.8 (↑33.0)</td><td>52.4 (↑28.8)</td><td>61.5 (↑16.1)</td></tr><tr><td><eq>\pi_0 \rightarrow \text{OpenVLA-OFT}</eq></td><td>43.6 (↑39.8)</td><td>37.2 (↑35.5)</td><td>41.5 (↑37.7)</td><td>48.4 (↑39.1)</td></tr></table>

$$
\hat {\mathbf {O}} _ {t} (\mathcal {M} _ {a d v}) = g \left(\mathbf {O} _ {t} (\mathcal {M} _ {a d v})\right), \quad g \sim \mathcal {T}, \tag {13}
$$

and $\hat { \mathbf { O } } _ { t }$ replaces $\mathbf { O } _ { t }$ in Eqs. (11) and (12) during training. This encourages the adversarial texture to remain effective across diverse and unseen physical rendering conditions, improving its sim-to-real transferability. EoT complements the multi-view optimization in Sec. 3.1: multi-view optimization improves robustness to viewpoint changes in simulation, whereas EoT further bridges the gap to physical deployment by modeling broader real-world transformations. Detailed analyses and pseudocode are in Appendices A and B. 


Table 3: Task failure rates (%) comparison across color variants and Gaussian perturbation on four LIBERO task suites.


<table><tr><td>Model</td><td>Variant</td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td></tr><tr><td rowspan="5">OpenVLA</td><td>Green</td><td>20.2 (↑4.6)</td><td>17.8 (↑6.0)</td><td>27.4 (↑3.8)</td><td>48.8 (↑3.4)</td></tr><tr><td>Red</td><td>23.2 (↑7.6)</td><td>18.6 (↑6.8)</td><td>26.6 (↑3.0)</td><td>50.6 (↑5.2)</td></tr><tr><td>Yellow</td><td>18.8 (↑3.2)</td><td>15.4 (↑3.6)</td><td>25.2 (↑1.6)</td><td>48.4 (↑3.0)</td></tr><tr><td>Blue</td><td>24.4 (↑8.8)</td><td>19.2 (↑7.4)</td><td>27.6 (↑4.0)</td><td>49.2 (↑3.8)</td></tr><tr><td>Gaussian</td><td>24.6 (↑9.0)</td><td>18.8 (↑7.0)</td><td>28.8 (↑5.2)</td><td>52.2 (↑6.8)</td></tr><tr><td rowspan="5">OpenVLA-OFT</td><td>Green</td><td>5.8 (↑2.0)</td><td>5.4 (↑3.7)</td><td>7.2 (↑3.4)</td><td>11.4 (↑2.1)</td></tr><tr><td>Red</td><td>6.8 (↑3.0)</td><td>6.8 (↑5.1)</td><td>8.6 (↑4.8)</td><td>12.8 (↑3.5)</td></tr><tr><td>Yellow</td><td>4.6 (↑0.8)</td><td>4.8 (↑3.1)</td><td>6.8 (↑3.0)</td><td>10.8 (↑1.5)</td></tr><tr><td>Blue</td><td>6.2 (↑2.4)</td><td>5.9 (↑4.2)</td><td>8.8 (↑5.0)</td><td>13.9 (↑4.6)</td></tr><tr><td>Gaussian</td><td>8.8 (↑5.0)</td><td>3.6 (↑1.9)</td><td>5.4 (↑1.6)</td><td>10.9 (↑1.6)</td></tr></table>

## 4 Experiments

### 4.1 Experimental Setup

Datasets & Threat Models. We conduct our experiments on the LIBERO [22] benchmark in simulation environments, which provides a standardized testbed for evaluating VLA models across four task categories: Spatial, Object, Goal, and Long-horizon tasks. These tasks cover progressively more challenging manipulation scenarios,ranging from simple spatial reasoning to multi-step planning. We evaluate four representative VLA models, OpenVLA, OpenVLA-OFT, $\pi_0$, and $\pi _ { 0 . 5 } .$ , and report results on all LIBERO task variants. 

Baselines. We use three variants with Tex3D: (1) Single-frame: perturbs the adversarial texture using only a single observation frame [40]; (2) Vertex Param.: removes temporal consistency and performs only vertex-based adversarial texture optimization; and (3) Tex+Temp.: introduces temporal consistency to the texture-only variant. Additionally, following [6, 30], we establish two potential baselines settings : the clean setting without perturbation (No Attack) and random Gaussian perturbations applied to the object texture (Gaussian). In Sec. 4.3, we further compare Tex3D against 2D patch-based adversarial attacks [37] in terms of robustness under digital geometric variations and physical-world deployment. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-08/14fc23a2-8e19-421f-87db-25b6a67475d5/fd762677cb1eda41942e58fb9c45dbe606209e78215f675eb2a1b5bcc0df835c.jpg)



Figure 3: Qualitative results of Tex3D on manipulation tasks. For each task, the green row shows the clean rollout,whereas the red row shows the adversarial rollout under Tex3D.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-08/14fc23a2-8e19-421f-87db-25b6a67475d5/8a6284fc3cdd049fb0b94f9fd10690de5076bfb2421256733ac848c9397ed131.jpg)



Figure 4: Robustness comparison of Tex3D and 2D patchbased baselines under varying camera angles, object rotations, object positions (digital simulation), and position offsets in the physical world. Task failure rate (%, ↑) is reported.


Evaluation Metric & Details. We evaluate Tex3D following the standard protocol adopted in the LIBERO benchmark. Each task is executed for 50 independent trials, and performance is measured by Task Failure Rate (FR), defined as the proportion of failed task completions over all trials. For each benchmark suite, we report the average FR across tasks to reflect overall attack effectiveness and generalization. Unless otherwise specified, all experiments are conducted on a single NVIDIA A100 GPU with 80GB memory. 

Physical Experiment Setting. For real-world experiments, we employ a robotic platform built around a Franka Emika Panda manipulator, controlled via 7-DoF end-effector delta commands. Observations are obtained from a single monocular RGB camera, and our method operates solely on raw RGB inputs without requiring additional sensory modalities. To facilitate systematic quantitative evaluation, we utilize 3D printing to fabricate both the objects used in pick-and-place tasks and the corresponding adversarial variants. All tasks are repeated for 100 independent trials each. Further implementation details and setups are provided in Appendix G. 

### 4.2 Main Results

Overall Attack Effectiveness. Tab. 1 quantitatively summarizes the overall effectiveness of Tex3D on LIBERO tasks. Tex3D consistently yields the highest task failure rates across all four victim models, all four task variants, and both untargeted and targeted settings. On OpenVLA, the average failure rate rises sharply from 24.1% to 88.1% and 90.5% under untargeted and targeted attacks, respectively, while OpenVLA-OFT increases from 4.7% to 76.0% and 79.3%. Similar trends are observed on $\pi _ { 0 } ,$ whose average failure rate rises from 4.6% to 71.8% and 73.3%, and on $\pi _ { 0 . 5 } .$ , where it increases from 2.8% to 69.3% and 71.2%. For targeted attacks, following the evaluation setting in [37], Tab. 4 further shows that the attacked actions remain close to the prescribed target actions, with average L1 distances of 0.0176, 0.0237, 0.0305, and 0.0372 on OpenVLA, OpenVLA-OFT, $\pi _ { 0 } ,$ and $\pi _ { 0 . 5 } ,$ respectively. The most pronounced increase appears on the Spatial task, where Tex3D raises the failure rate from 15.6% to 95.8%/96.7% on OpenVLA, and from 3.5% to 74.9%/75.9% on $\pi _ { 0 } .$ These results indicate that the strong attack performance does not come from simple noise injection or single-frame perturbation, but from jointly optimizing object-level adversarial textures with temporal consistency. This principled design enables Tex3D to produce stronger and more stable attacks than all baselines and ablations across diverse manipulation tasks. 
Cross-Model Transferability. Tab. 2 (bottom) reports cross-model transfer results to evaluate whether adversarial textures optimized on one VLA generalize to unseen victim models. Strong transfer appears in both same- and cross-family settings in practice. Within the same family, attacks transfer strongly between OpenVLA and OpenVLA-OFT, yielding 61.5%–70.6% failure rates in the Open-VLA → OpenVLA-OFT direction and 70.2%–75.7% in the reverse direction; transfer between $\pi _ { 0 }$ and $\pi _ { 0 . 5 }$ is likewise consistently high, at 49.2%–58.4% and 55.8%–63.7%, respectively. Across families, attacks crafted on OpenVLA transfer effectively to $\pi_0$ and $\pi_{0.5}$, reaching 34.5%–44.1% and 27.8%–36.5%, while textures optimized on $\pi_0$ remain effective on OpenVLA and OpenVLA-OFT, achieving 44.8%–61.5% and 37.2%–48.4%, respectively. Taken together, these results show that Tex3D learns robust object-level adversarial patterns rather than overfitting to a specific source model, making the attack also effective under diverse source-target model pairings. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-08/14fc23a2-8e19-421f-87db-25b6a67475d5/0c65ba4a051a3754427a4eeae80e0efa337e44a9b7f24b362732bdfa60ae0e83.jpg)



Figure 5: Physical-world qualitative comparison. The first row: clean samples, the second row :results under 2D patchbased attacks, and the third row: results under Tex3D.


Ruling Out Non-adversarial Factors. To verify the elevated failure rates in Tab. 1 arise from adversarial optimization rather than color variation or rendering artifacts, two control studies are performed. First, Fig. 1 and Tab. 3 show that neither color changes nor random Gaussian perturbations cause comparably severe failures. For instance, on the OpenVLA Spatial task, these controls remain within 18.8%–24.6%, far below the 95.8%/96.7% achieved by Tex3D under untargeted/targeted attacks. Second, Tab. 2 (top) shows that the MuJoCo+nvdiffrast cross-rendering pipeline introduces negligible discrepancies, with at most 1.2 points difference across all four task suites. Together, these results confirm that the high failure rates are primarily induced by the adversarial textures optimized by Tex3D, rather than ordinary color shifts or rendering distortion. 


Table 4: L1 distance (↓) between the attacked action and target action across task suites on four victim VLA models.


<table><tr><td>Model</td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td>Avg.</td></tr><tr><td>OpenVLA</td><td>0.0137</td><td>0.0205</td><td>0.0185</td><td>0.0176</td><td>0.0176</td></tr><tr><td>OpenVLA-OFT</td><td>0.0214</td><td>0.0281</td><td>0.0241</td><td>0.0212</td><td>0.0237</td></tr><tr><td><eq>\pi_0</eq></td><td>0.0284</td><td>0.0357</td><td>0.0312</td><td>0.0268</td><td>0.0305</td></tr><tr><td><eq>\pi_{0.5}</eq></td><td>0.0349</td><td>0.0423</td><td>0.0386</td><td>0.0331</td><td>0.0372</td></tr></table>


Table 5: Representative trajectory-aware dynamics-guided frame weights along a sample manipulation trajectory.


<table><tr><td>Stage</td><td>T1 Initial</td><td>T2 Reach</td><td>T3 Grasp</td><td>T4 Transfer</td><td>T5 Put down</td></tr><tr><td>Step</td><td>10</td><td>50</td><td>100</td><td>150</td><td>195</td></tr><tr><td>Weight</td><td>0.20</td><td>0.37</td><td>0.90</td><td>0.32</td><td>0.20</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-08/14fc23a2-8e19-421f-87db-25b6a67475d5/c04b268e865d6df21741f521d41bc961d020c3fb19906d7ae74b60f2538098a6.jpg)



Figure 6: Impact of input-space defenses on Tex3D.


### 4.3 Additional Analysis

Robustness against 2D Patch Attacks. To further systematically examine whether object-level adversarial textures provide better robustness than conventional 2D patch-based attacks [37], additional comparisons are conducted in both digital and physical settings. Fig. 4 shows that 2D patch-based attacks are highly brittle under digital geometric variations: their failure rate drops sharply from 100% to 67.4% under camera-angle perturbations and to 63.4% under object rotations. In contrast, Tex3D consistently remains much more stable, staying around 80.8%–88.1% across all three digital variations. A similar advantage is preserved in the physical world, where Tex3D maintains substantially higher failure rates (66.8%–67.6%) than both 2D patch (40.8%–50.8%) and 2D patch+EoT (49.8%–56.6%). Fig. 5 further provides qualitative evidence that 2D patches are more sensitive to viewpoint changes and placement mismatch, whereas the object-level adversarial textures generated by Tex3D preserve their attack effect after physical deployment. In addition, the first row of Fig. 7 shows that Tex3D also achieves better visual quality, reflected by lower LPIPS than the 2D patch-based baseline. Overall, these results indicate that optimizing textures directly on the target object yields attacks that are not only more transferable across geometric variations, but also more robust and consistently visually natural in real-world deployment scenarios, even under diverse and challenging real-world environments. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-08/14fc23a2-8e19-421f-87db-25b6a67475d5/82bb3b0154aa55f3b1eff66a3c81d5c8d9e915da2fe4a4f5dc6aba81f4770acd.jpg)



Figure 7: First column: qualitative perceptual-similarity direct comparison between Tex3D and 2D patch-based attacks. Second and third columns: rendering-consistency comparison at the same optimization step between the original renderer (MuJoCo) and our dual-renderer pipeline (Ours).


Robustness Against Common Defenses. Beyond the above geometric variations such as rotations and viewpoint changes, we further study the robustness of Tex3D against common defenses [7, 41, 45]. Fig. 6 shows that standard input-space defenses, including JPEG compression, additive noise, median blur, and bitdepth reduction, have little effect on mitigating our attack. Across all defense settings, the adversarial task failure rate under Tex3D remains nearly unchanged, staying around 86.6%–87.3%. These results suggest that Tex3D preserves strong attack effectiveness even under common defenses, further demonstrating its robustness. 

Rendering Consistency and Attack Trajectory. To further empirically verify that the proposed attack does not rely on renderer mismatch, the second and third columns of Fig. 7 compare samestep renderings from the original MuJoCo pipeline and our dualrenderer pipeline. The resulting SSIM scores are consistently close to 1, indicating that the dual-renderer composition remains highly consistent with the original simulator rendering while still enabling gradient-based optimization. At the same time, Fig. 3 shows that these visually consistent renderings can nevertheless induce substantial behavioral deviations over time: compared with the clean rollout, the adversarial trajectory gradually drifts away from the intended manipulation path and eventually leads to clear task failure. Together, these results show that the effectiveness of Tex3D comes from adversarial texture optimization rather than rendering artifacts, while our dual-renderer design preserves simulator fidelity and still produces consistently strong trajectory-level attack effects. 

Representative Keyframe Weights. To further illustrate how TAAO allocates optimization effort over time, Tab. 5 reports representative frame weights along a manipulation trajectory. The weights remain relatively low at stable stages such as initialization and put-down, become moderately larger during reaching and transfer, and peak sharply at the particularly critical grasping stage. In this example, grasping receives the highest weight, indicating that TAAO concentrates adversarial optimization on the most decision-sensitive moment of the task. This behavior is consistent with the intuition that errors near grasp formation are more likely to propagate to subsequent stages and cause final task failure. An ablation study on keyframe inversion is provided in Appendix C. 


Table 6: Ablation study on FBD and TAAO components of Tex3D. Fail. Rate: Task Failure Rate on OpenVLA (%, ↑); Time: wall-clock time per gradient optimization step (s, ↓).


<table><tr><td colspan="3">FBD</td><td colspan="3">TAAO</td><td rowspan="2">Fail. Rate (%)</td><td rowspan="2">Time (s)</td></tr><tr><td>MVP</td><td>Light.</td><td>Decouple</td><td>Rand.</td><td>Unif.</td><td>Dyn.</td></tr><tr><td colspan="8">FBD Component Ablation</td></tr><tr><td>-</td><td>√</td><td>√</td><td>-</td><td>-</td><td>√</td><td>65.8</td><td>~7.2</td></tr><tr><td>√</td><td>-</td><td>√</td><td>-</td><td>-</td><td>√</td><td>76.8</td><td>~7.2</td></tr><tr><td>√</td><td>√</td><td>-</td><td>-</td><td>-</td><td>√</td><td>84.6</td><td>~24.8</td></tr><tr><td colspan="8">TAAO Weighting Strategy Ablation</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>-</td><td>-</td><td>73.7</td><td>~7.2</td></tr><tr><td>√</td><td>√</td><td>√</td><td>-</td><td>√</td><td>-</td><td>82.1</td><td>~7.2</td></tr><tr><td>√</td><td>√</td><td>√</td><td>-</td><td>-</td><td>√</td><td>88.1</td><td>~7.2</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-08/14fc23a2-8e19-421f-87db-25b6a67475d5/d67732f39479d9c59f73b9e6fdabc0c5d2cafee17945d8696948cadcbcc29c15.jpg)



Figure 8: Tex3D performance across four perturbation levels. Each subplot shows task failure rates (%) from L0 to L3 settings, where L0 denotes the naturalness-constrained setting and L1–L3 progressively increase the perturbation budget.


### 4.4 Ablation Studies

Effect of FBD and TAAO Design. Tab. 6 studies which components drive the effectiveness and efficiency of Tex3D. Removing Decouple, i.e., rendering the whole scene in nvdiffrast instead of separating foreground and background, reduces task failure rate from 88.1% to 84.6% and increases per-step optimization time from ∼7.2s to ∼24.8s. Removing MVP or Lighting further lowers the task failure rate to 65.8% and 76.8%, respectively, confirming the importance of cross-renderer alignment and lighting modeling. On the TAAO side, dynamics-guided weighting achieves the highest task failure rate, outperforming random weighting (73.7%) and uniform weighting (82.1%). Overall, dual-renderer decoupling provides a favorable effectiveness-efficiency trade-off, while dynamics-guided weighting is the most effective choice for trajectory-aware optimization,consistently yielding better performance across tasks. 

Effect of Perturbation Level. To study how perturbation strength affects attack performance, we vary the perturbation level from L0 to L3 and report both white-box and cross-model transfer results in Fig. 8. Fig. 9 further provides a qualitative comparison of adversarial textures under different budgets. Here, L0 denotes the naturalness-constrained setting with $\varepsilon = 6 4 / 2 5 5$ and an additional MSE loss between adversarial and clean samples, while L1, L2, and L3 correspond to $\varepsilon$ = 16/255, 32/255, and 64/255, respectively. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-08/14fc23a2-8e19-421f-87db-25b6a67475d5/d4dd9f0839a1635bf5f53b2864fdd65f7b621e21f69844673fed3a07e3feee84.jpg)



Figure 9: The qualitative visual comparison under different perturbation budgets and the clean reference sample image.


Overall, all perturbation levels substantially increase task failure rates across all four LIBERO task suites for both victim models. A clear trend shows that stronger perturbations result in higher failure rates, especially from L1 to L3, in both white-box and transfer settings. Meanwhile, L0 yields slightly lower failure rates than unconstrained levels, indicating a trade-off between stealthiness and attack strength. Despite this, L0 remains highly effective, suggesting that visually natural perturbations can still induce severe failures. These results demonstrate that the proposed attack is robust across a wide range of perturbation budgets, with stronger perturbations further increasing failure rates and improving transferability. 

## 5 Discussion and Conclusion

Figs. 8 and 9 reveal a noteworthy phenomenon: even nearly imperceptible perturbations under the L0 and L1 settings can already cause sharp increases in VLA failure rates, with OpenVLA still exceeding 90% on the Spatial suite under strong attack. This fragile robustness may stem from the current data regime of VLA systems, as existing robot training corpora and evaluation benchmarks are still largely built in relatively clean and idealized environments [9] with limited coverage of subtle appearance shifts and adversarial visual variations. As a result, even small perturbations can induce substantial distribution shifts in the perception module. Our findings therefore highlight the urgent need to rethink VLA training for meaningful robustness against adversarial attacks. Such robustness may potentially be improved in two directions: (1) incorporating more diverse and realistic robotic scenes at the visual level; and (2) enforcing stronger structural constraints over joint dependencies at the action level to filter out implausible or harmful actions. 

In summary, this paper presents Tex3D, the first framework for end-to-end optimization of adversarial 3D textures directly in the VLA simulation environment. By introducing FBD, we establish a differentiable optimization path from VLA objectives back to object appearance while preserving rendering consistency with the original simulator. Built on this, TAAO emphasizes behaviorally critical frames to sustain attack effectiveness over long-horizon manipulation trajectories. Extensive experiments in both simulation and physical settings across multiple VLA models and LIBERO task suites show that Tex3D consistently achieves high task failure rates under both untargeted and targeted settings, transfers effectively across model architectures, and remains robust under geometric variations and common defenses. Taken together, these results show that object-level adversarial textures constitute a practical and highly effective attack surface for embodied agents,and underscore the need for more systematic robustness evaluation and more comprehensive training under persistent, physically grounded adversarial interactions in future real-world deployment settings and safety-critical scenarios, particularly in complex environments. 