"""OpenVLA 对抗纹理入口的纯配置 schema 与类型收窄。

该模块不导入 LIBERO、OpenVLA、nvdiffrast 或 wandb。Draccus 可以独立解码
``GenerateConfig``，单元测试也不会因为导入实验入口而修改 ``sys.path``。
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, TypeAlias, Union, cast


TextureParameterizationKind: TypeAlias = Literal[
    "legacy_vertex",
    "geometry_vertex",
    "fixed_support",
    "spectral",
]
FeatureObjectiveKind: TypeAlias = Literal[
    "last_hidden",
    "siglip_patch",
]
FeatureViewModeKind: TypeAlias = Literal[
    "primary",
    "primary_wrist",
]
SUPPORTED_TEXTURE_PARAMETERIZATIONS: frozenset[str] = frozenset(
    {"legacy_vertex", "geometry_vertex", "fixed_support", "spectral"}
)
SUPPORTED_FEATURE_OBJECTIVES: frozenset[str] = frozenset(
    {"last_hidden", "siglip_patch"}
)
SUPPORTED_FEATURE_VIEW_MODES: frozenset[str] = frozenset(
    {"primary", "primary_wrist"}
)


def resolve_texture_parameterization(
    raw_value: str,
) -> TextureParameterizationKind:
    """把 Draccus 可解码的字符串收窄为 renderer 强类型参数。

    当前环境中的 Draccus 无法直接解码 ``typing.Literal``，因此 CLI dataclass
    先接收 ``str``。所有合法值仍在实验入口统一校验，任意字符串不会继续传播
    到 Texture Parameterization 核心数据流。
    """
    if raw_value not in SUPPORTED_TEXTURE_PARAMETERIZATIONS:
        raise ValueError(
            f"未知纹理参数化 {raw_value!r}；可选值为 "
            f"{sorted(SUPPORTED_TEXTURE_PARAMETERIZATIONS)}"
        )
    return cast(TextureParameterizationKind, raw_value)


def resolve_feature_objective(raw_value: str) -> FeatureObjectiveKind:
    """把 Draccus 字符串收窄为内部 feature objective 类型。"""
    if raw_value not in SUPPORTED_FEATURE_OBJECTIVES:
        raise ValueError(
            f"未知 feature objective {raw_value!r}；可选值为 "
            f"{sorted(SUPPORTED_FEATURE_OBJECTIVES)}"
        )
    return cast(FeatureObjectiveKind, raw_value)


def resolve_feature_view_mode(raw_value: str) -> FeatureViewModeKind:
    """把 CLI feature 视角字符串收窄为内部强类型。"""
    if raw_value not in SUPPORTED_FEATURE_VIEW_MODES:
        raise ValueError(
            f"未知 feature view mode {raw_value!r}；可选值为 "
            f"{sorted(SUPPORTED_FEATURE_VIEW_MODES)}"
        )
    return cast(FeatureViewModeKind, raw_value)


@dataclass
class GenerateConfig:
    """OpenVLA LIBERO 对抗纹理训练与评估的命令行配置。

    资产相关数据流：

    ``object_name -> OBJECT_ASSETS -> XML / mesh / texture / search keywords``
    """

    model_family: str = "openvla"
    pretrained_checkpoint: Union[
        str,
        Path,
    ] = "./openvla-7b-finetuned-libero-spatial"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True

    object_name: str = "akita_black_bowl"
    override_mesh_path: Optional[str] = None
    override_texture_path: Optional[str] = None
    override_xml_path: Optional[str] = None

    task_suite_name: str = "libero_spatial"
    task_id: Optional[int] = 7
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    enable_attack: bool = True
    attack_iters: int = 10
    attack_lr: float = 0.05
    attack_surface_step: float = 2.0 / 255.0
    attack_epsilon: float = 128.0 / 255.0
    # Draccus 0.x 不支持直接解码 typing.Literal；eval_libero 会立即校验并收窄
    # 为 TextureParameterizationKind。
    texture_parameterization: str = "legacy_vertex"
    # ``fixed_support`` 唯一允许消费的不可变 Production Support NPZ。该路径
    # 只冻结参数空间；rho_nat/lambda_spec 校准通过前正式训练仍被入口阻止。
    fixed_support_path: Optional[str] = None
    # 固定 Support 的可学习空间仍不使用 ``spectral_basis_path``。以下四个
    # artifact 只定义 Spectral Naturalness Guard，并由正式 source trainer
    # 与已通过的两步 smoke 共同绑定。
    spectral_naturalness_basis_path: Optional[str] = None
    rho_nat_calibration_path: Optional[str] = None
    spectral_guard_manifest_path: Optional[str] = None
    fixed_support_training_smoke_manifest_path: Optional[str] = None
    # 仅用于5000轮训练已完成但后置evaluator/rollout中止的恢复流程。提供后主
    # 入口必须独立复核该正式manifest并直接使用其原bake，禁止再次训练。
    fixed_support_formal_training_manifest_path: Optional[str] = None
    # 正式 GPU 运行必须显式绑定即将执行的完整 Git commit；上游校准和 smoke
    # 可以来自较早 commit，但其文件 SHA-256 必须逐项匹配。
    code_commit: Optional[str] = None
    spectral_basis_path: Optional[str] = None
    spectral_basis_count: int = 128
    num_frames_to_attack: int = 20
    num_train_init_states: int = 10
    train_init_state_ids: Optional[str] = None
    eval_init_state_ids: Optional[str] = None
    require_disjoint_init_states: bool = True
    train_frames_per_state: int = 1
    alpha_action: float = 1.0
    alpha_feature: float = 10.0
    # 开启后，在每轮 batch 的谱系数空间中限制“加权 Feature 梯度 / 加权
    # Action 梯度”的 L2 范数比。默认关闭以保持已有实验行为。
    gradient_norm_protection_enabled: bool = False
    feature_gradient_norm_ratio_limit: float = 1.0
    # 保持 last_hidden 为默认值以复现现有行为；siglip_patch 直接攻击共享的
    # SigLIP patch features。Draccus 不支持 Literal，入口负责校验和收窄。
    feature_objective: str = "last_hidden"
    # primary 保留历史单视角行为；primary_wrist 仅扩展 Shared-SigLIP Feature
    # loss，Action loss 仍只使用 OpenVLA 的主视角。
    feature_view_mode: str = "primary"
    # Source-only 谱基梯度审计复用正常采帧与 loss 路径，但不更新纹理。
    # ``audit_only`` 启用时，入口在产物保存后跳过攻击训练和 held-out rollout。
    spectral_gradient_audit_enabled: bool = False
    spectral_gradient_audit_only: bool = False
    spectral_gradient_audit_top_k: int = 128
    # 缺省在零 Surface Delta 审计；提供谱系数 .pt 时只在该固定参考点求梯度，
    # 不更新参数。该字段不能替代迁移评估中的 Active Texture PNG。
    spectral_gradient_audit_reference_path: Optional[str] = None
    # 固定源 OpenVLA 训练状态上的动作响应诊断。该模式加载一份已训练谱系数，
    # 对比 clean/adv 的 teacher-forced margin、生成 token 与连续 action；只做
    # 前向并自动跳过纹理优化和 held-out rollout。
    source_action_response_audit_enabled: bool = False
    source_action_response_reference_path: Optional[str] = None
    frame_collect_with_policy: bool = False
    collect_grasp_frames: bool = False
    grasp_pre_frames: int = 40
    grasp_post_frames: int = 0
    grasp_max_steps: int = 400
    grasp_qpos_threshold: float = 0.02

    photometric_calib_frames: int = 5

    live_test_enabled: bool = True
    live_test_every_n_iters: int = 20
    live_test_resolution: int = 256
    live_test_max_steps: int = 300

    save_attack_artifacts: bool = True
    load_texture_path: Optional[str] = None
    local_log_dir: str = "./experiments/logs"

    use_wandb: bool = False
    wandb_project: str = "openvla_attack"
    wandb_entity: str = "user"

    seed: int = 7
    run_id_note: Optional[str] = None
    unnorm_key: Optional[str] = None


def validate_fixed_support_config(
    cfg: GenerateConfig,
    *,
    texture_parameterization: TextureParameterizationKind,
) -> None:
    """保护 Production Support 路径与其他参数化模式的互斥边界。"""

    if texture_parameterization == "fixed_support":
        if cfg.fixed_support_path is None:
            raise ValueError(
                "fixed_support 参数化必须提供 fixed_support_path"
            )
        if cfg.spectral_basis_path is not None:
            raise ValueError(
                "fixed_support 参数化不能把 spectral_basis_path 当作可学习参数空间"
            )
        required_artifacts = {
            "spectral_naturalness_basis_path": (
                cfg.spectral_naturalness_basis_path
            ),
            "rho_nat_calibration_path": cfg.rho_nat_calibration_path,
            "spectral_guard_manifest_path": cfg.spectral_guard_manifest_path,
            "fixed_support_training_smoke_manifest_path": (
                cfg.fixed_support_training_smoke_manifest_path
            ),
        }
        missing = [
            name for name, value in required_artifacts.items() if value is None
        ]
        if missing:
            raise ValueError(
                "fixed_support 正式训练缺少冻结artifact: "
                + ", ".join(missing)
            )
        if cfg.code_commit is None or len(cfg.code_commit) != 40 or any(
            character not in "0123456789abcdef"
            for character in cfg.code_commit
        ):
            raise ValueError("fixed_support 正式训练要求40位小写code_commit")
        return
    fixed_support_only_values = {
        "fixed_support_path": cfg.fixed_support_path,
        "spectral_naturalness_basis_path": (
            cfg.spectral_naturalness_basis_path
        ),
        "rho_nat_calibration_path": cfg.rho_nat_calibration_path,
        "spectral_guard_manifest_path": cfg.spectral_guard_manifest_path,
        "fixed_support_training_smoke_manifest_path": (
            cfg.fixed_support_training_smoke_manifest_path
        ),
        "fixed_support_formal_training_manifest_path": (
            cfg.fixed_support_formal_training_manifest_path
        ),
        "code_commit": cfg.code_commit,
    }
    configured = [
        name
        for name, value in fixed_support_only_values.items()
        if value is not None
    ]
    if configured:
        raise ValueError(
            "Fixed-Support正式训练字段只能与 "
            "texture_parameterization='fixed_support' 同用: "
            + ", ".join(configured)
        )


def validate_formal_fixed_support_experiment(
    cfg: GenerateConfig,
    *,
    texture_parameterization: TextureParameterizationKind,
) -> None:
    """冻结第一轮正式 source candidate 的方法与数据边界。

    该校验只在 ``fixed_support`` 路径生效。所有 legacy Feature、动态保护、
    audit、预训练纹理加载和 live-test 开关均被拒绝，防止 CLI 中看似无害的旧
    默认值悄悄改变已通过 smoke 的 Action+Spectral 数值语义。
    """

    if texture_parameterization != "fixed_support":
        return
    if not cfg.enable_attack:
        raise ValueError("正式Fixed-Support source训练要求enable_attack=True")
    frozen_values = {
        "model_family": (cfg.model_family, "openvla"),
        "object_name": (cfg.object_name, "akita_black_bowl"),
        "task_suite_name": (cfg.task_suite_name, "libero_spatial"),
        "task_id": (cfg.task_id, 0),
        "attack_iters": (cfg.attack_iters, 5000),
        "num_train_init_states": (cfg.num_train_init_states, 10),
        "num_trials_per_task": (cfg.num_trials_per_task, 10),
        "train_init_state_ids": (cfg.train_init_state_ids, "0-9"),
        "eval_init_state_ids": (cfg.eval_init_state_ids, "10-19"),
        "center_crop": (cfg.center_crop, True),
        "alpha_action": (cfg.alpha_action, 1.0),
        "alpha_feature": (cfg.alpha_feature, 0.0),
        "feature_view_mode": (cfg.feature_view_mode, "primary"),
        "unnorm_key": (cfg.unnorm_key, "libero_spatial_no_noops"),
        "live_test_enabled": (cfg.live_test_enabled, False),
        "load_in_8bit": (cfg.load_in_8bit, False),
        "load_in_4bit": (cfg.load_in_4bit, False),
        "override_mesh_path": (cfg.override_mesh_path, None),
        "override_texture_path": (cfg.override_texture_path, None),
        "override_xml_path": (cfg.override_xml_path, None),
        "save_attack_artifacts": (cfg.save_attack_artifacts, True),
    }
    mismatches = [
        f"{name}={actual!r}（要求{expected!r}）"
        for name, (actual, expected) in frozen_values.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError(
            "正式Fixed-Support source配置偏离冻结候选: "
            + "; ".join(mismatches)
        )
    if not cfg.require_disjoint_init_states:
        raise ValueError("正式source Gate要求train/eval states互斥")
    forbidden = {
        "load_texture_path": cfg.load_texture_path is not None,
        "gradient_norm_protection_enabled": (
            cfg.gradient_norm_protection_enabled
        ),
        "spectral_gradient_audit_enabled": cfg.spectral_gradient_audit_enabled,
        "spectral_gradient_audit_only": cfg.spectral_gradient_audit_only,
        "source_action_response_audit_enabled": (
            cfg.source_action_response_audit_enabled
        ),
        "frame_collect_with_policy": cfg.frame_collect_with_policy,
        "collect_grasp_frames": cfg.collect_grasp_frames,
    }
    enabled = [name for name, value in forbidden.items() if value]
    if enabled:
        raise ValueError(
            "正式Fixed-Support source路径禁止legacy/诊断开关: "
            + ", ".join(enabled)
        )
    if not math.isclose(
        cfg.attack_surface_step,
        2.0 / 255.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("正式source训练冻结Surface Step=2/255")
    if not math.isclose(
        cfg.attack_epsilon,
        128.0 / 255.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("正式source训练冻结Surface-Linf预算=128/255")


def validate_gradient_norm_protection(
    cfg: GenerateConfig,
    *,
    texture_parameterization: TextureParameterizationKind,
    feature_objective: FeatureObjectiveKind,
    feature_view_mode: FeatureViewModeKind,
) -> None:
    """校验第一版动态梯度范数保护的实验边界。

    当前功能只用于验证 K=256 双视角 Shared-SigLIP 的诊断结论。把范围限制在
    Spectral + Primary/Wrist，可以避免用户误以为 Geometry/Legacy 或历史
    last-hidden 基线也已经获得相同实验语义。
    """
    if not cfg.gradient_norm_protection_enabled:
        return
    if texture_parameterization != "spectral":
        raise ValueError("动态梯度范数保护当前只支持 spectral 参数化")
    if feature_objective != "siglip_patch":
        raise ValueError(
            "动态梯度范数保护当前要求 feature_objective='siglip_patch'"
        )
    if feature_view_mode != "primary_wrist":
        raise ValueError(
            "动态梯度范数保护当前要求 feature_view_mode='primary_wrist'"
        )
    if (
        not math.isfinite(cfg.feature_gradient_norm_ratio_limit)
        or cfg.feature_gradient_norm_ratio_limit <= 0.0
    ):
        raise ValueError("feature_gradient_norm_ratio_limit 必须为有限正数")
    if not math.isfinite(cfg.alpha_action) or cfg.alpha_action <= 0.0:
        raise ValueError("动态梯度范数保护要求 alpha_action 为有限正数")
    if not math.isfinite(cfg.alpha_feature) or cfg.alpha_feature <= 0.0:
        raise ValueError("动态梯度范数保护要求 alpha_feature 为有限正数")


def validate_source_action_response_audit(
    cfg: GenerateConfig,
    *,
    texture_parameterization: TextureParameterizationKind,
    feature_objective: FeatureObjectiveKind,
    feature_view_mode: FeatureViewModeKind,
) -> None:
    """校验源模型动作响应诊断的可比性边界。"""
    if (
        cfg.source_action_response_reference_path is not None
        and not cfg.source_action_response_audit_enabled
    ):
        raise ValueError(
            "source_action_response_reference_path 要求同时启用 "
            "source_action_response_audit_enabled"
        )
    if not cfg.source_action_response_audit_enabled:
        return
    if not cfg.enable_attack:
        raise ValueError("源动作响应诊断要求 enable_attack=True")
    if texture_parameterization != "spectral":
        raise ValueError("源动作响应诊断要求 texture_parameterization='spectral'")
    if feature_objective != "siglip_patch":
        raise ValueError("源动作响应诊断要求 feature_objective='siglip_patch'")
    if feature_view_mode != "primary_wrist":
        raise ValueError(
            "源动作响应诊断要求 feature_view_mode='primary_wrist'，"
            "以复现共享纹理的全部主视角实例"
        )
    if cfg.source_action_response_reference_path is None:
        raise ValueError("源动作响应诊断必须提供参考谱系数 .pt")
    if cfg.load_texture_path is not None:
        raise ValueError(
            "源动作响应诊断使用独立 reference path，不能同时设置 "
            "load_texture_path"
        )
    if cfg.spectral_gradient_audit_enabled:
        raise ValueError("源动作响应诊断不能与谱基梯度审计在同一次运行中启用")
