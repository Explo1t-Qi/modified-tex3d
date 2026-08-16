"""OpenVLA attack CLI 配置与强类型边界的回归测试。"""

from __future__ import annotations

import pytest
from draccus.parsers import decoding

from openvla.experiments.robot.libero.openvla_attack.configuration import (
    GenerateConfig,
    resolve_feature_objective,
    resolve_feature_view_mode,
    resolve_formal_training_variant,
    resolve_texture_parameterization,
    validate_fixed_support_config,
    validate_formal_fixed_support_experiment,
    validate_gradient_norm_protection,
    validate_source_action_response_audit,
)


def test_draccus_decodes_spectral_parameterization_from_cli_string() -> None:
    """复现用户命令的真实 Draccus dataclass decode seam。"""
    config = decoding.decode(
        GenerateConfig,
        {"texture_parameterization": "spectral"},
    )

    assert config.texture_parameterization == "spectral"
    assert resolve_texture_parameterization(
        config.texture_parameterization
    ) == "spectral"


def test_runtime_boundary_rejects_unknown_parameterization() -> None:
    with pytest.raises(ValueError, match="未知纹理参数化"):
        resolve_texture_parameterization("frequency_magic")


def test_draccus_decodes_frozen_support_path_and_runtime_narrows_kind() -> None:
    config = decoding.decode(
        GenerateConfig,
        {
            "texture_parameterization": "fixed_support",
            "fixed_support_path": "/tmp/production_fixed_support.npz",
            "spectral_naturalness_basis_path": "/tmp/basis.npz",
            "rho_nat_calibration_path": "/tmp/rho.npz",
            "spectral_guard_manifest_path": "/tmp/guard.json",
            "fixed_support_training_smoke_manifest_path": "/tmp/smoke.json",
            "code_commit": "a" * 40,
        },
    )

    assert resolve_texture_parameterization(
        config.texture_parameterization
    ) == "fixed_support"
    validate_fixed_support_config(
        config,
        texture_parameterization="fixed_support",
    )


def test_draccus_decodes_explicit_action_only_control_variant() -> None:
    config = decoding.decode(
        GenerateConfig,
        {"fixed_support_formal_training_variant": "action_only_control"},
    )

    assert resolve_formal_training_variant(
        config.fixed_support_formal_training_variant
    ) == "action_only_control"
    with pytest.raises(ValueError, match="正式训练变体"):
        resolve_formal_training_variant("legacy_action_feature")


def test_action_only_kappa_variant_freezes_preregistered_margin() -> None:
    config = decoding.decode(
        GenerateConfig,
        {
            "fixed_support_formal_training_variant": "action_only_kappa",
            "action_margin_kappa": 4.375,
        },
    )

    assert resolve_formal_training_variant(
        config.fixed_support_formal_training_variant
    ) == "action_only_kappa"

    config.action_margin_kappa = 4.0
    with pytest.raises(ValueError, match="4.375"):
        validate_formal_fixed_support_experiment(
            config,
            texture_parameterization="fixed_support",
        )


def test_action_only_kappa_cli_requires_accepted_smoke_for_formal_training() -> None:
    config = GenerateConfig(
        texture_parameterization="fixed_support",
        fixed_support_path="/tmp/support.npz",
        spectral_naturalness_basis_path="/tmp/naturalness.npz",
        rho_nat_calibration_path="/tmp/rho.npz",
        spectral_guard_manifest_path="/tmp/guard.json",
        fixed_support_training_smoke_manifest_path="/tmp/old-smoke.json",
        code_commit="a" * 40,
        task_id=0,
        attack_iters=2,
        num_trials_per_task=10,
        train_init_state_ids="0-9",
        eval_init_state_ids="10-19",
        alpha_feature=0.0,
        live_test_enabled=False,
        unnorm_key="libero_spatial_no_noops",
        fixed_support_formal_training_variant="action_only_kappa",
        action_margin_kappa=4.375,
        fixed_support_kappa_smoke_enabled=True,
    )

    validate_formal_fixed_support_experiment(
        config,
        texture_parameterization="fixed_support",
    )

    config.fixed_support_kappa_smoke_enabled = False
    config.attack_iters = 5000
    with pytest.raises(ValueError, match="κ smoke manifest"):
        validate_formal_fixed_support_experiment(
            config,
            texture_parameterization="fixed_support",
        )

    config.action_margin_kappa_smoke_manifest_path = "/tmp/kappa-smoke.json"
    validate_formal_fixed_support_experiment(
        config,
        texture_parameterization="fixed_support",
    )

    config.fixed_support_formal_training_variant = "action_only_control"
    config.action_margin_kappa = 0.0
    with pytest.raises(ValueError, match="只允许action_only_kappa"):
        validate_formal_fixed_support_experiment(
            config,
            texture_parameterization="fixed_support",
        )


def test_fixed_support_config_requires_exclusive_artifact_path() -> None:
    with pytest.raises(ValueError, match="必须提供"):
        validate_fixed_support_config(
            GenerateConfig(texture_parameterization="fixed_support"),
            texture_parameterization="fixed_support",
        )
    with pytest.raises(ValueError, match="只能与"):
        validate_fixed_support_config(
            GenerateConfig(fixed_support_path="/tmp/support.npz"),
            texture_parameterization="legacy_vertex",
        )
    with pytest.raises(ValueError, match="不能把 spectral_basis_path"):
        validate_fixed_support_config(
            GenerateConfig(
                texture_parameterization="fixed_support",
                fixed_support_path="/tmp/support.npz",
                spectral_basis_path="/tmp/basis.npz",
                spectral_naturalness_basis_path="/tmp/naturalness.npz",
                rho_nat_calibration_path="/tmp/rho.npz",
                spectral_guard_manifest_path="/tmp/guard.json",
                fixed_support_training_smoke_manifest_path="/tmp/smoke.json",
                code_commit="a" * 40,
            ),
            texture_parameterization="fixed_support",
        )


def test_fixed_support_config_requires_all_calibrated_artifacts_and_commit() -> None:
    with pytest.raises(ValueError, match="缺少冻结artifact"):
        validate_fixed_support_config(
            GenerateConfig(
                texture_parameterization="fixed_support",
                fixed_support_path="/tmp/support.npz",
            ),
            texture_parameterization="fixed_support",
        )


def test_formal_fixed_support_experiment_freezes_candidate_and_state_split() -> None:
    config = GenerateConfig(
        texture_parameterization="fixed_support",
        fixed_support_path="/tmp/support.npz",
        spectral_naturalness_basis_path="/tmp/naturalness.npz",
        rho_nat_calibration_path="/tmp/rho.npz",
        spectral_guard_manifest_path="/tmp/guard.json",
        fixed_support_training_smoke_manifest_path="/tmp/smoke.json",
        code_commit="a" * 40,
        task_id=0,
        attack_iters=5000,
        num_trials_per_task=10,
        train_init_state_ids="0-9",
        eval_init_state_ids="10-19",
        alpha_feature=0.0,
        live_test_enabled=False,
        unnorm_key="libero_spatial_no_noops",
        fixed_support_formal_training_manifest_path="/tmp/formal.json",
    )

    validate_formal_fixed_support_experiment(
        config,
        texture_parameterization="fixed_support",
    )
    assert config.fixed_support_formal_training_manifest_path == (
        "/tmp/formal.json"
    )

    config.eval_init_state_ids = "10-18"
    with pytest.raises(ValueError, match="偏离冻结候选"):
        validate_formal_fixed_support_experiment(
            config,
            texture_parameterization="fixed_support",
        )
    with pytest.raises(ValueError, match="40位小写code_commit"):
        validate_fixed_support_config(
            GenerateConfig(
                texture_parameterization="fixed_support",
                fixed_support_path="/tmp/support.npz",
                spectral_naturalness_basis_path="/tmp/naturalness.npz",
                rho_nat_calibration_path="/tmp/rho.npz",
                spectral_guard_manifest_path="/tmp/guard.json",
                fixed_support_training_smoke_manifest_path="/tmp/smoke.json",
                code_commit="not-a-commit",
            ),
            texture_parameterization="fixed_support",
        )

    config.fixed_support_formal_training_variant = "legacy_action_feature"
    with pytest.raises(ValueError, match="正式训练变体"):
        validate_formal_fixed_support_experiment(
            config,
            texture_parameterization="fixed_support",
        )


def test_draccus_decodes_and_runtime_narrows_siglip_objective() -> None:
    config = decoding.decode(
        GenerateConfig,
        {"feature_objective": "siglip_patch"},
    )

    assert config.feature_objective == "siglip_patch"
    assert resolve_feature_objective(
        config.feature_objective
    ) == "siglip_patch"


def test_runtime_boundary_rejects_unknown_feature_objective() -> None:
    with pytest.raises(ValueError, match="未知 feature objective"):
        resolve_feature_objective("target_model_magic")


def test_draccus_decodes_and_runtime_narrows_dual_feature_views() -> None:
    config = decoding.decode(
        GenerateConfig,
        {"feature_view_mode": "primary_wrist"},
    )

    assert config.feature_view_mode == "primary_wrist"
    assert resolve_feature_view_mode(config.feature_view_mode) == "primary_wrist"


def test_runtime_boundary_rejects_unknown_feature_view_mode() -> None:
    with pytest.raises(ValueError, match="未知 feature view mode"):
        resolve_feature_view_mode("target_model_views")


def test_draccus_decodes_source_only_spectral_gradient_audit_fields() -> None:
    config = decoding.decode(
        GenerateConfig,
        {
            "spectral_gradient_audit_enabled": True,
            "spectral_gradient_audit_only": True,
            "spectral_gradient_audit_top_k": 128,
            "spectral_gradient_audit_reference_path": "/tmp/final.pt",
        },
    )

    assert config.spectral_gradient_audit_enabled is True
    assert config.spectral_gradient_audit_only is True
    assert config.spectral_gradient_audit_top_k == 128
    assert config.spectral_gradient_audit_reference_path == "/tmp/final.pt"


def test_draccus_decodes_dynamic_gradient_norm_protection_fields() -> None:
    config = decoding.decode(
        GenerateConfig,
        {
            "gradient_norm_protection_enabled": True,
            "feature_gradient_norm_ratio_limit": 1.0,
        },
    )

    assert config.gradient_norm_protection_enabled is True
    assert config.feature_gradient_norm_ratio_limit == 1.0


def test_gradient_norm_protection_accepts_only_first_candidate_scope() -> None:
    config = GenerateConfig(
        gradient_norm_protection_enabled=True,
        feature_gradient_norm_ratio_limit=1.0,
        alpha_action=0.1,
        alpha_feature=4.0,
    )

    validate_gradient_norm_protection(
        config,
        texture_parameterization="spectral",
        feature_objective="siglip_patch",
        feature_view_mode="primary_wrist",
    )
    with pytest.raises(ValueError, match="primary_wrist"):
        validate_gradient_norm_protection(
            config,
            texture_parameterization="spectral",
            feature_objective="siglip_patch",
            feature_view_mode="primary",
        )


def test_gradient_norm_protection_rejects_nonpositive_ratio_limit() -> None:
    config = GenerateConfig(
        gradient_norm_protection_enabled=True,
        feature_gradient_norm_ratio_limit=0.0,
        alpha_action=0.1,
        alpha_feature=4.0,
    )

    with pytest.raises(ValueError, match="必须为有限正数"):
        validate_gradient_norm_protection(
            config,
            texture_parameterization="spectral",
            feature_objective="siglip_patch",
            feature_view_mode="primary_wrist",
        )


def test_draccus_decodes_source_action_response_fields() -> None:
    config = decoding.decode(
        GenerateConfig,
        {
            "source_action_response_audit_enabled": True,
            "source_action_response_reference_path": "/tmp/final.pt",
        },
    )

    assert config.source_action_response_audit_enabled is True
    assert config.source_action_response_reference_path == "/tmp/final.pt"


def test_source_action_response_requires_exact_dual_view_spectral_scope() -> None:
    config = GenerateConfig(
        enable_attack=True,
        source_action_response_audit_enabled=True,
        source_action_response_reference_path="/tmp/final.pt",
    )
    validate_source_action_response_audit(
        config,
        texture_parameterization="spectral",
        feature_objective="siglip_patch",
        feature_view_mode="primary_wrist",
    )

    with pytest.raises(ValueError, match="primary_wrist"):
        validate_source_action_response_audit(
            config,
            texture_parameterization="spectral",
            feature_objective="siglip_patch",
            feature_view_mode="primary",
        )


def test_source_action_response_rejects_missing_reference() -> None:
    config = GenerateConfig(
        enable_attack=True,
        source_action_response_audit_enabled=True,
    )

    with pytest.raises(ValueError, match="参考谱系数"):
        validate_source_action_response_audit(
            config,
            texture_parameterization="spectral",
            feature_objective="siglip_patch",
            feature_view_mode="primary_wrist",
        )
