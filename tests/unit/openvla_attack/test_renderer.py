"""OpenVLA 可微 renderer module 的无 GPU 接口测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.renderer import (
    AdversarialTextureLoadResult,
    DifferentiableRenderer,
    RendererEvidence,
    resolve_position_offset,
)
from openvla_attack.production_support import (
    PRODUCTION_SUPPORT_SCHEMA_VERSION,
    FrozenProductionSupport,
    ProductionSupportProvenance,
    array_sha256,
    file_sha256,
    write_production_support_artifact,
)
from openvla_attack.spectral_geometry import (
    load_obj_geometry,
    mesh_array_sha256,
    save_spectral_basis,
)


def test_renderer_module_can_be_imported_without_creating_cuda_context() -> None:
    """导入类定义不应提前构造 nvdiffrast CUDA context。"""
    assert issubclass(DifferentiableRenderer, nn.Module)


def test_renderer_unspecified_position_offset_is_exact_zero(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """未显式校准时不得给所有物体注入历史经验平移。

    该测试通过替换唯一的 CUDA-only 构造步骤，在 CPU 上覆盖真实
    ``DifferentiableRenderer.__init__`` 数据流。不存在的 mesh 会触发 renderer
    自带的薄盒 fallback；本测试只关心最终注册的 model-space offset。
    """
    monkeypatch.setattr(
        "openvla_attack.renderer.dr.RasterizeCudaContext",
        lambda: object(),
    )
    monkeypatch.setattr(
        DifferentiableRenderer,
        "_sample_uv_texture_at_vertices",
        lambda renderer: torch.zeros(
            (renderer.num_vertices, 3),
            dtype=torch.float32,
            device=renderer.device,
        ),
    )

    renderer = DifferentiableRenderer(
        mesh_path=tmp_path / "missing.obj",
        device=torch.device("cpu"),
        pos_offset=None,
    )

    torch.testing.assert_close(
        renderer.pos_offset,
        torch.zeros(3, dtype=torch.float32),
        rtol=0.0,
        atol=0.0,
    )


def test_renderer_preserves_explicit_position_offset() -> None:
    np.testing.assert_allclose(
        resolve_position_offset([0.02, 0.01, 0.025]),
        [0.02, 0.01, 0.025],
    )


@pytest.mark.parametrize(
    "invalid_offset",
    ([0.0, 0.0], [0.0, float("nan"), 0.0]),
)
def test_renderer_rejects_invalid_position_offset(
    invalid_offset: list[float],
) -> None:
    with pytest.raises(ValueError, match="三个有限"):
        resolve_position_offset(invalid_offset)


def _minimal_cpu_renderer() -> DifferentiableRenderer:
    """绕过 CUDA 构造，建立只覆盖纹理加载 interface 的 renderer。"""
    renderer = DifferentiableRenderer.__new__(DifferentiableRenderer)
    nn.Module.__init__(renderer)
    renderer.device = torch.device("cpu")
    renderer.epsilon = 0.5
    renderer.texture_parameterization_kind = "legacy_vertex"
    renderer.surface_parameterization = None
    renderer.adv_noise = nn.Parameter(
        torch.zeros((2, 3), dtype=torch.float32)
    )
    # orig_texture: float32 NHWC [1, texture_height, texture_width, 3]。
    renderer.orig_texture = torch.zeros(
        (1, 1, 2, 3),
        dtype=torch.float32,
    )
    renderer.orig_vertex_colors = torch.zeros(
        (2, 3),
        dtype=torch.float32,
    )
    return renderer


def test_renderer_loads_adversarial_parameter_tensor(
    tmp_path: Path,
) -> None:
    renderer: DifferentiableRenderer = _minimal_cpu_renderer()
    saved_noise = torch.tensor(
        [[0.2, -0.4, 0.0], [0.1, 0.3, -0.2]],
        dtype=torch.float32,
    )
    noise_path: Path = tmp_path / "noise.pt"
    torch.save(saved_noise, noise_path)

    result: AdversarialTextureLoadResult = (
        renderer.load_adversarial_texture(noise_path)
    )

    torch.testing.assert_close(renderer.adv_noise.detach(), saved_noise)
    expected_delta = torch.tanh(saved_noise) * renderer.epsilon
    assert result.source_kind == "parameter"
    assert result.max_absolute_delta == expected_delta.abs().max().item()
    assert result.nonzero_percentage == (
        (expected_delta.abs() > 1e-3).float().mean().item() * 100.0
    )


def test_renderer_converts_baked_png_to_vertex_noise(
    monkeypatch,
    tmp_path: Path,
) -> None:
    renderer: DifferentiableRenderer = _minimal_cpu_renderer()
    original_texture: torch.Tensor = renderer.orig_texture
    image_pixels = np.array(
        [[[255, 0, 128], [64, 192, 0]]],
        dtype=np.uint8,
    )
    texture_path: Path = tmp_path / "texture.png"
    Image.fromarray(image_pixels).save(texture_path)

    def sample_current_texture() -> torch.Tensor:
        # 模拟真实 UV sampler：每个测试顶点分别读取一个 texture pixel。
        return renderer.orig_texture.reshape(-1, 3)

    monkeypatch.setattr(
        renderer,
        "_sample_uv_texture_at_vertices",
        sample_current_texture,
    )
    result: AdversarialTextureLoadResult = (
        renderer.load_adversarial_texture(texture_path)
    )

    # loader 临时替换 orig_texture 完成采样后，必须恢复原 buffer。
    assert renderer.orig_texture is original_texture
    loaded_delta = torch.tanh(renderer.adv_noise.detach()) * renderer.epsilon
    expected_vertices = torch.from_numpy(
        image_pixels.astype(np.float32) / 255.0
    ).reshape(-1, 3)
    expected_delta = expected_vertices.clamp(
        -renderer.epsilon + 1e-6,
        renderer.epsilon - 1e-6,
    )
    torch.testing.assert_close(loaded_delta, expected_delta)
    assert result.source_kind == "baked_texture"
    assert result.max_absolute_delta == expected_delta.abs().max().item()


def test_geometry_vertex_renderer_builds_surface_parameterization(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """新 adapter 应使用 OBJ 几何顶点，而非无界 legacy 参数。"""
    mesh_path = tmp_path / "triangle.obj"
    mesh_path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "f 1 2 3",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "openvla_attack.renderer.dr.RasterizeCudaContext",
        lambda: object(),
    )
    monkeypatch.setattr(
        DifferentiableRenderer,
        "_sample_uv_texture_at_vertices",
        lambda renderer: torch.zeros(
            (renderer.num_vertices, 3),
            dtype=torch.float32,
            device=renderer.device,
        ),
    )

    renderer = DifferentiableRenderer(
        mesh_path=mesh_path,
        device=torch.device("cpu"),
        texture_parameterization="geometry_vertex",
    )

    assert renderer.get_texture_parameterization_name() == "geometry_vertex"
    assert renderer.get_texture_param().shape == (3, 3)
    torch.testing.assert_close(
        renderer.get_surface_delta(),
        torch.zeros((3, 3)),
    )
    torch.testing.assert_close(
        renderer.get_geometry_surface_delta(),
        torch.zeros((3, 3)),
    )
    torch.testing.assert_close(
        renderer.get_render_to_geometry_mapping(),
        torch.arange(3),
    )


def test_fixed_support_renderer_consumes_frozen_compact_coordinates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Renderer 只加载冻结 Support，并把紧凑参数散射回完整几何顶点。"""

    mesh_path = tmp_path / "triangle.obj"
    mesh_path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "f 1 2 3",
            ]
        ),
        encoding="utf-8",
    )
    geometry_vertices, geometry_faces = load_obj_geometry(mesh_path)
    support_mask = np.asarray([True, False, True], dtype=np.bool_)
    render_to_geometry = np.arange(3, dtype=np.int64)
    support = FrozenProductionSupport(
        schema_version=PRODUCTION_SUPPORT_SCHEMA_VERSION,
        provenance=ProductionSupportProvenance(
            code_commit="1" * 40,
            object_name="akita_black_bowl",
            source_support_manifest_sha256="2" * 64,
            source_candidate_artifact_sha256="3" * 64,
            source_coverage_artifact_sha256="4" * 64,
            seed_score_manifest_sha256="5" * 64,
            seed_score_artifact_sha256="6" * 64,
            visibility_manifest_sha256="7" * 64,
            visibility_metrics_sha256="8" * 64,
            mesh_file_sha256=file_sha256(mesh_path),
            mesh_array_sha256=mesh_array_sha256(
                geometry_vertices,
                geometry_faces,
            ),
            renderer_faces_sha256=array_sha256(
                geometry_faces.astype(np.int32)
            ),
            render_to_geometry_sha256=array_sha256(render_to_geometry),
        ),
        selected_candidate_index=0,
        num_geometry_vertices=3,
        support_mask=support_mask,
        support_vertex_indices=np.asarray([0, 2], dtype=np.int64),
        support_mask_sha256=array_sha256(support_mask),
        compact_coordinate_order="ascending_geometry_vertex_id",
        num_regions=1,
        seed_vertex_ids=(0,),
        total_surface_area=10.0,
        target_area_fraction=0.1,
        target_mass=1.0,
        actual_mass=1.0,
        actual_area_fraction=0.1,
        primary_coverage_min_threshold=0.2,
        primary_state_ids=(0,),
        primary_statuses=("valid",),
        primary_source_coverage=(0.3,),
        primary_effective_coverage=(0.3,),
        wrist_state_ids=(),
        wrist_statuses=(),
        wrist_source_coverage=(),
        wrist_effective_coverage=(),
        naturalness_k_nonconstant=128,
        production_support_constructed=True,
        fixed_support_frozen=True,
        rho_nat_calibrated=False,
        lambda_spec_calibrated=False,
        formal_training_allowed=False,
    )
    support_path = tmp_path / "production_fixed_support.npz"
    write_production_support_artifact(support_path, support)
    monkeypatch.setattr(
        "openvla_attack.renderer.dr.RasterizeCudaContext",
        lambda: object(),
    )
    monkeypatch.setattr(
        DifferentiableRenderer,
        "_sample_uv_texture_at_vertices",
        lambda renderer: torch.zeros(
            (renderer.num_vertices, 3),
            dtype=torch.float32,
            device=renderer.device,
        ),
    )

    renderer = DifferentiableRenderer(
        mesh_path=mesh_path,
        device=torch.device("cpu"),
        texture_parameterization="fixed_support",
        fixed_support_path=support_path,
    )

    assert renderer.get_texture_parameterization_name() == "fixed_support"
    assert renderer.production_support is not None
    assert renderer.production_support.support_mask_sha256 == array_sha256(
        support_mask
    )
    assert renderer.get_texture_param().shape == (2, 3)
    with torch.no_grad():
        renderer.get_texture_param()[0] = torch.tensor([0.2, -0.1, 0.3])
    geometry_delta = renderer.get_surface_delta()
    assert geometry_delta.shape == (3, 3)
    torch.testing.assert_close(
        renderer.get_geometry_surface_delta(),
        geometry_delta,
    )
    torch.testing.assert_close(
        geometry_delta[1],
        torch.zeros(3),
        rtol=0.0,
        atol=0.0,
    )


def test_legacy_renderer_rejects_missing_strict_geometry_mapping() -> None:
    renderer: DifferentiableRenderer = _minimal_cpu_renderer()

    with pytest.raises(RuntimeError, match="不提供严格 render_to_geometry"):
        renderer.get_render_to_geometry_mapping()


@pytest.mark.parametrize("return_clean", [False, True])
def test_render_tuple_interface_delegates_to_explicit_evidence(
    monkeypatch,
    return_clean: bool,
) -> None:
    renderer: DifferentiableRenderer = _minimal_cpu_renderer()
    adversarial = torch.full((1, 2, 2, 3), 0.6)
    clean = torch.full((1, 2, 2, 3), 0.5)
    mask = torch.ones((1, 2, 2, 1))
    raster = torch.zeros((1, 2, 2, 4))
    evidence = RendererEvidence(
        adversarial_rgb=adversarial,
        clean_rgb=clean,
        visibility_mask=mask,
        raster=raster,
    )
    calls: list[
        tuple[torch.Tensor, tuple[int, int], torch.Tensor | None]
    ] = []

    def fake_render_evidence(
        mvp: torch.Tensor,
        resolution: tuple[int, int],
        model_rot: torch.Tensor | None,
    ) -> RendererEvidence:
        calls.append((mvp, resolution, model_rot))
        return evidence

    monkeypatch.setattr(renderer, "render_evidence", fake_render_evidence)
    mvp = torch.eye(4)
    rotation = torch.eye(3)

    result = renderer.render(
        mvp,
        resolution=(2, 2),
        return_clean=return_clean,
        model_rot=rotation,
    )

    assert calls == [(mvp, (2, 2), rotation)]
    if return_clean:
        assert len(result) == 3
        assert result[0] is adversarial
        assert result[1] is clean
        assert result[2] is mask
    else:
        assert len(result) == 2
        assert result[0] is adversarial
        assert result[1] is mask


def test_spectral_renderer_exposes_basis_and_matching_eigenvalues(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "triangle.obj"
    mesh_path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "f 1 2 3",
            ]
        ),
        encoding="utf-8",
    )
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    basis_with_constant = np.asarray(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, -1.0, -1.0],
        ],
        dtype=np.float64,
    )
    basis_path = tmp_path / "basis.npz"
    save_spectral_basis(
        basis_path,
        vertices=vertices,
        faces=faces,
        mass=np.ones(3, dtype=np.float64),
        eigenvalues_with_constant=np.asarray(
            [0.0, 0.25, 0.5],
            dtype=np.float64,
        ),
        basis_with_constant=basis_with_constant,
    )
    monkeypatch.setattr(
        "openvla_attack.renderer.dr.RasterizeCudaContext",
        lambda: object(),
    )
    monkeypatch.setattr(
        DifferentiableRenderer,
        "_sample_uv_texture_at_vertices",
        lambda renderer: torch.zeros(
            (renderer.num_vertices, 3),
            dtype=torch.float32,
            device=renderer.device,
        ),
    )

    renderer = DifferentiableRenderer(
        mesh_path=mesh_path,
        device=torch.device("cpu"),
        texture_parameterization="spectral",
        spectral_basis_path=basis_path,
        spectral_basis_count=2,
    )
    basis, eigenvalues = renderer.get_spectral_basis_and_eigenvalues()

    torch.testing.assert_close(
        basis,
        torch.from_numpy(basis_with_constant[:, 1:]).float(),
    )
    torch.testing.assert_close(
        eigenvalues,
        torch.tensor([0.25, 0.5]),
    )
