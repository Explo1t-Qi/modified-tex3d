import os
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List

import draccus
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import wandb
from PIL import Image
from torch.cuda.amp import autocast

# print("[INFO] Setting up OSMesa for CPU Rendering...")
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# try:
#     ctypes.CDLL("libOSMesa.so")
# except Exception as e:
#     print(f"[WARNING] Failed to load libOSMesa.so: {e}")

# 直接执行该文件和通过 importlib 加载该文件时，sys.path 的初始值不同。
# 先显式加入当前目录，确保两种入口都能导入 OpenVLA 专用 module。
LIBERO_EXPERIMENT_DIR = str(Path(__file__).parent)
if LIBERO_EXPERIMENT_DIR not in sys.path:
    sys.path.append(LIBERO_EXPERIMENT_DIR)

from openvla_attack.assets import LIBERO_ROOT, OBJECT_ASSETS, parse_mesh_scale

if LIBERO_ROOT not in sys.path:
    sys.path.append(LIBERO_ROOT)

from libero.libero import benchmark
import imageio

from libero_utils import (
    get_libero_dummy_action, get_libero_env, get_libero_image,
    quat2axisangle, save_rollout_video,
)
from openvla_attack.compositing import (
    SingleViewFrame,
    build_single_view_samples,
    render_and_composite,
)
from openvla_attack.evaluation import LiberoEpisodeRunner
from openvla_attack.frame_collection import TrainingFrame, TrainingFrameCollector
from openvla_attack.objective import get_attack_loss
from openvla_attack.renderer import DifferentiableRenderer
from openvla_attack.scene import (
    compute_render_mvp,
    find_target_body_pose,
    render_background_without_target,
)

sys.path.append(str(Path(__file__).parent.parent))

OPENVLA_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if OPENVLA_REPO_ROOT not in sys.path:
    sys.path.insert(0, OPENVLA_REPO_ROOT)

from openvla_utils import get_processor
from robot_utils import (
    DATE_TIME, get_action, get_image_resize_size, get_model,
    invert_gripper_action, normalize_gripper_action, set_seed_everywhere,
)

def train_adversarial_texture(
        cfg, model, processor, renderer,
        initial_obs_state, task, task_description,
        save_dir, episode_idx,
        search_keywords_list: List[List[str]],
        xml_path,
        num_iters=20,
        init_states=None,
):
    print(f"[ATTACK] Training Ep {episode_idx} | {cfg.num_frames_to_attack}-Frame Optimization...")
    os.makedirs(save_dir, exist_ok=True)

    RENDER_RES = 256
    model_input_size = get_image_resize_size(cfg)
    device = model.device

    frame_collector = TrainingFrameCollector(
        cfg=cfg,
        model=model,
        processor=processor,
        renderer=renderer,
        search_keywords=search_keywords_list,
        render_resolution=RENDER_RES,
    )
    frame_pool: list[TrainingFrame] = frame_collector.collect(
        task=task,
        task_description=task_description,
        fallback_initial_state=initial_obs_state,
        initial_states=init_states,
    )
    pool_size = len(frame_pool)
    batch_size = min(cfg.num_frames_to_attack, pool_size)

    pgd_step = cfg.attack_lr
    loss_history = []

    grad_log_path = os.path.join(save_dir, f"Ep{episode_idx}_gradient_log.txt")
    with open(grad_log_path, "w") as f:
        f.write("Iter | Total Loss | Action Loss | Feature Loss | Grad Norm | LR\n")

    def _bake_and_inject_eval_texture(tag):
        with torch.no_grad():
            baked_tex = renderer.get_baked_adv_texture()
        tex_path = os.path.join(save_dir, f"Ep{episode_idx}_LiveTexture_{tag}.png")
        Image.fromarray(
            (baked_tex.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        ).save(tex_path)

        tree = ET.parse(xml_path)
        root = tree.getroot()
        tex_name = f"tex-{cfg.object_name}"
        mat_name = f"mat-{cfg.object_name}"
        for asset_elem in root.findall("asset"):
            for tex_elem in asset_elem.findall("texture"):
                if tex_elem.get("name") == tex_name:
                    tex_elem.set("file", str(Path(tex_path).resolve()))
                    tex_elem.set("type", "2d")
                    break
        for mat_elem in root.findall(".//material"):
            if mat_elem.get("name") == mat_name:
                mat_elem.set("texuniform", "false")
        tree.write(xml_path)
        return tex_path

    def _run_live_test(i):
        tex_path = _bake_and_inject_eval_texture(f"iter{i:04d}")

        noise_pt_path = os.path.join(save_dir, f"Ep{episode_idx}_noise_iter{i:04d}.pt")
        torch.save(renderer.adv_noise.data.cpu(), noise_pt_path)

        if init_states is not None and len(init_states) > 1:
            chosen_state = init_states[np.random.randint(len(init_states))]
        else:
            chosen_state = initial_obs_state

        test_env, _ = get_libero_env(
            task, cfg.model_family, resolution=cfg.live_test_resolution
        )
        test_env.reset()
        test_obs = test_env.set_init_state(chosen_state)
        test_env.env.sim.forward()

        t2, max_steps, done2 = 0, cfg.live_test_max_steps, False
        frames = []
        while t2 < max_steps + cfg.num_steps_wait:
            if t2 < cfg.num_steps_wait:
                test_obs, _, _, _ = test_env.step(get_libero_dummy_action(cfg.model_family))
                t2 += 1
                continue

            img_np = get_libero_image(test_obs, cfg.live_test_resolution)
            bg_tensor = (torch.from_numpy(img_np).float().to(device) / 255.0
                         ).permute(2, 0, 1).unsqueeze(0)

            target_pose = find_target_body_pose(
                test_env,
                search_keywords_list,
                device,
            )
            model_matrix = target_pose.model_matrix
            body_id = target_pose.body_id
            mvp = (
                compute_render_mvp(
                    test_env,
                    model_matrix,
                    resolution=(
                        cfg.live_test_resolution,
                        cfg.live_test_resolution,
                    ),
                )
                if body_id != -1
                else None
            )

            _bg_no_obj = (
                render_background_without_target(
                    test_env,
                    body_id,
                    cfg.live_test_resolution,
                )
                if body_id != -1
                else None
            )
            composite_bg = (
                (torch.from_numpy(_bg_no_obj.copy()).float().to(device) / 255.0
                 ).permute(2, 0, 1).unsqueeze(0)
                if _bg_no_obj is not None else bg_tensor
            )

            with torch.no_grad():
                composited = render_and_composite(
                    renderer,
                    composite_bg,
                    mvp,
                    resolution=(
                        cfg.live_test_resolution,
                        cfg.live_test_resolution,
                    ),
                    model_rotation=model_matrix[:3, :3],
                )

            perceived_np = (
                    composited[0].permute(1, 2, 0).detach().clamp(0, 1).cpu().numpy() * 255
            ).astype(np.uint8)
            frames.append(img_np)

            img_model = np.array(
                Image.fromarray(perceived_np).resize((model_input_size, model_input_size))
            )
            observation = {
                "full_image": img_model,
                "state": np.concatenate((
                    test_obs["robot0_eef_pos"],
                    quat2axisangle(test_obs["robot0_eef_quat"]),
                    test_obs["robot0_gripper_qpos"],
                )),
            }
            act = get_action(cfg, model, observation, task_description, processor=processor)
            act = normalize_gripper_action(act, binarize=True)
            if cfg.model_family == "openvla":
                act = invert_gripper_action(act)

            test_obs, _, done2, _ = test_env.step(act.tolist())
            if done2:
                break
            t2 += 1
        test_env.close()

        video_path = os.path.join(
            save_dir, f"Ep{episode_idx}_LiveTest_iter{i:04d}_success={done2}.mp4"
        )
        writer = imageio.get_writer(video_path, fps=30)
        for fr in frames:
            writer.append_data(fr)
        writer.close()

        print(f"[LIVE-TEST] iter={i} success={done2} | video={video_path}")
        return done2

    iterator = tqdm.tqdm(range(num_iters), desc="Optimizing", leave=False)
    for i in iterator:
        renderer.adv_noise.grad = None
        avg_total = avg_action = avg_feat = 0.0
        valid = 0

        batch_idx = np.random.choice(pool_size, batch_size, replace=False)
        batch_frames = [frame_pool[j] for j in batch_idx]

        for t, fdata in enumerate(batch_frames):
            frame_mvp = fdata["mvp"]
            if frame_mvp is None:
                continue

            w_t = torch.tensor(1.0 / batch_size, device=device)

            # collector 允许 mvp=None 表示目标定位失败；经过上面的收窄后，
            # 这里构造 compositing module 所需的必选 MVP interface。
            single_view_frame: SingleViewFrame = {
                "mvp": frame_mvp,
                "bg_tensor": fdata["bg_tensor"],
                "bg_tensor_no_obj": fdata["bg_tensor_no_obj"],
                "model_rot": fdata["model_rot"],
            }
            adv_samples = build_single_view_samples(
                renderer,
                single_view_frame,
                render_resolution=RENDER_RES,
            )

            sample_action_losses = []
            sample_feat_losses = []

            for sample_idx, adv_img in enumerate(adv_samples):
                adv_224 = F.interpolate(
                    adv_img,
                    size=(fdata["model_input_size"], fdata["model_input_size"]),
                    mode="bilinear", align_corners=False,
                )

                pv = torch.cat(
                    [(adv_224 - fdata["siglip_mean"]) / fdata["siglip_std"],
                     (adv_224 - fdata["dino_mean"]) / fdata["dino_std"]], dim=1
                )

                with autocast(dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=fdata["clean_output_ids"],
                        attention_mask=torch.ones_like(fdata["clean_output_ids"]),
                        pixel_values=pv.to(torch.bfloat16),
                        output_hidden_states=True,
                    )

                sample_action_losses.append(get_attack_loss(outputs.logits, fdata["clean_output_ids"]))
                sample_feat_losses.append(-F.mse_loss(outputs.hidden_states[-1], fdata["clean_hidden"]))

            loss_action = torch.stack(sample_action_losses).mean()
            loss_feature = torch.stack(sample_feat_losses).mean()

            frame_loss = w_t * (
                    cfg.alpha_action * loss_action +
                    cfg.alpha_feature * loss_feature
            )
            frame_loss.backward()

            avg_total += frame_loss.item()
            avg_action += loss_action.item()
            avg_feat += loss_feature.item()
            valid += 1

        if valid > 0:
            avg_total /= valid
            avg_action /= valid
            avg_feat /= valid
        loss_history.append(avg_total)

        grad = renderer.adv_noise.grad
        g_norm = grad.norm().item() if grad is not None else 0.0

        with open(grad_log_path, "a") as f:
            f.write(f"{i:02d} | {avg_total:.6f} | {avg_action:.6f} | "
                    f"{avg_feat:.6f} | {g_norm:.6e} | {pgd_step:.6e}\n")

        with torch.no_grad():
            renderer.adv_noise.data -= pgd_step * grad.sign()
        iterator.set_postfix(
            act=f"{avg_action:.4f}",
            feat=f"{avg_feat:.4f}",
            gnorm=f"{g_norm:.4f}",
        )

        if (cfg.live_test_enabled and cfg.live_test_every_n_iters > 0
                and (i + 1) % cfg.live_test_every_n_iters == 0):
            _run_live_test(i + 1)

    if cfg.save_attack_artifacts:
        torch.save(
            renderer.get_texture_param().detach().cpu(),
            os.path.join(save_dir, f"Ep{episode_idx}_Vertex_Noise.pt"),
        )
        Image.fromarray(
            (renderer.get_baked_adv_texture().squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        ).save(os.path.join(save_dir, f"Ep{episode_idx}_UV_Map.png"))
        np.save(
            os.path.join(save_dir, f"Ep{episode_idx}_loss_history.npy"),
            np.array(loss_history),
        )

    return env, loss_history


@dataclass
class GenerateConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "./openvla-7b-finetuned-libero-spatial"
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
    num_frames_to_attack: int = 20
    num_train_init_states: int = 10
    train_frames_per_state: int = 1
    alpha_action: float = 1.0
    alpha_feature: float = 10.0
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


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    set_seed_everywhere(cfg.seed)

    if cfg.object_name not in OBJECT_ASSETS:
        raise ValueError(
            f"未知物体 '{cfg.object_name}'，可选: {list(OBJECT_ASSETS.keys())}"
        )
    obj_cfg = OBJECT_ASSETS[cfg.object_name]
    mesh_path = cfg.override_mesh_path or obj_cfg["mesh"]
    texture_path = cfg.override_texture_path or obj_cfg["texture"]
    xml_path = cfg.override_xml_path or obj_cfg["xml"]
    search_kw = obj_cfg["search"]
    task_suite_name = cfg.task_suite_name or obj_cfg["task_suite"]

    scale_xyz = parse_mesh_scale(xml_path)

    run_id = f"EVAL-{task_suite_name}-{DATE_TIME}"
    if cfg.run_id_note:
        run_id = f"{cfg.run_id_note}-{run_id}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    artifact_dir = os.path.join(cfg.local_log_dir, "attack_artifacts", run_id)
    if cfg.enable_attack and cfg.save_attack_artifacts:
        os.makedirs(artifact_dir, exist_ok=True)
    log_file = open(os.path.join(cfg.local_log_dir, run_id + ".txt"), "w")
    original_xml = Path(xml_path)

    if cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, name=run_id)

    if not original_xml.exists():
        raise FileNotFoundError(f"XML asset not found at {original_xml}")
    global_clean_backup = original_xml.with_name(
        f"{original_xml.stem}_clean_backup_{DATE_TIME}{original_xml.suffix}"
    )
    shutil.copy(original_xml, global_clean_backup)

    _real_tex_path = Path(texture_path)
    if not _real_tex_path.is_absolute():
        _real_tex_path = (Path.cwd() / _real_tex_path).resolve()
    _real_tex_backup = None
    if _real_tex_path.exists():
        _real_tex_backup = _real_tex_path.with_name(f"texture_clean_backup_{DATE_TIME}.png")
        shutil.copy(_real_tex_path, _real_tex_backup)
    else:
        print(f"[WARNING] Real MuJoCo texture not found at {_real_tex_path}, MuJoCo video may look clean")

    import signal, atexit

    def _restore_clean_assets(context: str, remove_backups: bool = False):
        if global_clean_backup.exists():
            shutil.copy(global_clean_backup, original_xml)
            if remove_backups:
                global_clean_backup.unlink(missing_ok=True)
            print(f"[INFO] {context} Original XML restored.")
        if _real_tex_backup is not None and _real_tex_backup.exists():
            shutil.copy(_real_tex_backup, _real_tex_path)
            if remove_backups:
                _real_tex_backup.unlink(missing_ok=True)
            print(f"[INFO] {context} Real MuJoCo texture restored.")

    def _restore_xml_on_exit():
        _restore_clean_assets("(exit handler)", remove_backups=True)

    def _sig_handler(signum, frame):
        _restore_xml_on_exit()
        raise SystemExit(f"Caught signal {signum}, exiting cleanly.")

    atexit.register(_restore_xml_on_exit)
    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT, _sig_handler)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_obj = benchmark_dict[task_suite_name]()
    target_tasks = ([cfg.task_id] if cfg.task_id is not None
                    else range(task_suite_obj.n_tasks))

    model = get_model(cfg)
    processor = get_processor(cfg) if cfg.model_family == "openvla" else None

    renderer = None
    if cfg.enable_attack:
        print("[INFO] Initializing DifferentiableRenderer...")
        renderer = DifferentiableRenderer(
            mesh_path=mesh_path,
            orig_texture_path=texture_path,
            device=str(model.device),
            scale_xyz=scale_xyz,
        ).to(model.device)
    total_episodes = 0
    total_successes = 0
    video_resolution = 512
    episode_runner = LiberoEpisodeRunner(
        cfg=cfg,
        model=model,
        processor=processor,
        renderer=renderer,
        search_keywords=search_kw,
        video_resolution=video_resolution,
        max_steps=300,
    )

    try:
        for task_id in tqdm.tqdm(target_tasks, desc="Tasks"):
            _restore_clean_assets(f"Before Task {task_id}", remove_backups=False)

            task = task_suite_obj.get_task(task_id)
            init_states = task_suite_obj.get_task_init_states(task_id)

            if cfg.enable_attack and cfg.load_texture_path is not None:
                print(f"[INFO] Loading pre-trained adversarial noise from {cfg.load_texture_path}")
                if cfg.load_texture_path.endswith(".pt"):
                    noise_t = torch.load(cfg.load_texture_path,
                                         map_location=renderer.adv_noise.device)
                    renderer.adv_noise.data.copy_(noise_t)
                    with torch.no_grad():
                        delta = torch.tanh(noise_t) * renderer.epsilon
                    print(f"[INFO] adv_noise loaded from .pt: "
                          f"max_delta={delta.abs().max():.4f}, "
                          f"nonzero={(delta.abs() > 1e-3).float().mean() * 100:.1f}%")
                else:
                    baked_np = np.array(Image.open(cfg.load_texture_path)).astype(np.float32) / 255.0
                    baked_t = torch.from_numpy(baked_np).unsqueeze(0).to(renderer.adv_noise.device)
                    baked_stored = torch.flip(baked_t, dims=[1])
                    saved_orig = renderer.orig_texture
                    renderer.orig_texture = baked_stored.contiguous()
                    adv_vc_loaded = renderer._sample_uv_texture_at_vertices()
                    renderer.orig_texture = saved_orig
                    delta = (adv_vc_loaded - renderer.orig_vertex_colors).clamp(
                        -renderer.epsilon + 1e-6, renderer.epsilon - 1e-6)
                    noise_t = torch.atanh(delta / renderer.epsilon)
                    renderer.adv_noise.data.copy_(noise_t)
                    print(f"[INFO] adv_noise loaded from PNG: "
                          f"max_delta={delta.abs().max():.4f}, "
                          f"nonzero={(delta.abs() > 1e-3).float().mean() * 100:.1f}%")
                adv_tex_inj_path = os.path.join(artifact_dir, f"task_{task_id}_adv_tex_loaded.png")
                with torch.no_grad():
                    baked_vis = renderer.get_baked_adv_texture()
                Image.fromarray(
                    (baked_vis.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                ).save(adv_tex_inj_path)
                tree = ET.parse(original_xml)
                root = tree.getroot()
                for asset_elem in root.findall("asset"):
                    for tex_elem in asset_elem.findall("texture"):
                        if tex_elem.get("name") == f"tex-{cfg.object_name}":
                            tex_elem.set("file", str(Path(adv_tex_inj_path).resolve()))
                            tex_elem.set("type", "2d")
                            break
                for mat_elem in root.findall(".//material"):
                    if mat_elem.get("name") == f"mat-{cfg.object_name}":
                        mat_elem.set("texuniform", "false")
                tree.write(original_xml)
                print(f"[INFO] MuJoCo XML updated with adversarial texture → {adv_tex_inj_path}")
                if _real_tex_path.exists() or _real_tex_backup is not None:
                    shutil.copy(adv_tex_inj_path, _real_tex_path)
                    print(f"[INFO] Real MuJoCo texture overwritten → {_real_tex_path}")

            if cfg.enable_attack and cfg.load_texture_path is None:
                print(f"[INFO] Attack training for Task {task_id}...")
                renderer.reset_texture()

                dummy_env, train_task_desc = get_libero_env(
                    task, cfg.model_family, resolution=256
                )
                dummy_env.close()

                _, _ = train_adversarial_texture(
                    cfg, model, processor, renderer,
                    init_states[0], task, train_task_desc,
                    artifact_dir, episode_idx=task_id,
                    search_keywords_list=search_kw,
                    xml_path=original_xml,
                    num_iters=cfg.attack_iters,
                    init_states=init_states,
                )

                with torch.no_grad():
                    baked_tex = renderer.get_baked_adv_texture()
                    trained_tex_path = os.path.join(
                        artifact_dir, f"task_{task_id}_adv_texture_{DATE_TIME}.png"
                    )
                    Image.fromarray(
                        (baked_tex.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                    ).save(trained_tex_path)
                print(f"[INFO] Task {task_id} texture saved → {trained_tex_path}")

                tree = ET.parse(original_xml)
                root = tree.getroot()
                tex_name = f"tex-{cfg.object_name}"
                mat_name = f"mat-{cfg.object_name}"
                for asset_elem in root.findall("asset"):
                    for tex_elem in asset_elem.findall("texture"):
                        if tex_elem.get("name") == tex_name:
                            tex_elem.set("file", str(Path(trained_tex_path).resolve()))
                            tex_elem.set("type", "2d")
                            break
                for mat_elem in root.findall(".//material"):
                    if mat_elem.get("name") == mat_name:
                        mat_elem.set("texuniform", "false")
                tree.write(original_xml)
                print(f"[INFO] XML updated for Task {task_id}.")
                if _real_tex_path.exists() or _real_tex_backup is not None:
                    shutil.copy(trained_tex_path, _real_tex_path)
                    print(f"[INFO] Real MuJoCo texture overwritten → {_real_tex_path}")

            n_eval = min(cfg.num_trials_per_task, len(init_states))
            for ep in tqdm.tqdm(range(n_eval),
                                desc=f"Task {task_id} Episodes"):
                rollout_result = episode_runner.run(
                    task=task,
                    initial_state=init_states[ep],
                    task_id=task_id,
                    episode_index=ep,
                )
                if rollout_result.success:
                    total_successes += 1

                total_episodes += 1
                log_str = (
                    f"Task: {task_id} | Ep: {ep} | "
                    f"Success: {rollout_result.success}"
                )
                print(log_str)
                log_file.write(log_str + "\n")
                log_file.flush()

                save_rollout_video(
                    rollout_result.replay_images,
                    total_episodes,
                    success=rollout_result.success,
                    task_description=rollout_result.task_description,
                    log_file=log_file,
                )

        avg_sr = total_successes / total_episodes if total_episodes > 0 else 0.0
        print(f"\n[DONE] Episodes: {total_episodes} | Attack success rate: {avg_sr:.2%}")
        log_file.write(f"\nFINAL AVG SUCCESS RATE: {avg_sr:.2%}\n")

    finally:
        _restore_clean_assets("Final cleanup", remove_backups=True)
        log_file.close()
        if cfg.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    eval_libero()
