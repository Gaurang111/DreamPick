"""Takes the world model's imagined joint_pos trajectory and replays it
through the REAL MuJoCo scene/renderer - from any camera, including
viewer_cam, the nice diagonal view - rather than the model's own image
decoder, which only knows how to reconstruct the two camera views it was
trained on (wrist_image, overhead_image) and can't render a new angle.

Since the real simulator is driving this (not the model's image decoder),
we can render ALL THREE cameras from the SAME predicted pose each frame:
viewer_cam full-screen as the main view, with wrist_cam/overhead_cam as
small reference insets in the corners - same composition as
scripts/random_rollout.py.

Mirrors WorldModel.video_pred()'s approach: observes the first few real steps
from a sampled replay batch (grounding the latent state in reality), then
imagines forward using the dataset's recorded actions for the rest - but
instead of decoding to images, decodes to joint_pos at every step and uses
that to pose the real simulator.

Caveat: only the arm's pose is a decoded quantity of this model - the cube's
position isn't, so it stays at its fixed resting position throughout (it
won't visually react to an imagined grasp/lift). To see the cube move too,
use generate_dream_video.py instead, which shows the model's own (lower
resolution, only wrist/overhead angle) image predictions.

Usage (from src/dreamer/):
    python generate_imagined_arm_video.py --logdir ../../outputs/dreamer_logdir --near_step 60000
"""

import argparse
import pathlib
import sys

import imageio
import mujoco
import numpy as np
import torch
from PIL import Image

sys.path.append(str(pathlib.Path(__file__).parent))
import models  # noqa: E402
import tools  # noqa: E402
from envs.so101 import SO101Dreamer  # noqa: E402
from generate_dream_video import load_config  # noqa: E402
from generate_progression_video import find_milestones  # noqa: E402

to_np = lambda x: x.detach().cpu().numpy()

SCENE_PATH = pathlib.Path(__file__).resolve().parents[2] / "assets" / "scene_pick_cube.xml"
CONTEXT_STEPS = 5  # real/observed steps to ground the latent state before imagining forward

MAIN_W, MAIN_H = 960, 720  # up from 640x480 - the earlier version looked blurry at that size
INSET_SIZE = 220
INSET_MARGIN = 14


def pick_checkpoint(logdir: pathlib.Path, near_step: int | None):
    """near_step=None -> latest.pt (most recent, full agent). Otherwise the
    milestone (world-model-only snapshot) whose step is closest to near_step."""
    if near_step is None:
        path = logdir / "latest.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {path} - train first (train.py saves latest.pt "
                f"every config.checkpoint_every steps)."
            )
        return "latest", path, True  # True = full agent state_dict (needs "_wm." stripped)

    milestones = find_milestones(logdir / "milestones")
    if not milestones:
        raise FileNotFoundError(f"No milestone checkpoints in {logdir / 'milestones'}.")
    step, path = min(milestones, key=lambda sp: abs(sp[0] - near_step))
    return step, path, False  # False = world-model-only state_dict, used as-is


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True, help="Same --logdir the training run used.")
    parser.add_argument("--split", choices=["train", "eval"], default="train")
    parser.add_argument("--camera", default="viewer_cam",
                         help="Main full-screen camera. wrist_cam/overhead_cam always render as corner insets.")
    parser.add_argument("--example_idx", type=int, default=0,
                         help="Which trajectory from the sampled batch to render (0 to batch_size-1).")
    parser.add_argument("--out", default=None, help="Output mp4 path (default: <logdir>/imagined_arm_view.mp4)")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--near_step", type=int, default=None,
                         help="Use the milestone checkpoint closest to this env step, instead of the latest "
                              "(full-agent) checkpoint. Requires milestones to exist.")
    parser.add_argument("--data_until_step", type=int, default=None,
                         help="Only sample the reference episode/actions from data collected up to this many env "
                              "steps (chronologically) - e.g. match --near_step so the video reflects what the "
                              "model had actually seen by that point, not later (already-better) experience.")
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir).expanduser()
    config = load_config(logdir)

    ckpt_label, checkpoint_path, is_full_agent = pick_checkpoint(logdir, args.near_step)
    print(f"Using checkpoint: {ckpt_label} ({checkpoint_path.name})")

    env = SO101Dreamer(reward_stage=config.reward_stage, seed=0)
    obs_space, act_space = env.observation_space, env.action_space
    env.close()

    wm = models.WorldModel(obs_space, act_space, 0, config).to(config.device)
    if is_full_agent:
        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
        wm_state = {
            k[len("_wm."):]: v
            for k, v in checkpoint["agent_state_dict"].items()
            if k.startswith("_wm.")
        }
        wm.load_state_dict(wm_state)
    else:
        wm.load_state_dict(torch.load(checkpoint_path, map_location=config.device, weights_only=True))
    wm.eval()

    episode_dir = logdir / f"{args.split}_eps"
    if args.data_until_step is not None:
        episodes = tools.load_episodes(episode_dir, limit=args.data_until_step, reverse=False)
        print(f"Restricted to episodes collected up to step {args.data_until_step} ({len(episodes)} episodes).")
    else:
        episodes = tools.load_episodes(episode_dir, limit=None)
    if not episodes:
        raise RuntimeError(f"No saved episodes in {episode_dir} yet - let training run longer first.")
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    batch = next(dataset)

    # Mirrors WorldModel.video_pred(): observe the first CONTEXT_STEPS real
    # steps, then imagine the rest forward using the recorded actions - but
    # decode joint_pos instead of image.
    with torch.no_grad():
        data = wm.preprocess(batch)
        embed = wm.encoder(data)
        ex = slice(args.example_idx, args.example_idx + 1)

        states, _ = wm.dynamics.observe(
            embed[ex, :CONTEXT_STEPS], data["action"][ex, :CONTEXT_STEPS], data["is_first"][ex, :CONTEXT_STEPS]
        )
        context_joint_pos = wm.heads["decoder"](wm.dynamics.get_feat(states))["joint_pos"].mode()

        init = {k: v[:, -1] for k, v in states.items()}
        prior = wm.dynamics.imagine_with_action(data["action"][ex, CONTEXT_STEPS:], init)
        imagined_joint_pos = wm.heads["decoder"](wm.dynamics.get_feat(prior))["joint_pos"].mode()

        joint_pos_trajectory = torch.cat([context_joint_pos, imagined_joint_pos], dim=1)

    joint_pos_trajectory = to_np(joint_pos_trajectory)[0]  # (T, 6), single example
    n_context, n_imagined = CONTEXT_STEPS, joint_pos_trajectory.shape[0] - CONTEXT_STEPS
    print(f"Imagined joint_pos trajectory: {n_context} real context steps + {n_imagined} imagined steps")

    # Replay through the real sim: pure forward kinematics (mj_forward), no
    # physics stepping - we're visualizing a predicted pose sequence, not
    # simulating contact dynamics. Three cameras rendered per frame since
    # this is real rendering, not a learned decoder limited to trained views.
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data_mj = mujoco.MjData(model)
    joint_qpos_adr = np.array([model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)])
    joint_low, joint_high = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]

    main_renderer = mujoco.Renderer(model, height=MAIN_H, width=MAIN_W)
    inset_renderer = mujoco.Renderer(model, height=INSET_SIZE, width=INSET_SIZE)

    frames = []
    for t in range(joint_pos_trajectory.shape[0]):
        pose = np.clip(np.nan_to_num(joint_pos_trajectory[t]), joint_low, joint_high)
        data_mj.qpos[joint_qpos_adr] = pose
        mujoco.mj_forward(model, data_mj)

        main_renderer.update_scene(data_mj, camera=args.camera)
        frame = Image.fromarray(main_renderer.render()).convert("RGB")

        inset_renderer.update_scene(data_mj, camera="wrist_cam")
        wrist_inset = Image.fromarray(inset_renderer.render())
        inset_renderer.update_scene(data_mj, camera="overhead_cam")
        overhead_inset = Image.fromarray(inset_renderer.render())

        frame.paste(wrist_inset, (INSET_MARGIN, INSET_MARGIN))
        frame.paste(overhead_inset, (MAIN_W - INSET_SIZE - INSET_MARGIN, INSET_MARGIN))
        frames.append(np.array(frame))

    main_renderer.close()
    inset_renderer.close()

    out_path = pathlib.Path(args.out) if args.out else logdir / "imagined_arm_view.mp4"
    # quality=10 (max, near-lossless) + macro_block_size=1 - the default
    # ffmpeg compression looked noticeably blurry at this resolution.
    imageio.mimsave(out_path, frames, fps=args.fps, quality=10, macro_block_size=1)
    print(f"saved {out_path} ({len(frames)} frames, main={args.camera}, insets=wrist_cam+overhead_cam)")
    print("Note: only the arm's pose is predicted - the cube stays at its fixed resting "
          "position throughout (its position isn't a decoded quantity of this model).")


if __name__ == "__main__":
    main()
