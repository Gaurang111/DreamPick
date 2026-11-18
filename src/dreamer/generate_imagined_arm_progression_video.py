"""Combines generate_progression_video.py (render every milestone checkpoint
to show improvement over training) with generate_imagined_arm_video.py
(decode joint_pos instead of images, replay through the REAL simulator so
any camera angle - including viewer_cam - can be used, not just the two
trained camera views).

For each milestone, in order: observes the same fixed real context, imagines
forward using the same recorded actions, decodes joint_pos at every step, and
poses the real sim/renders viewer_cam for each frame - then concatenates all
milestones' clips into one labeled video. Lets you watch the model's own
sense of "how the arm should move" get less erratic / more purposeful over
training, from a clean full-resolution angle.

Caveat: only the arm's pose is predicted - the cube isn't a decoded quantity
of this model, so it stays at its fixed resting position throughout in every
clip.

Requires milestone checkpoints (train.py saves these to <logdir>/milestones/
every config.milestone_every steps).

Usage (from src/dreamer/):
    python generate_imagined_arm_progression_video.py --logdir ../../outputs/dreamer_logdir
"""

import argparse
import pathlib
import sys

import imageio
import mujoco
import numpy as np
import torch

sys.path.append(str(pathlib.Path(__file__).parent))
import models  # noqa: E402
import tools  # noqa: E402
from envs.so101 import SO101Dreamer  # noqa: E402
from generate_dream_video import load_config  # noqa: E402
from generate_progression_video import find_milestones, label_frame  # noqa: E402

to_np = lambda x: x.detach().cpu().numpy()

SCENE_PATH = pathlib.Path(__file__).resolve().parents[2] / "assets" / "scene_pick_cube.xml"
CONTEXT_STEPS = 5


def build_batch_from_episode(episode, batch_length, start=0):
    """A batch_size=1 'batch' dict from one window of a single episode,
    shaped the way tools.from_generator's output is - used for
    --episode_selection earliest, where we want a SPECIFIC (the very first
    collected) episode rather than tools.sample_episodes' weighted random
    pick across the whole replay buffer (which, with a fixed seed, usually
    lands on some arbitrary, often much-later, episode - not representative
    of early/mostly-random behavior)."""
    return {
        key: value[start : start + batch_length][None]  # (T, ...) -> (1, T, ...)
        for key, value in episode.items()
        if "log_" not in key
    }


def decode_joint_pos_trajectory(wm, batch, example_idx):
    """Observe the first CONTEXT_STEPS real steps, imagine the rest forward
    using the recorded actions, decode joint_pos at every step. Mirrors
    WorldModel.video_pred()'s structure but for joint_pos instead of images."""
    data = wm.preprocess(batch)
    embed = wm.encoder(data)
    ex = slice(example_idx, example_idx + 1)

    states, _ = wm.dynamics.observe(
        embed[ex, :CONTEXT_STEPS], data["action"][ex, :CONTEXT_STEPS], data["is_first"][ex, :CONTEXT_STEPS]
    )
    context_joint_pos = wm.heads["decoder"](wm.dynamics.get_feat(states))["joint_pos"].mode()

    init = {k: v[:, -1] for k, v in states.items()}
    prior = wm.dynamics.imagine_with_action(data["action"][ex, CONTEXT_STEPS:], init)
    imagined_joint_pos = wm.heads["decoder"](wm.dynamics.get_feat(prior))["joint_pos"].mode()

    return to_np(torch.cat([context_joint_pos, imagined_joint_pos], dim=1))[0]  # (T, 6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True, help="Same --logdir the training run used.")
    parser.add_argument("--split", choices=["train", "eval"], default="train")
    parser.add_argument("--camera", default="viewer_cam",
                         help="Any camera in scene_pick_cube.xml: viewer_cam, overhead_cam, wrist_cam.")
    parser.add_argument("--example_idx", type=int, default=0)
    parser.add_argument("--out", default=None, help="Output mp4 path (default: <logdir>/imagined_arm_progression.mp4)")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0,
                         help="Fixes which episode/window gets sampled, so every milestone uses the same context+actions.")
    parser.add_argument("--episode_selection", choices=["random", "earliest"], default="random",
                         help="'random' (default): fixed-seed weighted sample across ALL saved episodes - "
                              "usually lands on a mid/late-training (already decent) one. 'earliest': use the "
                              "very first episode ever collected (prefill, essentially random actions) instead - "
                              "shows what the model predicts for genuinely early/chaotic behavior.")
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir).expanduser()
    config = load_config(logdir)

    milestones = find_milestones(logdir / "milestones")
    if not milestones:
        raise FileNotFoundError(
            f"No milestone checkpoints in {logdir / 'milestones'} - train.py saves these "
            f"every config.milestone_every steps (default 5000)."
        )
    print(f"Found {len(milestones)} milestones: steps {[s for s, _ in milestones]}")

    env = SO101Dreamer(reward_stage=config.reward_stage, seed=0)
    obs_space, act_space = env.observation_space, env.action_space
    env.close()

    wm = models.WorldModel(obs_space, act_space, 0, config).to(config.device)

    episode_dir = logdir / f"{args.split}_eps"
    if args.episode_selection == "earliest":
        # reverse=False -> chronological order, so the first item is the
        # very first episode ever collected (prefill, near-random actions).
        episodes = tools.load_episodes(episode_dir, limit=None, reverse=False)
        if not episodes:
            raise RuntimeError(f"No saved episodes in {episode_dir} yet.")
        first_episode = next(iter(episodes.values()))
        batch = build_batch_from_episode(first_episode, config.batch_length, start=0)
        print("Using the earliest collected episode as the fixed reference (near-random actions).")
    else:
        episodes = tools.load_episodes(episode_dir, limit=None)
        if not episodes:
            raise RuntimeError(f"No saved episodes in {episode_dir} yet.")
        # Fixed seed -> same sampled context/actions for every milestone, so
        # differences across the video are purely about the model, not the
        # data - but which episode that lands on is essentially arbitrary
        # across the whole replay buffer (see --episode_selection earliest).
        generator = tools.sample_episodes(episodes, config.batch_length, seed=args.seed)
        dataset = tools.from_generator(generator, config.batch_size)
        batch = next(dataset)

    # Real sim/renderer, shared across all milestones - pure forward
    # kinematics (mj_forward), no physics stepping.
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data_mj = mujoco.MjData(model)
    joint_qpos_adr = np.array([model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)])
    joint_low, joint_high = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    renderer = mujoco.Renderer(model, height=480, width=640)

    all_frames = []
    for step, path in milestones:
        wm.load_state_dict(torch.load(path, map_location=config.device, weights_only=True))
        wm.eval()
        with torch.no_grad():
            joint_pos_trajectory = decode_joint_pos_trajectory(wm, batch, args.example_idx)

        label = f"step {step}"
        for t in range(joint_pos_trajectory.shape[0]):
            pose = np.clip(np.nan_to_num(joint_pos_trajectory[t]), joint_low, joint_high)
            data_mj.qpos[joint_qpos_adr] = pose
            mujoco.mj_forward(model, data_mj)
            renderer.update_scene(data_mj, camera=args.camera)
            all_frames.append(label_frame(renderer.render(), label))
        print(f"  rendered {label} ({joint_pos_trajectory.shape[0]} frames)")

    renderer.close()

    out_path = pathlib.Path(args.out) if args.out else logdir / "imagined_arm_progression.mp4"
    imageio.mimsave(out_path, all_frames, fps=args.fps)
    print(f"saved {out_path} ({len(all_frames)} total frames across {len(milestones)} milestones, camera={args.camera})")
    print("Note: only the arm's pose is predicted - the cube stays at its fixed resting "
          "position throughout every clip.")


if __name__ == "__main__":
    main()
