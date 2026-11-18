"""Generate a standalone reconstruction/imagination video from a trained
checkpoint. Run this on demand (after training, or at any point you have a
checkpoint) - NOT during training, which only logs this to TensorBoard.

Loads <logdir>/latest.pt, samples a real batch of trajectories from the saved
replay episodes, and calls WorldModel.video_pred() - the same truth vs.
reconstruction vs. error comparison the training loop sends to TensorBoard -
but writes it to a standalone mp4 instead.

Usage (from src/dreamer/):
    python generate_dream_video.py --logdir ../../outputs/dreamer_logdir
"""

import argparse
import pathlib
import sys

import imageio
import numpy as np
import ruamel.yaml as yaml
import torch

sys.path.append(str(pathlib.Path(__file__).parent))
import models  # noqa: E402
import tools  # noqa: E402
from envs.so101 import SO101Dreamer  # noqa: E402

to_np = lambda x: x.detach().cpu().numpy()


def load_config(logdir: pathlib.Path):
    """Rebuilds the same config dreamer-training used: defaults +
    so101_pick_cube from configs.yaml next to this script."""
    configs = yaml.safe_load((pathlib.Path(__file__).parent / "configs.yaml").read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    defaults = {}
    for name in ["defaults", "so101_pick_cube"]:
        recursive_update(defaults, configs[name])

    class Config:
        pass

    config = Config()
    for k, v in defaults.items():
        setattr(config, k, v)
    config.num_actions = 6
    config.logdir = str(logdir)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True, help="Same --logdir the training run used.")
    parser.add_argument("--split", choices=["train", "eval"], default="train",
                         help="Which saved episodes to sample trajectories from.")
    parser.add_argument("--out", default=None, help="Output mp4 path (default: <logdir>/dream_video.mp4)")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir).expanduser()
    config = load_config(logdir)

    checkpoint_path = logdir / "latest.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path} - train first (train.py saves "
            f"latest.pt every config.checkpoint_every steps)."
        )

    # Only need obs/action spaces here, not a live rollout.
    env = SO101Dreamer(reward_stage=config.reward_stage, seed=0)
    obs_space, act_space = env.observation_space, env.action_space
    env.close()

    wm = models.WorldModel(obs_space, act_space, 0, config).to(config.device)

    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    # agent_state_dict covers the whole Dreamer module (_wm, _task_behavior,
    # _expl_behavior); pull out just the world model's own weights.
    wm_state = {
        k[len("_wm."):]: v
        for k, v in checkpoint["agent_state_dict"].items()
        if k.startswith("_wm.")
    }
    wm.load_state_dict(wm_state)
    wm.eval()

    episode_dir = logdir / f"{args.split}_eps"
    episodes = tools.load_episodes(episode_dir, limit=None)
    if not episodes:
        raise RuntimeError(
            f"No saved episodes in {episode_dir} yet - let training run a bit "
            f"longer (at least one full episode) before generating a video."
        )
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    batch = next(dataset)

    with torch.no_grad():
        openl = wm.video_pred(batch)

    value = to_np(openl)
    if np.issubdtype(value.dtype, np.floating):
        value = np.clip(255 * value, 0, 255).astype(np.uint8)
    # (B, T, 3*H, W, C): B example trajectories, each truth/recon/error
    # stacked vertically - tile the B examples side by side into one video.
    B, T, H, W, C = value.shape
    frames = value.transpose(1, 2, 0, 3, 4).reshape(T, H, B * W, C)

    out_path = pathlib.Path(args.out) if args.out else logdir / "dream_video.mp4"
    imageio.mimsave(out_path, frames, fps=args.fps)
    print(f"saved {out_path} ({T} frames, {H}x{B*W})")


if __name__ == "__main__":
    main()
