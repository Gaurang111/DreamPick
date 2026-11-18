"""Shows how the world model's reconstructions improved over training, by
rendering the SAME fixed episode through every saved milestone checkpoint in
sequence and concatenating the clips into one video - literally watching it
learn, rather than a single end-of-training snapshot.

Requires milestone checkpoints (train.py saves these to <logdir>/milestones/
every config.milestone_every steps, in addition to the frequent latest.pt
used for crash-resume). If you only have latest.pt, use generate_dream_video.py
instead for a single-checkpoint reconstruction video.

Usage (from src/dreamer/):
    python generate_progression_video.py --logdir ../../outputs/dreamer_logdir
"""

import argparse
import pathlib
import re
import sys

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.append(str(pathlib.Path(__file__).parent))
import models  # noqa: E402
import tools  # noqa: E402
from envs.so101 import SO101Dreamer  # noqa: E402
from generate_dream_video import load_config  # noqa: E402

to_np = lambda x: x.detach().cpu().numpy()

STEP_RE = re.compile(r"step_(\d+)\.pt$")


def find_milestones(milestones_dir: pathlib.Path):
    found = []
    for path in milestones_dir.glob("step_*.pt"):
        match = STEP_RE.search(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found, key=lambda x: x[0])


def label_frame(frame: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    # Small black outline behind white text so it stays readable over any
    # background color.
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        draw.text((4 + dx, 2 + dy), text, fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255))
    return np.array(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True, help="Same --logdir the training run used.")
    parser.add_argument("--split", choices=["train", "eval"], default="train")
    parser.add_argument("--out", default=None, help="Output mp4 path (default: <logdir>/progression_video.mp4)")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0,
                         help="Fixes which episode/window gets sampled, so every milestone renders the same clip.")
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir).expanduser()
    config = load_config(logdir)

    milestones = find_milestones(logdir / "milestones")
    if not milestones:
        raise FileNotFoundError(
            f"No milestone checkpoints in {logdir / 'milestones'} - train.py saves these "
            f"every config.milestone_every steps (default 5000). Either let training run "
            f"longer, or use generate_dream_video.py for a single-checkpoint video."
        )
    print(f"Found {len(milestones)} milestones: steps {[s for s, _ in milestones]}")

    env = SO101Dreamer(reward_stage=config.reward_stage, seed=0)
    obs_space, act_space = env.observation_space, env.action_space
    env.close()

    wm = models.WorldModel(obs_space, act_space, 0, config).to(config.device)

    episode_dir = logdir / f"{args.split}_eps"
    episodes = tools.load_episodes(episode_dir, limit=None)
    if not episodes:
        raise RuntimeError(f"No saved episodes in {episode_dir} yet.")
    # Fixed seed -> the SAME sampled window every time, so each milestone is
    # judged on an identical clip (otherwise "improvement" could just be a
    # difference in how hard the sampled clip happened to be).
    generator = tools.sample_episodes(episodes, config.batch_length, seed=args.seed)
    dataset = tools.from_generator(generator, config.batch_size)
    batch = next(dataset)

    all_frames = []
    for step, path in milestones:
        wm.load_state_dict(torch.load(path, map_location=config.device, weights_only=True))
        wm.eval()
        with torch.no_grad():
            openl = wm.video_pred(batch)

        value = to_np(openl)
        if np.issubdtype(value.dtype, np.floating):
            value = np.clip(255 * value, 0, 255).astype(np.uint8)
        B, T, H, W, C = value.shape
        frames = value.transpose(1, 2, 0, 3, 4).reshape(T, H, B * W, C)

        label = f"step {step}"
        frames = [label_frame(frame, label) for frame in frames]
        all_frames.extend(frames)
        print(f"  rendered {label} ({T} frames)")

    out_path = pathlib.Path(args.out) if args.out else logdir / "progression_video.mp4"
    imageio.mimsave(out_path, all_frames, fps=args.fps)
    print(f"saved {out_path} ({len(all_frames)} total frames across {len(milestones)} milestones)")


if __name__ == "__main__":
    main()
