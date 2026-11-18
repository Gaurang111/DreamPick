"""Run random actions in SO101PickCube-v0 and save what happened to outputs/.

For each reward stage ("reach", "grasp", "lift") this runs a couple of episodes
with random actions (no learning - just sanity-checking the env end to end) and
saves:
  - an MP4 per episode: the full-screen "viewer_cam" diagonal view (the one a
    human would actually want to watch), with the two observation cameras
    (wrist_image, overhead_image) shown as small reference insets in the
    corners - they are NOT the main view, just a reminder of what the agent
    itself sees
  - a few PNG snapshots from the first episode of each stage
  - per-step reward/info printed to the console

Random actions rarely reach/grasp/lift the cube - that's expected. This is here
to verify the env *runs* correctly end to end, not to demonstrate a policy.
"""

import sys
from pathlib import Path

import imageio
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from so101_env import SO101PickCubeEnv  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
EPISODES_PER_STAGE = 2
MAX_STEPS = 100
FPS = 20
INSET_SIZE = 160  # small reference thumbnail size, in pixels
INSET_MARGIN = 10


def make_frame(main_image: np.ndarray, obs: dict) -> np.ndarray:
    """viewer_cam full-screen, with wrist/overhead observation images as small
    corner insets - just for reference, not the main thing to look at."""
    frame = Image.fromarray(main_image).convert("RGB")

    wrist_inset = Image.fromarray(obs["wrist_image"]).resize((INSET_SIZE, INSET_SIZE), Image.NEAREST)
    overhead_inset = Image.fromarray(obs["overhead_image"]).resize((INSET_SIZE, INSET_SIZE), Image.NEAREST)

    width, _ = frame.size
    frame.paste(wrist_inset, (INSET_MARGIN, INSET_MARGIN))
    frame.paste(overhead_inset, (width - INSET_SIZE - INSET_MARGIN, INSET_MARGIN))
    return np.array(frame)


def run_episode(env: SO101PickCubeEnv, episode_idx: int, stage: str, rng: np.random.Generator):
    obs, info = env.reset(seed=1000 * episode_idx + hash(stage) % 1000)
    frames = [make_frame(env.render(), obs)]
    print(f"\n--- {stage} / episode {episode_idx} / step 0 (reset) --- info={info}")

    snapshots = []
    if episode_idx == 0:
        snapshots.append((0, Image.fromarray(obs["overhead_image"])))

    total_reward = 0.0
    for step in range(1, MAX_STEPS + 1):
        action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        frames.append(make_frame(env.render(), obs))

        if step % 20 == 0 or terminated:
            print(
                f"{stage} / ep {episode_idx} / step {step}: reward={reward:.4f} "
                f"terminated={terminated} truncated={truncated} info={info}"
            )
        if episode_idx == 0 and step in (MAX_STEPS // 2, MAX_STEPS):
            snapshots.append((step, Image.fromarray(obs["overhead_image"])))

        if terminated or truncated:
            break

    print(f"{stage} / ep {episode_idx}: total_reward={total_reward:.3f} over {step} steps")

    video_path = OUTPUT_DIR / f"rollout_{stage}_ep{episode_idx}.mp4"
    # Default ffmpeg quality is heavily compressed and looks blurry; quality=10 is
    # the max for imageio's ffmpeg writer (near-lossless, bigger file, still fine
    # at this length).
    imageio.mimsave(video_path, frames, fps=FPS, quality=10, macro_block_size=1)
    print(f"saved {video_path}")

    for step_idx, img in snapshots:
        png_path = OUTPUT_DIR / f"rollout_{stage}_ep{episode_idx}_step{step_idx}.png"
        img.save(png_path)
        print(f"saved {png_path}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)

    for stage in ("reach", "grasp", "lift"):
        env = SO101PickCubeEnv(reward_stage=stage, render_mode="rgb_array")
        for episode_idx in range(EPISODES_PER_STAGE):
            run_episode(env, episode_idx, stage, rng)
        env.close()


if __name__ == "__main__":
    main()
