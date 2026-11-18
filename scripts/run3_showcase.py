"""Record a run-3 state policy doing the whole task in the simulator, with a verdict per episode.

Each episode is one video segment (300 steps = 15 s at 20 Hz) with a caption bar: episode, phase (reaching / grasping /
lifting / holding), time, cube start, cube height and sideways shift, and at the end the same pass/fail verdict the
stage gates use. Episodes use seeds the training and gates never saw.

Usage (project root):
    .venv\\Scripts\\python.exe scripts\\run3_showcase.py --checkpoint outputs\\run3\\s4c_occlusion\\stage_best.pt --difficulty wide --episodes 8
"""
import argparse
import pathlib
import sys

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "dreamer"))
sys.path.insert(0, str(ROOT / "scripts"))
from run3_gate_eval import DreamerPolicy  # noqa: E402
from so101_env.metrics import episode_metrics  # noqa: E402
from so101_env.state_pick_env import SO101StatePickEnv  # noqa: E402

W, H, BAR = 960, 540, 64


def phase(info):
    if info["cube_height"] >= 0.05:
        return "HOLDING at 5 cm or higher"
    if info["cube_height"] > 0.008:
        return "LIFTING"
    if info["is_grasped"]:
        return "GRASPED"
    if info["gripper_cube_distance"] < 0.04:
        return "GRASPING"
    return "REACHING"


def font(size):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def caption(frame, lines, color=(170, 200, 255)):
    img = Image.new("RGB", (W, H + BAR), (20, 20, 24))
    img.paste(Image.fromarray(frame), (0, 0))
    d = ImageDraw.Draw(img)
    d.text((16, H + 8), lines[0], fill=(255, 255, 255), font=font(24))
    d.text((16, H + 36), lines[1], fill=color, font=font(18))
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--difficulty", default="wide")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "run3_showcase"))
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = SO101StatePickEnv("lift", args.difficulty)
    policy = DreamerPolicy(args.checkpoint, env, args.device)
    renderer = mujoco.Renderer(env.model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.18, 0.0, 0.07]
    cam.distance, cam.azimuth, cam.elevation = 0.6, -90, -20

    all_frames, wins = [], 0
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        policy.reset()
        start = env.cube_pos()[:2] * 100
        heights, shifts, tables, joints, frames, done, step = [], [], [], [], [], False, 0
        while not done:
            obs, _, _, done, info = env.step(policy.act(obs))
            step += 1
            heights.append(info["cube_height"]); shifts.append(info["cube_shift"])
            tables.append(info["table_force"]); joints.append(info["joint_pos"])
            renderer.update_scene(env.data, camera=cam)
            frames.append(caption(renderer.render(), [
                f"Episode {ep + 1}/{args.episodes}    {phase(info)}",
                f"t = {step / 20:4.1f} s    cube start x={start[0]:.1f} cm, y={start[1]:+.1f} cm    "
                f"height {info['cube_height'] * 100:.1f} cm    sideways shift {info['cube_shift'] * 100:.1f} cm",
            ]))
        m = episode_metrics(heights, shifts, tables, joints)
        wins += bool(m["success"])
        verdict = (f"SUCCESS: held at 5 cm or higher for {m['held_steps'] / 20:.1f} s" if m["success"]
                   else f"FAILED: peak {m['peak_cm']:.1f} cm, held {m['held_steps'] / 20:.1f} s, "
                        f"sideways shift while held {m['shift_while_held_cm']:.1f} cm")
        color = (120, 230, 140) if m["success"] else (255, 130, 120)
        end = caption(frames[-1][:H, :, :], [f"Episode {ep + 1}/{args.episodes}    RESULT", verdict], color)
        frames += [end] * 20
        imageio.mimsave(out / f"episode_{ep + 1:02d}.mp4", frames, fps=20, macro_block_size=1)
        all_frames += frames
        print(f"episode {ep + 1}: cube ({start[0]:.1f}, {start[1]:+.1f}) cm -> {verdict}", flush=True)

    imageio.mimsave(out / "all_episodes.mp4", all_frames, fps=20, macro_block_size=1)
    print(f"{wins}/{args.episodes} successes. saved {out / 'all_episodes.mp4'} ({len(all_frames) / 20:.0f} s)")


if __name__ == "__main__":
    main()
