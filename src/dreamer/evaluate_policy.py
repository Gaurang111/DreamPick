"""Test the trained policies in the sim: no exploration noise, fresh episodes.

For each stage's checkpoint (reach / grasp / lift) this runs N episodes of the
policy's most-likely action (actor.mode(), exactly what training's eval does)
and reports what physically happens - gripper-to-cube distance, how long the
cube is pinched, how high it is lifted - the same for every stage, whatever
reward that stage trained on. A random-action baseline gives a reference for
what "no skill" looks like. Optionally records a viewer-camera video per stage.

Only the world model + actor are built (not the full Dreamer class, which opens
the logdir's joint_pos_trace.bin for writing), so nothing in the logdir is
touched.

Usage (from src/dreamer/):
    python evaluate_policy.py --base_logdir ../../outputs/dreamer_logdir --episodes 20 --video_dir ../../outputs/policy_test
"""

import argparse
import json
import pathlib
import sys

import imageio
import mujoco
import numpy as np
import torch

sys.path.append(str(pathlib.Path(__file__).parent))
import models  # noqa: E402
from envs.so101 import SO101Dreamer  # noqa: E402
from generate_dream_video import load_config  # noqa: E402


class PolicyRunner:
    """World model + actor from a checkpoint, acting the way training's eval does."""

    def __init__(self, checkpoint_path, config, obs_space, act_space, sample=False):
        self.sample = sample
        self.wm = models.WorldModel(obs_space, act_space, 0, config).to(config.device)
        self.behavior = models.ImagBehavior(config, self.wm).to(config.device)
        ckpt = torch.load(checkpoint_path, map_location=config.device, weights_only=False)["agent_state_dict"]
        self.wm.load_state_dict({k[len("_wm."):]: v for k, v in ckpt.items() if k.startswith("_wm.")})
        self.behavior.load_state_dict(
            {k[len("_task_behavior."):]: v for k, v in ckpt.items() if k.startswith("_task_behavior.")}
        )
        self.wm.eval()
        self.behavior.eval()
        self.reset()

    def reset(self):
        self.latent, self.action = None, None

    @torch.no_grad()
    def act(self, obs):
        batch = {k: np.asarray(v)[None] for k, v in obs.items()}
        o = self.wm.preprocess(batch)
        embed = self.wm.encoder(o)
        self.latent, _ = self.wm.dynamics.obs_step(self.latent, self.action, embed, o["is_first"])
        feat = self.wm.dynamics.get_feat(self.latent)
        dist = self.behavior.actor(feat)
        action = dist.sample() if self.sample else dist.mode()
        self.latent = {k: v.detach() for k, v in self.latent.items()}
        self.action = action.detach()
        return action[0].cpu().numpy()


def run_episode(env, act_fn, renderer=None, frames=None):
    obs = env.reset()
    reach_d, grasped, height, cube_path, ret = [], [], [], [], 0.0
    done = False
    while not done:
        obs, reward, done, info = env.step(act_fn(obs))
        reach_d.append(info["gripper_cube_distance"])
        grasped.append(info["is_grasped"])
        height.append(info["cube_height"])
        cube_path.append(env._env.data.xpos[env._env._cube_body_id].copy())
        ret += reward
        if renderer is not None:
            renderer.update_scene(env._env.data, camera="viewer_cam")
            frames.append(renderer.render())
    d, g, h = np.array(reach_d), np.array(grasped), np.array(height)
    return {
        "return": ret,
        "min_dist_cm": d.min() * 100,
        "time_within_2cm": (d < 0.02).mean(),
        "time_grasped": g.mean(),
        "max_height_cm": h.max() * 100,
        "time_at_5cm": (h >= 0.05).mean(),
        "success": bool((h >= 0.05).any()),
        "final_height_cm": h[-1] * 100,
        "cube_path": np.round(np.array(cube_path), 4).tolist(),
    }


def summarize(results):
    keys = ["min_dist_cm", "time_within_2cm", "time_grasped", "max_height_cm", "time_at_5cm", "final_height_cm", "return"]
    out = {k: float(np.mean([r[k] for r in results])) for k in keys}
    out["reached_5cm"] = float(np.mean([r["success"] for r in results]))
    out["best_height_cm"] = float(np.max([r["max_height_cm"] for r in results]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_logdir", required=True)
    parser.add_argument("--stages", nargs="+", default=["reach", "grasp", "lift"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234, help="Different from training's seed 0.")
    parser.add_argument("--video_dir", default=None, help="If set, record the first 3 episodes of each stage here.")
    parser.add_argument("--checkpoint", choices=["best", "latest"], default="best",
                        help="Which saved weights to test: the best-scoring evaluation's (falls back to the last "
                             "ones if a stage has no best.pt, e.g. an older run) or always the last.")
    parser.add_argument("--device", default=None,
                        help="e.g. cpu: run the policy on the CPU instead of the training config's GPU (used while "
                             "training is running, so the test can't take GPU memory from it).")
    parser.add_argument("--json_out", default=None, help="Write per-episode results and the summary table here.")
    parser.add_argument("--sample", action="store_true",
                        help="Sample actions from the policy's distribution (what training's practice episodes do) "
                             "instead of taking its most likely action.")
    args = parser.parse_args()

    base = pathlib.Path(args.base_logdir)
    config = load_config(base)
    if args.device:
        config.device = args.device
    env = SO101Dreamer(reward_stage="lift", seed=args.seed)  # lift reward = the full reward, comparable across stages
    obs_space, act_space = env.observation_space, env.action_space
    rng = np.random.default_rng(args.seed)

    renderer, video_dir = None, None
    if args.video_dir:
        video_dir = pathlib.Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(env._env.model, height=540, width=960)

    table, per_episode = {}, {}
    def weights(stage):
        best = base / stage / "best.pt"
        return best if args.checkpoint == "best" and best.exists() else base / stage / "latest.pt"

    runs = [("random", None)] + [(s, weights(s)) for s in args.stages]
    for name, ckpt in runs:
        if ckpt is not None and not ckpt.exists():
            print(f"skipping {name}: no {ckpt}")
            continue
        if ckpt is not None:
            print(f"  {name}: testing {ckpt.parent.name}/{ckpt.name}")
        policy = None if ckpt is None else PolicyRunner(ckpt, config, obs_space, act_space, sample=args.sample)
        results, frames = [], []
        for ep in range(args.episodes):
            if policy is None:
                act_fn = lambda o: rng.uniform(-1, 1, size=act_space.shape).astype(np.float32)
            else:
                policy.reset()
                act_fn = policy.act
            rec = renderer if (renderer is not None and ep < 3) else None
            results.append(run_episode(env, act_fn, rec, frames))
            print(f"  {name} episode {ep + 1}/{args.episodes}: return {results[-1]['return']:.0f}, "
                  f"max height {results[-1]['max_height_cm']:.1f}cm, grasped {results[-1]['time_grasped'] * 100:.0f}%", flush=True)
        table[name] = summarize(results)
        per_episode[name] = results
        if video_dir is not None and frames:
            path = video_dir / f"{name}_policy.mp4"
            imageio.mimsave(path, frames, fps=20, macro_block_size=1)
            print(f"  saved {path}")
        del policy
        torch.cuda.empty_cache()

    cols = [("min_dist_cm", "closest (cm)", "{:.1f}"), ("time_within_2cm", "within 2cm", "{:.0%}"),
            ("time_grasped", "gripping", "{:.0%}"), ("max_height_cm", "avg peak lift (cm)", "{:.1f}"),
            ("best_height_cm", "best lift (cm)", "{:.1f}"), ("reached_5cm", "episodes at 5cm", "{:.0%}"),
            ("return", "return (max ~800)", "{:.0f}")]
    mode = "sampled actions" if args.sample else "deterministic actions"
    print(f"\nResults over {args.episodes} episodes each ({mode}):")
    print(f"  {'policy':<8}" + "".join(f"{c[1]:>20}" for c in cols))
    for name, s in table.items():
        print(f"  {name:<8}" + "".join(f"{c[2].format(s[c[0]]):>20}" for c in cols))
    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(
            {"episodes": args.episodes, "sampled_actions": args.sample, "summary": table, "per_episode": per_episode}, indent=2))
        print(f"wrote {args.json_out}")
    env.close()


if __name__ == "__main__":
    main()
