"""Gate evaluation for run 3: play a checkpoint as a policy on many fixed-seed episodes and report success metrics,
sliced by every randomised factor. A stage passes or fails on these numbers, not on the training score.

    .venv\\Scripts\\python.exe scripts\\run3_gate_eval.py --checkpoint <agent .pt> --stage lift --difficulty wide --episodes 200
    .venv\\Scripts\\python.exe scripts\\run3_gate_eval.py --expert --stage lift --difficulty full --episodes 100      # the scripted solver
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import ruamel.yaml as yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dreamer"))
sys.path.insert(0, str(ROOT / "src"))
from so101_env.expert import ScriptedExpert  # noqa: E402
from so101_env.metrics import episode_metrics  # noqa: E402
from so101_env.state_pick_env import SO101StatePickEnv  # noqa: E402


def load_state_config(device="cuda:0"):
    configs = yaml.safe_load((ROOT / "src" / "dreamer" / "configs.yaml").read_text())

    def rec(base, upd):
        for k, v in upd.items():
            if isinstance(v, dict) and k in base:
                rec(base[k], v)
            else:
                base[k] = v

    d = {}
    for name in ("defaults", "so101_state"):
        rec(d, configs[name])

    class Config:
        pass

    c = Config()
    for k, v in d.items():
        setattr(c, k, v)
    c.num_actions, c.device, c.compile = 6, device, False
    return c


class DreamerPolicy:
    """World model + actor from a full-agent checkpoint, acting the way training's evaluation does (deterministic)."""

    def __init__(self, checkpoint, env, device="cuda:0"):
        import torch
        import models
        self.torch = torch
        config = load_state_config(device)
        self.wm = models.WorldModel(env.observation_space, env.action_space, 0, config).to(device)
        self.behavior = models.ImagBehavior(config, self.wm).to(device)
        sd = torch.load(checkpoint, map_location=device, weights_only=False)["agent_state_dict"]
        self.wm.load_state_dict({k[len("_wm."):]: v for k, v in sd.items() if k.startswith("_wm.")})
        self.behavior.load_state_dict({k[len("_task_behavior."):]: v for k, v in sd.items() if k.startswith("_task_behavior.")})
        self.wm.eval(); self.behavior.eval()
        self.reset()

    def reset(self):
        self.latent, self.action, self.first = None, None, True

    def act(self, obs):
        torch = self.torch
        batch = {k: np.asarray(v)[None] for k, v in obs.items()}
        batch["is_first"] = np.array([self.first], dtype=np.float32)
        batch["is_terminal"] = np.array([False], dtype=np.float32)
        with torch.no_grad():
            o = self.wm.preprocess(batch)
            embed = self.wm.encoder(o)
            self.latent, _ = self.wm.dynamics.obs_step(self.latent, self.action, embed, o["is_first"])
            feat = self.wm.dynamics.get_feat(self.latent)
            action = self.behavior.actor(feat).mode()
        self.latent = {k: v.detach() for k, v in self.latent.items()}
        self.action, self.first = action.detach(), False
        return action[0].cpu().numpy()


def slices(results):
    """Success rate within each half / bin of every randomised factor (only bins with >= 20 episodes)."""
    out = {}

    def add(name, values, edges=None, labels=None):
        v = np.array(values, float)
        ok = np.array([r["success"] for r in results])
        if edges is None:
            groups = {f"{name}={g:g}": v == g for g in sorted(set(v))}
        else:
            idx = np.digitize(v, edges)
            groups = {f"{name} {labels[i]}": idx == i for i in range(len(labels))}
        for label, mask in groups.items():
            if mask.sum() >= 20:
                out[label] = (float(ok[mask].mean()), int(mask.sum()))

    P = [r["params"] for r in results]
    add("cube x", [p["x"] * 100 for p in P], [24, 27], ["21-24 cm", "24-27 cm", "27-31 cm"])
    add("cube |y|", [abs(p["y"]) * 100 for p in P], [3.5, 7], ["<3.5 cm", "3.5-7 cm", ">7 cm"])
    add("start pose", [p.get("start_u", 0) for p in P], [0.33, 0.66], ["folded", "middle", "home"])
    if any(p.get("kp", 1) != 1 for p in P):
        add("servo gain", [p["kp"] for p in P], [0.55, 0.78], ["low", "mid", "high"])
        add("servo damping", [p["kv"] for p in P], [0.8, 1.2], ["low", "mid", "high"])
        add("gripper force", [p["force"] for p in P], [0.8, 1.0], ["low", "mid", "high"])
        add("friction", [p["friction"] for p in P], [0.9, 1.2], ["low", "mid", "high"])
        add("cube mass", [p["mass"] for p in P], [0.95, 1.25], ["light", "mid", "heavy"])
        add("cube size", [p["size_delta"] * 1000 for p in P], [-0.5, 0.5], ["small", "nominal", "large"])
    if any(p.get("delay", 0) for p in P):
        add("action delay", [p["delay"] for p in P])
        add("joint offset", [p["joint_offset_deg"] for p in P], [1.5, 2.5], ["<1.5 deg", "1.5-2.5 deg", ">2.5 deg"])
        add("camera offset", [p["cam_offset_mm"] for p in P], [4, 7], ["<4 mm", "4-7 mm", ">7 mm"])
    return out


def evaluate(policy, stage, difficulty, episodes, seed):
    env = SO101StatePickEnv(stage, difficulty)
    expert = ScriptedExpert(env) if policy is None else None
    results = []
    for i in range(episodes):
        obs, info = env.reset(seed=seed + i)
        if expert:
            expert.reset()
        else:
            policy.reset()
        heights, shifts, tables, joints, dists, grasped, drags = [], [], [], [], [], [], []
        total, done = 0.0, False
        while not done:
            a = expert.act() if expert else policy.act(obs)
            obs, r, _, done, info = env.step(a)
            total += r
            heights.append(info["cube_height"]); shifts.append(info["cube_shift"]); tables.append(info["table_force"])
            joints.append(info["joint_pos"]); dists.append(info["gripper_cube_distance"]); grasped.append(info["is_grasped"])
            drags.append(info["cube_drag"])
        m = episode_metrics(heights, shifts, tables, joints)
        g = np.array(grasped)
        run = best = 0
        for x in g:
            run = run + 1 if x else 0
            best = max(best, run)
        m.update(params=dict(env.params), min_dist_cm=float(np.min(dists) * 100), pinched=bool(best >= 10),
                 max_drag_cm=float(np.max(drags) * 100), ret=total)
        results.append(m)
    return results


def summarize(results, difficulty):
    n = len(results)
    mean = lambda key: float(np.mean([r[key] for r in results]))
    flips = [r["hold_flips"] for r in results if not np.isnan(r["hold_flips"])]
    out = {
        "episodes": n, "difficulty": difficulty,
        "success": mean("success"), "reached": mean("reached"), "held_at_end": mean("held_at_end"),
        "reach_within_1.5cm": float(np.mean([r["min_dist_cm"] <= 1.5 for r in results])),
        "pinched": mean("pinched"), "peak_cm": mean("peak_cm"), "return": mean("ret"),
        "dragged_over_3cm": float(np.mean([r["max_drag_cm"] > 3 for r in results])),
        "hard_table_hits": float(np.mean([r["table_force"] > 40 for r in results])),
        "hold_flips": float(np.mean(flips)) if flips else float("nan"),
        "peak_accel_p95": float(np.percentile([r["peak_accel"] for r in results], 95)),
        "slices": slices(results),
    }
    s = [v[0] for v in out["slices"].values()]
    out["worst_slice"] = float(min(s)) if s else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--expert", action="store_true")
    ap.add_argument("--stage", default="lift")
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=100000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()
    env = SO101StatePickEnv(args.stage, args.difficulty)
    policy = None if args.expert else DreamerPolicy(args.checkpoint, env, args.device)
    res = evaluate(policy, args.stage, args.difficulty, args.episodes, args.seed)
    out = summarize(res, args.difficulty)
    print(f"{args.difficulty}: success {out['success']:.0%} of {out['episodes']} | reached {out['reached']:.0%} | held at end {out['held_at_end']:.0%} | "
          f"reach<=1.5cm {out['reach_within_1.5cm']:.0%} | pinched {out['pinched']:.0%} | dragged>3cm {out['dragged_over_3cm']:.0%} | "
          f"hard table hits {out['hard_table_hits']:.0%} | hold flips {out['hold_flips']:.0%} | peak accel p95 {out['peak_accel_p95']:.1f} rad/s^2 | "
          f"worst slice {out['worst_slice']:.0%}")
    for k, (rate, n) in sorted(out["slices"].items(), key=lambda kv: kv[1][0])[:5]:
        print(f"   weakest: {k}: {rate:.0%} (n={n})")
    if args.json_out:
        pathlib.Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.json_out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
