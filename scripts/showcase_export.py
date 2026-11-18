"""Showcase prep, step 2 (project venv, needs the GPU): take the joint path that was sent to the real arm in a recorded run, replay it physically in the
simulator (the sim twin), and let the trained world model WATCH it: at every step the model gets what it gets in the real pipeline (cube pose + joint
angles), updates its internal state, and we record what it IMAGINES for the next 15 steps (decoded joint angles and predicted reward) next to what really
happens. Also records the sim state for rendering.

    .venv\\Scripts\\python.exe scripts\\showcase_export.py run_20261115_212001
"""
import json
import math
import pathlib
import sys

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dreamer"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import torch  # noqa: E402
from run3_follow_real import place_arm, place_cube  # noqa: E402
from run3_gate_eval import DreamerPolicy  # noqa: E402
from so101_env.state_pick_env import REST_QPOS, SO101StatePickEnv  # noqa: E402

run = sys.argv[1]
FORCE_ACTIONS = "--forced" in sys.argv                 # tell the model the action actually executed at each step (helps some runs, hurts others)
H = 15
plan = json.loads((ROOT / "outputs" / "real_live" / f"plan_{run.replace('run_', '')}.json").read_text())
prep = json.loads((ROOT / "outputs" / "showcase" / run / "prep.json").read_text())
cx, cy = plan["cube_cm"]
yaw = prep["placement"]["lock"][2]
goals = np.array(plan["goals"])                                   # 20 Hz joint targets of the plan (what the real arm followed, played slower)

env = SO101StatePickEnv("lift", "wide_exact")
policy = DreamerPolicy(str(ROOT / "outputs" / "final_policy" / "run3_state_policy.pt"), env, "cuda:0")
q0 = REST_QPOS.copy()
q0[5] = 0.4


def imagine():
    wm, beh = policy.wm, policy.behavior
    js, rs = [], []
    with torch.no_grad():
        state = {k: v.clone() for k, v in policy.latent.items()}
        for _ in range(H):
            feat = wm.dynamics.get_feat(state)
            a = beh.actor(feat).mode()
            state = wm.dynamics.img_step(state, a, sample=False)
            f2 = wm.dynamics.get_feat(state)
            js.append(wm.heads["decoder"](f2)["proprio"].mode()[0].cpu().numpy())
            rs.append(float(wm.heads["reward"](f2).mode().reshape(-1)[0].cpu()))
    return np.array(js), np.array(rs)


env.reset(seed=plan["seed"])
place_cube(env, cx, cy, yaw)
place_arm(env, q0)
obs = env._observe()
policy.reset()
out = dict(goals=goals, qpos=[env.data.qpos.copy()], joints=[env.data.qpos[env._adr].copy()], imag_j=[], imag_r=[], height=[0.0], cube=[env.cube_pos().copy()],
           obs_cube=[], obs_proprio=[], action=[])
prev = goals[0]
for t in range(1, len(goals)):
    a = policy.act(obs)                                           # the model takes in the current observation (updates its state) and picks an action
    j, r = imagine()                                              # ... and imagines the next H steps from there
    # what was really executed at this step (from the plan's joint motion), so the model's memory of its last action matches reality
    v = (goals[t] - goals[t - 1]) / env.dt
    a_exec = np.clip(v / env.v_max, -1.0, 1.0)
    a_exec = np.sign(a_exec) * np.where(np.abs(a_exec) > 1e-4, np.abs(a_exec) * (1 - 0.07) + 0.07, 0.0)
    if FORCE_ACTIONS:
        policy.action = torch.tensor(a_exec[None], dtype=policy.action.dtype, device=policy.action.device)
    out["imag_j"].append(j); out["imag_r"].append(r); out["obs_cube"].append(obs["cube"].copy()); out["obs_proprio"].append(obs["proprio"].copy()); out["action"].append(np.array(a))
    for k in range(env.frame_skip):                               # the plan's goal path, interpolated inside the step like the smoother did
        w = (k + 1) / env.frame_skip
        env.data.ctrl[:] = np.clip((1 - w) * prev + w * goals[t], env._low, env._high)
        mujoco.mj_step(env.model, env.data)
    prev = goals[t]
    env._target = np.clip(goals[t], env._low, env._high)
    info = env._info()
    obs = env._observe()
    out["qpos"].append(env.data.qpos.copy()); out["joints"].append(info["joint_pos"]); out["height"].append(info["cube_height"]); out["cube"].append(env.cube_pos().copy())
out = {k: np.array(v) for k, v in out.items()}
np.savez(ROOT / "outputs" / "showcase" / run / "model.npz", **out)
h = out["height"]
print(f"RESULT {run}: cube at ({cx:.1f}, {cy:+.1f}) yaw {yaw:+.0f}; sim twin peak height {h.max() * 100:.1f} cm, at the end {h[-1] * 100:.1f} cm, held >= 5 cm for {int((h >= 0.05).sum())} of {len(h)} steps")
ij, aj = out["imag_j"], out["joints"]
err = [np.degrees(np.abs(ij[t, :min(H, len(aj) - 1 - t)] - aj[t + 1:t + 1 + min(H, len(aj) - 1 - t)])).mean() for t in range(len(ij) - H)]
print(f"RESULT imagination vs what happened (mean over the next {H} steps, joint angles): {np.mean(err):.1f} deg average error")
