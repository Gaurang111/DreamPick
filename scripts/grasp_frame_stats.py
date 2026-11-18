"""Where a grasped cube sits in the FIXED jaw pad's own frame (sim, scripted-expert grasps): the reference the kinematic calibration uses,
because the fixed jaw does not move with the gripper angle, so it is immune to the sim-vs-real jaw-angle scale.
Writes outputs/phase1/grasp_in_fixed_pad.json.   .venv\\Scripts\\python.exe scripts\\grasp_frame_stats.py
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from so101_env.expert import ScriptedExpert  # noqa: E402
from so101_env.state_pick_env import SO101StatePickEnv  # noqa: E402

env = SO101StatePickEnv("lift", "easy")
expert = ScriptedExpert(env)
rows, tilt = [], []
for seed in range(60):
    env.reset(seed=500 + seed)
    expert.reset()
    done = False
    while not done:
        _, _, _, done, info = env.step(expert.act())
        if info["is_grasped"] and info["cube_height"] >= 0.03:
            R = env.data.xmat[env._fixed_pad].reshape(3, 3)
            rows.append(R.T @ (env.cube_pos() - env.data.xpos[env._fixed_pad]))        # cube centre in the fixed pad's frame (m)
            tilt.append(R[2, 2])                           # world-up component of the pad's closing axis: 0 when the jaw faces are vertical
a = np.array(rows) * 1000
mean, sd = a.mean(axis=0), a.std(axis=0)
print(f"RESULT cube centre in the fixed pad's frame (mm): mean {np.round(mean, 1).tolist()}  sd {np.round(sd, 1).tolist()}  ({len(a)} held steps)")
tilt = np.array(tilt)
print(f"RESULT closing axis world-up component: mean {tilt.mean():+.3f} sd {tilt.std():.3f}  (0.017 = 1 deg of tilt)")
(ROOT / "outputs" / "phase1").mkdir(parents=True, exist_ok=True)
(ROOT / "outputs" / "phase1" / "grasp_in_fixed_pad.json").write_text(json.dumps(dict(v_local_m=(mean / 1000).tolist(), sd_mm=sd.tolist(),
                                                                                       tilt_mean=float(tilt.mean()), tilt_sd=float(tilt.std()))))
