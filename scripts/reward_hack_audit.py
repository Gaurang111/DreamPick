"""Reward-hack audit: every known way of cheating must earn LESS than an honest pick-up, in the actual reward function.

Runs the scripted solver and five deliberate cheats on the same cube positions and compares total episode reward
(lift-stage reward, easy and wide difficulty). Exit code 1 if any cheat gets within MARGIN of the honest return.

    .venv\\Scripts\\python.exe scripts\\reward_hack_audit.py
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from so101_env.expert import GRIP_CLOSED, GRIP_OPEN, ScriptedExpert  # noqa: E402
from so101_env.metrics import episode_metrics  # noqa: E402
from so101_env.state_pick_env import SO101StatePickEnv  # noqa: E402

MARGIN = 0.85          # a cheat must score below this fraction of the honest return


class CheatExpert(ScriptedExpert):
    """The honest solver up to the grasp; then one of several ways of cheating."""

    def __init__(self, env, cheat, **kw):
        super().__init__(env, **kw)
        self.cheat = cheat

    def reset(self):
        ok = super().reset()
        self.cube0 = self.xy.copy()
        base_z = self.plan[3][1]
        lift = self.plan[5]
        away = -self.radial                     # toward the arm's base
        left = np.array([-self.radial[1], self.radial[0]])
        if self.cheat == "drag_toward_base":
            self.cheat_goals = [(self.cube0 + 0.12 * away, base_z + 0.002)]
        elif self.cheat == "sideways_carry":
            self.cheat_goals = [(self.cube0, 0.040), (self.cube0 + 0.15 * left, 0.040)]
        elif self.cheat == "grasp_and_hold_low":
            self.cheat_goals = [(self.cube0, base_z)]
        elif self.cheat == "push_away":
            # no grasp: open jaws, start behind the cube, sweep it forward along the table
            self.cheat_goals = [(self.cube0 + 0.15 * self.radial, base_z + 0.004)]
        elif self.cheat == "hover":
            self.cheat_goals = [(self.cube0, 0.075)]
        self.cheat_phase = 0
        return ok

    def act(self):
        name = self.plan[self.phase][0]
        if self.cheat in ("hover",) and name != "approach":
            self.phase = 0                                # stay in the approach: never descend or grasp
        if self.cheat == "push_away" and name in ("close", "lift", "hold"):
            self.phase = 3                                # keep the jaws open, keep pushing
        if name in ("lift", "hold") and self.cheat in ("drag_toward_base", "sideways_carry", "grasp_and_hold_low"):
            xy, z = self.cheat_goals[min(self.cheat_phase, len(self.cheat_goals) - 1)]
            self.xy = np.array(xy)
            p = list(self.plan[5])
            p[1], p[3] = z, p[3]
            self.plan[5] = tuple(p)
            self.plan[6] = tuple(list(self.plan[6][:1]) + [z] + list(self.plan[6][2:]))
            pad = self.env.pad_mid()
            if np.linalg.norm(pad[:2] - self.xy) < 0.01 and self.cheat_phase < len(self.cheat_goals) - 1:
                self.cheat_phase += 1
        if self.cheat == "push_away" and name.startswith("descend"):
            self.xy = self.cube0 - 0.05 * self.radial if self.phase < 3 else self.cheat_goals[0][0]
        return super().act()


def run(env, agent, seed):
    env.reset(seed=seed)
    agent.reset()
    total, heights, shifts = 0.0, [], []
    done = False
    while not done:
        _, r, _, done, info = env.step(agent.act())
        total += r
        heights.append(info["cube_height"]); shifts.append(info["cube_shift"])
    return total, episode_metrics(heights, shifts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--difficulty", default="easy")
    args = ap.parse_args()
    bad = False
    env = SO101StatePickEnv("lift", args.difficulty)
    honest = [run(env, ScriptedExpert(env), 3000 + i) for i in range(args.episodes)]
    h_ret = float(np.mean([r for r, _ in honest]))
    print(f"honest pick-up        return {h_ret:7.0f}   success {np.mean([m['success'] for _, m in honest]):.0%}")
    for cheat in ("drag_toward_base", "sideways_carry", "grasp_and_hold_low", "push_away", "hover"):
        res = [run(env, CheatExpert(env, cheat), 3000 + i) for i in range(args.episodes)]
        ret = float(np.mean([r for r, _ in res]))
        verdict = "ok" if ret < MARGIN * h_ret else "REWARD HACK"
        bad |= verdict != "ok"
        print(f"{cheat:21s} return {ret:7.0f}   = {ret / h_ret:4.0%} of honest   success {np.mean([m['success'] for _, m in res]):.0%}   {verdict}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
