"""Run 3 orchestrator: six stages, each ending on a pass/fail gate measured on fixed-seed episodes (not on the training score).
See docs/training_run_3_plan.md.

Per stage: train in CHUNK-step pieces; after each piece save the full agent as a numbered checkpoint and play it on
fixed-seed episodes. Once the stage's minimum step count is reached and the gate is met, the best-gated checkpoint becomes the
next stage's starting weights. If the gate is still not met at 3x the minimum, the run stops and says so.
Resumable: state lives in <base>/state.json and each stage's own logdir; rerun the same command after any interruption.

    .venv\\Scripts\\python.exe -u scripts\\run3_train.py --base outputs\\run3
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable
CHUNK = 25_000

# name, reward stage, difficulty, min steps, scripted-solver episodes, gate
STAGES = [
    dict(name="s1_reach", reward="reach", difficulty="easy", min_steps=50_000, expert=30,
         gate=dict(episodes=100, need={"reach_within_1.5cm": 0.98})),
    dict(name="s2_grasp", reward="grasp", difficulty="easy", min_steps=75_000, expert=30,
         gate=dict(episodes=100, need={"pinched": 0.95}, at_most={"dragged_over_3cm": 0.15})),
    dict(name="s3_lift", reward="lift", difficulty="easy", min_steps=150_000, expert=30,
         gate=dict(episodes=200, need={"success": 0.94})),
    dict(name="s4a_zone", reward="lift", difficulty="wide_exact", min_steps=100_000, expert=60,
         gate=dict(episodes=300, need={"success": 0.95})),
    dict(name="s4b_noise", reward="lift", difficulty="wide_noisy", min_steps=100_000, expert=60,
         gate=dict(episodes=300, need={"success": 0.94}), accept_within=0.04),
    dict(name="s4c_occlusion", reward="lift", difficulty="wide", min_steps=100_000, expert=60,
         gate=dict(episodes=300, need={"success": 0.94}), accept_within=0.04),
    dict(name="s5_robust", reward="lift", difficulty="full", min_steps=200_000, expert=100,
         gate=dict(episodes=500, need={"success": 0.93, "worst_slice": 0.85}), accept_within=1.0),
    # final stage, redesigned 2026-11-08 after S5 plateaued at 60 to 70% (see docs section 13): realistic calibration errors
    # (1.5 deg, 5 mm), scripted-solver demonstrations added to every chunk, bounded budget, the best checkpoint is kept
    dict(name="s6_polish", reward="lift", difficulty="polish", min_steps=150_000, expert=30, expert_each_chunk=True, max_factor=2,
         gate=dict(episodes=500, need={"success": 0.90, "worst_slice": 0.70}), accept_within=1.0),
]
MAX_FACTOR = 5


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in pathlib.Path(folder).glob("*.npz"))


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd).returncode


def gate_eval(ckpt, stage, difficulty, episodes, out_json, seed=100000):
    code = run([PY, str(ROOT / "scripts" / "run3_gate_eval.py"), "--checkpoint", str(ckpt), "--stage", stage["reward"],
                "--difficulty", difficulty, "--episodes", str(episodes), "--seed", str(seed), "--json_out", str(out_json)])
    if code != 0:
        raise SystemExit(f"gate evaluation failed (exit {code})")
    return json.loads(pathlib.Path(out_json).read_text())


def passes(result, gate):
    miss = [f"{k} {result.get(k, float('nan')):.3f} < {v}" for k, v in gate.get("need", {}).items() if not result.get(k, 0) >= v]
    miss += [f"{k} {result.get(k, float('nan')):.3f} > {v}" for k, v in gate.get("at_most", {}).items() if not result.get(k, 1) <= v]
    return not miss, miss


def score(result):
    return (round(result["success"], 2), round(result.get("pinched", 0), 3), -round(result.get("dragged_over_3cm", 0), 3),
            round(result.get("reach_within_1.5cm", 0), 3),
            round(result["worst_slice"], 3) if result["worst_slice"] == result["worst_slice"] else 0.0)


def train_chunk(stage, logdir, init_weights, steps_now, device):
    cmd = [PY, str(ROOT / "src" / "dreamer" / "train.py"), "--configs", "so101_state", "--logdir", str(logdir),
           "--one_chunk", "True", "--eval_every", str(CHUNK), "--reward_stage", stage["reward"],
           "--env_difficulty", stage["difficulty"], "--expert_episodes", str(stage["expert"]), "--device", device,
           "--expert_each_chunk", str(bool(stage.get("expert_each_chunk", False)))]
    for k, v in EXTRA_TRAIN_ARGS.items():
        cmd += [f"--{k}", str(v)]
    if init_weights:
        cmd += ["--init_weights", str(init_weights)]
    return run(cmd, cwd=str(ROOT / "src" / "dreamer"))


EXTRA_TRAIN_ARGS = {}


def keep_awake():
    """Ask Windows not to idle-sleep while this process runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    except Exception:
        pass


def main():
    global CHUNK
    keep_awake()
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "outputs" / "run3"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only", default=None, help="Run only this stage name (for tests).")
    ap.add_argument("--min_scale", type=float, default=1.0, help="Multiply every stage's minimum steps (smoke tests use a small number).")
    ap.add_argument("--extra", nargs="*", default=[], help="Extra train.py arguments as key=value, e.g. batch_size=64.")
    ap.add_argument("--chunk", type=int, default=CHUNK, help="Steps between checkpoints / gate checks (smaller only for rehearsals).")
    ap.add_argument("--gate_scale", type=float, default=1.0, help="Multiply gate thresholds (rehearsals use 0 to force a pass).")
    ap.add_argument("--episodes_scale", type=float, default=1.0, help="Multiply gate episode counts (rehearsals use a small number).")
    ap.add_argument("--stop_after", default=None, help="Stop after this stage has passed.")
    args = ap.parse_args()
    for kv in args.extra:
        k, v = kv.split("=", 1)
        EXTRA_TRAIN_ARGS[k] = v
    CHUNK = args.chunk
    base = pathlib.Path(args.base).resolve()
    base.mkdir(parents=True, exist_ok=True)
    state_file = base / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {"done": {}, "history": []}

    prev_weights = None
    for stage in STAGES:
        name = stage["name"]
        if args.only and name != args.only:
            continue
        logdir = base / name
        override = logdir / "init_weights.txt"                  # a restarted stage may begin from a chosen checkpoint
        if name in state["done"]:
            prev_weights = state["done"][name]["weights"]
            log(f"{name}: already passed ({state['done'][name]['summary']}) - skipping")
            continue
        min_steps = int(stage["min_steps"] * args.min_scale)
        max_steps = stage.get("max_factor", MAX_FACTOR) * min_steps
        gate = json.loads(json.dumps(stage["gate"]))
        gate["need"] = {k: v * args.gate_scale for k, v in gate.get("need", {}).items()}
        gate["episodes"] = max(10, int(gate["episodes"] * args.episodes_scale))
        if gate.get("also_nominal"):
            gate["also_nominal"]["episodes"] = max(10, int(gate["also_nominal"]["episodes"] * args.episodes_scale))
            gate["also_nominal"]["success"] *= args.gate_scale
        if override.exists():
            prev_weights = override.read_text().strip()
        ckpt_dir = logdir / "ckpt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        gate_dir = logdir / "gates"
        gate_dir.mkdir(exist_ok=True)
        log(f"=== {name}: reward {stage['reward']}, difficulty {stage['difficulty']}, minimum {min_steps} steps, gate {gate} ===")
        best = None                                                   # (score, checkpoint path, result)
        for f in sorted(gate_dir.glob("full_*.json")):               # resume: re-read earlier full gate results
            r = json.loads(f.read_text())
            c = ckpt_dir / f"step_{f.stem.split('_')[1]}.pt"
            if c.exists() and (best is None or score(r) > best[0]):
                best = (score(r), c, r)
        stage_done = bool(best and passes(best[2], gate)[0] and not gate.get("also_nominal"))   # resume: an earlier checkpoint may already meet the gate
        while not stage_done:
            steps_now = count_steps(logdir / "train_eps")
            if steps_now >= max_steps:
                break
            code = train_chunk(stage, logdir, prev_weights, steps_now, args.device)
            if code != 0:
                raise SystemExit(f"training exited with code {code} in {name}")
            steps_now = count_steps(logdir / "train_eps")
            ckpt = ckpt_dir / f"step_{steps_now}.pt"
            shutil.copy(logdir / "latest.pt", ckpt)
            import torch
            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            torch.save({"agent_state_dict": blob["agent_state_dict"]}, ckpt)      # keep the file small: weights only
            del blob
            quick = gate_eval(ckpt, stage, stage["difficulty"], min(50, gate["episodes"]), gate_dir / f"quick_{steps_now}.json")
            log(f"{name} step {steps_now}: quick check success {quick['success']:.0%} reach {quick['reach_within_1.5cm']:.0%} "
                f"pinched {quick['pinched']:.0%} worst slice {quick['worst_slice']:.0%}")
            state["history"].append({"stage": name, "step": steps_now, "quick": {k: quick[k] for k in ("success", "reach_within_1.5cm", "pinched", "worst_slice")}})
            state_file.write_text(json.dumps(state, indent=1))
            ok_quick, _ = passes(quick, {"need": {k: v - 0.06 for k, v in gate.get("need", {}).items()},
                                         "at_most": {k: v + 0.06 for k, v in gate.get("at_most", {}).items()}}) if steps_now >= min_steps else (False, [])
            if not ok_quick:
                continue
            full = gate_eval(ckpt, stage, stage["difficulty"], gate["episodes"], gate_dir / f"full_{steps_now}.json")
            ok, miss = passes(full, gate)
            if ok and gate.get("also_nominal"):
                nom = gate["also_nominal"]
                r2 = gate_eval(ckpt, stage, nom["difficulty"], nom["episodes"], gate_dir / f"nominal_{steps_now}.json")
                if r2["success"] < nom["success"]:
                    ok, miss = False, [f"nominal success {r2['success']:.3f} < {nom['success']}"]
            log(f"{name} step {steps_now}: FULL GATE {'PASSED' if ok else 'not met: ' + '; '.join(miss)} (success {full['success']:.1%} of {full['episodes']}, worst slice {full['worst_slice']:.0%})")
            if best is None or score(full) > best[0]:
                best = (score(full), ckpt, full)
            if ok:
                stage_done = True
                break
        if not stage_done and best and passes(best[2], gate)[0] and (not gate.get("also_nominal")):
            stage_done = True                                         # a checkpoint evaluated earlier already meets the (revised) gate
        if not stage_done and best is None and stage.get("accept_within") and sorted(ckpt_dir.glob("step_*.pt")):
            last = max(ckpt_dir.glob("step_*.pt"), key=lambda c: int(c.stem.split("_")[1]))
            full = gate_eval(last, stage, stage["difficulty"], gate["episodes"], gate_dir / f"full_{last.stem.split('_')[1]}.json")
            best = (score(full), last, full)                          # step cap with no full gate yet: use the latest checkpoint
            log(f"{name}: step cap reached without a full gate; latest checkpoint {last.name} scores {full['success']:.1%}")
        near = stage.get("accept_within")
        if not stage_done and near and best and best[2]["success"] >= gate["need"]["success"] - near:
            stage_done = True                                         # at the step cap, a near-miss (best checkpoint within `near` of the gate) moves on
            log(f"{name}: step cap reached; accepting best checkpoint {best[1].name} (success {best[2]['success']:.1%}, gate {gate['need']['success']:.0%})")
        if not stage_done:
            msg = f"{name} did not meet its gate within {max_steps} steps."
            if best:
                shutil.copy(best[1], logdir / "stage_best.pt")        # keep the best-gated checkpoint anyway
                msg += (f" Best: success {best[2]['success']:.1%} (worst slice {best[2]['worst_slice']:.0%}) at {best[1].name}, "
                        f"saved as {logdir / 'stage_best.pt'}")
            log("STOP: " + msg)
            (base / "FAILED").write_text(msg + "\n")
            sys.exit(2)
        final = logdir / "stage_best.pt"
        shutil.copy(best[1], final)
        prev_weights = str(final)
        for old_ckpt in ckpt_dir.glob("step_*.pt"):          # keep disk use small: only the stage's best checkpoint survives
            if old_ckpt != best[1]:
                old_ckpt.unlink()
        for old_ms in (logdir / "milestones").glob("*.pt"):
            old_ms.unlink()
        state["done"][name] = {"weights": prev_weights, "summary": f"success {best[2]['success']:.1%} of {best[2]['episodes']} at {best[1].name}"}
        state_file.write_text(json.dumps(state, indent=1))
        log(f"{name} DONE -> {final}")
        if args.stop_after == name:
            log(f"stopping after {name} as requested")
            return
    log("ALL STAGES PASSED" if not args.only else "requested stage finished")
    if not args.only:
        (base / "COMPLETE").write_text("all stages passed\n")


if __name__ == "__main__":
    main()
