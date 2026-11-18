"""Where run 3 is: current stage, steps, latest gate numbers, finished stages.

    .venv\Scripts\python.exe scripts\run3_status.py [--base outputs/run3]
"""
import argparse
import json
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
STAGES = ["s1_reach", "s2_grasp", "s3_lift", "s4a_zone", "s4b_noise", "s4c_occlusion", "s5_robust", "s6_polish"]


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in pathlib.Path(folder).glob("*.npz"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "outputs" / "run3"))
    base = pathlib.Path(ap.parse_args().base)
    state = json.loads((base / "state.json").read_text()) if (base / "state.json").exists() else {"done": {}, "history": []}
    if (base / "COMPLETE").exists():
        print("RUN COMPLETE: all stages passed.")
    if (base / "FAILED").exists():
        print("RUN STOPPED: " + (base / "FAILED").read_text().strip())
    for name in STAGES:
        d = base / name
        if name in state["done"]:
            print(f"{name:10s} DONE  {state['done'][name]['summary']}")
            continue
        if not d.exists():
            print(f"{name:10s} not started")
            continue
        steps = count_steps(d / "train_eps")
        hist = [h for h in state["history"] if h["stage"] == name]
        last = hist[-1] if hist else None
        age = time.time() - max([f.stat().st_mtime for f in (d / "train_eps").glob("*.npz")] or [0])
        line = f"{name:10s} RUNNING  {steps:,} steps (last data {age / 60:.0f} min ago)"
        if last:
            q = last["quick"]
            line += f" | quick check at {last['step']:,}: success {q['success']:.0%}, reach {q['reach_within_1.5cm']:.0%}, pinched {q['pinched']:.0%}"
        print(line)
        for h in hist[-6:]:
            q = h["quick"]
            print(f"    step {h['step']:>8,}: success {q['success']:.0%} reach {q['reach_within_1.5cm']:.0%} pinched {q['pinched']:.0%} worst slice {q['worst_slice']:.0%}")
        break
    log = ROOT / "outputs" / "run3_console.log"
    if log.exists():
        gate = [l for l in log.read_text(errors="ignore").splitlines() if "GATE" in l or "STOP" in l or "DONE" in l]
        if gate:
            print("latest gate line: " + gate[-1])


if __name__ == "__main__":
    main()
