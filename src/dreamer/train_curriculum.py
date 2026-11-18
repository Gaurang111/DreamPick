"""Runs the full reach -> grasp -> lift curriculum sequentially. Each stage
trains in its own subdirectory (so the replay buffer/checkpoints/milestones
stay stage-specific and comparable), but the NEURAL NETWORK WEIGHTS carry
over between stages via --init_weights: each stage after the first starts
from the previous stage's final latest.pt instead of random initialization,
so the world model doesn't have to relearn basic scene dynamics (what the
arm/cube look like, how joints move) from scratch every stage - only the
actor/critic need to adapt to the new reward.

Each stage trains until it plateaus (early stopping - see configs.yaml's
early_stop_patience/early_stop_min_improvement/early_stop_min_steps) or hits
--steps, whichever comes first. A stage ends at the first evaluation past the
cap (every 10,000 steps), so --steps 140000 means a stage runs to about
151,000 steps at most.

The next stage starts from the previous stage's BEST-scoring weights
(<stage>/best.pt, saved at the best evaluation) rather than its last ones, which
aren't necessarily its best.

While running, this asks Windows not to idle-sleep (released automatically when
the process exits). It does not stop the laptop sleeping when the lid is closed.
Launch it with scripts/start_training.ps1 so it runs independently of the
terminal or chat session that started it.

This is resume-safe the same way a single train.py run is: if interrupted
mid-stage, rerunning this script will resume that stage from its own
latest.pt (not restart it), then continue to the remaining stages. Rerunning
after a stage already finished just re-confirms it has plateaued (a little
wasted overhead, not incorrect) before moving on.

Usage (from src/dreamer/):
    python train_curriculum.py --base_logdir ../../outputs/dreamer_logdir --steps 140000
"""

import argparse
import pathlib
import subprocess
import sys

STAGES = ["reach", "grasp", "lift"]


def keep_awake():
    """Ask Windows not to idle-sleep while this process runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    except Exception:
        pass  # not Windows, or not allowed - training just runs without the hold


def main():
    keep_awake()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_logdir", required=True, help="Each stage gets a subdirectory under this.")
    parser.add_argument("--steps", type=int, default=140000,
                         help="Per-stage step cap - early stopping may end a stage sooner.")
    parser.add_argument("--early_stop_min_steps", type=int, default=None,
                         help="Override the config's earliest step at which early stopping may end a stage "
                              "(used when extending a finished stage).")
    parser.add_argument("--init_weights", default=None,
                         help="Weights file to start the FIRST stage that runs from (instead of the previous stage's "
                              "best.pt) - used to retrain one stage from an already-trained policy.")
    parser.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES,
                         help="Which stages to run, in order.")
    args = parser.parse_args()

    base_logdir = pathlib.Path(args.base_logdir).expanduser()
    train_py = pathlib.Path(__file__).parent / "train.py"
    python = sys.executable

    prev_stage_weights = pathlib.Path(args.init_weights) if args.init_weights else None
    for stage in args.stages:
        stage_logdir = base_logdir / stage
        done_marker = stage_logdir / "STAGE_DONE"

        if done_marker.exists():
            # Finished in an earlier launch (an interruption later on, then a restart): don't re-run it.
            print(f"\n=== Stage {stage} already finished - skipping ===\n", flush=True)
        else:
            print(f"\n=== Starting stage: {stage} (logdir: {stage_logdir}) ===\n", flush=True)
            cmd = [
                python, str(train_py),
                "--configs", "so101_pick_cube",
                "--reward_stage", stage,
                "--steps", str(args.steps),
                "--logdir", str(stage_logdir),
            ]
            if args.early_stop_min_steps is not None:
                cmd += ["--early_stop_min_steps", str(args.early_stop_min_steps)]
            if prev_stage_weights is not None:
                cmd += ["--init_weights", str(prev_stage_weights)]

            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"Stage {stage} exited with code {result.returncode} - stopping curriculum here.", flush=True)
                sys.exit(result.returncode)
            done_marker.write_text("finished\n")
            print(f"\n=== Finished stage: {stage} ===\n", flush=True)

        best, latest = stage_logdir / "best.pt", stage_logdir / "latest.pt"
        prev_stage_weights = best if best.exists() else latest
        if not prev_stage_weights.exists():
            print(f"Stage {stage} finished but no weights were found in {stage_logdir} - stopping.", flush=True)
            sys.exit(1)
        print(f"Next stage will start from {prev_stage_weights.name}", flush=True)

    print("\n=== Curriculum complete ===")


if __name__ == "__main__":
    main()
