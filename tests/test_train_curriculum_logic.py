"""Unit tests for the curriculum warm-start (init_weights) and early-stopping
decision logic in src/dreamer/train.py. These two functions are pure (no
torch/env/logger I/O), extracted specifically so they're testable without a
GPU, a real checkpoint, or a live training run.

src/dreamer isn't a package (no __init__.py - its scripts import each other
via sys.path.append(own dir), same pattern used here) so it's added to
sys.path directly in this file rather than conftest.py, to avoid the very
generic module names in src/dreamer (tools, models, ...) shadowing anything
for unrelated tests.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "dreamer"))

from train import compute_early_stop, determine_checkpoint_source, is_new_best, read_best_eval, read_eval_history  # noqa: E402
import tools  # noqa: E402


# --- determine_checkpoint_source ---------------------------------------


def test_checkpoint_source_fresh_when_nothing_exists(tmp_path):
    source, path = determine_checkpoint_source(tmp_path, init_weights="")
    assert source == "fresh"
    assert path is None


def test_checkpoint_source_init_weights_when_no_local_checkpoint(tmp_path):
    prev_stage = tmp_path / "reach" / "latest.pt"
    prev_stage.parent.mkdir(parents=True)
    prev_stage.write_bytes(b"fake checkpoint")

    source, path = determine_checkpoint_source(tmp_path / "grasp", init_weights=str(prev_stage))
    assert source == "init_weights"
    assert path == prev_stage


def test_checkpoint_source_resume_takes_priority_over_init_weights(tmp_path):
    # Same-dir latest.pt means this stage is being resumed after an
    # interruption - it must win even if --init_weights was also passed
    # (e.g. a curriculum rerun where this stage already has progress).
    stage_dir = tmp_path / "grasp"
    stage_dir.mkdir()
    local_checkpoint = stage_dir / "latest.pt"
    local_checkpoint.write_bytes(b"local checkpoint")
    prev_stage = tmp_path / "reach" / "latest.pt"
    prev_stage.parent.mkdir()
    prev_stage.write_bytes(b"fake checkpoint")

    source, path = determine_checkpoint_source(stage_dir, init_weights=str(prev_stage))
    assert source == "resume"
    assert path == local_checkpoint


def test_checkpoint_source_fresh_when_init_weights_path_missing(tmp_path):
    # init_weights pointing at a nonexistent file (e.g. a typo, or the
    # previous stage never finished) shouldn't silently resolve to "fresh" -
    # determine_checkpoint_source only decides WHICH path to use, not
    # whether it's loadable, so it should still report 'init_weights' and
    # let the actual torch.load() in main() raise.
    source, path = determine_checkpoint_source(
        tmp_path, init_weights=str(tmp_path / "does_not_exist" / "latest.pt")
    )
    assert source == "init_weights"
    assert path == tmp_path / "does_not_exist" / "latest.pt"


# --- compute_early_stop --------------------------------------------------


def test_early_stop_no_history_yet():
    should_stop, improvement = compute_early_stop([1.0], patience=5, min_improvement=0.02)
    assert should_stop is False
    assert improvement is None


def test_early_stop_disabled_when_patience_zero():
    history = [1.0] * 10
    should_stop, improvement = compute_early_stop(history, patience=0, min_improvement=0.02)
    assert should_stop is False
    assert improvement is None


def test_early_stop_triggers_on_plateau():
    # baseline (index -(patience+1)) = 100, best of the last `patience`
    # evals barely moves - well under the 2% improvement threshold.
    history = [100.0, 100.5, 100.2, 100.8, 100.1, 100.6]
    should_stop, improvement = compute_early_stop(history, patience=5, min_improvement=0.02)
    assert should_stop is True
    assert improvement == (100.8 - 100.0) / 100.0


def test_early_stop_does_not_trigger_on_real_improvement():
    history = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]
    should_stop, improvement = compute_early_stop(history, patience=5, min_improvement=0.02)
    assert should_stop is False
    assert improvement == (150.0 - 100.0) / 100.0


def test_early_stop_uses_best_not_last_in_window():
    # A dip on the very last eval shouldn't trigger a stop if an earlier eval
    # within the patience window was still a real improvement over baseline.
    history = [100.0, 150.0, 90.0]
    should_stop, improvement = compute_early_stop(history, patience=2, min_improvement=0.02)
    assert should_stop is False
    assert improvement == (150.0 - 100.0) / 100.0


def test_early_stop_handles_near_zero_baseline():
    # baseline ~0 would divide-by-zero without the abs(baseline) floor.
    history = [0.0, 0.01, 0.02]
    should_stop, improvement = compute_early_stop(history, patience=2, min_improvement=0.02)
    assert should_stop is False
    assert improvement == (0.02 - 0.0) / 1e-6


# --- read_eval_history -------------------------------------------------------


def test_eval_history_survives_logger_write(tmp_path):
    # tools.simulate logs eval_return and immediately calls logger.write(),
    # which clears the logger's scalars - so early stopping must read the
    # history from metrics.jsonl, not from the logger.
    logger = tools.Logger(tmp_path, 0)
    for step, ret in [(1000, 5.0), (11000, 99.0), (21000, 111.0)]:
        logger.step = step
        logger.scalar("eval_return", ret)
        logger.write()
        logger.scalar("train_return", 1.0)  # non-eval lines in between are ignored
        logger.write()
    assert read_eval_history(tmp_path / "metrics.jsonl") == [5.0, 99.0, 111.0]


def test_eval_history_skips_truncated_lines_and_missing_file(tmp_path):
    path = tmp_path / "metrics.jsonl"
    assert read_eval_history(path) == []
    path.write_text('{"step": 1, "eval_return": 3.0}\n{"step": 2, "eval_ret\n{"step": 3, "eval_return": 7.5}\n')
    assert read_eval_history(path) == [3.0, 7.5]


# --- minimum steps, best checkpoint ------------------------------------------------


def test_early_stop_waits_for_the_minimum_number_of_steps():
    flat = [100.0] * 8  # a clear plateau
    stop, _ = compute_early_stop(flat, patience=5, min_improvement=0.02, step=30000, min_steps=40000)
    assert stop is False
    stop, _ = compute_early_stop(flat, patience=5, min_improvement=0.02, step=40000, min_steps=40000)
    assert stop is True


def test_early_stop_ignores_min_steps_when_not_given():
    # old call style still works: no step => no minimum applies
    stop, _ = compute_early_stop([100.0] * 8, patience=5, min_improvement=0.02)
    assert stop is True


def test_best_eval_bookkeeping(tmp_path):
    path = tmp_path / "best.json"
    assert read_best_eval(path) == (None, None)
    assert is_new_best(5.0, None) is True
    path.write_text(json.dumps({"eval_return": 412.5, "step": 37000}))
    assert read_best_eval(path) == (412.5, 37000)
    assert is_new_best(430.0, 412.5) is True
    assert is_new_best(412.5, 412.5) is False  # a tie is not an improvement
    assert is_new_best(300.0, 412.5) is False


def test_best_eval_survives_a_corrupt_file(tmp_path):
    path = tmp_path / "best.json"
    path.write_text("{not json")
    assert read_best_eval(path) == (None, None)
