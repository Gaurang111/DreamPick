"""Builds the documentation assets for a finished (or stopped) training run:
learning-curve and policy-test charts, and every kind of video, into one folder.

    python scripts/make_training_docs.py --base_logdir outputs/dreamer_logdir --out docs/training_run_1

Everything is read-only with respect to the logdir. Steps are skipped when
their output already exists (use --force to redo), and a failing step is
logged and skipped so one bad video doesn't lose the rest. Run it with the
project venv BEFORE changing the scene, camera or reward - the world-model
videos render with the scene as it is now.

Steps (--only takes any of these names):
  curves      learning curves per stage, from each stage's metrics.jsonl
  evaluate    policies tested in the sim (deterministic + sampled actions) -> JSON + charts
  views       policy rollouts with the viewer, overhead and wrist cameras side by side
  eye         what the model saw: stored training episodes from early to late, per stage
  worldmodel  reconstruction/dream, milestone-progression and imagined-arm videos, per stage
  timelapse   per-update joint-position timelapse, per stage
  slideshow   TensorBoard train_openl slideshow GIFs, per stage
  data        copy metrics, console log and config
"""

import argparse
import glob
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DREAMER = ROOT / "src" / "dreamer"
PY = sys.executable
STAGES = ["reach", "grasp", "lift"]

# First three slots of the validated categorical palette (blue, orange, aqua).
STAGE_COLOR = {"reach": "#2a78d6", "grasp": "#eb6834", "lift": "#1baf7a"}
POLICY_COLOR = {"random": "#8a8982", **STAGE_COLOR}
STAGE_MAX_RETURN = {"reach": 200, "grasp": 400, "lift": 800}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

log_lines = []


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_lines.append(line)


def run(cmd, cwd=DREAMER, what=""):
    t = time.time()
    log(f"running: {what}")
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if p.returncode:
        log(f"  FAILED ({what}): {(p.stderr or p.stdout)[-600:]}")
        return False
    log(f"  done in {time.time() - t:.0f}s")
    return True


def need(path, force):
    if path.exists() and not force:
        log(f"skip (exists): {path.name}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    return True


# --- charts ---------------------------------------------------------------


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def read_metrics(path):
    rows = []
    for line in pathlib.Path(path).read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def series(rows, key):
    pts = [(r["step"], r[key]) for r in rows if key in r]
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def make_curves(base, out, force):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = out / "plots" / "learning_curves.png"
    if not need(path, force):
        return
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), facecolor=SURFACE)
    for i, stage in enumerate(STAGES):
        mpath = base / stage / "metrics.jsonl"
        if not mpath.exists():
            continue
        rows = read_metrics(mpath)
        c = STAGE_COLOR[stage]
        ax = axes[i, 0]
        style_axes(ax)
        x, y = series(rows, "model_loss")
        ax.plot(x / 1000, y, color=c, linewidth=2, marker="o", markersize=4)
        use_log = y.max() / y.min() > 8
        if use_log:
            from matplotlib.ticker import ScalarFormatter
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(ScalarFormatter())
        ax.set_ylabel(f"{stage}\nworld-model loss" + (" (log scale)" if use_log else ""), color=INK, fontsize=10)
        if i == 0:
            ax.set_title("World model: how well it predicts", color=INK, fontsize=12, loc="left")
        ax.annotate(f"{y[-1]:.0f}", (x[-1] / 1000, y[-1]), textcoords="offset points", xytext=(4, 6), color=INK, fontsize=9)

        ax = axes[i, 1]
        style_axes(ax)
        xt, yt = series(rows, "train_return")
        if len(yt):
            k = 10
            smooth = np.convolve(yt, np.ones(k) / k, mode="valid")
            ax.plot(xt[k - 1:] / 1000, smooth, color=c, alpha=0.35, linewidth=1.5, label="practice episodes (10-episode average)")
        xe, ye = series(rows, "eval_return")
        ax.plot(xe / 1000, ye, color=c, linewidth=2.2, marker="o", markersize=5, label="evaluation (3 episodes)")
        ax.axhline(STAGE_MAX_RETURN[stage], color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.text(0.99, STAGE_MAX_RETURN[stage], " best possible ≈ %d" % STAGE_MAX_RETURN[stage], transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", color=INK2, fontsize=8)
        ax.set_ylim(0, STAGE_MAX_RETURN[stage] * 1.1)
        ax.set_ylabel("episode return", color=INK, fontsize=10)
        if i == 0:
            ax.set_title("Policy: score per 10-second episode", color=INK, fontsize=12, loc="left")
        ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=INK2)
        if i == 2:
            axes[i, 0].set_xlabel("environment steps in this stage (thousands)", color=INK2, fontsize=9)
            axes[i, 1].set_xlabel("environment steps in this stage (thousands)", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    log(f"wrote {path.name}")


def make_policy_charts(out, force):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jpath = out / "data" / "policy_test_deterministic.json"
    if not jpath.exists():
        return
    data = json.loads(jpath.read_text())
    summ, eps = data["summary"], data["per_episode"]
    names = list(summ)

    path = out / "plots" / "policy_test.png"
    if need(path, force):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.2, 1]})
        style_axes(a1)
        w = 0.38
        xs = np.arange(len(names))
        a1.bar(xs - w / 2, [summ[n]["time_within_2cm"] * 100 for n in names], w, color=[POLICY_COLOR[n] for n in names], label="within 2 cm of the cube")
        a1.bar(xs + w / 2, [summ[n]["time_grasped"] * 100 for n in names], w, color=[POLICY_COLOR[n] for n in names], alpha=0.45,
               label="cube pinched between both jaws")
        for x, n in zip(xs, names):
            a1.text(x - w / 2, summ[n]["time_within_2cm"] * 100 + 1.5, f"{summ[n]['time_within_2cm'] * 100:.0f}%", ha="center", color=INK, fontsize=9)
            a1.text(x + w / 2, summ[n]["time_grasped"] * 100 + 1.5, f"{summ[n]['time_grasped'] * 100:.0f}%", ha="center", color=INK, fontsize=9)
        a1.set_xticks(xs, names)
        a1.set_ylabel("share of each 10-second episode", color=INK2, fontsize=9)
        a1.set_ylim(0, 105)
        a1.set_title("Reaching and gripping (solid = near, light = gripping)", color=INK, fontsize=11, loc="left")
        style_axes(a2)
        a2.bar(xs - w / 2, [summ[n]["max_height_cm"] for n in names], w, color=[POLICY_COLOR[n] for n in names], label="average best height")
        a2.bar(xs + w / 2, [summ[n]["best_height_cm"] for n in names], w, color=[POLICY_COLOR[n] for n in names], alpha=0.45, label="best single episode")
        a2.axhline(5, color=INK, linewidth=1.2, linestyle=(0, (4, 3)))
        a2.text(len(names) - 0.5, 5.1, "5 cm target", ha="right", va="bottom", color=INK, fontsize=9)
        for x, n in zip(xs, names):
            a2.text(x - w / 2, summ[n]["max_height_cm"] + 0.1, f"{summ[n]['max_height_cm']:.1f}", ha="center", color=INK, fontsize=9)
            a2.text(x + w / 2, summ[n]["best_height_cm"] + 0.1, f"{summ[n]['best_height_cm']:.1f}", ha="center", color=INK, fontsize=9)
        a2.set_xticks(xs, names)
        a2.set_ylabel("cube lifted above the mat (cm)", color=INK2, fontsize=9)
        a2.set_ylim(0, 6)
        a2.set_title("Lifting (solid = average, light = best)", color=INK, fontsize=11, loc="left")
        fig.suptitle(f"Policies tested in the sim: {data['episodes']} fresh episodes each", color=INK, fontsize=13, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(path, dpi=140, facecolor=SURFACE)
        plt.close(fig)
        log(f"wrote {path.name}")

    path = out / "plots" / "cube_dragged_toward_base.png"
    if need(path, force):
        fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
        style_axes(ax)
        t = np.arange(200) / 20
        label_y = {}
        for n in names:
            paths = np.array([e["cube_path"] for e in eps[n]])  # episodes x steps x 3
            x_cm = paths[:, :, 0] * 100
            m, lo, hi = x_cm.mean(0), np.percentile(x_cm, 10, axis=0), np.percentile(x_cm, 90, axis=0)
            ax.fill_between(t[: len(m)], lo, hi, color=POLICY_COLOR[n], alpha=0.12, linewidth=0)
            ax.plot(t[: len(m)], m, color=POLICY_COLOR[n], linewidth=2.2)
            label_y[n] = m[-1]
        placed = []
        for n in sorted(names, key=lambda k: label_y[k]):  # keep the end labels from overlapping
            y = label_y[n]
            while any(abs(y - q) < 0.9 for q in placed):
                y += 0.9
            placed.append(y)
            ax.text(t[-1] + 0.12, y, n, color=INK, fontsize=10, va="center")
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.text(0.1, 0.5, "arm base", color=INK2, fontsize=8, va="bottom")
        ax.set_xlim(0, 11.2)
        ax.set_xlabel("seconds into the episode", color=INK2, fontsize=9)
        ax.set_ylabel("cube distance from the arm base (cm)", color=INK2, fontsize=9)
        ax.set_title("Every trained policy pulls the cube in toward the arm", color=INK, fontsize=12, loc="left")
        fig.tight_layout()
        fig.savefig(path, dpi=140, facecolor=SURFACE)
        plt.close(fig)
        log(f"wrote {path.name}")


# --- policy evaluation and camera-view videos ---------------------------------


def evaluate(base, out, force):
    for sampled in (False, True):
        name = "policy_test_sampled.json" if sampled else "policy_test_deterministic.json"
        path = out / "data" / name
        if not need(path, force):
            continue
        cmd = [PY, "evaluate_policy.py", "--base_logdir", base, "--episodes", 20, "--json_out", path]
        if sampled:
            cmd.append("--sample")
        run(cmd, what=f"policy evaluation ({'sampled' if sampled else 'deterministic'} actions)")


def label(frame, text, height=22):
    from PIL import Image, ImageDraw
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, height], fill=(0, 0, 0))
    d.text((6, 5), text, fill=(255, 255, 255))
    return np.array(im)


def policy_views(base, out, force):
    video_dir = out / "videos" / "policy_rollouts"
    targets = {n: video_dir / f"{n}_policy_three_cameras.mp4" for n in ["random"] + STAGES}
    grid_path = video_dir / "all_policies_side_by_side.mp4"
    if not any(need(p, force) for p in list(targets.values()) + [grid_path]):
        return
    sys.path.insert(0, str(DREAMER))
    import imageio
    import mujoco
    from PIL import Image
    from envs.so101 import SO101Dreamer
    from evaluate_policy import PolicyRunner
    from generate_dream_video import load_config

    config = load_config(base)
    env = SO101Dreamer("lift", seed=2024)
    e = env._env
    renderer = mujoco.Renderer(e.model, height=360, width=640)
    rng = np.random.default_rng(2024)
    first_episode = {}
    for name in ["random"] + STAGES:
        ckpt = base / name / "latest.pt"
        if name != "random" and not ckpt.exists():
            continue
        pol = None if name == "random" else PolicyRunner(ckpt, config, env.observation_space, env.action_space)
        writer = imageio.get_writer(targets[name], fps=20, macro_block_size=1)
        for ep in range(2):
            obs = env.reset()
            if pol:
                pol.reset()
            done, t, ret = False, 0, 0.0
            while not done:
                action = rng.uniform(-1, 1, 6).astype(np.float32) if pol is None else pol.act(obs)
                obs, reward, done, info = env.step(action)
                ret += reward
                t += 1
                renderer.update_scene(e.data, camera="viewer_cam")
                view = renderer.render()
                ov = np.array(Image.fromarray(obs["overhead_image"]).resize((360, 360), Image.NEAREST))
                wr = np.array(Image.fromarray(obs["wrist_image"]).resize((360, 360), Image.NEAREST))
                head = (f"{name} policy, episode {ep + 1}   t={t / 20:4.1f}s   return {ret:5.0f}   gripping: {'YES' if info['is_grasped'] else 'no '}   "
                        f"cube lifted {info['cube_height'] * 100:3.1f} cm")
                panel = label(np.concatenate([view, ov, wr], axis=1), head)
                writer.append_data(panel)
                if ep == 0:
                    first_episode.setdefault(name, []).append(view.copy())
        writer.close()
        log(f"wrote {targets[name].name}")
        del pol
    names = [n for n in ["random"] + STAGES if n in first_episode]
    if len(names) == 4:
        writer = imageio.get_writer(grid_path, fps=20, macro_block_size=1)
        for i in range(min(len(first_episode[n]) for n in names)):
            tiles = [label(first_episode[n][i], f"{n} policy  t={i / 20:4.1f}s") for n in names]
            writer.append_data(np.concatenate([np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:], axis=1)], axis=0))
        writer.close()
        log(f"wrote {grid_path.name}")


def policy_eye(base, out, force):
    import imageio
    from PIL import Image
    for stage in STAGES:
        files = sorted(glob.glob(str(base / stage / "train_eps" / "*.npz")), key=os.path.getmtime)
        path = out / "videos" / "training_progression" / f"{stage}_what_the_model_saw_early_to_late.mp4"
        if not files or not need(path, force):
            continue
        picks = np.linspace(0, len(files) - 1, 5).round().astype(int)
        writer = imageio.get_writer(path, fps=20, macro_block_size=1)
        for i in picks:
            ep = np.load(files[i])
            ov, wr, rew = ep["overhead_image"], ep["wrist_image"], ep["reward"]
            ret = np.cumsum(rew)
            for t in range(len(rew)):
                frame = np.concatenate([
                    np.array(Image.fromarray(ov[t]).resize((320, 320), Image.NEAREST)),
                    np.array(Image.fromarray(wr[t]).resize((320, 320), Image.NEAREST)),
                ], axis=1)
                frame = label(frame, f"{stage} stage, practice episode {i + 1} of {len(files)} (about step {i * 200})  t={t / 20:4.1f}s  return {ret[t]:5.0f}")
                writer.append_data(frame)
        writer.close()
        log(f"wrote {path.name}")


# --- the existing generator scripts, per stage ----------------------------------


def worldmodel(base, out, force):
    wm = out / "videos" / "world_model"
    for stage in STAGES:
        d = base / stage
        if not (d / "latest.pt").exists():
            continue
        jobs = [
            ("generate_dream_video.py", f"{stage}_reconstruction_and_dream.mp4", []),
            ("generate_progression_video.py", f"{stage}_world_model_progression_across_milestones.mp4", []),
            ("generate_imagined_arm_video.py", f"{stage}_imagined_arm_final_model.mp4", []),
            ("generate_imagined_arm_progression_video.py", f"{stage}_imagined_arm_progression.mp4", ["--episode_selection", "earliest"]),
        ]
        for script, fname, extra in jobs:
            path = wm / fname
            if need(path, force):
                run([PY, script, "--logdir", d, "--out", path, *extra], what=f"{script} [{stage}]")


def timelapse(base, out, force):
    for stage in STAGES:
        d = base / stage
        path = out / "videos" / "training_progression" / f"{stage}_arm_motion_timelapse_every_update.mp4"
        if (d / "joint_pos_trace.bin").exists() and need(path, force):
            run([PY, "generate_joint_pos_timelapse_video.py", "--logdir", d, "--stride", 12, "--out", path, "--fps", 30],
                what=f"joint-position timelapse [{stage}]")


def slideshow(base, out, force):
    for stage in STAGES:
        d = base / stage
        for tag in ("train_openl", "eval_openl"):
            path = out / "videos" / "world_model" / f"{stage}_{tag}_slideshow.gif"
            if d.exists() and need(path, force):
                run([PY, "generate_train_openl_slideshow.py", "--logdir", d, "--tag", tag, "--out", path], what=f"{tag} slideshow [{stage}]")


def copy_data(base, out, force):
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    for stage in STAGES:
        src = base / stage / "metrics.jsonl"
        if src.exists():
            shutil.copy2(src, data / f"metrics_{stage}.jsonl")
    for src in (base.parent / f"{base.name}_console.log", DREAMER / "configs.yaml"):
        if src.exists():
            shutil.copy2(src, data / src.name)
    log("copied metrics, console log and config")


STEPS = {
    "curves": make_curves, "evaluate": evaluate, "views": policy_views, "eye": policy_eye,
    "worldmodel": worldmodel, "timelapse": timelapse, "slideshow": slideshow, "data": copy_data,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_logdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="+", choices=list(STEPS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    base, out = pathlib.Path(args.base_logdir).resolve(), pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    order = ["data", "curves", "evaluate", "views", "eye", "worldmodel", "slideshow", "timelapse"]
    for name in [s for s in order if not args.only or s in args.only]:
        log(f"=== {name}")
        try:
            STEPS[name](base, out, args.force)
        except Exception as e:  # keep going - one broken video shouldn't lose the rest
            import traceback
            log(f"  STEP {name} FAILED: {e}\n{traceback.format_exc()[-800:]}")
    make_policy_charts(out, args.force)
    (out / "data" / "build_log.txt").write_text("\n".join(log_lines))
    log("finished")


if __name__ == "__main__":
    main()
