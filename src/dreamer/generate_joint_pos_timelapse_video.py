"""Renders the joint_pos_trace.bin log (one predicted joint_pos trajectory per
training update, logged continuously - see joint_pos_trace.py and
Dreamer._record_joint_pos_probe in train.py) into a smooth, continuous
progression video: one frame per logged training update, not per sparse
milestone, so there's no jump-cut feel between snapshots.

Each logged record holds a full (T,6) trajectory (context + imagined); pick
one representative step from it per frame with --frame_select. Use --stride
to subsample if the full-density video would be too long, and --until_step
to cut the video off at (approximately) a given env step - e.g. to isolate
just the early, chaotic part of training before it stabilized. "Approximate"
because _update_count resets on every process restart; see
step_calibration.py for how update values get mapped back to env steps.

Renders all three cameras per frame: viewer_cam full-screen (main) with
wrist_cam/overhead_cam as corner insets, same composition as
scripts/random_rollout.py - since this drives the real simulator directly
(not the model's own image decoder), any camera angle works.

Usage (from src/dreamer/):
    python generate_joint_pos_timelapse_video.py --logdir ../../outputs/dreamer_logdir --until_step 62000
"""

import argparse
import pathlib
import sys

import imageio
import mujoco
import numpy as np
from PIL import Image

sys.path.append(str(pathlib.Path(__file__).parent))
from generate_dream_video import load_config  # noqa: E402
from generate_progression_video import label_frame  # noqa: E402
from joint_pos_trace import read_trace  # noqa: E402
from step_calibration import build_update_to_step_fn  # noqa: E402

SCENE_PATH = pathlib.Path(__file__).resolve().parents[2] / "assets" / "scene_pick_cube.xml"

INSET_MARGIN = 14


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True, help="Same --logdir the training run used.")
    parser.add_argument("--camera", default="viewer_cam", help="Main full-screen camera.")
    parser.add_argument("--stride", type=int, default=1,
                         help="Use every Nth logged update (1 = every single one, full density).")
    parser.add_argument("--until_step", type=int, default=None,
                         help="Only render updates whose calibrated env step is <= this value.")
    parser.add_argument("--max_frames", type=int, default=None,
                         help="Keep only the first N frames after filtering/striding - e.g. to crop to the first "
                              "X minutes at a given --fps (frames = minutes * 60 * fps).")
    parser.add_argument("--frame_select", default="last",
                         help="Which timestep of each update's trajectory to render: 'last', 'first', or an integer index.")
    parser.add_argument("--out", default=None, help="Output mp4 path (default: <logdir>/joint_pos_timelapse.mp4)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960, help="Main view width. Must fit within the scene's "
                                                                 "<visual><global offwidth/offheight> buffer size.")
    parser.add_argument("--height", type=int, default=720, help="Main view height.")
    args = parser.parse_args()
    main_w, main_h = args.width, args.height
    inset_size = int(main_w * 220 / 960)  # keep insets the same proportion of the frame regardless of resolution

    logdir = pathlib.Path(args.logdir).expanduser()
    config = load_config(logdir)

    trace_path = logdir / "joint_pos_trace.bin"
    if not trace_path.exists():
        raise FileNotFoundError(
            f"No trace file at {trace_path} - this is written by train.py as it trains "
            f"(one record per training update), nothing to render until training has run."
        )
    # batch_length is read FROM the trace file's own header, not assumed from
    # config - the two can differ (e.g. --configs debug overrides it).
    batch_length, updates, trajectories = read_trace(trace_path)
    print(f"Loaded {len(updates)} logged updates, batch_length={batch_length}")

    approx_step_fn, eras = build_update_to_step_fn(logdir, updates)
    print(f"Detected {len(eras)} training era(s) (process launches): {eras}")

    era_index = np.zeros(len(updates), dtype=int)
    for i, (start, end) in enumerate(eras):
        era_index[start:end] = i
    approx_steps = np.array([approx_step_fn(u, e) for u, e in zip(updates, era_index)])

    keep = np.ones(len(updates), dtype=bool)
    if args.until_step is not None:
        keep &= approx_steps <= args.until_step
    updates, trajectories, approx_steps = updates[keep], trajectories[keep], approx_steps[keep]
    print(f"After --until_step filter: {len(updates)} updates remain "
          f"(approx step {approx_steps[0]:.0f} to {approx_steps[-1]:.0f})" if len(updates) else "no updates remain")

    if args.frame_select == "last":
        idx = batch_length - 1
    elif args.frame_select == "first":
        idx = 0
    else:
        idx = int(args.frame_select)
    poses = trajectories[:: args.stride, idx]  # (N_selected, 6)
    selected_steps = approx_steps[:: args.stride]
    if args.max_frames is not None:
        poses, selected_steps = poses[: args.max_frames], selected_steps[: args.max_frames]
    print(f"Rendering {len(poses)} frames (stride={args.stride}, frame_select={args.frame_select}, "
          f"{main_w}x{main_h})")

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data_mj = mujoco.MjData(model)
    joint_qpos_adr = np.array([model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)])
    joint_low, joint_high = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]

    main_renderer = mujoco.Renderer(model, height=main_h, width=main_w)
    inset_renderer = mujoco.Renderer(model, height=inset_size, width=inset_size)

    out_path = pathlib.Path(args.out) if args.out else logdir / "joint_pos_timelapse.mp4"
    # Stream frames straight to the encoder instead of collecting them in a
    # list first - at this frame count (tens of thousands x 960x720x3 bytes),
    # holding them all in memory before one mimsave() call ballooned to many
    # GB and never finished.
    writer = imageio.get_writer(out_path, fps=args.fps, quality=10, macro_block_size=1)
    n_frames = 0
    for approx_step, pose in zip(selected_steps, poses):
        pose = np.clip(np.nan_to_num(pose), joint_low, joint_high)
        data_mj.qpos[joint_qpos_adr] = pose
        mujoco.mj_forward(model, data_mj)

        main_renderer.update_scene(data_mj, camera=args.camera)
        frame = label_frame(main_renderer.render(), f"step ~{approx_step:.0f}")
        frame = Image.fromarray(frame).convert("RGB")

        inset_renderer.update_scene(data_mj, camera="wrist_cam")
        wrist_inset = Image.fromarray(inset_renderer.render())
        inset_renderer.update_scene(data_mj, camera="overhead_cam")
        overhead_inset = Image.fromarray(inset_renderer.render())

        frame.paste(wrist_inset, (INSET_MARGIN, INSET_MARGIN))
        frame.paste(overhead_inset, (main_w - inset_size - INSET_MARGIN, INSET_MARGIN))
        writer.append_data(np.array(frame))
        n_frames += 1
        if n_frames % 1000 == 0:
            print(f"  ...{n_frames} frames rendered so far", flush=True)

    main_renderer.close()
    inset_renderer.close()
    writer.close()

    duration_s = n_frames / args.fps
    print(f"saved {out_path} ({n_frames} frames, {duration_s:.1f}s at {args.fps}fps, main={args.camera})")
    print("Note: only the arm's pose is predicted - the cube stays at its fixed resting position throughout.")
    print("Note: step labels are approximate (interpolated from update_count, which resets on each training restart).")


if __name__ == "__main__":
    main()
