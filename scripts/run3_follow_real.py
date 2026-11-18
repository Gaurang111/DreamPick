"""Sim first, then the real arm copies the sim arm (phase 2: smooth sim -> real).

1. The arm's current pose is read (torque switched on holding it, after your ENTER).
2. The final policy picks a cube placed at a random spot in the SIMULATOR, starting from that pose. The joint targets
   pass through the smoother (so101_env.target_smoothing.py) and the whole episode is planned before anything moves.
3. You see the plan (cube spot, success, speeds, roughness) and press ENTER; after a countdown the real arm replays the
   sim arm's goals at 20 Hz through sim_real_bridge.Bridge (speed cap, range limits, rollover guard, sag compensation).
   The real arm does not feed back into the policy: it only follows. Tracking error is watched and the run stops if a
   joint falls 18 deg behind.
4. A report compares real against sim (sync error) and rates how rough the real motion was.

Servers/venvs (same as run3_real_client.py): policy server in the project venv; this script in the LeRobot venv.
    --dry   no hardware: plan only (works in either venv), good for checking the smoothing and the cube spots.
Put the real cube where the plan says (printed x, y) if you want the arm to really grasp it; otherwise keep the mat clear.
"""
import argparse
import json
import math
import pathlib
import sys
import time

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from run3_real_client import HZ, MAX_TRACK_ERR_DEG, OUT, RealArm, Server  # noqa: E402
from so101_env.metrics import episode_metrics  # noqa: E402
from so101_env.target_smoothing import make_smoother, smoothness  # noqa: E402
from so101_env.state_pick_env import CUBE_SCALE, CUBE_X0, REST_QPOS, SO101StatePickEnv  # noqa: E402


def place_arm(env, q):
    q = np.clip(np.asarray(q, float), env._low, env._high)
    env.data.qpos[env._adr] = q
    env.data.qvel[:] = 0.0
    env._target = q.copy()
    env._v[:] = 0.0
    env.data.ctrl[:] = q
    mujoco.mj_forward(env.model, env.data)
    return env._observe()


def ease_to_start_pose(arm, gripper_rad=0.0, tol_deg=1.5, seconds=10.0):
    """Move the real arm (torque on, speed-capped by the bridge) to the pose every training episode of the folded kind starts
    from (REST_QPOS: lift -99.7, elbow 91.7, wrist 57.3 deg, all inside the usable range), so the sim and real arms start identical
    and clear of the stops. The real rest position sits 3 to 16 deg beyond that, on its hard stops."""
    target = REST_QPOS.copy()
    target[5] = gripper_rad                                   # training starts the gripper anywhere in 0..0.7 rad; 0 is the closed end
    print("easing the real arm to the sim's start pose ...", flush=True)
    t_end = time.time() + seconds
    while time.time() < t_end:
        q = arm.read()                                        # raises if a joint passed its rollover point
        arm.command(target)
        if np.degrees(np.abs(q - target))[:5].max() < tol_deg:
            break
        time.sleep(1.0 / HZ)
    q = arm.read()
    print("start pose now (deg):", np.degrees(q).round(1), " sim start:", np.degrees(target).round(1))
    return q


def place_cube(env, x_cm, y_cm, yaw_deg):
    """Put the sim cube where the real one was measured (policy frame), instead of the random spot env.reset picked."""
    x, y, yaw = x_cm / 100.0, y_cm / 100.0, math.radians(yaw_deg)
    env.data.qpos[env._cube_adr:env._cube_adr + 7] = [x, y, env._cube_rest_z, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    env.data.qvel[:] = 0.0
    env._cube_start_xy, env._cube_start_radius = np.array([x, y]), float(np.hypot(x, y))
    env._cube_hist.clear()                                    # forget the random spot's observation history
    mujoco.mj_forward(env.model, env.data)


def close_step_of(goals):
    """Index of the plan step at which the gripper starts closing for the grasp (first step after its widest point that is 8 deg shut)."""
    g = np.degrees(goals[:, 5])
    top = int(np.argmax(g))
    return next((i for i in range(top, len(g)) if g[i] < g.max() - 8.0), len(g) - 1)


def pause_picture(cap, env, q_real, cube_pose, path):
    """Top-down camera picture of the real arm and cube at the pause, with where the SIM thinks the pads and the cube are drawn on it."""
    import cv2
    import locate_cube as lc
    from so101_env import camera_map
    for _ in range(10):                                       # flush buffered frames
        cap.read()
    ok, img = cap.read()
    if not ok:
        print("  (could not read the camera)")
        return None
    fl, _ = lc.get_flattener(img)
    m = camera_map.load()
    A, b = (np.array(m["A"]), np.array(m["b"])) if m else (np.eye(2), np.zeros(2))

    def to_px(x_cm, y_cm):                                    # policy frame (cm out, cm left) -> pixel in the raw picture
        cx, cy = np.linalg.solve(A, np.array([x_cm, y_cm]) - b)
        px = int(np.clip(lc.RECT_W / 2 - cy * lc.PX_PER_CM, 0, lc.RECT_W - 1))
        py = int(np.clip(lc.RECT_H - cx * lc.PX_PER_CM, 0, lc.RECT_H - 1))
        return int(fl.map_x[py, px]), int(fl.map_y[py, px])

    env.data.qpos[env._adr] = np.clip(np.asarray(q_real, float), env._low, env._high)
    mujoco.mj_forward(env.model, env.data)
    mid = env.pad_mid()[:2] * 100
    fixed, moving = env.data.xpos[env._fixed_pad][:2] * 100, env.data.xpos[env._moving_pad][:2] * 100
    cv2.drawMarker(img, to_px(*mid), (255, 255, 0), cv2.MARKER_CROSS, 28, 2)               # cyan cross: sim says the pads' centre is here
    cv2.circle(img, to_px(*fixed), 7, (0, 255, 255), 2)                                      # yellow: fixed pad
    cv2.circle(img, to_px(*moving), 7, (255, 0, 255), 2)                                     # magenta: moving pad
    cv2.drawMarker(img, to_px(cube_pose[0], cube_pose[1]), (0, 0, 255), cv2.MARKER_SQUARE, 30, 2)   # red square: where the camera measured the cube
    cv2.putText(img, "cross = pads centre (sim kinematics of the real joints), red square = cube as measured", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(path), img)
    res, _ = lc.analyse(img.copy(), 0.0, 0.0, fl)
    seen = camera_map.to_arm_frame(res["x_cm"], res["y_cm"], res["yaw_deg"]) if res else None
    return mid, fixed, moving, seen


class TopRecorder:
    """Records the overhead camera to a video file from a background thread, so it never slows the 20 Hz control loop."""

    def __init__(self, cap, path):
        self.cap, self.path, self.ts, self.running, self.writer = cap, path, [], False, None
        self.latest = None                                    # the newest frame, for the main thread to look at (e.g. to find the cube)
        self.stride = 1                                       # write every Nth frame (long waits are recorded sparsely, the run itself fully)
        self.detections = []                                  # (time, cube pose in the policy frame or None) while waiting for the cube
        self.t_lock = None                                    # when the cube was locked onto

    def start(self):
        import threading
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        import cv2
        n = 0
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            self.latest = frame
            n += 1
            if n % self.stride:
                continue
            if self.writer is None:
                h, w = frame.shape[:2]
                self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
            self.writer.write(frame)
            self.ts.append(time.perf_counter())

    def stop(self):
        self.running = False
        self.thread.join(timeout=3.0)
        if self.writer is not None:
            self.writer.release()


def wait_for_cube(rec, in_zone, settle_s=2.0, max_std_cm=0.15):
    """The camera is already recording: wait until a cube sits still on the mat inside the zone and return its pose in the policy frame.
    (Your hand placing the cube is on the video; a moving cube or hand just keeps it waiting.)"""
    import locate_cube as lc
    from so101_env import camera_map
    fl, hist, told = None, [], False
    seen_empty, empty_since, warned_old = False, None, False
    print("\nThe camera is recording. If a cube is on the mat, take it off; then PLACE THE CUBE ON THE MAT (inside the cyan zone) and let go ...", flush=True)
    while True:
        f = rec.latest
        if f is None:
            time.sleep(0.1)
            continue
        if fl is None:
            fl, _ = lc.get_flattener(f)
        res, _ = lc.analyse(f.copy(), 0.0, 0.0, fl)
        now = time.time()
        rec.detections.append((time.perf_counter(), camera_map.to_arm_frame(res["x_cm"], res["y_cm"], res["yaw_deg"]) if res else None))   # for the real->sim video
        if not seen_empty:                                    # a cube seen before the mat was empty is the OLD one: never lock onto it
            if res is None:
                empty_since = empty_since or now
                if now - empty_since >= 1.5:
                    seen_empty = True
                    print("  mat is empty - now place the cube.", flush=True)
            else:
                empty_since = None
                if not warned_old:
                    print("  a cube is already on the mat: take it off first (I must see the mat empty), then place it again.", flush=True)
                    warned_old = True
            continue
        if res is None:                                       # no cube in view (e.g. a hand is placing it): start over
            hist = []
            continue
        hist.append((now, res["x_cm"], res["y_cm"], res["yaw_deg"]))
        hist = [h for h in hist if now - h[0] <= 6.0]         # one frame takes about a second to analyse, so the window is long
        if len(hist) >= 3 and now - hist[0][0] >= settle_s:
            a = np.array([h[1:] for h in hist])
            if a[:, 0].std() <= max_std_cm and a[:, 1].std() <= max_std_cm:
                yaw = float(np.degrees(np.angle(np.mean(np.exp(1j * 4 * np.radians(a[:, 2]))))) / 4.0)
                pose = camera_map.to_arm_frame(float(np.median(a[:, 0])), float(np.median(a[:, 1])), yaw)
                if in_zone(pose[0], pose[1]):
                    rec.t_lock = time.perf_counter()
                    return pose
                if not told:
                    print(f"  the cube is at {pose[0]:.1f} cm out, {pose[1]:+.1f} cm left: outside the zone (21-31 cm out, +-10 cm). Move it inside the box.", flush=True)
                    told = True
                hist = []


def current_cube(rec, n=3):
    """The cube's pose now (policy frame), median of n camera frames from the running recorder; None if it is not seen."""
    import locate_cube as lc
    from so101_env import camera_map
    fl, got = None, []
    for _ in range(n):
        f = rec.latest
        if f is None:
            continue
        if fl is None:
            fl, _ = lc.get_flattener(f)
        res, _ = lc.analyse(f.copy(), 0.0, 0.0, fl)
        if res:
            got.append((res["x_cm"], res["y_cm"], res["yaw_deg"]))
    if len(got) < 2:
        return None
    a = np.array(got)
    yaw = float(np.degrees(np.angle(np.mean(np.exp(1j * 4 * np.radians(a[:, 2]))))) / 4.0)
    return camera_map.to_arm_frame(float(np.median(a[:, 0])), float(np.median(a[:, 1])), yaw)


def make_real_to_sim(plan, env, rec, folder):
    """Video 1, real -> sim: the real top camera while the cube is being placed, next to a sim whose cube sits where the camera sees it."""
    import cv2
    if rec.t_lock is None or not rec.ts:
        return None
    W, H = 768, 432
    ts = np.array(rec.ts)
    n_frames = int(np.searchsorted(ts, rec.t_lock + 0.7))
    renderer = mujoco.Renderer(env.model, H, W)
    vdata = mujoco.MjData(env.model)
    vdata.qpos[:] = plan["qpos"][0]
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.22, 0.0, 0.06]
    cam.distance, cam.azimuth, cam.elevation = 0.8, 145, -28
    w = cv2.VideoWriter(str(folder / "real_to_sim.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 7.0, (2 * W, H))
    src = cv2.VideoCapture(str(rec.path))
    t0 = ts[0]
    for i in range(n_frames):
        ok, f = src.read()
        if not ok:
            break
        seen = None
        for td, pose in rec.detections:                       # the newest camera detection at or before this frame
            if td <= ts[i]:
                seen = pose
            else:
                break
        if seen is None:
            vdata.qpos[env._cube_adr:env._cube_adr + 7] = [0.0, 0.0, -0.5, 1, 0, 0, 0]       # not seen: out of sight under the table
        else:
            x, y, yaw = seen[0] / 100.0, seen[1] / 100.0, math.radians(seen[2])
            vdata.qpos[env._cube_adr:env._cube_adr + 7] = [x, y, env._cube_rest_z, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_forward(env.model, vdata)
        renderer.update_scene(vdata, camera=cam)
        sim = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        real = cv2.resize(f, (W, H))
        cv2.putText(real, f"real: placing the cube  t={ts[i] - t0:4.1f}s", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        label = f"sim: cube where the camera sees it ({seen[0]:.1f}, {seen[1]:+.1f} cm)" if seen else "sim: camera does not see the cube"
        cv2.putText(sim, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        w.write(np.hstack([real, sim]))
    src.release()
    w.release()
    return folder / "real_to_sim.mp4"


def make_videos(plan, env, rec, step_ts, folder, stretch=1.0):
    """After the run: the sim replay as its own video, and the real top camera next to it, matched step by step."""
    import cv2
    W, H = 768, 432
    folder.mkdir(parents=True, exist_ok=True)
    need = {}                                                     # plan step -> index of the camera frame nearest in time
    ts = np.array(rec.ts)
    for k, t in enumerate(step_ts):
        need[k + 1] = int(np.argmin(np.abs(ts - t))) if len(ts) else 0
    need[0] = need.get(1, 0)
    wanted = set(need.values())
    top, cap = {}, cv2.VideoCapture(str(rec.path))
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in wanted:
            top[i] = cv2.resize(f, (W, H))
        i += 1
    cap.release()
    renderer = mujoco.Renderer(env.model, H, W)
    vdata = mujoco.MjData(env.model)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.22, 0.0, 0.06]
    cam.distance, cam.azimuth, cam.elevation = 0.8, 145, -28
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    sim_w = cv2.VideoWriter(str(folder / "sim.mp4"), fourcc, 20.0, (W, H))
    both_w = cv2.VideoWriter(str(folder / "sim_to_real.mp4"), fourcc, 20.0, (2 * W, H))
    last_top = next(iter(top.values())) if top else np.zeros((H, W, 3), np.uint8)
    for k in range(len(step_ts) + 1):
        vdata.qpos[:] = plan["qpos"][min(int(k / stretch), len(plan["qpos"]) - 1)]      # the replay is `stretch` times slower than the plan
        mujoco.mj_forward(env.model, vdata)
        renderer.update_scene(vdata, camera=cam)
        sim = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        real = top.get(need.get(k, -1), last_top)
        last_top = real
        sim, real = sim.copy(), real.copy()
        cv2.putText(sim, f"sim plan  t={k / HZ:4.1f}s", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(real, f"real arm, top camera  t={k / HZ:4.1f}s", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        sim_w.write(sim)
        both_w.write(np.hstack([real, sim]))
    sim_w.release()
    both_w.release()
    return folder


def plan_episode(env, server, smoother, seed, q0, cube_pose=None):
    """One policy episode in the simulator. Returns dict with the servo-goal path (T, 6 rad) and how it went."""
    obs, _ = env.reset(seed=seed)
    if cube_pose is not None:
        place_cube(env, *cube_pose)
        obs = env._observe()
    if q0 is not None:
        obs = place_arm(env, q0)
    smoother.reset(env._target)
    env.ctrl_hook = smoother.step
    server.ask("reset", obs["proprio"], cube_vec=obs["cube"])
    goals, joints, h, sh, tb, done = [env._target.copy()], [], [], [], [], False
    cube_xy = env.cube_pos()[:2] * 100
    qpos_traj = [env.data.qpos.copy()]                        # the whole sim state each step, to play the pick back on screen
    actions = []
    while not done:
        reply = server.ask("step", obs["proprio"], cube_vec=obs["cube"])
        actions.append(np.array(reply["action"]))
        obs, _, _, done, info = env.step(np.array(reply["action"]))
        goals.append(smoother.x.copy()); joints.append(info["joint_pos"]); qpos_traj.append(env.data.qpos.copy())
        h.append(info["cube_height"]); sh.append(info["cube_shift"]); tb.append(info["table_force"])
    env.ctrl_hook = None
    m = episode_metrics(h, sh, tb, joints)
    goals = np.array(goals)
    rough = smoothness(goals, np.vstack([goals[:1], joints]), env.dt)
    return dict(seed=seed, goals=goals, success=bool(m["success"]), peak_cm=float(m["peak_cm"]),
                cube_cm=(float(cube_xy[0]), float(cube_xy[1])), rough=rough,
                hold_gripper_deg=float(np.degrees(joints[-1][5])),      # where the sim gripper ends up (stopped by the cube when it holds one)
                qpos=np.array(qpos_traj), actions=np.array(actions),
                max_speed_deg_s=float(np.degrees(np.abs(np.diff(goals[:, :5], axis=0))).max() * HZ))


def restrict_env_to_real(env, bridge):
    """Plan inside the joint ranges the real arm can actually use (its calibrated range cut at the encoder wrap, in sim angles through the
    fitted joint offsets), so a plan never asks for what the real arm would silently clip (the wrist stops 9 deg short of the policy's range)."""
    from sim_real_bridge import JOINTS
    for i, j in enumerate(JOINTS[:5]):
        a, b = bridge.real_to_sim(j, bridge.real_lo[j]), bridge.real_to_sim(j, bridge.real_hi[j])
        lo, hi = min(a, b), max(a, b)
        old = (math.degrees(env._low[i]), math.degrees(env._high[i]))
        env._low[i], env._high[i] = max(env._low[i], lo), min(env._high[i], hi)
        new = (math.degrees(env._low[i]), math.degrees(env._high[i]))
        if abs(old[0] - new[0]) > 0.2 or abs(old[1] - new[1]) > 0.2:
            print(f"  planning limit {j}: {old[0]:.1f}..{old[1]:.1f} -> {new[0]:.1f}..{new[1]:.1f} deg (sim angles)")


def robust_replay(env, smoother, plan, cube_pose, q0, offset):
    """Replay a plan's recorded actions open loop (as the real arm will) with the cube displaced by offset = (dx mm, dy mm, dyaw deg). Success?"""
    dx, dy, dyaw = offset
    env.reset(seed=plan["seed"])
    place_cube(env, cube_pose[0] + dx / 10.0, cube_pose[1] + dy / 10.0, cube_pose[2] + dyaw)
    place_arm(env, q0)
    smoother.reset(env._target)
    env.ctrl_hook = smoother.step
    h, sh, tb, joints = [], [], [], []
    for a in plan["actions"]:
        _, _, _, done, info = env.step(a)
        h.append(info["cube_height"]); sh.append(info["cube_shift"]); tb.append(info["table_force"]); joints.append(info["joint_pos"])
        if done:
            break
    env.ctrl_hook = None
    return bool(episode_metrics(h, sh, tb, joints)["success"])


def stretch_goals(goals, s):
    """The same joint path played s times slower (linear interpolation between the plan's 20 Hz goals)."""
    if s == 1.0:
        return np.asarray(goals)
    T = len(goals)
    t = np.arange(int((T - 1) * s) + 1) / s
    i0 = np.minimum(t.astype(int), T - 1)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (t - i0)[:, None]
    return (1 - w) * np.asarray(goals)[i0] + w * np.asarray(goals)[i1]


def replay_goals_in_sim(env, goals, cube_pose, q0, seed=0, offset=(0, 0, 0)):
    """Drive the sim arm with a goal path (no policy), as the real arm will be driven, and report whether the cube ends up lifted and held."""
    dz = offset[3] if len(offset) > 3 else 0.0                # mm: the whole arm held this much higher (+) or lower (-) than the plan assumes
    base = env.model.body("base").id
    z0 = float(env.model.body_pos[base][2])
    env.model.body_pos[base][2] = z0 + dz / 1000.0
    try:
        env.reset(seed=seed)
        place_cube(env, cube_pose[0] + offset[0] / 10.0, cube_pose[1] + offset[1] / 10.0, cube_pose[2] + offset[2])
        place_arm(env, q0)
        h, sh, tb, joints = [], [], [], []
        for g in goals[1:]:
            env._target = np.clip(np.asarray(g, float), env._low, env._high)
            env.data.ctrl[:] = env._target
            for _ in range(env.frame_skip):
                mujoco.mj_step(env.model, env.data)
            info = env._info()
            h.append(info["cube_height"]); sh.append(info["cube_shift"]); tb.append(info["table_force"]); joints.append(info["joint_pos"])
        return bool(episode_metrics(h, sh, tb, joints)["success"])
    finally:
        env.model.body_pos[base][2] = z0
        mujoco.mj_forward(env.model, env.data)


def choose_plan(env, server, smoother, args, seed0, q0, cube_pose):
    """Plan several times, keep the ones that pick the cube, then pick the plan that still works when the cube is really a few mm / degrees
    off from where the camera put it (the calibration is good to about 6 mm)."""
    cands = []
    for i in range(args.tries):
        p = plan_episode(env, server, smoother, seed0 + i, q0, cube_pose)
        print(describe(p), flush=True)
        if p["success"] or args.allow_fail:
            cands.append(p)
            if args.seed is not None or len(cands) >= args.robust_k:
                break
    if not cands:
        return None
    if cube_pose is None or args.robust_mm <= 0 or len(cands) == 1:
        return cands[0]
    s = args.robust_mm
    offsets = [(0, 0, 0, 0)] + [(dx * s, dy * s, 0, 0) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))] + [(0, 0, 10, 0), (0, 0, -10, 0)]
    offsets += [(0, 0, 0, dz) for dz in (-8, 4, 8, 12)]       # the real gripper tends to sit higher than the plan: heights up to 12 mm too high, 8 mm too low
    print(f"\nchecking {len(cands)} plans against the cube being up to {s} mm / 10 deg off and the arm 8 mm too low .. 12 mm too high ...", flush=True)
    for p in cands:                                           # as the real arm will be driven: the (slowed) goal path, open loop
        wins = [replay_goals_in_sim(env, stretch_goals(p["goals"], args.stretch), cube_pose, q0, seed=p["seed"], offset=o) for o in offsets]
        p["robust"] = float(np.mean(wins))
        print(f"  seed {p['seed']}: picks the cube in {sum(wins)} of {len(wins)} shifted cases", flush=True)
    return max(cands, key=lambda p: p["robust"])


def describe(p):
    r = p["rough"]["goal"]
    return (f"seed {p['seed']}: cube at {p['cube_cm'][0]:.1f} cm out, {p['cube_cm'][1]:+.1f} cm left | sim "
            f"{'SUCCESS' if p['success'] else 'failed'} (peak {p['peak_cm']:.1f} cm) | goal path: top speed {p['max_speed_deg_s']:.0f} deg/s, "
            f"biggest jump {r['jump_max_deg']:.1f} deg/step, accel p99 {r['accel_p99']:.1f} rad/s2, jerk p99 {r['jerk_p99']:.0f} rad/s3")


def report(goals, real_q, dt):
    n = min(len(goals), len(real_q))
    g, rq = np.asarray(goals)[:n], np.asarray(real_q)[:n]
    err = np.degrees(np.abs(g - rq))
    print("\nsync, real arm vs the sim arm's goal (deg; gripper in sim-angle deg):")
    for i, name in enumerate(("pan", "lift", "elbow", "wrist flex", "wrist roll", "gripper")):
        print(f"  {name:<11} mean {err[:, i].mean():5.1f}   max {err[:, i].max():5.1f}")
    r = smoothness(g, rq, dt)
    print("how rough: goal path vs real arm (arm joints)")
    for key, label in (("goal", "goal path"), ("meas", "real arm ")):
        s = r[key]
        print(f"  {label}: accel p99 {s['accel_p99']:6.1f} rad/s2 | jerk p99 {s['jerk_p99']:7.0f} rad/s3 | max jerk {s['jerk_max']:7.0f} | "
              f"biggest step {s['jump_max_deg']:.1f} deg | direction flips {s['flips_per_s']:.2f}/s")
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry", action="store_true", help="Plan only, no hardware.")
    ap.add_argument("--difficulty", default=None, help="Sim conditions for planning (default: polish; wide_exact with --real_cube, since the cube position is measured).")
    ap.add_argument("--real_cube", action="store_true", help="Measure the REAL cube with the overhead camera (through the phase-1 map) and plan the pick for that spot.")
    ap.add_argument("--seed", type=int, default=None, help="Cube spot / episode seed (default: random).")
    ap.add_argument("--tries", type=int, default=30, help="Plan up to this many seeds while looking for plans that pick the cube.")
    ap.add_argument("--allow_fail", action="store_true", help="Accept a plan where the sim pick fails.")
    ap.add_argument("--smooth", default="crit", choices=["hold", "interp", "ema", "crit"])
    ap.add_argument("--tau", type=float, default=0.05, help="Smoother time constant, s.")
    ap.add_argument("--servo_accel", type=int, default=40, help="Servo Acceleration register for the arm joints (LeRobot default 254; ~30 matches the policy's own limit).")
    ap.add_argument("--start_gripper", type=float, default=0.4, help="Gripper angle at the start, sim rad (0 = nearly closed, 0.7 = wide open; training used 0..0.7).")
    ap.add_argument("--robust_k", type=int, default=6, help="Keep this many successful plans and pick the most tolerant one.")
    ap.add_argument("--robust_mm", type=float, default=8.0, help="Test plans against the cube being this many mm off (and 10 deg in yaw); 0 = skip.")
    ap.add_argument("--no_view", action="store_true", help="Do not open the sim window.")
    ap.add_argument("--no_record", action="store_true", help="Do not record the top camera / sim videos (outputs/real_live/run_<time>/).")
    ap.add_argument("--pause_at_close", action="store_true", help="Pause just before the gripper closes and take a picture with the pads and cube marked (needs --real_cube).")
    ap.add_argument("--stretch", type=float, default=2.0, help="Play the replay this many times slower than planned (the servos track a slow path much better).")
    ap.add_argument("--lead", type=int, default=4, help="Send the arm joints' goals this many control steps (50 ms) early; measured servo lag is 4.")
    ap.add_argument("--grip_lead", type=int, default=3, help="Same for the gripper (measured lag 3).")
    ap.add_argument("--max_step", type=float, default=3.0)
    ap.add_argument("--server_port", type=int, default=8770)
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--robot_id", default="my_follower")
    args = ap.parse_args()

    env = SO101StatePickEnv("lift", args.difficulty or ("wide_exact" if args.real_cube else "polish"))
    smoother = make_smoother(env.n, env.dt, env.frame_skip, args.smooth, args.tau)
    server = Server(args.server_port)
    arm, q0, cube_pose, cap = None, None, None, None
    viewer, post, rec = None, None, None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = OUT / f"run_{stamp}"
    if args.real_cube:
        from run3_real_client import in_zone, measure_cube
        if not args.no_record and not args.pause_at_close and not args.dry:
            import cv2
            import locate_cube as lc
            print("opening the camera and starting the recording ...", flush=True)       # recording starts BEFORE the cube is placed
            cap = cv2.VideoCapture(lc.find_camera(), cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            for _ in range(20):
                cap.read()
            run_dir.mkdir(parents=True, exist_ok=True)
            rec = TopRecorder(cap, run_dir / "top_camera.mp4")
            rec.stride = 3                                    # waiting and planning are recorded at about 7 frames/s to keep the file small
            rec.start()
            cube_pose = wait_for_cube(rec, in_zone)
        else:
            print("Looking for the real cube ...", flush=True)
            cube_pose = measure_cube()                        # policy frame, through the camera -> arm map
            if not in_zone(cube_pose[0], cube_pose[1]):
                raise SystemExit("The cube is outside the trained zone (21-31 cm out, +-10 cm). Move it (see the zone picture) and run again.")
        print(f"cube (policy frame): {cube_pose[0]:.1f} cm out, {cube_pose[1]:+.1f} cm left, turned {cube_pose[2]:+.0f} deg", flush=True)
    armed = False
    if not args.dry:
        arm = RealArm(args.port, args.robot_id, args.max_step)      # only builds the bridge; nothing is opened until arm.start()
        arm.bridge.servo_accel = args.servo_accel
        print(f"servo acceleration for the arm joints: {args.servo_accel} (LeRobot default 254)")
        restrict_env_to_real(env, arm.bridge)
    else:
        try:                                                  # a dry run in the LeRobot venv still plans inside the real arm's ranges
            from sim_real_bridge import Bridge
            restrict_env_to_real(env, Bridge(args.port, args.robot_id, args.max_step))
        except ImportError:
            pass
    try:
        q0 = REST_QPOS.copy()                                 # the pose the real arm is eased to before the run; the plan starts there
        q0[5] = args.start_gripper
        seed0 = args.seed if args.seed is not None else int(np.random.default_rng().integers(1_000_000, 9_000_000))
        plan = choose_plan(env, server, smoother, args, seed0, q0, cube_pose)
        if plan is None:
            print("No successful plan found; nothing was moved.")
            return

        def cube_moved():
            """Re-measure the cube; if it is more than 6 mm from where the plan assumes (hand, bump, anything), return the new pose."""
            if rec is None or cube_pose is None:
                return None
            now = current_cube(rec)
            if now is None:
                print("  (the camera cannot see the cube right now - keeping the plan)", flush=True)
                return None
            d = math.hypot(now[0] - cube_pose[0], now[1] - cube_pose[1]) * 10
            return now if d > 6.0 else None

        for _ in range(3):                                    # the cube may have been touched while planning: re-plan for where it is now
            moved = cube_moved()
            if moved is None:
                break
            print(f"  the cube moved: {cube_pose[0]:.1f},{cube_pose[1]:+.1f} -> {moved[0]:.1f},{moved[1]:+.1f} cm. Re-planning ...", flush=True)
            cube_pose = moved
            plan = choose_plan(env, server, smoother, args, seed0, q0, cube_pose)
            if plan is None:
                print("No successful plan found for the new cube position; nothing was moved.")
                return
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"plan_{stamp}.json").write_text(json.dumps({"seed": plan["seed"], "cube_cm": plan["cube_cm"], "goals": plan["goals"].tolist(),
                                                              "smooth": [args.smooth, args.tau], "servo_accel": args.servo_accel}))
        if args.dry:
            print("dry run: done (nothing moved).")
            return
        print(f"\nPlanned. The real arm will copy the sim arm: {len(plan['goals']) / HZ:.0f} s. Cube spot in the sim: "
              f"{plan['cube_cm'][0]:.1f} cm out, {plan['cube_cm'][1]:+.1f} cm left (put the real cube there to really pick it, or keep the mat clear).")
        if not arm.start():                                   # ENTER -> torque on, holding the current pose
            return
        armed = True
        sag_file = OUT / "sag_k.json"                         # starting sag compensation fitted from earlier runs (scripts/fit_sag.py)
        if sag_file.exists():
            arm.bridge.sag_k.update(json.loads(sag_file.read_text()))
            print("sag compensation starts from:", {j: round(v, 2) for j, v in arm.bridge.sag_k.items()})
        q_real = ease_to_start_pose(arm, args.start_gripper)
        print("arm pose (sim rad):", np.round(q_real, 2))
        for _ in range(3):                                    # last look before the arm moves: still the same cube position?
            moved = cube_moved()
            if moved is None:
                break
            print(f"  the cube moved: {cube_pose[0]:.1f},{cube_pose[1]:+.1f} -> {moved[0]:.1f},{moved[1]:+.1f} cm. Re-planning (the arm waits) ...", flush=True)
            cube_pose = moved
            plan = choose_plan(env, server, smoother, args, seed0, q0, cube_pose)
            if plan is None:
                print("No successful plan found for the new cube position; stopping.")
                return
        if np.degrees(np.abs(q_real - plan["qpos"][0][env._adr]))[:5].max() > 3.0:
            print("  WARNING: the real start pose differs from the planned start pose by more than 3 deg; the plan assumed the sim start pose.")
        goals, real_q, stopped = stretch_goals(plan["goals"], args.stretch), [], None      # played `stretch` times slower: smaller servo errors at the grasp
        T = len(goals)
        print(f"replay: {T / HZ:.0f} s ({args.stretch:g}x slower than planned)")
        viewer, vdata = None, None
        if not args.no_view:                                   # the planned pick, drawn on screen in step with the real arm
            import mujoco.viewer
            vdata = mujoco.MjData(env.model)
            vdata.qpos[:] = plan["qpos"][0]
            mujoco.mj_forward(env.model, vdata)
            viewer = mujoco.viewer.launch_passive(env.model, vdata)
            viewer.cam.lookat[:] = [0.22, 0.0, 0.06]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.8, 145, -28
            viewer.sync()
            print("sim window open: the planned pick plays there while the real arm follows.", flush=True)
        pause_k = int(close_step_of(plan["goals"]) * args.stretch) - args.grip_lead - 1 if (args.pause_at_close and cube_pose is not None) else None
        step_ts = []
        if rec is None and (pause_k is not None or not args.no_record):     # not already recording: open the camera now, for the pause picture and/or the recording
            import cv2
            import locate_cube as lc
            print("opening the camera ...", flush=True)
            cap = cv2.VideoCapture(lc.find_camera(), cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            for _ in range(20):
                cap.read()
            if pause_k is None and not args.no_record:
                run_dir.mkdir(parents=True, exist_ok=True)
                rec = TopRecorder(cap, run_dir / "top_camera.mp4")
        print("Starting in 5 s - hand near the power switch, Ctrl-C stops.", flush=True)
        if rec is not None:
            rec.stride = 1                                     # full frame rate for the run itself
            if not rec.running:
                rec.start()
        for s in range(5, 0, -1):
            print(f"  starting in {s} ...", flush=True)
            time.sleep(1.0)
        try:
            for k in range(1, T):
                t0 = time.perf_counter()
                step_ts.append(t0)
                real_q.append(arm.read())
                if viewer is not None and viewer.is_running():
                    vdata.qpos[:] = plan["qpos"][min(int(k / args.stretch), len(plan["qpos"]) - 1)]
                    mujoco.mj_forward(env.model, vdata)
                    viewer.sync()
                cmd = goals[min(k + args.lead, T - 1)].copy()               # the servos trail their goal by ~0.2 s: send it early
                cmd[5] = goals[min(k + args.grip_lead, T - 1)][5]
                arm.command(cmd)
                if pause_k is not None and k == pause_k:
                    for _ in range(int(1.5 * HZ)):                          # keep holding this goal so the arm settles
                        real_q_hold = arm.read()
                        arm.command(cmd)
                        time.sleep(1.0 / HZ)
                    info = pause_picture(cap, env, real_q_hold, cube_pose, OUT / "pause_before_closing.jpg")
                    print(f"\nPAUSED just before the gripper closes (plan step {k} of {T}).")
                    if info:
                        mid, fixed, moving, seen = info
                        print(f"  pads centre (sim kinematics of the real joints): {mid[0]:.1f} cm out, {mid[1]:+.1f} cm left  | fixed pad {fixed[0]:.1f},{fixed[1]:+.1f}  moving pad {moving[0]:.1f},{moving[1]:+.1f}")
                        print(f"  cube as measured before the run: {cube_pose[0]:.1f} cm out, {cube_pose[1]:+.1f} cm left", end="")
                        print(f"  | camera sees it now at {seen[0]:.1f}, {seen[1]:+.1f}" if seen else "  | the camera cannot see it now (hidden by the gripper)")
                        print(f"  picture: {OUT / 'pause_before_closing.jpg'}")
                    input("Look at the gripper and the cube now. Press ENTER to close the gripper and lift (Ctrl-C stops here)... ")
                    t0 = time.perf_counter()
                if arm.tracking_error_deg() > MAX_TRACK_ERR_DEG:
                    stopped = f"stopped at step {k}: a joint is {arm.tracking_error_deg():.0f} deg from its goal"
                    break
                time.sleep(max(0.0, 1.0 / HZ - (time.perf_counter() - t0)))
        except KeyboardInterrupt:
            stopped = "stopped by Ctrl-C"
        if rec is not None:
            time.sleep(1.5)                                    # a little of the held pose on video
            rec.stop()
            def post():                                        # rendered after the arm is released, not while it holds the cube
                p1 = make_real_to_sim(plan, env, rec, run_dir)
                p2 = make_videos(plan, env, rec, step_ts, run_dir, args.stretch)
                print("videos:", p1, "and", run_dir / "sim_to_real.mp4", "(+ sim.mp4, top_camera.mp4)")
        if stopped:
            print(stopped)
        if real_q:
            rep = report(goals[1:len(real_q) + 1], real_q, env.dt)
            g_end = float(np.degrees(real_q[-1][5]))
            print(f"\ngripper at the end: {g_end:.0f} deg (sim, holding the cube: {plan['hold_gripper_deg']:.0f} deg; shut on nothing: about -7 deg)  ->  "
                  f"{'HOLDING SOMETHING' if g_end > -7 + 0.5 * (plan['hold_gripper_deg'] + 7) else 'closed on nothing: the cube was missed'}")
            (OUT / f"follow_{stamp}.json").write_text(json.dumps({"plan": f"plan_{stamp}.json", "real_q": np.array(real_q).tolist(),
                                                                    "stopped": stopped, "rough": rep, "stretch": args.stretch, "lead": args.lead}))
    finally:
        if cap is not None:
            cap.release()
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass
        if arm is not None and armed:
            arm.finish()
    if post is not None:
        print("rendering the sim and side-by-side videos (about 30 s) ...", flush=True)
        post()


if __name__ == "__main__":
    main()
