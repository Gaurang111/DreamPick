"""Drive the real SO-101 with a run-3 policy: measured joint angles + the overhead camera's cube pose in, joint targets out.

Three programs (see docs/training_run_3_plan.md, section 14):
  1. policy server (project venv):  .venv\\Scripts\\python.exe scripts\\run3_policy_server.py --checkpoint <stage_best.pt>
  2. this client (LeRobot venv, it has OpenCV, the motor library and mujoco):
       C:\\Users\\goran\\PycharmProjects\\so101\\lerobot\\venv\\Scripts\\python.exe scripts\\run3_real_client.py
  3. nothing else may hold COM3 or camera 1 (close scripts\\locate_cube.py --live first).

Per step (20 Hz): read the real joint angles -> ask the server for the next joint target -> send it through
sim_real_bridge.Bridge (speed cap, usable range, rollover guard, sag compensation, all unchanged). The cube pose is
measured once, before the arm moves (median of ~2 s of camera frames) and held for the run, which is what the policy was
trained to expect once the gripper covers the cube.

Sequence: camera finds the cube and it must sit inside the trained zone -> you confirm -> torque on holding the current
pose (ENTER, as in the bridge) -> 5 s countdown -> the policy runs 15 s -> the arm holds -> ENTER opens the gripper
(the cube drops) and the arm eases back to where it started, torque off. Ctrl-C at any time does the same.
Keep a hand near the arm's power switch.

    --mock    no hardware: a simulated arm (lag included) plays the real arm and a simulated cube plays the camera, so the
              whole loop, the server and the logging can be tested. Needs the project venv.
"""
import argparse
import json
import math
import pathlib
import socket
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs" / "real_live"
HZ = 20
ZONE_X, ZONE_Y = (21.0, 31.0), (-10.0, 10.0)           # cm: where the policy was trained to find the cube
MAX_TRACK_ERR_DEG = 18.0                               # stop if a joint (not the gripper) is this far from its goal
OPEN_GRIPPER_RAD = 0.45


class Server:
    def __init__(self, port):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.f = self.sock.makefile("rw")

    def ask(self, cmd, q, cube=None, cube_vec=None):
        msg = {"cmd": cmd, "q": [float(v) for v in q], "cube": [float(v) for v in (cube if cube is not None else (0, 0, 0))]}
        if cube_vec is not None:
            msg["cube_vec"] = [float(v) for v in cube_vec]
        self.f.write(json.dumps(msg) + "\n")
        self.f.flush()
        reply = json.loads(self.f.readline())
        if not reply.get("ok"):
            raise RuntimeError(f"policy server: {reply.get('error')}")
        return reply


def in_zone(x, y):
    return ZONE_X[0] <= x <= ZONE_X[1] and ZONE_Y[0] <= y <= ZONE_Y[1]


# ---------------------------------------------------------------- the arms

class RealArm:
    def __init__(self, port, robot_id, max_step):
        from sim_real_bridge import JOINTS, Bridge
        self.joints = JOINTS
        self.bridge = Bridge(port, robot_id, max_step)
        self.start_pose = None
        self.goal = None

    def start(self):
        self.start_pose = self.bridge.start_torque()          # asks for ENTER; None if Ctrl-C
        return self.start_pose is not None

    def read(self):
        raw = self.bridge.read_raw()
        if not self.bridge.check_rollover(raw):
            raise RuntimeError("a joint passed its rollover point")
        self.present = self.bridge.raw_to_real(raw)
        return np.array([self.bridge.real_to_sim(j, self.present[j]) for j in self.joints])

    def command(self, target_rad):
        target = {j: self.bridge.sim_to_real(j, float(r)) for j, r in zip(self.joints, target_rad)}
        self.goal = self.bridge.send_real(target, self.present)

    def tracking_error_deg(self):
        if self.goal is None:
            return 0.0
        return max(abs(self.goal[j] - self.present[j]) for j in self.joints if j != "gripper")

    def finish(self):
        try:
            input("\nHolding. Press ENTER to open the gripper (the cube drops) and ease the arm back...")
            q = self.read()
            for _ in range(int(2 * HZ)):                       # open the gripper in place first
                q_t = q.copy()
                q_t[5] = OPEN_GRIPPER_RAD
                self.read()
                self.command(q_t)
                time.sleep(1.0 / HZ)
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            try:
                self.bridge.park(self.start_pose)
            finally:
                self.bridge.release()


class MockArm:
    """A simulated arm with the real arm's kind of lag, and a camera that sees the simulated cube (3 mm noise)."""

    def __init__(self, seed, difficulty):
        from so101_env.state_pick_env import SO101StatePickEnv
        self.env = SO101StatePickEnv("lift", difficulty)
        self.env.reset(seed=seed)
        self.heights, self.shifts, self.tables, self.joints_hist = [], [], [], []
        self.rng = np.random.default_rng(seed)

    def camera_pose(self):
        c, yaw = self.env.cube_pos(), self.env.cube_yaw()
        return (c[0] * 100 + self.rng.normal(0, 0.3), c[1] * 100 + self.rng.normal(0, 0.3), math.degrees(yaw))

    def start(self):
        return True

    def read(self):
        return self.env.data.qpos[self.env._adr].copy()

    def command(self, target_rad):
        import mujoco
        env = self.env
        env._target = np.clip(np.asarray(target_rad, float), env._low, env._high)
        env.data.ctrl[:] = env._target
        for _ in range(env.frame_skip):
            mujoco.mj_step(env.model, env.data)
        info = env._info()
        self.heights.append(info["cube_height"]); self.shifts.append(info["cube_shift"])
        self.tables.append(info["table_force"]); self.joints_hist.append(info["joint_pos"])

    def tracking_error_deg(self):
        return 0.0

    def verdict(self):
        from so101_env.metrics import episode_metrics
        m = episode_metrics(self.heights, self.shifts, self.tables, self.joints_hist)
        return (f"success={bool(m['success'])}  peak {m['peak_cm']:.1f} cm  held {m['held_steps'] / HZ:.1f} s  "
                f"sideways shift while held {m['shift_while_held_cm']:.1f} cm")

    def finish(self):
        pass


# ---------------------------------------------------------------- the camera

def measure_cube(seconds=2.0, base_x=None, base_y=None):
    """Median cube pose (x_cm, y_cm, yaw_deg) over `seconds` of camera frames, or raise if not seen steadily."""
    import cv2
    import locate_cube as lc
    from so101_env import camera_map
    # Default call: raw camera position (arm origin 0/0), then the phase-1 camera -> arm-frame map if one is saved.
    # Explicit base_x / base_y (the calibration collector passes 0.0) gives the raw, uncorrected camera reading.
    use_map = base_x is None and base_y is None and camera_map.available()
    base_x = (0.0 if use_map else lc.BASE_X_CM) if base_x is None else base_x
    base_y = (0.0 if use_map else lc.BASE_Y_CM) if base_y is None else base_y
    cap = cv2.VideoCapture(lc.find_camera(), cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    try:
        for _ in range(20):
            cap.read()
            time.sleep(0.03)
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f"could not read the overhead camera")
        flattener, _ = lc.get_flattener(img)
        poses, t0, frames = [], time.time(), 0
        while time.time() - t0 < seconds:
            ok, img = cap.read()
            if not ok:
                continue
            frames += 1
            result, _ = lc.analyse(img, base_x, base_y, flattener)
            if result:
                poses.append((result["x_cm"], result["y_cm"], result["yaw_deg"]))
    finally:
        cap.release()
    if frames == 0 or len(poses) < 0.8 * frames:
        raise RuntimeError(f"the cube was found in only {len(poses)} of {frames} camera frames")
    a = np.array(poses)
    if a[:, 0].std() > 0.4 or a[:, 1].std() > 0.4:
        raise RuntimeError(f"the cube position is not steady (std {a[:, 0].std():.2f}, {a[:, 1].std():.2f} cm)")
    yaw = np.degrees(np.angle(np.mean(np.exp(1j * 4 * np.radians(a[:, 2]))))) / 4.0     # circular median (the square repeats every 90 deg)
    x, y, yaw = float(np.median(a[:, 0])), float(np.median(a[:, 1])), float(yaw)
    return camera_map.to_arm_frame(x, y, yaw) if use_map else (x, y, yaw)


# ---------------------------------------------------------------- the run

def run(arm, server, cube, steps, countdown, log_name, realtime=True):
    q = arm.read()
    server.ask("reset", q, cube)
    if countdown:
        for s in range(countdown, 0, -1):
            print(f"  starting in {s} ...", flush=True)
            time.sleep(1.0)
    log = {"cube_cm_cm_deg": list(cube), "q": [], "target": [], "action": [], "goal_deg": []}
    stopped = None
    try:
        for step in range(steps):
            t0 = time.perf_counter()
            q = arm.read()
            reply = server.ask("step", q, cube)
            arm.command(reply["target"])
            log["q"].append(q.tolist()); log["target"].append(reply["target"]); log["action"].append(reply["action"])
            err = arm.tracking_error_deg()
            if err > MAX_TRACK_ERR_DEG:
                stopped = f"stopped at step {step}: a joint is {err:.0f} deg from its goal (arm blocked or lagging)"
                break
            if realtime:
                time.sleep(max(0.0, 1.0 / HZ - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        stopped = "stopped by Ctrl-C"
    log["stopped"] = stopped
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{log_name}.json"
    path.write_text(json.dumps(log))
    print(f"log saved: {path}")
    if stopped:
        print(stopped)
    return log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="No hardware: simulated arm and cube.")
    ap.add_argument("--mock_difficulty", default="polish")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--server_port", type=int, default=8770)
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--robot_id", default="my_follower")
    ap.add_argument("--max_step", type=float, default=3.0, help="Bridge speed cap, deg per 1/20 s step (gripper 0-100).")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--base_x_cm", type=float, default=None)
    ap.add_argument("--base_y_cm", type=float, default=None)
    args = ap.parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")

    if args.mock:
        arm = MockArm(args.seed, args.mock_difficulty)
        cube = arm.camera_pose()
        print(f"mock cube seen at {cube[0]:.1f} cm out, {cube[1]:+.1f} cm left, yaw {cube[2]:+.0f} deg")
    else:
        print("Looking for the cube ...")
        cube = measure_cube(base_x=args.base_x_cm, base_y=args.base_y_cm)
        print(f"cube: {cube[0]:.1f} cm out, {cube[1]:+.1f} cm left, turned {cube[2]:+.0f} deg")
        if not in_zone(cube[0], cube[1]):
            raise SystemExit(f"The cube is outside the trained zone (x {ZONE_X}, y {ZONE_Y} cm). Move it and run again.")
        arm = RealArm(args.port, args.robot_id, args.max_step)
    server = Server(args.server_port)
    if not arm.start():
        return
    try:
        run(arm, server, cube, args.steps, countdown=0 if args.mock else 5, log_name=f"live_{stamp}", realtime=not args.mock)
        if args.mock:
            print("result:", arm.verdict())
    finally:
        arm.finish()


if __name__ == "__main__":
    main()
