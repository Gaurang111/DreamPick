"""Links the MuJoCo sim to the physical SO-101 follower arm. Not part of training.

Run with the LeRobot venv (it has `lerobot`; `mujoco` was added to it):
    C:\\Users\\goran\\PycharmProjects\\so101\\lerobot\\venv\\Scripts\\python.exe scripts/sim_real_bridge.py --mode mirror

Modes:
  mirror  Real -> sim. Motor torque stays OFF the whole time; move the arm by
          hand and the sim copies it. Zero-risk check of every joint's
          direction and zero. Do this first.
  drive   Sim -> real. Drag the actuator sliders in the viewer's right-hand
          "Control" panel; the sim arm moves and the real arm follows. On exit
          the real arm is eased back to the pose it started in, then torque
          is released.

Joint mapping: the scene uses the `so101_new_calib` model, built to match
LeRobot's calibration, so sim 0 rad == the middle of each calibrated range ==
LeRobot 0 deg. Body joints convert radians <-> degrees; the gripper maps its
sim range linearly onto LeRobot's 0-100. If mirror mode shows a joint moving
the wrong way or offset, fix it in JOINT_SIGN / JOINT_OFFSET_DEG below.

Encoder rollover: each STS3215 reports (encoder - Homing_Offset) without
wrapping, so a joint whose calibrated range crosses the encoder's 4095 -> 0
point reads a large negative number past that point (this arm: wrist_flex
past ~+71 deg, shoulder_lift past ~+82 deg). LeRobot's own normalization
doesn't correct for this, so this script talks to the motor bus in raw ticks:
reads are unwrapped back to the true position, and drive mode never commands a
joint past its rollover point - a goal there would make the motor turn the
long way round, through its mechanical stop.
"""

import argparse
import json
import math
import os
import pathlib
import time

import mujoco
import mujoco.viewer
import numpy as np
from lerobot.motors.feetech import OperatingMode
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

SCENE = pathlib.Path(__file__).resolve().parents[1] / "assets" / "scene_pick_cube.xml"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

JOINT_SIGN = {j: 1.0 for j in JOINTS}
JOINT_OFFSET_DEG = {j: 0.0 for j in JOINTS}

TICKS = 4096
MAX_TICK = TICKS - 1
BODY_MARGIN_DEG = 3.0
GRIPPER_MARGIN = 2.0
CONTROL_HZ = 20

# Sag compensation: with LeRobot's soft servo gains, gravity-loaded joints
# settle a few degrees short of their goal, so the real gripper ends up lower
# than the sim's. Sag scales with the gravity load on the joint, which changes
# a lot with pose (shoulder_lift carries ~2x the ready pose's load at the
# grasp), so each joint learns degrees-of-sag per N*m of load, and the goal
# gets (that) x (the sim's gravity load at the target pose) added. Learned only
# while the target is nearly still - learning during motion would wind up on
# ordinary tracking lag and overshoot afterwards. Bounded, and the result is
# still clipped to the usable range and the per-step cap.
SAG_JOINTS = ("shoulder_lift", "elbow_flex", "wrist_flex")
SAG_GAIN = 0.05             # fraction of the error corrected per control step
SAG_LIMIT_DEG = 8.0         # max correction added to any goal
SAG_K_MAX = 30.0            # max learned sag, deg per N*m
SAG_MIN_LOAD = 0.05         # N*m; below this a joint's sag is too small to learn from
SAG_STILL_DEG = 0.3         # target counts as "nearly still" below this change per step
SAG_DEADBAND_DEG = 0.3

# How far a goal may lead the joint's measured position (deg; gripper 0-100).
# The per-step speed cap applies to the goal itself (vs the previous goal), not
# to the measured position: capping vs the measured position stalls any joint
# whose sag exceeds the cap - the goal can never get far enough above the joint
# to lift it, so it droops further every step. This lead cap is what still
# bounds the force when a joint is blocked (e.g. the gripper squeezing the cube).
MAX_LEAD = 15.0


class Bridge:
    def __init__(self, port, robot_id, max_step):
        self.max_step = max_step
        self.servo_accel = None          # set before connect() to override the servo Acceleration register (counts of ~8.8 deg/s^2)
        # Starts from JOINT_OFFSET_DEG; callers may add a run-time correction
        # (e.g. sim_real_demo.py's height correction from --align).
        self.joint_offset = dict(JOINT_OFFSET_DEG)
        kin = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "phase1" / "kinematics.json"      # joint zeros fitted from hand grasps (scripts/phase1_fit_kinematics.py)
        if kin.exists() and not os.environ.get("NO_KINEMATICS"):
            self.joint_offset.update({j: float(v) for j, v in json.loads(kin.read_text())["joint_offset_deg"].items() if j in JOINTS})
        self.reset_tracking()
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.qadr = {j: self.model.joint(j).qposadr[0] for j in JOINTS}
        self.act = {j: self.model.actuator(j).id for j in JOINTS}
        self.sim_lo = {j: self.model.actuator_ctrlrange[self.act[j], 0] for j in JOINTS}
        self.sim_hi = {j: self.model.actuator_ctrlrange[self.act[j], 1] for j in JOINTS}

        # Only used for its configured motor bus and loaded calibration; the
        # robot's own connect/get_observation/send_action are bypassed (see
        # "Encoder rollover" above).
        self.robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, use_degrees=True))
        self.bus = self.robot.bus
        self.cal = self.robot.calibration
        assert all(self.cal[j].drive_mode == 0 for j in JOINTS), "drive_mode != 0 is not handled"

        # Usable command range per joint, in real units (deg, gripper 0-100):
        # the calibrated range cut at the encoder rollover points, minus a margin.
        self.real_lo, self.real_hi = {}, {}
        for j in JOINTS:
            c = self.cal[j]
            enc_lo, enc_hi = self.encoder_window(j)
            margin = GRIPPER_MARGIN if j == "gripper" else BODY_MARGIN_DEG
            self.real_lo[j] = self.ticks_to_real(j, max(c.range_min, enc_lo)) + margin
            self.real_hi[j] = self.ticks_to_real(j, min(c.range_max, enc_hi)) - margin

    def encoder_window(self, j):
        """Position ticks the motor can report without rolling over."""
        off = self.cal[j].homing_offset
        return -off, MAX_TICK - off

    # --- unit conversions -------------------------------------------------

    def mid(self, j):
        return (self.cal[j].range_min + self.cal[j].range_max) / 2

    def unwrap(self, j, raw):
        return raw + TICKS * round((self.mid(j) - raw) / TICKS)

    def ticks_to_real(self, j, ticks):
        c = self.cal[j]
        if j == "gripper":
            return (min(c.range_max, max(c.range_min, ticks)) - c.range_min) / (c.range_max - c.range_min) * 100
        return (ticks - self.mid(j)) * 360 / MAX_TICK

    def real_to_ticks(self, j, val):
        c = self.cal[j]
        if j == "gripper":
            return int(round(val / 100 * (c.range_max - c.range_min) + c.range_min))
        return int(round(val * MAX_TICK / 360 + self.mid(j)))

    def sim_to_real(self, j, rad):
        if j == "gripper":
            frac = (rad - self.sim_lo[j]) / (self.sim_hi[j] - self.sim_lo[j])
            val = 100.0 * (frac if JOINT_SIGN[j] > 0 else 1.0 - frac)
        else:
            val = JOINT_SIGN[j] * math.degrees(rad) + self.joint_offset[j]
        return float(np.clip(val, self.real_lo[j], self.real_hi[j]))

    def real_to_sim(self, j, val):
        if j == "gripper":
            frac = val / 100.0 if JOINT_SIGN[j] > 0 else 1.0 - val / 100.0
            rad = self.sim_lo[j] + frac * (self.sim_hi[j] - self.sim_lo[j])
        else:
            rad = math.radians((val - self.joint_offset[j]) * JOINT_SIGN[j])
        return float(np.clip(rad, self.sim_lo[j], self.sim_hi[j]))

    # --- hardware ---------------------------------------------------------

    def connect(self):
        """Open the bus and apply LeRobot's SO follower motor settings with
        torque left OFF (SOFollower.connect re-enables torque at the end)."""
        for attempt in range(1, 7):                            # the handshake pings each motor once; a marginal bus misses one now and then
            try:
                self.bus.connect()
                break
            except RuntimeError as e:
                if attempt == 6:
                    raise
                print(f"  motor check missed a servo (attempt {attempt}/6), trying again ...", flush=True)
                try:
                    if self.bus.is_connected:
                        self.bus.disconnect(disable_torque=False)
                except Exception:
                    pass
                time.sleep(0.5)
        if not self.bus.is_calibrated:
            self.bus.disconnect(disable_torque=False)
            raise SystemExit("The motors' stored calibration differs from the calibration file. "
                             "Run LeRobot's calibration for this arm first.")
        self.bus.disable_torque()
        self.bus.configure_motors()
        cfg = self.robot.config
        for m in self.bus.motors:
            self.bus.write("Operating_Mode", m, OperatingMode.POSITION.value)
            self.bus.write("P_Coefficient", m, cfg.position_p_coefficient)
            self.bus.write("I_Coefficient", m, cfg.position_i_coefficient)
            self.bus.write("D_Coefficient", m, cfg.position_d_coefficient)
        if self.servo_accel is not None:                       # gentler servo acceleration than LeRobot's 254 (arm joints only)
            for m in self.bus.motors:
                if m != "gripper":
                    self.bus.write("Acceleration", m, int(self.servo_accel))
        self.bus.write("Max_Torque_Limit", "gripper", 500)
        self.bus.write("Protection_Current", "gripper", 250)
        self.bus.write("Overload_Torque", "gripper", 25)

    def read_raw(self):
        return self.bus.sync_read("Present_Position", normalize=False, num_retry=2)

    def raw_to_real(self, raw):
        return {j: self.ticks_to_real(j, self.unwrap(j, raw[j])) for j in JOINTS}

    def read_real(self):
        return self.raw_to_real(self.read_raw())

    def near_rollover(self, raw, margin_deg=0.0):
        """Joints whose true position is past (or within margin_deg of) the
        point where their reading rolls over."""
        m = margin_deg * MAX_TICK / 360
        out = []
        for j in JOINTS:
            u = self.unwrap(j, raw[j])
            lo, hi = self.encoder_window(j)
            if u < lo + m or u > hi - m:
                out.append(j)
        return out

    def rollover_limit_deg(self, j, raw):
        """The rollover point nearest this joint's reading, in degrees."""
        lo, hi = self.encoder_window(j)
        u = self.unwrap(j, raw[j])
        edge = hi if abs(u - hi) < abs(u - lo) else lo
        return (edge - self.mid(j)) * 360 / MAX_TICK

    def reset_tracking(self):
        self.sag_k = {j: 0.0 for j in SAG_JOINTS}
        self.sag = {j: 0.0 for j in SAG_JOINTS}
        self.prev_want = None
        self.prev_goal = None
        self.track_err = {j: [] for j in JOINTS}

    def gravity_load(self, real):
        """Gravity torque (N*m) each joint holds at this pose, from the sim model."""
        if not hasattr(self, "_grav_data"):
            self._grav_data = mujoco.MjData(self.model)
        g = self._grav_data
        for j in JOINTS:
            g.qpos[self.qadr[j]] = self.real_to_sim(j, real[j])
        g.qvel[:] = 0
        mujoco.mj_forward(self.model, g)
        return {j: float(g.qfrc_bias[self.model.joint(j).dofadr[0]]) for j in SAG_JOINTS}

    def send_real(self, target, present):
        """Step each joint toward `target` (real units): sag-compensated, inside
        the usable range, at most max_step from the previous goal and MAX_LEAD
        from the measured position."""
        want = {j: float(np.clip(target[j], self.real_lo[j], self.real_hi[j])) for j in JOINTS}
        load = self.gravity_load(want)
        goal = {}
        for j in JOINTS:
            err = want[j] - present[j]
            self.track_err[j].append(err)
            v = want[j]
            if j in SAG_JOINTS:
                still = self.prev_want is not None and abs(want[j] - self.prev_want[j]) < SAG_STILL_DEG
                if still and abs(err) > SAG_DEADBAND_DEG and abs(load[j]) > SAG_MIN_LOAD:
                    self.sag_k[j] = float(np.clip(self.sag_k[j] + SAG_GAIN * err / load[j], -SAG_K_MAX, SAG_K_MAX))
                self.sag[j] = float(np.clip(self.sag_k[j] * load[j], -SAG_LIMIT_DEG, SAG_LIMIT_DEG))
                v += self.sag[j]
            v = float(np.clip(v, self.real_lo[j], self.real_hi[j]))
            last = present[j] if self.prev_goal is None else self.prev_goal[j]
            v = float(np.clip(v, last - self.max_step, last + self.max_step))
            v = float(np.clip(v, present[j] - MAX_LEAD, present[j] + MAX_LEAD))
            goal[j] = v
        self.prev_want = want
        self.prev_goal = goal
        self.bus.sync_write("Goal_Position", {j: self.real_to_ticks(j, goal[j]) for j in JOINTS}, normalize=False)
        return goal

    def tracking_report(self):
        """How far the real arm trailed its target, per joint (deg; gripper 0-100)."""
        lines = [f"  {'joint':<14}{'mean |err|':>11}{'max |err|':>11}{'sag deg/Nm':>12}{'last offset':>13}"]
        for j in JOINTS:
            e = np.abs(np.array(self.track_err[j] or [0.0]))
            k = f"{self.sag_k[j]:+.1f}" if j in SAG_JOINTS else "-"
            sag = f"{self.sag[j]:+.1f}" if j in SAG_JOINTS else "-"
            lines.append(f"  {j:<14}{e.mean():>11.1f}{e.max():>11.1f}{k:>12}{sag:>13}")
        return "\n".join(lines)

    # --- sim --------------------------------------------------------------

    def set_sim_pose(self, real):
        for j in JOINTS:
            # Show the sim arm only inside the range the real arm can really use (a real arm resting on its stop can read a
            # few degrees past it, which would push the sim wrist into the table and the links into each other).
            rad = self.real_to_sim(j, float(np.clip(real[j], self.real_lo[j], self.real_hi[j])))
            self.data.qpos[self.qadr[j]] = rad
            self.data.ctrl[self.act[j]] = rad
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def limit_sim_sliders_to_real(self):
        """Narrow each actuator's slider range to what the real arm can safely
        reach, so the sim can't be driven somewhere the real arm won't follow."""
        for j in JOINTS:
            a, b = self.real_to_sim(j, self.real_lo[j]), self.real_to_sim(j, self.real_hi[j])
            self.model.actuator_ctrlrange[self.act[j]] = [min(a, b), max(a, b)]

    def print_table(self, real, label):
        print(f"\n{label}")
        print(f"  {'joint':<14}{'real':>9}{'sim (deg)':>12}{'usable real range':>22}")
        for j in JOINTS:
            sim_deg = math.degrees(self.data.qpos[self.qadr[j]])
            print(f"  {j:<14}{real[j]:>9.1f}{sim_deg:>12.1f}{self.real_lo[j]:>12.1f} .. {self.real_hi[j]:.1f}")

    # --- modes ------------------------------------------------------------

    def run_mirror(self):
        self.connect()
        print("\nTorque is OFF. Move the arm by hand; the sim should copy every joint.")
        print("Check each joint moves the same direction in both. Close the viewer or Ctrl-C to stop.")
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                last_print = 0.0
                while viewer.is_running():
                    real = self.read_real()
                    self.set_sim_pose(real)
                    viewer.sync()
                    if time.time() - last_print > 0.5:
                        self.print_table(real, "mirror (torque off)")
                        last_print = time.time()
                    time.sleep(1.0 / CONTROL_HZ)
        except KeyboardInterrupt:
            pass
        finally:
            self.release()

    def wait_for_safe_start(self):
        """Torque is off here. A joint past (or close to) its rollover point
        must be moved back by hand: a goal sent from there would make the
        motor turn the long way round, through its mechanical stop. Checked
        again at the instant of ENTER, since gravity can pull it back."""
        while True:
            raw = self.read_raw()
            bad = self.near_rollover(raw, BODY_MARGIN_DEG)
            if bad:
                print("\nTorque is off. Move these joints by hand away from the limit shown, and hold them there:")
                while bad:
                    real = self.raw_to_real(raw)
                    print("  " + " | ".join(
                        f"{j}: {real[j]:.1f} deg (keep at least {BODY_MARGIN_DEG:.0f} deg inside "
                        f"{self.rollover_limit_deg(j, raw):.1f})" for j in bad) + "    ", end="\r")
                    time.sleep(0.2)
                    raw = self.read_raw()
                    bad = self.near_rollover(raw, BODY_MARGIN_DEG)
                print("\nGood - keep holding it there.")
            print("\nLive angles below (deg). Hold the arm still, wrist_flex best at about 55-60. Press ENTER to switch torque on")
            print("(the arm will hold its current pose); Ctrl-C cancels.")
            try:
                import msvcrt
            except ImportError:
                msvcrt = None
            while msvcrt is not None:
                raw = self.read_raw()
                real = self.raw_to_real(raw)
                ok = not self.near_rollover(raw, BODY_MARGIN_DEG)
                limit = self.rollover_limit_deg("wrist_flex", raw)
                print(f"  wrist_flex {real['wrist_flex']:6.1f}  elbow {real['elbow_flex']:6.1f}  lift {real['shoulder_lift']:6.1f}   "
                      f"{'OK - press ENTER' if ok else 'TOO CLOSE TO THE LIMIT (' + format(limit, '.1f') + ')'}      ", end="\r", flush=True)
                if msvcrt.kbhit() and msvcrt.getwch() in ("\r", "\n"):
                    print()
                    break
                time.sleep(0.1)
            if msvcrt is None:
                input("Press ENTER to switch torque on (the arm will hold its current pose)...")
            raw = self.read_raw()
            if not self.near_rollover(raw, BODY_MARGIN_DEG):
                return raw
            print("A joint moved back toward its rollover point before torque came on - again.")

    def start_torque(self):
        """Connect, get every joint clear of its rollover point, then switch
        torque on holding the current pose. Returns the starting pose (real
        units), or None if the user pressed Ctrl-C."""
        self.connect()
        self.reset_tracking()
        try:
            raw = self.wait_for_safe_start()
            # Goal registers reset to 0 at power-up, so pin every goal to where
            # the arm already is before torque comes on. Clamped into the
            # calibrated range: a joint resting on its stop can read a few ticks
            # past its Position_Limit, and the motor refuses torque-on with an
            # out-of-limit goal (seen on elbow_flex: present 4170, limit 4167).
            pin = {j: int(np.clip(raw[j], self.cal[j].range_min, self.cal[j].range_max)) for j in JOINTS}
            # A body joint within 8 deg of its rollover point is pinned 5 deg further inside instead: with the soft servo
            # gains a gravity-loaded joint sags a few degrees toward its stop as torque comes on, and that was tripping the
            # rollover guard right after ENTER (wrist_flex held at 69 deg sagged past 70.9).
            near, back = 8.0 * MAX_TICK / 360, 5.0 * MAX_TICK / 360
            for j in JOINTS:
                if j == "gripper":
                    continue
                u, (lo, hi) = self.unwrap(j, raw[j]), self.encoder_window(j)
                if u > hi - near:
                    pin[j] = int(np.clip(u - back, self.cal[j].range_min, self.cal[j].range_max))
                elif u < lo + near:
                    pin[j] = int(np.clip(u + back, self.cal[j].range_min, self.cal[j].range_max))
            self.bus.sync_write("Goal_Position", pin, normalize=False)
            self.bus.enable_torque(num_retry=2)
        except BaseException as e:
            self.release()
            if isinstance(e, KeyboardInterrupt):
                return None
            raise
        return self.raw_to_real(raw)

    def run_drive(self):
        start = self.start_torque()
        if start is None:
            return
        self.set_sim_pose(start)
        self.limit_sim_sliders_to_real()
        self.print_table(start, "torque ON - starting pose (sim synced to real)")
        print("\nYou can let go. Open the viewer's right panel > Control and drag the sliders to move the arm.")
        print("Close the viewer or Ctrl-C to stop; the arm eases back to this starting pose first.")

        substeps = max(1, round(1.0 / CONTROL_HZ / self.model.opt.timestep))
        safe = True
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                last_print = 0.0
                while viewer.is_running():
                    tick = time.perf_counter()
                    viewer.sync()
                    for _ in range(substeps):
                        mujoco.mj_step(self.model, self.data)
                    raw = self.read_raw()
                    safe = self.check_rollover(raw)
                    if not safe:
                        break
                    present = self.raw_to_real(raw)
                    target = {j: self.sim_to_real(j, self.data.qpos[self.qadr[j]]) for j in JOINTS}
                    self.send_real(target, present)
                    if time.time() - last_print > 0.5:
                        self.print_table(present, "drive (real = measured)")
                        last_print = time.time()
                    time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - tick)))
        except KeyboardInterrupt:
            pass
        finally:
            try:
                if safe:
                    self.park(start)
            finally:
                self.release()

    def release(self):
        """Best-effort: torque off on every motor that still answers, then
        close the port - even if one motor has stopped responding."""
        if not self.bus.is_connected:
            return
        for m in JOINTS:
            try:
                self.bus.write("Torque_Enable", m, 0, num_retry=2)
            except Exception as e:
                print(f"Could not switch off torque on {m}: {e}")
        try:
            self.bus.disconnect(disable_torque=False)
        except Exception:
            pass

    def check_rollover(self, raw):
        """Mid-run guard: if a joint was forced past its rollover point, any
        goal could send it the long way round - release torque instead."""
        bad = self.near_rollover(raw)
        if bad:
            print(f"\nSTOPPED: {', '.join(bad)} passed its rollover point. Releasing torque.")
            self.release()
        return not bad

    def park(self, start):
        print("\nEasing back to the starting pose...")
        # Joints that started resting on a stop sit just outside the usable range.
        target = {j: float(np.clip(start[j], self.real_lo[j], self.real_hi[j])) for j in JOINTS}
        deadline = time.time() + 6.0
        while time.time() < deadline:
            raw = self.read_raw()
            if not self.check_rollover(raw):
                return
            present = self.raw_to_real(raw)
            if all(abs(present[j] - target[j]) < 1.5 for j in JOINTS):
                break
            self.send_real(target, present)
            time.sleep(1.0 / CONTROL_HZ)
        time.sleep(0.5)
        print("Done. Releasing torque.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["mirror", "drive"], required=True)
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--robot_id", default="my_follower")
    parser.add_argument("--max_step", type=float, default=4.0,
                        help="Max change per joint per 1/20s control step, in degrees (gripper: 0-100 units).")
    args = parser.parse_args()

    bridge = Bridge(args.port, args.robot_id, args.max_step)
    bridge.run_mirror() if args.mode == "mirror" else bridge.run_drive()


if __name__ == "__main__":
    main()
