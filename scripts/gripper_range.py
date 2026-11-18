"""Finds the gripper's real closed and open stops, gently (torque limited to ~15%), and prints them. Read-only about the
calibration file (use --apply to write the stops into it and into the servo's position limits).

    C:\\Users\\goran\\PycharmProjects\\so101\\lerobot\\venv\\Scripts\\python.exe scripts\\gripper_range.py [--apply]

Only the gripper servo gets torque; the arm joints stay limp. Keep the jaws clear of fingers and objects.
"""
import argparse
import json
import time

from lerobot.motors import MotorCalibration
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

G = "gripper"
TORQUE_TEST, TORQUE_NORMAL = 150, 500          # of 1000
STEP, DT = 8, 0.12                             # ticks per goal step, seconds per step
MARGIN = 12                                    # ticks kept off a stop when it becomes the range end

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true")
ap.add_argument("--set_range", type=int, nargs=2, metavar=("MIN", "MAX"), help="Write this range without moving anything.")
args = ap.parse_args()
robot = SO101Follower(SO101FollowerConfig(port="COM3", id="my_follower", use_degrees=True))
bus = robot.bus
for attempt in range(6):
    try:
        bus.connect()
        break
    except RuntimeError:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
        time.sleep(0.5)
else:
    raise SystemExit("could not connect to all six servos")


def pos():
    return int(bus.read("Present_Position", G, normalize=False, num_retry=3))


def load():
    return int(bus.read("Present_Load", G, normalize=False, num_retry=3))


def sweep(direction, limit):
    """Move the goal step by step until the jaw stops following it (stall) or the position limit is reached."""
    p = pos()
    goal, lag_hits = p, 0
    while True:
        goal += direction * STEP
        if (direction < 0 and goal <= limit) or (direction > 0 and goal >= limit):
            return pos(), "reached the position limit"
        bus.write("Goal_Position", G, goal, normalize=False)
        time.sleep(DT)
        p = pos()
        lag_hits = lag_hits + 1 if abs(goal - p) > 4 * STEP else 0
        if lag_hits >= 3 or abs(load()) > 110:
            return p, "stalled"


if args.set_range:                                       # no motion: just write a range measured earlier (closed stop 2043, open >= 3163)
    cal = robot.calibration[G]
    robot.calibration[G] = MotorCalibration(id=cal.id, drive_mode=cal.drive_mode, homing_offset=cal.homing_offset,
                                            range_min=args.set_range[0], range_max=args.set_range[1])
    bus.write_calibration({G: robot.calibration[G]}, cache=True)
    robot._save_calibration()
    print(f"GRIP range written: {cal.range_min}..{cal.range_max} -> {args.set_range[0]}..{args.set_range[1]} (file and servo)")
    bus.disconnect(disable_torque=False)
    raise SystemExit

try:
    err = bus.packet_handler.read1ByteTxRx(bus.port_handler, 6, 40)[2]
    print(f"GRIP servo error bits before: {err} (0 = clear)", flush=True)
    cal = robot.calibration[G]
    start = pos()
    bus.write("Torque_Enable", G, 0)
    bus.write("Torque_Limit", G, TORQUE_TEST, normalize=False)
    bus.write("Goal_Position", G, start, normalize=False)
    bus.write("Torque_Enable", G, 1)
    closed, how_c = sweep(-1, cal.range_min + 5)
    print(f"GRIP closed stop at reading {closed} ({how_c})", flush=True)
    time.sleep(0.3)
    opened, how_o = sweep(+1, cal.range_max - 5)
    print(f"GRIP open stop at reading {opened} ({how_o})", flush=True)
    mid = (closed + opened) // 2
    bus.write("Goal_Position", G, mid, normalize=False)
    time.sleep(1.0)
finally:
    try:
        bus.write("Torque_Enable", G, 0)
        bus.write("Torque_Limit", G, TORQUE_NORMAL, normalize=False)
    except Exception as e:
        print("GRIP could not restore torque settings:", e, flush=True)

new_min, new_max = closed + MARGIN, opened - MARGIN
print(f"GRIP old range {cal.range_min}..{cal.range_max}  ->  new range {new_min}..{new_max}  ({(new_max - new_min) * 360 / 4095:.0f} deg of travel)")
if args.apply:
    robot.calibration[G] = MotorCalibration(id=cal.id, drive_mode=cal.drive_mode, homing_offset=cal.homing_offset,
                                            range_min=new_min, range_max=new_max)
    bus.write_calibration({G: robot.calibration[G]}, cache=True) if hasattr(bus, "write_calibration") else None
    robot._save_calibration()
    print("GRIP calibration file and servo limits updated")
bus.disconnect(disable_torque=False)
