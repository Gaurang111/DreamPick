"""Live bus check: pings the six follower servos over and over and prints how often each one answers (read-only, no torque).

Run with the LeRobot venv while the arm is powered, then wiggle / reseat connectors one at a time and watch which servo's
rate changes:
    C:\\Users\\goran\\PycharmProjects\\so101\\lerobot\\venv\\Scripts\\python.exe scripts\\bus_monitor.py --port COM3 --seconds 60
"""
import argparse
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

ap = argparse.ArgumentParser()
ap.add_argument("--port", default="COM3")
ap.add_argument("--seconds", type=float, default=60.0)
args = ap.parse_args()
bus = SO101Follower(SO101FollowerConfig(port=args.port, id="my_follower", use_degrees=True)).bus
bus._connect(handshake=False)
names = {i + 1: n for i, n in enumerate(["pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper"])}
hits, tries, t0, last = {i: 0 for i in names}, {i: 0 for i in names}, time.time(), time.time()
while time.time() - t0 < args.seconds:
    for i in names:
        tries[i] += 1
        try:
            hits[i] += bus.ping(i, num_retry=0) is not None
        except Exception:
            pass
    if time.time() - last >= 2.0:
        print("  ".join(f"{names[i]} {100 * hits[i] // max(1, tries[i]):3d}%" for i in names), flush=True)
        hits, tries, last = {i: 0 for i in names}, {i: 0 for i in names}, time.time()
bus.port_handler.closePort()
print("done")
