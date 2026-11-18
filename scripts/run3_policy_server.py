"""Serves a run-3 state policy over a local socket, for the real-arm client (scripts/run3_real_client.py).

Project venv (it has torch):
    .venv\\Scripts\\python.exe scripts\\run3_policy_server.py --checkpoint outputs\\run3\\s6_polish\\stage_best.pt

One JSON object per line in, one per line out, on 127.0.0.1:8770:
    {"cmd": "reset", "q": [6 joint angles, rad], "cube": [x_cm, y_cm, yaw_deg]}  -> {"ok": true, "target": [6 rad]}
    {"cmd": "step",  "q": [...],                "cube": [x_cm, y_cm, yaw_deg]}  -> {"ok": true, "target": [...], "action": [...]}
The server keeps the policy's memory and the joint-target integrator between steps (so101_env/deploy.py).
"""
import argparse
import json
import pathlib
import socketserver
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "dreamer"))
sys.path.insert(0, str(ROOT / "scripts"))
from run3_gate_eval import DreamerPolicy  # noqa: E402
from so101_env.deploy import LiveController, observation  # noqa: E402
from so101_env.state_pick_env import SO101StatePickEnv  # noqa: E402


class Session:
    def __init__(self, checkpoint, device):
        env = SO101StatePickEnv("lift", "easy")                # only for its limits, timing and the policy's input/output spaces
        self.policy = DreamerPolicy(checkpoint, env, device)
        self.ctl = LiveController.from_env(env)

    def handle(self, msg):
        q = np.asarray(msg["q"], dtype=np.float64)
        if "cube_vec" in msg:                                   # the simulator's own observation vector (sim-in-the-loop planning)
            obs = {"proprio": q.astype(np.float32), "cube": np.asarray(msg["cube_vec"], dtype=np.float32)}
        else:
            obs = observation(q, msg["cube"])
        if msg["cmd"] == "reset":
            self.policy.reset()
            return {"ok": True, "target": self.ctl.reset(q).tolist()}
        action = np.asarray(self.policy.act(obs), dtype=np.float64).reshape(-1)
        return {"ok": True, "target": self.ctl.step(action).tolist(), "action": action.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    session = Session(args.checkpoint, args.device)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            for line in self.rfile:
                try:
                    reply = session.handle(json.loads(line))
                except Exception as e:                          # keep serving; the client decides what to do
                    reply = {"ok": False, "error": repr(e)}
                self.wfile.write((json.dumps(reply) + "\n").encode())
                self.wfile.flush()

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", args.port), Handler) as srv:
        print(f"policy server ready on 127.0.0.1:{args.port}  ({args.checkpoint})", flush=True)
        srv.serve_forever()


if __name__ == "__main__":
    main()
