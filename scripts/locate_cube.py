"""Find the real cube on the mat from the overhead camera and say where it is in the sim's coordinates.

Run with the LeRobot venv (it has OpenCV):
    C:\\Users\\goran\\PycharmProjects\\so101\\lerobot\\venv\\Scripts\\python.exe scripts/locate_cube.py            # live camera
    ... scripts/locate_cube.py --image outputs/real_replay/camera/view2.png                                          # saved picture
    ... scripts/locate_cube.py --live          # live view in a web page: http://127.0.0.1:8765, with a Save button

How: the green mat (60 x 45 cm, printed on it) is found in the picture and flattened to a top-down view of just
the mat, so the background is ignored. The cube is the orange-wood blob on it. Positions are reported in the sim's
frame: x = centimetres out from the mat edge the arm stands on (forward), y = centimetres to the left of the arm's
middle line (right is negative), plus the cube's rotation. The arm's rotation axis is taken from where the protractor lines on the mat meet
(BASE_X_CM / BASE_Y_CM, about +-1 cm); outputs/real_replay/camera_calibration.json (written by
calibrate_camera_with_arm.py, when it exists) replaces that guess with a fit made from the real arm itself.

The run-3 policy was trained with the cube 21-31 cm out and within +-10 cm of the arm's middle line; the live preview draws
that zone so the cube can just be placed inside it.
"""
import argparse
import json
import math
import pathlib
import time

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "real_replay"
CAMERA_INDEX = 1
MAT_W_CM, MAT_H_CM = 61.0, 46.0         # printed on the mat: 61 x 46 cm (the sim uses the same size)
PX_PER_CM = 20
RECT_W, RECT_H = int(MAT_W_CM * PX_PER_CM), int(MAT_H_CM * PX_PER_CM)
# The arm's rotation axis in the flattened picture, from where the protractor's radiating lines meet (about +-1 cm).
BASE_X_CM, BASE_Y_CM = 0.4, 0.35          # cm out from the mat edge / cm left of the mat's middle
RING_MAT_FRACTION = 0.8                   # share of the ring around the cube that must be green mat
ZONE = dict(x=(21.0, 31.0), y=(-10.0, 10.0))     # cm: the cube start zone the run-3 policy was trained on


_CAMERA_FOUND = None


def find_camera(candidates=(1, 0, 2, 3)):
    """Index of the overhead camera: the first one whose picture is not black and is mostly the green mat. Windows hands out
    camera numbers in a different order after a USB re-plug (it was 1, then 0), so this is looked up instead of assumed."""
    global _CAMERA_FOUND
    if _CAMERA_FOUND is not None:
        return _CAMERA_FOUND
    remembered = OUT / "camera_index.txt"                       # the index that worked last time goes first (opening the black one is slow)
    try:
        first = int(remembered.read_text().strip())
        candidates = (first,) + tuple(c for c in candidates if c != first)
    except (OSError, ValueError):
        pass
    for i in candidates:
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        frame = None
        for _ in range(25):
            ok, f = cap.read()
            if ok:
                frame = f
            time.sleep(0.03)
        cap.release()
        if frame is None or frame.mean() < 25:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if (cv2.inRange(hsv, (55, 25, 20), (110, 255, 255)) > 0).mean() > 0.2:     # a big green area = the mat
            _CAMERA_FOUND = i
            try:
                OUT.mkdir(parents=True, exist_ok=True)
                remembered.write_text(str(i))
            except OSError:
                pass
            return i
    raise SystemExit("No camera shows the green mat (tried indices %s)." % (candidates,))


def grab(index=None):
    index = find_camera() if index is None else index
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    for _ in range(20):                 # let auto-exposure settle
        cap.read()
        time.sleep(0.03)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read camera {index}.")
    return frame


F_PX = 700.0                       # nominal focal length used to scale the distortion fit (k1/k2 absorb the rest)
CENTRE = (640.0, 360.0)
MODEL_FILE = OUT / "camera_model.json"
# Which parts of each edge of the mat are visible (fractions of the mat's bounding box). The arm hides part of the
# bottom edge and the lamp hides part of the top-right; edit these if the rig changes.
EDGE_WINDOWS = {
    "left": (0.12, 0.88),          # fractions of the height
    "right": (0.40, 0.88),         # below the lamp
    "top": [(0.03, 0.62)],         # fractions of the width, left of the lamp
    "bottom": [(0.03, 0.22), (0.64, 0.97)],   # either side of the arm
}


def green_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (55, 25, 20), (110, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))   # bridge the white grid lines
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        raise SystemExit("No green mat found in the picture.")
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == keep).astype(np.uint8) * 255


def edge_points(mask):
    """Points on the four edges of the mat, from the parts of the mask outline that are not hidden."""
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0, y1 - y0
    pts = {"left": [], "right": [], "top": [], "bottom": []}
    for y in range(int(y0 + EDGE_WINDOWS["left"][0] * h), int(y0 + EDGE_WINDOWS["left"][1] * h), 4):
        row = np.nonzero(mask[y])[0]
        if len(row):
            pts["left"].append((row.min(), y))
    for y in range(int(y0 + EDGE_WINDOWS["right"][0] * h), int(y0 + EDGE_WINDOWS["right"][1] * h), 4):
        row = np.nonzero(mask[y])[0]
        if len(row):
            pts["right"].append((row.max(), y))
    for lo, hi in EDGE_WINDOWS["top"]:
        for x in range(int(x0 + lo * w), int(x0 + hi * w), 4):
            col = np.nonzero(mask[:, x])[0]
            if len(col):
                pts["top"].append((x, col.min()))
    for lo, hi in EDGE_WINDOWS["bottom"]:
        for x in range(int(x0 + lo * w), int(x0 + hi * w), 4):
            col = np.nonzero(mask[:, x])[0]
            if len(col):
                pts["bottom"].append((x, col.max()))
    return {k: np.array(v, dtype=np.float64) for k, v in pts.items()}


def _camera_matrix():
    return np.array([[F_PX, 0, CENTRE[0]], [0, F_PX, CENTRE[1]], [0, 0, 1]], dtype=np.float64)


def _undistort(pts, k1, k2):
    """Picture points -> undistorted picture points (same pixel scale)."""
    out = cv2.undistortPoints(pts.reshape(-1, 1, 2), _camera_matrix(), np.array([k1, k2, 0, 0], dtype=np.float64), P=_camera_matrix())
    return out.reshape(-1, 2)


def _fit_line(pts):
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    n = vt[1]
    return n, float(n @ c)                               # n . p = d


def _line_cost(edges, k):
    cost = 0.0
    for p in edges.values():
        u = _undistort(p, k[0], k[1])
        n, d = _fit_line(u)
        cost += float(np.sum((u @ n - d) ** 2))
    return cost


def fit_camera_model(img):
    """Lens distortion (k1, k2) that makes the mat's edges straight, and the mat's corners once straightened."""
    edges = edge_points(green_mask(img))
    # Pattern search over (k1, k2): try a step in each direction, keep any improvement, halve the step when stuck.
    k, step = np.zeros(2), 0.2
    best = _line_cost(edges, k)
    while step > 1e-5:
        improved = False
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            trial = k + step * np.array(d, dtype=float)
            c = _line_cost(edges, trial)
            if c < best:
                k, best, improved = trial, c, True
        if not improved:
            step /= 2
    k1, k2 = k
    res = type("R", (), {"fun": best})()
    lines = {}
    for name, p in edges.items():
        lines[name] = _fit_line(_undistort(p, k1, k2))
    def meet(a, b):
        (n1, d1), (n2, d2) = lines[a], lines[b]
        return np.linalg.solve(np.array([n1, n2]), np.array([d1, d2]))
    corners = np.array([meet("top", "left"), meet("top", "right"), meet("bottom", "right"), meet("bottom", "left")])
    rms_before = math.sqrt(_line_cost(edges, [0.0, 0.0]) / sum(len(p) for p in edges.values()))
    rms_after = math.sqrt(res.fun / sum(len(p) for p in edges.values()))
    model = {"k1": float(k1), "k2": float(k2), "corners_undistorted": corners.tolist(),
             "edge_straightness_rms_px": {"before": rms_before, "after": rms_after}}
    return model


class Flattener:
    """Turns the camera picture into a top-down picture of just the mat (RECT_W x RECT_H, PX_PER_CM per cm)."""

    def __init__(self, model):
        self.model = model
        dst = np.array([[0, 0], [RECT_W, 0], [RECT_W, RECT_H], [0, RECT_H]], dtype=np.float32)
        self.H = cv2.getPerspectiveTransform(np.array(model["corners_undistorted"], dtype=np.float32), dst)
        Hinv = np.linalg.inv(self.H)
        u, v = np.meshgrid(np.arange(RECT_W, dtype=np.float32), np.arange(RECT_H, dtype=np.float32))
        pts = np.stack([u.ravel(), v.ravel(), np.ones(u.size, np.float32)])
        src = Hinv @ pts
        xu = ((src[0] / src[2]) - CENTRE[0]) / F_PX           # undistorted, normalised
        yu = ((src[1] / src[2]) - CENTRE[1]) / F_PX
        r2 = xu * xu + yu * yu
        f = 1 + model["k1"] * r2 + model["k2"] * r2 * r2
        self.map_x = (xu * f * F_PX + CENTRE[0]).reshape(RECT_H, RECT_W).astype(np.float32)
        self.map_y = (yu * f * F_PX + CENTRE[1]).reshape(RECT_H, RECT_W).astype(np.float32)

    def __call__(self, img, interpolation=cv2.INTER_LINEAR):
        return cv2.remap(img, self.map_x, self.map_y, interpolation)


def get_flattener(img, refit=False):
    if MODEL_FILE.exists() and not refit:
        return Flattener(json.loads(MODEL_FILE.read_text())), json.loads(MODEL_FILE.read_text())
    model = fit_camera_model(img)
    MODEL_FILE.write_text(json.dumps(model, indent=1))
    return Flattener(model), model


def find_cube(rect, mat_mask=None):
    """(cx_px, cy_px, yaw_off_deg, box) of the cube in the flattened picture, or None.
    The cube must lie ON the mat: when mat_mask (the flattened green-mat mask) is given, the ring of pixels just
    around a candidate has to be green mat. That rejects wood-coloured things at the mat's edge or off it (the
    camera stand's base, the table, the arm) while a cube sitting on the mat is surrounded by mat."""
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (5, 80, 130), (24, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        area = cv2.contourArea(c)
        (cx, cy), (w, h), _ = cv2.minAreaRect(c)
        if 0.6 * (2.5 * PX_PER_CM) ** 2 < area < 1.8 * (4.2 * PX_PER_CM) ** 2 and 0.6 < w / max(h, 1e-6) < 1.7:
            if mat_mask is not None:
                blob = np.zeros(mat_mask.shape, np.uint8)
                cv2.drawContours(blob, [c], -1, 255, -1)
                ring = cv2.subtract(cv2.dilate(blob, np.ones((61, 61), np.uint8)), cv2.dilate(blob, np.ones((13, 13), np.uint8)))
                if (mat_mask[ring > 0] > 0).mean() < RING_MAT_FRACTION:
                    continue
            if best is None or area > best[0]:
                best = (area, c)
    if best is None:
        return None
    rect_box = cv2.minAreaRect(best[1])
    box = cv2.boxPoints(rect_box)
    e = box[1] - box[0]                                  # one edge, in picture coordinates (x right, y down)
    forward, left = -e[1], -e[0]                         # arm forward = up in the picture, left = left
    yaw = (math.degrees(math.atan2(left, forward)) + 45.0) % 90.0 - 45.0
    return float(rect_box[0][0]), float(rect_box[0][1]), float(yaw), box


def to_sim_cm(cx, cy, base_x_cm=0.0, base_y_cm=0.0):
    """Position in the flattened picture -> (forward cm, left cm) in the sim's frame."""
    forward = (RECT_H - cy) / PX_PER_CM - base_x_cm
    left = (RECT_W / 2 - cx) / PX_PER_CM - base_y_cm
    return forward, left


def zone_status(x_cm, y_cm):
    inside = ZONE["x"][0] <= x_cm <= ZONE["x"][1] and ZONE["y"][0] <= y_cm <= ZONE["y"][1]
    if inside:
        return "inside the zone the policy was trained on"
    notes = []
    if x_cm > ZONE["x"][1]:
        notes.append(f"{x_cm - ZONE['x'][1]:.0f} cm too far out")
    if x_cm < ZONE["x"][0]:
        notes.append(f"{ZONE['x'][0] - x_cm:.0f} cm too close")
    if y_cm > ZONE["y"][1]:
        notes.append(f"{y_cm - ZONE['y'][1]:.0f} cm too far left")
    if y_cm < ZONE["y"][0]:
        notes.append(f"{ZONE['y'][0] - y_cm:.0f} cm too far right")
    return "OUTSIDE the trained zone: " + ", ".join(notes)


def draw_zone(rect, base_x_cm=0.0, base_y_cm=0.0):
    x0 = int(RECT_W / 2 - (ZONE["y"][1] + base_y_cm) * PX_PER_CM)
    x1 = int(RECT_W / 2 - (ZONE["y"][0] + base_y_cm) * PX_PER_CM)
    y0 = int(RECT_H - (ZONE["x"][1] + base_x_cm) * PX_PER_CM)
    y1 = int(RECT_H - (ZONE["x"][0] + base_x_cm) * PX_PER_CM)
    cv2.rectangle(rect, (x0, y0), (x1, y1), (255, 200, 0), 3)
    cv2.putText(rect, "place the cube inside this box", (x0 - 40, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)


def analyse(img, base_x_cm, base_y_cm, flattener):
    rect = flattener(img)
    mat = flattener(green_mask(img), cv2.INTER_NEAREST)
    found = find_cube(rect, mat)
    shown = rect.copy()
    draw_zone(shown, base_x_cm, base_y_cm)
    result = None
    if found:
        cx, cy, yaw, box = found
        fx, fy = to_sim_cm(cx, cy, base_x_cm, base_y_cm)
        cv2.polylines(shown, [box.astype(int)], True, (0, 0, 255), 3)
        label = f"cube: {fx:.1f} cm out, {fy:+.1f} cm left, turned {yaw:+.0f} deg"
        cv2.putText(shown, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(shown, zone_status(fx, fy), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        result = {"x_cm": fx, "y_cm": fy, "yaw_deg": yaw, "pixel": [cx, cy], "zone": zone_status(fx, fy)}
    else:
        cv2.putText(shown, "cube not found", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    return result, shown


PAGE = """<!doctype html><meta charset="utf-8"><title>Cube on the mat</title>
<style>body{font-family:system-ui,sans-serif;margin:16px;background:#14161a;color:#eee}
img{max-width:100%;border:1px solid #444}.row{display:flex;gap:12px;flex-wrap:wrap}.row>div{flex:1 1 420px}h3{margin:6px 0;font-weight:500}button{font-size:18px;padding:10px 18px;margin:8px 0}
#st{font-size:18px;margin:6px 0}.ok{color:#6f6}.bad{color:#f88}</style>
<h2>Cube on the mat (live)</h2>
<div id="st">starting...</div>
<button onclick="save()">Save this cube position</button> <span id="saved"></span>
<div class="row"><div><h3>Real mat (camera)</h3><img src="/stream"></div>
<div><h3>Sim from above: its arm copies the real arm, its cube is where the camera sees it</h3><img src="/sim"></div>
<div><h3>Sim side view (the policy's own side camera)</h3><img src="/side"></div></div>
<script>
async function poll(){try{const r=await (await fetch('/status')).json();
 const st=document.getElementById('st');
 if(r.found){st.className=r.inside?'ok':'bad';st.textContent=`cube ${r.x_cm.toFixed(1)} cm out, ${r.y_cm.toFixed(1)} cm left, turned ${r.yaw_deg.toFixed(0)} deg - ${r.zone}`}
 else{st.className='bad';st.textContent='cube not found'}}catch(e){}
 setTimeout(poll,400)}
async function save(){const r=await (await fetch('/save')).json();document.getElementById('saved').textContent=r.message}
poll()
</script>"""


# The real arm's rest pose (deg), used when the arm cannot be read: shoulder_pan, lift, elbow, wrist_flex, roll, gripper.
REST_DEG = {"shoulder_pan": 13.0, "shoulder_lift": -100.0, "elbow_flex": 96.0, "wrist_flex": 61.0, "wrist_roll": -6.0, "gripper": -10.0}


class ArmMirror:
    """Reads the real arm's joint angles (read only: torque is left exactly as it is) so the sim arm can copy it.
    Falls back to the rest pose if the arm is not reachable. NOTE: while this holds the serial port, nothing else
    (the replay script, the bridge) can use it."""

    def __init__(self, port="COM3", robot_id="my_follower"):
        self.bridge = None
        self.live = False
        try:
            from sim_real_bridge import Bridge
            self.bridge = Bridge(port, robot_id, 4.0)
            self.bridge.bus.connect()
            self.live = True
            print("Sim arm mirrors the real arm on", port, flush=True)
        except Exception as e:
            print("Real arm not readable, the sim arm shows the rest pose instead:", str(e)[:100], flush=True)

    def pose(self):
        """{joint name: sim angle in rad}"""
        if self.live:
            try:
                real = self.bridge.read_real()
                return {j: self.bridge.real_to_sim(j, real[j]) for j in real}
            except Exception as e:
                print("lost the real arm:", str(e)[:80], flush=True)
                self.live = False
        return {j: math.radians(v) if j != "gripper" else math.radians(v) for j, v in REST_DEG.items()}

    def close(self):
        try:
            if self.bridge is not None and self.bridge.bus.is_connected:
                self.bridge.bus.disconnect(disable_torque=False)
        except Exception:
            pass


def make_sim_view(arm=None, width=960, height=720):
    """Returns f(result) -> (JPEG of the sim seen from above, JPEG of the sim from the policy's side camera), with
    the sim cube where the camera sees the real one; None if the sim cannot be drawn. The sim arm stays in the policy's start pose. Must be called from the thread
    that renders."""
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "scene_pick_cube.xml"))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
    except Exception as e:                      # no OpenGL, or mujoco missing: the camera view still works
        print("sim view unavailable:", e, flush=True)
        return None
    cube_adr = model.joint("cube_freejoint").qposadr[0]
    cube_z = float(model.body("cube").pos[2])
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.23, 0.0, 0.0]
    cam.distance, cam.azimuth, cam.elevation = 0.68, 0, -90     # straight down, arm's forward = up in the picture

    def render(result):
        pose = arm.pose() if arm is not None else {j: math.radians(v) for j, v in REST_DEG.items()}
        for name, v in pose.items():
            data.qpos[model.joint(name).qposadr[0]] = v
            data.ctrl[model.actuator(name).id] = v
        x, y, yaw = result["x_cm"] / 100.0, result["y_cm"] / 100.0, math.radians(result["yaw_deg"])
        data.qpos[cube_adr:cube_adr + 7] = [x, y, cube_z, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        mujoco.mj_forward(model, data)
        label = f"sim cube: {result['x_cm']:.1f} cm out, {result['y_cm']:+.1f} cm left, turned {result['yaw_deg']:+.0f} deg"
        out = []
        for camera, text in ((cam, label), ("overhead_cam", "side view - what the policy's side camera sees")):
            renderer.update_scene(data, camera=camera)
            frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            cv2.putText(frame, text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            out.append(cv2.imencode(".jpg", cv2.resize(frame, None, fx=0.7, fy=0.7), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
        return out[0], out[1]

    return render


def serve(args, port=8765):
    """Live view in a web page (the LeRobot venv's OpenCV has no window support): http://127.0.0.1:PORT"""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    state = {"jpg": None, "sim_jpg": None, "side_jpg": None, "result": None, "stop": False}

    def capture():
        cap = cv2.VideoCapture(find_camera(), cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        flattener = None
        arm = ArmMirror()
        sim = make_sim_view(arm)
        while not state["stop"]:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            if flattener is None:
                flattener, _ = get_flattener(frame, refit=args.refit)
            result, shown = analyse(frame, args.base_x_cm, args.base_y_cm, flattener)
            state["result"] = result
            if sim is not None and result is not None:
                state["sim_jpg"], state["side_jpg"] = sim(result)
            state["jpg"] = cv2.imencode(".jpg", cv2.resize(shown, None, fx=0.6, fy=0.6), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        cap.release()
        arm.close()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self._send(200, "text/html; charset=utf-8", PAGE.encode())
            elif self.path == "/status":
                r = state["result"]
                body = {"found": r is not None}
                if r:
                    body.update(r, inside=r["zone"].startswith("inside"))
                self._send(200, "application/json", json.dumps(body).encode())
            elif self.path == "/save":
                r = state["result"]
                if r:
                    (OUT / "cube_pose.json").write_text(json.dumps(r, indent=1))
                    msg = f"saved: {r['x_cm']:.1f} cm out, {r['y_cm']:+.1f} cm left, turned {r['yaw_deg']:+.0f} deg"
                    print(msg, flush=True)
                else:
                    msg = "not saved: no cube found"
                self._send(200, "application/json", json.dumps({"message": msg}).encode())
            elif self.path in ("/stream", "/sim", "/side"):
                key = {"/stream": "jpg", "/sim": "sim_jpg", "/side": "side_jpg"}[self.path]
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while not state["stop"]:
                        if state[key]:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + state[key] + b"\r\n")
                        time.sleep(0.15)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
            else:
                self._send(404, "text/plain", b"not found")

    threading.Thread(target=capture, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Live view: http://127.0.0.1:{port}   (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state["stop"] = True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=None, help="Use a saved picture instead of the camera.")
    ap.add_argument("--refit", action="store_true", help="Fit the lens model again from this picture (do this if the camera moved).")
    ap.add_argument("--live", action="store_true", help="Live view as a web page at http://127.0.0.1:8765 (a Save button stores the pose).")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--base_x_cm", type=float, default=BASE_X_CM, help="Arm's rotation axis, cm out from the mat edge.")
    ap.add_argument("--base_y_cm", type=float, default=BASE_Y_CM, help="Arm's rotation axis, cm left of the mat's middle.")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.live:
        serve(args, args.port)
        return

    img = cv2.imread(args.image) if args.image else grab()
    if img is None:
        raise SystemExit(f"Could not read {args.image}")
    stamp = "image" if args.image else time.strftime("%Y%m%d_%H%M%S")
    flattener, model = get_flattener(img, refit=args.refit)
    print(f"lens distortion k1={model['k1']:.4f} k2={model['k2']:.4f}; mat edges' straightness error "
          f"{model['edge_straightness_rms_px']['before']:.1f} px -> {model['edge_straightness_rms_px']['after']:.1f} px")
    result, shown = analyse(img, args.base_x_cm, args.base_y_cm, flattener)
    cv2.imwrite(str(OUT / "camera" / f"mat_topdown_{stamp}.png"), shown)
    cv2.imwrite(str(OUT / "camera" / f"mat_flat_{stamp}.png"), flattener(img))
    print("saved the top-down pictures in outputs/real_replay/camera/")
    if result:
        print(f"cube: {result['x_cm']:.1f} cm out from the mat edge, {result['y_cm']:+.1f} cm to the left, "
              f"turned {result['yaw_deg']:+.0f} deg  ->  {result['zone']}")
        (OUT / "cube_pose.json").write_text(json.dumps(result, indent=1))
    else:
        print("cube not found")


if __name__ == "__main__":
    main()
