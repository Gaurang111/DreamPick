"""Showcase videos (LeRobot venv): two 1080p cuts of the real-arm pick, built from a recorded run + scripts/showcase_prep.py + scripts/showcase_export.py.

    ...\\lerobot\\venv\\Scripts\\python.exe scripts\\make_showcase.py A run_20261115_212001      cinematic sequence
    ...\\lerobot\\venv\\Scripts\\python.exe scripts\\make_showcase.py B run_20261115_213201      single-screen pipeline
Output: outputs/showcase/<name>.mp4 (H.264) and a poster frame.
"""
import json
import math
import pathlib
import subprocess
import sys
import time

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import locate_cube as lc  # noqa: E402
from so101_env import camera_map  # noqa: E402

W, H, FPS = 1920, 1080, 30
BG, PANEL, LINE = (9, 13, 22), (17, 24, 39), (38, 52, 78)
CYAN, ORANGE, GREEN, PINK, YELLOW = (64, 224, 255), (255, 159, 28), (84, 224, 140), (255, 99, 160), (255, 226, 92)
WHITE, GRAY, DIM = (240, 244, 252), (150, 162, 186), (88, 100, 126)
JOINT_COLORS = [(120, 170, 255), CYAN, GREEN, ORANGE, PINK, YELLOW]          # pan, lift, elbow, wrist flex, wrist roll, gripper
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist", "roll", "gripper"]
FONT_DIR = pathlib.Path(r"C:\Windows\Fonts")
_fonts = {}


def font(size, weight="regular"):
    key = (size, weight)
    if key not in _fonts:
        name = {"regular": "segoeui.ttf", "bold": "segoeuib.ttf", "light": "segoeuil.ttf", "semibold": "seguisb.ttf"}[weight]
        _fonts[key] = ImageFont.truetype(str(FONT_DIR / name), size)
    return _fonts[key]


# ---------------------------------------------------------------- data
class Run:
    def __init__(self, name, stretch=2.0):
        self.name, self.stretch = name, stretch
        base = ROOT / "outputs" / "real_live"
        stamp = name.replace("run_", "")
        self.dir = base / name
        self.prep = json.loads((ROOT / "outputs" / "showcase" / name / "prep.json").read_text())
        self.m = dict(np.load(ROOT / "outputs" / "showcase" / name / "model.npz"))
        self.plan = json.loads((base / f"plan_{stamp}.json").read_text())
        self.follow = json.loads((base / f"follow_{stamp}.json").read_text())
        self.goals_deg = np.degrees(self.m["goals"])                    # (301, 6) plan-time joint targets
        self.lock = self.prep["placement"]["lock"]                      # cube pose (x, y, yaw) in the policy frame
        self.align = np.array(self.prep["align"])                       # replay step -> camera frame index
        self.n_steps = len(self.align)
        h = self.m["height"]
        g = self.goals_deg[:, 5]
        top = int(np.argmax(g))
        self.t_close = next((i for i in range(top, len(g)) if g[i] < g.max() - 8.0), len(g) - 1)      # plan step where the gripper starts closing
        self.t_lift = int(np.argmax(h >= 0.01))
        self.t_hold = int(np.argmax(h >= 0.05))
        self.peak_cm = float(h.max() * 100)


class SeqReader:
    """Frames of a video, asked for in increasing order only (cheap sequential decoding)."""

    def __init__(self, path):
        self.cap, self.pos, self.last = cv2.VideoCapture(str(path)), -1, None

    def get(self, i):
        if i < self.pos:
            raise ValueError("SeqReader only moves forward")
        while self.pos < i:
            ok, f = self.cap.read()
            if not ok:
                break
            self.pos += 1
            self.last = f
        return self.last


class SimRenderer:
    def __init__(self, w, h):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "scene_pick_cube.xml"))
        self.data = mujoco.MjData(self.model)
        self.r = mujoco.Renderer(self.model, h, w)
        self.cube_adr = self.model.joint("cube_freejoint").qposadr[0]
        self.cam = mujoco.MjvCamera()

    def render(self, qpos, az=140.0, el=-32.0, dist=0.5, lookat=(0.17, 0.0, 0.07)):
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        self.cam.lookat[:] = lookat
        self.cam.distance, self.cam.azimuth, self.cam.elevation = dist, az, el
        self.r.update_scene(self.data, camera=self.cam)
        return cv2.cvtColor(self.r.render(), cv2.COLOR_RGB2BGR)

    def set_cube(self, qpos, pose):
        q = np.array(qpos, float)
        if pose is None:
            q[self.cube_adr:self.cube_adr + 7] = [0, 0, -0.5, 1, 0, 0, 0]
        else:
            x, y, yaw = pose[0] / 100.0, pose[1] / 100.0, math.radians(pose[2])
            q[self.cube_adr:self.cube_adr + 7] = [x, y, 0.015, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        return q


def interp_qpos(qpos, s):
    s = float(np.clip(s, 0, len(qpos) - 1))
    i, w = int(s), s - int(s)
    j = min(i + 1, len(qpos) - 1)
    return (1 - w) * qpos[i] + w * qpos[j]


# ---------------------------------------------------------------- drawing helpers
def ease(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def background(t=0.0):
    return Image.new("RGB", (W, H), BG)


def glow_text(img, xy, text, f, fill, glow=None, anchor="la"):
    if glow:
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).text(xy, text, font=f, fill=glow + (200,), anchor=anchor)
        layer = layer.filter(ImageFilter.GaussianBlur(10))
        img.paste(layer, (0, 0), layer)
    ImageDraw.Draw(img).text(xy, text, font=f, fill=fill, anchor=anchor)


def chip(img, xy, text, color, active=True, f=None, pad=(18, 8)):
    f = f or font(26, "semibold")
    d = ImageDraw.Draw(img)
    l, t, r, b = d.textbbox(xy, text, font=f)
    box = (l - pad[0], t - pad[1], r + pad[0], b + pad[1])
    if active:
        d.rounded_rectangle(box, radius=(box[3] - box[1]) // 2, fill=tuple(int(c * 0.28) for c in color), outline=color, width=2)
        d.text(xy, text, font=f, fill=color)
    else:
        d.rounded_rectangle(box, radius=(box[3] - box[1]) // 2, outline=DIM, width=2)
        d.text(xy, text, font=f, fill=DIM)
    return box


def paste_bgr(img, arr, xy, size, border=CYAN, radius=18, label=None, label_color=None, dim=1.0):
    w, h = size
    a = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
    if dim < 1.0:
        a = (a.astype(np.float32) * dim).astype(np.uint8)
    p = Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=255)
    img.paste(p, xy, mask)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((xy[0], xy[1], xy[0] + w - 1, xy[1] + h - 1), radius=radius, outline=border, width=3)
    if label:
        chip(img, (xy[0] + 18, xy[1] + 14), label, label_color or border, f=font(18, "semibold"), pad=(12, 5))


def crop_real(f):
    """The 1280x720 camera frame cut to the mat around the arm and the cube (16:9)."""
    return f[110:614, 150:1030]


def to_np(img):
    return np.asarray(img, dtype=np.uint8)


# the camera's pixel for a policy-frame position (for the cube marker on the real picture)
_fl = None


def cube_pixel(x_cm, y_cm):
    global _fl
    if _fl is None:
        _fl, _ = lc.get_flattener(cv2.imread(str(ROOT / "outputs" / "phase1" / "top_cube_raw.jpg")))
    m = camera_map.load()
    A, b = np.array(m["A"]), np.array(m["b"])
    cx, cy = np.linalg.solve(A, np.array([x_cm, y_cm]) - b)
    px = int(np.clip(lc.RECT_W / 2 - cy * lc.PX_PER_CM, 0, lc.RECT_W - 1))
    py = int(np.clip(lc.RECT_H - cx * lc.PX_PER_CM, 0, lc.RECT_H - 1))
    return int(_fl.map_x[py, px]), int(_fl.map_y[py, px])


def draw_detection(frame_bgr, pose, t):
    """Cube marker on a 1280x720 camera frame: corner brackets, pulsing."""
    out = frame_bgr.copy()
    if pose is None:
        return out
    px, py = cube_pixel(pose[0], pose[1])
    r = int(46 + 6 * math.sin(t * 7))
    c = (255, 224, 64)
    for sx in (-1, 1):
        for sy in (-1, 1):
            x0, y0 = px + sx * r, py + sy * r
            cv2.line(out, (x0, y0), (x0 - sx * 22, y0), c, 4, cv2.LINE_AA)
            cv2.line(out, (x0, y0), (x0, y0 - sy * 22), c, 4, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------- charts
def chart_joints(img, box, run, step_plan, upto_only=True, labels=True, title=None):
    """All six joint-target curves of the plan, drawn as far as the cursor, with the cursor and a glow on the head."""
    x0, y0, x1, y1 = box
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(box, radius=16, fill=PANEL, outline=LINE, width=2)
    g = run.goals_deg
    n = len(g)
    pad = 18
    inner = (x0 + 100 if labels else x0 + pad, y0 + pad + (26 if title else 0), x1 - pad, y1 - pad)
    if title:
        d.text((x0 + 16, y0 + 10), title, font=font(18), fill=GRAY)
    for j in range(6):
        col = g[:, j]
        lo, hi = col.min(), col.max()
        span = max(hi - lo, 1.0)
        ys = inner[3] - (col - lo) / span * (inner[3] - inner[1])
        xs = inner[0] + np.arange(n) / (n - 1) * (inner[2] - inner[0])
        k = int(min(step_plan, n - 1)) if upto_only else n - 1
        pts = list(zip(xs[:k + 1], ys[:k + 1]))
        if len(pts) > 1:
            d.line(pts, fill=JOINT_COLORS[j], width=3)
        if upto_only and k < n - 1:
            d.line(list(zip(xs[k:], ys[k:])), fill=tuple(int(c * 0.22) for c in JOINT_COLORS[j]), width=2)
        d.ellipse((xs[k] - 6, ys[k] - 6, xs[k] + 6, ys[k] + 6), fill=JOINT_COLORS[j])
        if labels:
            d.text((x0 + 22, inner[1] + 6 + j * ((inner[3] - inner[1] - 12) / 6)), JOINT_NAMES[j], font=font(20), fill=JOINT_COLORS[j])
    cx = inner[0] + min(step_plan, n - 1) / (n - 1) * (inner[2] - inner[0])
    d.line([(cx, inner[1] - 6), (cx, inner[3] + 6)], fill=WHITE, width=2)
    # phase ticks
    for t, name, col in ((run.t_close, "grasp", ORANGE), (run.t_lift, "lift", GREEN)):
        tx = inner[0] + t / (n - 1) * (inner[2] - inner[0])
        d.line([(tx, inner[3] - 4), (tx, inner[3] + 4)], fill=col, width=3)
        if labels:
            d.text((tx + 6, inner[3] - 24), name, font=font(18), fill=col)


def chart_imagination(img, box, run, t, rows=(1, 2, 3), window=40):
    """The world model's imagination: for three joints, the past (solid), what the model imagines for the next 15 steps (dotted, glowing) and what really happened (thin)."""
    x0, y0, x1, y1 = box
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(box, radius=16, fill=PANEL, outline=LINE, width=2)
    m = run.m
    t = int(np.clip(t, 1, len(m["imag_j"]) - 1))
    joints = np.degrees(m["joints"])                         # (301, 6)
    imag = np.degrees(m["imag_j"][t])                        # (15, 6)
    H_ = imag.shape[0]
    row_h = (y1 - y0 - 60) / len(rows)
    for r_i, j in enumerate(rows):
        ry0 = y0 + 44 + r_i * row_h
        ry1 = ry0 + row_h - 14
        col = JOINT_COLORS[j]
        lo_t, hi_t = max(0, t - window), min(len(joints) - 1, t + H_)
        seg = joints[lo_t:hi_t + 1, j]
        allv = np.concatenate([seg, imag[:, j]])
        lo, hi = allv.min() - 2, allv.max() + 2
        span = max(hi - lo, 6.0)
        px = lambda i: x0 + 90 + (i - (t - window)) / (window + H_) * (x1 - x0 - 120)
        py = lambda v: ry1 - (v - lo) / span * (ry1 - ry0)
        d.text((x0 + 20, ry0 + 6), JOINT_NAMES[j], font=font(22, "semibold"), fill=col)
        d.line([(x0 + 84, ry1 + 2), (x1 - 20, ry1 + 2)], fill=LINE, width=1)
        past = [(px(i), py(joints[i, j])) for i in range(lo_t, t + 1)]
        if len(past) > 1:
            d.line(past, fill=col, width=4)
        fut = [(px(i), py(joints[i, j])) for i in range(t, hi_t + 1)]
        if len(fut) > 1:
            d.line(fut, fill=(110, 120, 140), width=2)                 # what really happened
        ip = [(px(t), py(joints[t, j]))] + [(px(t + 1 + h), py(imag[h, j])) for h in range(H_)]
        glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.line(ip, fill=col + (160,), width=12)
        glow = glow.filter(ImageFilter.GaussianBlur(7))
        img.paste(glow, (0, 0), glow)
        d = ImageDraw.Draw(img)
        for a, b in zip(ip[:-1:2], ip[1::2]):                           # dotted: the imagined path
            d.line([a, b], fill=WHITE, width=4)
        d.ellipse((px(t) - 7, py(joints[t, j]) - 7, px(t) + 7, py(joints[t, j]) + 7), fill=WHITE)
    d.text((x0 + 20, y0 + 10), "dotted: imagined    grey: real", font=font(19), fill=GRAY)


def chart_reward(img, box, run, t):
    x0, y0, x1, y1 = box
    d = ImageDraw.Draw(img)
    t = int(np.clip(t, 0, len(run.m["imag_r"]) - 1))
    r = run.m["imag_r"][t]
    r = np.clip(r, -1, 8)
    d.text((x0, y0), "expected reward over the imagined steps", font=font(20), fill=GRAY)
    bw = (x1 - x0) / len(r)
    for i, v in enumerate(r):
        h = (y1 - y0 - 30) * min(1.0, max(0.04, v / 6.0))
        d.rectangle((x0 + i * bw + 2, y1 - h, x0 + (i + 1) * bw - 2, y1), fill=(int(40 + 40 * i / len(r)), int(160 + 60 * i / len(r)), 255))


def draw_network(img, box, t, pulse_speed=0.9):
    """A stylised network: layers of nodes with activations travelling left to right (decoration, no data)."""
    x0, y0, x1, y1 = box
    d = ImageDraw.Draw(img)
    layers = [5, 7, 8, 7, 5]
    rng = np.random.default_rng(3)
    xs = np.linspace(x0 + 40, x1 - 40, len(layers))
    nodes = [[(xs[i], y0 + 40 + (j + 0.5) * (y1 - y0 - 80) / n) for j in range(n)] for i, n in enumerate(layers)]
    phase = (t * pulse_speed) % 1.0
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for i in range(len(layers) - 1):
        for a in nodes[i]:
            for b in nodes[i + 1]:
                if rng.random() < 0.55:
                    d.line([a, b], fill=(34, 52, 86), width=1)
    for i in range(len(layers)):
        lit = np.exp(-((i / (len(layers) - 1) - phase) ** 2) / 0.02)
        for (x, y) in nodes[i]:
            r = 9 + 4 * lit
            c = tuple(int(DIM[k] + (CYAN[k] - DIM[k]) * lit) for k in range(3))
            gd.ellipse((x - r - 6, y - r - 6, x + r + 6, y + r + 6), fill=CYAN + (int(150 * lit),))
            d.ellipse((x - r, y - r, x + r, y + r), fill=c, outline=(20, 30, 50))
    glow = glow.filter(ImageFilter.GaussianBlur(8))
    img.paste(glow, (0, 0), glow)


def arrow(img, a, b, color, t=None, active=True, width=5):
    d = ImageDraw.Draw(img)
    col = color if active else DIM
    d.line([a, b], fill=col, width=width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    for s in (-0.5, 0.5):
        d.line([b, (b[0] - 22 * math.cos(ang + s), b[1] - 22 * math.sin(ang + s))], fill=col, width=width)
    if active and t is not None:                                   # a travelling packet
        k = (t * 1.3) % 1.0
        px, py = a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k
        d.ellipse((px - 8, py - 8, px + 8, py + 8), fill=WHITE)


def start_encoder(path):
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


# ---------------------------------------------------------------- shared pieces
def real_placement_frame(run, rd, idx_frames, dets, t_clip, speed_fps=7.0):
    """The real camera during the placement clip at clip time t_clip (s): picture + detection marker + the detected pose."""
    n = len(idx_frames)
    k = int(min(n - 1, t_clip * speed_fps))
    f = rd.get(idx_frames[k])
    pose = dets[k]["pose"]
    return draw_detection(f, pose, t_clip), pose, k


def sim_cube_twin(sim, run, pose, t, az0=140.0):
    q0 = run.m["qpos"][0]
    q = sim.set_cube(q0, pose)
    return sim.render(q, az=az0 + 6 * math.sin(t * 0.8), dist=0.44)


def readout(img, xy, pose, f=None, color=CYAN):
    f = f or font(40, "bold")
    if pose is None:
        glow_text(img, xy, "searching for the cube ...", f, GRAY)
        return
    glow_text(img, xy, f"cube   x {pose[0]:.1f} cm    y {pose[1]:+.1f} cm    yaw {pose[2]:+.0f}°", f, WHITE, glow=color)
