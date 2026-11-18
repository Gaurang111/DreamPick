"""Chooses the fixed (overhead) camera by measurement.

Renders the real 64x64 observation at many candidate camera poses, across
arm-and-cube situations that matter (approaching, gripping and carrying the
cube, with the cube at several spots across its random start zone), and counts
how many pixels of the cube and of each arm link the camera actually sees. A
good camera never loses the cube behind the arm (the old behind-the-arm view
did exactly that when the gripper reached it) and always shows every link.

    python scripts/camera_search.py [--top 8] [--csv path] [--sheet path]

Candidates are applied to the model in memory only; no project file is changed.
"""

import argparse
import itertools
import math
import pathlib

import mujoco
import numpy as np
from PIL import Image, ImageDraw

SCENE = pathlib.Path(__file__).resolve().parents[1] / "assets" / "scene_pick_cube.xml"
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# Joint ranges the real arm can use (deg): calibrated range minus margins, cut at the encoder rollover.
LIMITS_DEG = {"shoulder_pan": (-109.9, 109.9), "shoulder_lift": (-101.8, 79.1), "elbow_flex": (-93.7, 93.7),
              "wrist_flex": (-98.0, 65.9), "wrist_roll": (-177.0, 169.0)}
LINKS = {"base": "base", "pan link": "shoulder", "upper arm": "upper_arm", "forearm": "lower_arm",
         "wrist": "wrist", "gripper": "gripper", "moving jaw": "moving_jaw_so101_v1"}
CUBE_HALF = 0.015
CUBE_SPOTS = [(0.22, 0.0), (0.18, 0.0), (0.26, 0.0), (0.22, 0.04), (0.22, -0.04), (0.18, 0.04), (0.26, -0.04)]
REST = [13, -100, 96, 61, -6, -10]
READY = [0, -51.6, 40.1, 57.3, 0, -10]

m = mujoco.MjModel.from_xml_path(str(SCENE))
d = mujoco.MjData(m)
cam_id = m.camera("overhead_cam").id
target_id = m.body("overhead_target_ref").id
site = m.site("gripperframe").id
qadr = [m.joint(j).qposadr[0] for j in J]
dof = [m.joint(j).dofadr[0] for j in J[:4]]
cube_adr = m.joint("cube_freejoint").qposadr[0]
lo = np.radians([LIMITS_DEG[j][0] for j in J[:4]])
hi = np.radians([LIMITS_DEG[j][1] for j in J[:4]])
body_of_geom = m.geom_bodyid
link_body = {k: m.body(v).id for k, v in LINKS.items()}
cube_body = m.body("cube").id
renderer = mujoco.Renderer(m, 64, 64)
renderer.enable_segmentation_rendering()
rgb_renderer = mujoco.Renderer(m, 64, 64)


def ik(target, tilt_deg, q0, iters=250):
    """Joints 0-3 putting the grasp point at `target` with the gripper pointing
    down, tilted `tilt_deg` from vertical. Only the tilt magnitude is constrained,
    not its compass heading: the arm's links sit a little off the pan axis, so
    the heading toward a sideways cube isn't exactly reachable and shouldn't matter."""
    t = math.radians(tilt_deg)
    dd = mujoco.MjData(m)
    q = np.array(q0, dtype=float)

    def want_axis(a):
        h = a[:2] / max(np.linalg.norm(a[:2]), 1e-6)
        return np.array([math.sin(t) * h[0], math.sin(t) * h[1], -math.cos(t)])

    for _ in range(iters):
        dd.qpos[qadr[:4]] = q[:4]
        mujoco.mj_forward(m, dd)
        a = dd.site_xmat[site].reshape(3, 3)[:, 0]
        e = np.concatenate([target - dd.site_xpos[site], 0.3 * np.cross(a, want_axis(a))])
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, dd, jp, jr, site)
        jac = np.vstack([jp, 0.3 * jr])[:, dof]
        q[:4] = np.clip(q[:4] + jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), e), lo, hi)
    ok = np.linalg.norm(target - dd.site_xpos[site]) < 1e-3 and np.linalg.norm(np.cross(a, want_axis(a))) < 0.02
    return q, ok


def solve(target, q0):
    for tilt in range(0, 61, 2):
        q, ok = ik(target, tilt, q0)
        if ok:
            return q
    return None


def max_grasp_height(cx, cy, tilt_max=60):
    """Highest the grasp point can be held (m) above a cube at (cx, cy) with the real arm's joint limits,
    gripper pointing mostly downward (tilt up to tilt_max degrees). A 5cm lift needs about 6.5cm.
    Returns None if the cube can't even be reached. See docs/training_run_1/README.md, section 11."""
    seed = np.radians(READY[:4])
    q, t = None, 0
    for t in range(0, tilt_max + 1, 2):
        q, ok = ik(np.array([cx - CUBE_HALF, cy, 0.015]), t, seed)
        if ok:
            break
    else:
        return None
    cur, last = q, 0.015
    for i in range(1, 12):
        z = 0.015 + 0.0125 * i
        hit = None
        for tilt in range(t, tilt_max + 1, 2):
            qq, ok = ik(np.array([cx - CUBE_HALF, cy, z]), tilt, cur)
            if ok:
                hit = (qq, tilt)
                break
        if hit is None:
            break
        cur, t = hit
        last = z
    return last


def build_cases():
    """(name, joint pose in radians (6), cube position or 'held')."""
    cases = [("rest", np.radians(REST), None), ("ready", np.radians(READY), None)]
    seed = np.radians(READY[:4])
    heights = [0.015 + 0.0125 * i for i in range(7)]  # 1.5cm (grasp) ... 9cm in 1.25cm steps
    for cx, cy in CUBE_SPOTS:
        # Solve at the grasp, then climb in small steps - each from the last solution, using the
        # smallest forward tilt the wrist can reach - like the real descent in reverse. A single
        # jump to the higher poses doesn't converge within the wrist limit.
        q = solve(np.array([cx - CUBE_HALF, cy, heights[0]]), seed)
        if q is None:
            continue
        found = {"grasp": q}
        cur, tilt0 = q, 0
        for i, z in enumerate(heights[1:], 1):
            hit = None
            for tilt in range(tilt0, 61, 2):
                qq, ok = ik(np.array([cx - CUBE_HALF, cy, z]), tilt, cur)
                if ok:
                    hit = (qq, tilt)
                    break
            if hit is None:
                break
            cur, tilt0 = hit
            if i == 4:
                found["lift5"] = cur  # 6.5cm: cube held 5cm above its resting height
            if i == 6:
                found["above"] = cur
        spec = {"grasp": ("pads", 10), "lift5": ("pads", 10), "above": (False, 40)}
        for label, qj in found.items():
            held, grip = spec[label]
            cases.append((f"{label}@({cx:.2f},{cy:+.2f})", np.append(qj, [0.0, math.radians(grip)]), held or ("fixed", cx, cy)))
    return cases


def set_scene(q, cube):
    d.qpos[qadr] = q
    mujoco.mj_forward(m, d)
    if cube == "pads":
        mid = 0.5 * (d.xpos[m.body("fixed_jaw_pad").id] + d.xpos[m.body("moving_jaw_pad").id])
        d.qpos[cube_adr:cube_adr + 3] = mid
    elif cube is None:
        d.qpos[cube_adr:cube_adr + 3] = [0.22, 0.0, CUBE_HALF]
    else:
        d.qpos[cube_adr:cube_adr + 3] = [cube[1], cube[2], CUBE_HALF]
    d.qpos[cube_adr + 3:cube_adr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)


JOINT_ANCHORS = [m.joint(j).id for j in J]


def joint_margin(fovy):
    """Smallest distance (pixels, of 64) from any joint axis or the grasp point to the image border; negative = outside."""
    R = d.cam_xmat[cam_id].reshape(3, 3)
    f = 32.0 / math.tan(math.radians(fovy) / 2)
    pts = [d.xanchor[j] for j in JOINT_ANCHORS] + [d.site_xpos[site], np.zeros(3)]
    worst = 99.0
    for p in pts:
        rel = p - d.cam_xpos[cam_id]
        depth = -rel @ R[:, 2]
        if depth <= 0.01:
            return -99.0
        px, py = 32 + f * (rel @ R[:, 0]) / depth, 32 - f * (rel @ R[:, 1]) / depth
        worst = min(worst, px, 63 - px, py, 63 - py)
    return worst


def counts(camera_pose):
    pos, tgt, fovy = camera_pose
    m.cam_pos[cam_id] = pos
    m.body_pos[target_id] = tgt
    m.cam_fovy[cam_id] = fovy
    mujoco.mj_forward(m, d)  # not just mj_camlight: a moved aim point only reaches the camera via kinematics
    renderer.update_scene(d, camera="overhead_cam")
    seg = renderer.render()
    geom = seg[..., 0][seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)]
    bodies = body_of_geom[geom]
    out = {"cube": int((bodies == cube_body).sum()), "margin": joint_margin(fovy)}
    for k, b in link_body.items():
        out[k] = int((bodies == b).sum())
    return out


def evaluate(camera_pose, cases):
    worst_cube, cube_px, worst_link, worst_margin = 10**9, [], ("", 10**9), 99.0
    unocc_ref = {}
    for name, q, cube in cases:
        set_scene(q, cube)
        c = counts(camera_pose)
        if cube not in (None,) and name not in unocc_ref:
            # reference: the same cube placement with the arm folded away
            keep = d.qpos[cube_adr:cube_adr + 3].copy()
            d.qpos[qadr] = np.radians(REST)
            mujoco.mj_forward(m, d)
            d.qpos[cube_adr:cube_adr + 3] = keep
            mujoco.mj_forward(m, d)
            unocc_ref[name] = max(counts(camera_pose)["cube"], 1)
        elif cube is None:
            unocc_ref[name] = max(c["cube"], 1)
        worst_margin = min(worst_margin, c["margin"])
        ratio = c["cube"] / unocc_ref[name]
        if name not in ("rest", "ready"):
            worst_cube = min(worst_cube, ratio)
            cube_px.append(c["cube"])
        for k in LINKS:
            if c[k] < worst_link[1]:
                worst_link = (f"{k} in {name}", c[k])
    return worst_cube, float(np.mean(cube_px)), worst_link, worst_margin


def candidates():
    centre = np.array([0.15, 0.0, 0.08])
    for az, el, dist, fovy in itertools.product([-90, -75, -60, -45, -30, 30, 45, 60, 75, 90], [30, 45, 60], [0.45, 0.55, 0.65, 0.8], [50, 60, 70]):
        a, e = math.radians(az), math.radians(el)
        pos = centre + dist * np.array([math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)])
        yield (np.round(pos, 3), centre, fovy), (az, el, dist, fovy)


def contact_sheet(camera_pose, cases, path, title):
    tiles = []
    pick = [c for c in cases if c[0] in ("rest", "ready") or c[0].startswith(("above@(0.22,+0.00)", "grasp@(0.22,+0.00)", "lift5@(0.22,+0.00)",
                                                                              "grasp@(0.18,+0.04)", "grasp@(0.26,-0.04)", "lift5@(0.22,-0.04)"))]
    for name, q, cube in pick:
        set_scene(q, cube)
        counts(camera_pose)
        rgb_renderer.update_scene(d, camera="overhead_cam")
        im = Image.fromarray(rgb_renderer.render()).resize((256, 256), Image.NEAREST)
        ImageDraw.Draw(im).text((4, 4), name, fill=(255, 255, 255))
        tiles.append(np.array(im))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + 4], axis=1) for i in range(0, len(tiles), 4)]
    img = Image.fromarray(np.concatenate(rows, axis=0))
    ImageDraw.Draw(img).text((6, img.height - 14), title, fill=(255, 255, 0))
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--sheet_prefix", default=None, help="Write contact sheets for the top 3 candidates to <prefix>_N.png")
    args = ap.parse_args()

    cases = build_cases()
    print(f"{len(cases)} situations; current camera for reference:")
    current = (np.array([-0.05, 0.0, 0.55]), np.array([0.3, 0.0, 0.02]), 55.0)
    w, mean_px, wl, mg = evaluate(current, cases)
    print(f"  current overhead camera: worst cube visibility {w:.0%}, mean cube pixels {mean_px:.0f}, joint margin {mg:.0f}px, least-visible part: {wl[0]} ({wl[1]} px)")

    rows = []
    for pose, (az, el, dist, fovy) in candidates():
        w, mean_px, wl, mg = evaluate(pose, cases)
        rows.append((w, mean_px, wl, pose, (az, el, dist, fovy), mg))
    # Every joint must sit inside the image (3px margin) and every arm link show 15+ px in every
    # situation; then prefer the best worst-case cube visibility (capped at 90%), then a bigger cube.
    ok = lambda r: r[5] >= 3 and r[2][1] >= 15
    rows.sort(key=lambda r: (ok(r), min(r[0], 0.9), r[1]), reverse=True)
    print(f"{sum(ok(r) for r in rows)} of {len(rows)} candidates keep every joint in frame and every link visible")
    print(f"\ntop {args.top} candidates (az = angle around the workspace from straight ahead of the arm; el = elevation):")
    print(f"  {'az':>4} {'el':>3} {'dist':>5} {'fov':>4}   {'camera position (x,y,z m)':<28} {'worst cube':>10} {'mean cube px':>13} {'joint margin':>13}   least-visible arm part")
    for w, mean_px, wl, pose, (az, el, dist, fovy), mg in rows[: args.top]:
        print(f"  {az:>4} {el:>3} {dist:>5} {fovy:>4}   {str(tuple(pose[0])):<28} {w:>9.0%} {mean_px:>13.0f} {mg:>11.0f}px   {wl[0]} ({wl[1]} px)")
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["az", "el", "dist", "fovy", "x", "y", "z", "worst_cube_visibility", "mean_cube_px", "joint_margin_px", "least_visible_part", "least_visible_px"])
            for w, mean_px, wl, pose, (az, el, dist, fovy), mg in rows:
                wr.writerow([az, el, dist, fovy, *pose[0], round(w, 3), round(mean_px, 1), round(mg, 1), wl[0], wl[1]])
    if args.sheet_prefix:
        contact_sheet(current, cases, f"{args.sheet_prefix}_current.png", "CURRENT overhead camera")
        for n, (w, mean_px, wl, pose, p, mg) in enumerate(rows[:3], 1):
            contact_sheet(pose, cases, f"{args.sheet_prefix}_{n}.png", f"candidate {n}: az {p[0]} el {p[1]} dist {p[2]} fov {p[3]}  pos {tuple(pose[0])}")
        print(f"wrote contact sheets with prefix {args.sheet_prefix}")


if __name__ == "__main__":
    main()
