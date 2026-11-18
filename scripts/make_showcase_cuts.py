"""The two cuts. See make_showcase.py.   ...\\lerobot\\venv\\Scripts\\python.exe scripts\\make_showcase_cuts.py A run_20261115_212001"""
import sys
import time

from make_showcase import *  # noqa: F401,F403

SPEED = 1.5                                                      # the replay (30 s) is shown 1.5x faster
import os
STILL_TIMES = [float(v) for v in os.environ["STILLS"].split(",")] if os.environ.get("STILLS") else None      # test mode: render only these moments as PNGs


def hold_poses(dets, keep=4):
    """The camera loses the cube when a hand covers it: hold the last detection for a few frames."""
    out, last, age = [], None, 99
    for d in dets:
        if d["pose"] is not None:
            last, age = d["pose"], 0
        else:
            age += 1
        out.append(last if age <= keep else None)
    return out


class Frames:
    def __init__(self, enc):
        self.enc, self.n = enc, 0

    def put(self, img):
        if STILL_TIMES is not None:
            img.save(str(ROOT / 'outputs' / 'showcase' / 'sheets' / f'C_still_{self.n:02d}.png'))
        else:
            self.enc.stdin.write(to_np(img).tobytes())
        self.n += 1

    def poster(self, img, path):
        img.save(path)


def header(img, number, title, sub, color=CYAN):
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((60, 62, 134, 136), radius=20, fill=tuple(int(c * 0.25) for c in color), outline=color, width=3)
    d.text((97, 99), str(number), font=font(46, "bold"), fill=color, anchor="mm")
    ImageDraw.Draw(img).text((160, 58), title, font=font(58, "bold"), fill=WHITE)
    d.text((162, 128), sub, font=font(30), fill=GRAY)


# ---------------------------------------------------------------- cut A: cinematic sequence
def cut_A(run, out):
    enc = start_encoder(out)
    fr = Frames(enc)
    sim = SimRenderer(870, 489)
    rd = SeqReader(run.dir / "top_camera.mp4")
    pl = run.prep["placement"]
    idx_frames = list(range(pl["first"], pl["last"] + 1))
    dets = pl["detections"]
    held = hold_poses(dets)
    poster = None

    # ---- title (3.4 s)
    n = int(3.4 * FPS)
    for i in range(n):
        t = i / FPS
        img = background()
        a = ease(t / 0.9)
        ImageDraw.Draw(img).text((960, 420), "A real arm picking up a real cube", font=font(88, "bold"), fill=tuple(int(c * a) for c in WHITE), anchor="mm")
        ImageDraw.Draw(img).text((960, 535), "planned entirely in a simulator", font=font(88, "bold"), fill=tuple(int(c * ease((t - 0.5) / 0.9)) for c in CYAN), anchor="mm")
        d = ImageDraw.Draw(img)
        d.text((960, 690), "The model never trained on the real robot, only in simulation", font=font(38, "light"), fill=tuple(int(c * ease((t - 1.4) / 0.8)) for c in GRAY), anchor="mm")
        w = int(900 * ease((t - 0.2) / 1.6))
        d.line([(960 - w // 2, 770), (960 + w // 2, 770)], fill=CYAN, width=3)
        fr.put(img)

    # ---- 1  SEE (6 s clip + 1.2 s locked)
    t_clip_total = len(idx_frames) / pl["fps"]
    n = int((t_clip_total + 1.2) * FPS)
    last_real = None
    for i in range(n):
        t = i / FPS
        clip_t = min(t, t_clip_total - 1e-3)
        k = int(clip_t * pl["fps"])
        f = rd.get(idx_frames[k]) if t < t_clip_total else last_real
        last_real = f
        pose = held[k]
        locked = t >= t_clip_total - 0.3
        shown = crop_real(draw_detection(f, pose, t))
        simimg = sim_cube_twin(sim, run, pose, t)
        img = background()
        header(img, 1, "Camera finds the cube", "Its position goes into a simulated copy of the setup")
        paste_bgr(img, shown, (60, 250), (870, 489), border=CYAN, label="real, overhead camera", label_color=CYAN)
        paste_bgr(img, simimg, (990, 250), (870, 489), border=ORANGE, label="simulated copy", label_color=ORANGE)
        arrow(img, (934, 494), (986, 494), CYAN, t=t)
        if pose:
            glow_text(img, (960, 850), f"cube at x {pose[0]:.1f} cm, y {pose[1]:+.1f} cm, yaw {pose[2]:+.0f}°", font(44, "semibold"), WHITE, anchor="mm")
        else:
            glow_text(img, (960, 850), "waiting for the cube", font(44, "semibold"), GRAY, anchor="mm")
        if locked:
            chip(img, (960, 940), "position sent to the world model", GREEN, f=font(30, "semibold"), pad=(26, 12))
        fr.put(img)
    seen_pose = run.lock

    # ---- 2  THINK (7 s)
    n = int(7.0 * FPS)
    m = run.m
    for i in range(n):
        t = i / FPS
        sweep = ease((t - 1.0) / 5.4)
        tp = 1 + sweep * 150                                       # plan step shown (reach, grasp, start of the lift)
        img = background()
        header(img, 2, "The world model plans the pick", "It takes the cube position and the arm's joint angles, then imagines what happens next", ORANGE)
        d = ImageDraw.Draw(img)
        # input cards
        d.rounded_rectangle((60, 230, 600, 470), radius=20, fill=PANEL, outline=CYAN, width=2)
        d.text((90, 252), "input: cube position", font=font(26, "semibold"), fill=CYAN)
        d.text((90, 306), f"x  {seen_pose[0]:.1f} cm", font=font(44, "bold"), fill=WHITE)
        d.text((90, 362), f"y  {seen_pose[1]:+.1f} cm", font=font(44, "bold"), fill=WHITE)
        d.text((350, 334), f"yaw {seen_pose[2]:+.0f}°", font=font(38, "bold"), fill=WHITE)
        d.rounded_rectangle((60, 500, 600, 800), radius=20, fill=PANEL, outline=GREEN, width=2)
        d.text((90, 522), "input: arm joint angles", font=font(26, "semibold"), fill=GREEN)
        ti = int(min(tp, len(m["obs_proprio"]) - 1))
        for j in range(6):
            v = math.degrees(m["obs_proprio"][ti][j])
            d.text((90, 574 + j * 36), f"{JOINT_NAMES[j]}", font=font(24), fill=JOINT_COLORS[j])
            d.text((560, 574 + j * 36), f"{v:+6.1f}°", font=font(26, "semibold"), fill=WHITE, anchor="ra")
        # world model
        arrow(img, (604, 350), (676, 420), CYAN, t=t)
        arrow(img, (604, 650), (676, 560), GREEN, t=t + 0.4)
        draw_network(img, (680, 250, 1280, 700), t)
        ImageDraw.Draw(img).text((980, 740), "world model", font=font(44, "bold"), fill=WHITE, anchor="mm")
        d.text((980, 792), "trained in simulation", font=font(26, "light"), fill=GRAY, anchor="mm")
        arrow(img, (1284, 475), (1346, 475), ORANGE, t=t + 0.2)
        chart_imagination(img, (1350, 230, 1860, 810), run, tp)
        # output
        d.rounded_rectangle((60, 850, 1860, 1010), radius=20, fill=PANEL, outline=ORANGE, width=2)
        d.text((90, 872), "output: joint targets for the arm, 20 per second", font=font(26, "semibold"), fill=ORANGE)
        gi = int(min(tp, len(run.goals_deg) - 1))
        for j in range(6):
            d.text((90 + j * 296, 930), JOINT_NAMES[j], font=font(24), fill=JOINT_COLORS[j])
            d.text((90 + j * 296, 962), f"{run.goals_deg[gi][j]:+6.1f}°", font=font(34, "bold"), fill=WHITE)
        fr.put(img)

    # ---- 3  ACT (20 s)
    sim2 = SimRenderer(870, 489)
    n_steps = run.n_steps
    n = int(n_steps / 20.0 / SPEED * FPS)
    for i in range(n):
        t = i / FPS
        k = min(n_steps - 1, int(t * SPEED * 20))
        f = rd.get(int(run.align[k]))
        sp = k / run.stretch
        q = interp_qpos(run.m["qpos"], sp)
        simimg = sim2.render(q, az=134 + 14 * (i / n))
        img = background()
        header(img, 3, "The real arm follows the plan", f"played {SPEED:g}x faster than it happened", GREEN)
        paste_bgr(img, crop_real(f), (60, 215), (870, 489), border=CYAN, label="real arm", label_color=CYAN)
        paste_bgr(img, simimg, (990, 215), (870, 489), border=ORANGE, label="simulated copy", label_color=ORANGE)
        # phase chips
        phases = [("reach", sp < run.t_close, CYAN), ("grasp", run.t_close <= sp < run.t_lift, ORANGE), ("lift", run.t_lift <= sp < run.t_hold, GREEN), ("hold", sp >= run.t_hold, GREEN)]
        x = 60
        for name, active, col in phases:
            b = chip(img, (x, 730), name, col, active=active, f=font(34, "bold"), pad=(26, 10))
            x = b[2] + 26
        hgt = float(interp_qpos(run.m["cube"], sp)[2] - 0.015) * 100 if len(run.m["cube"]) else 0
        ImageDraw.Draw(img).text((1860, 745), f"cube lifted {max(0.0, hgt):4.1f} cm", font=font(38, "semibold"), fill=WHITE, anchor="ra")
        chart_joints(img, (60, 800, 1860, 1040), run, sp)
        if poster is None and sp >= run.t_hold + 20:
            poster = img.copy()
        fr.put(img)

    # ---- end card (6 s)
    n = int(6.0 * FPS)
    for i in range(n):
        t = i / FPS
        img = background()
        d = ImageDraw.Draw(img)
        ImageDraw.Draw(img).text((960, 190), "What it took", font=font(72, "bold"), fill=WHITE, anchor="mm")
        stats = [("3", "training runs"), ("8 days", "of training on one laptop GPU"), ("5M+", "simulated steps"), ("84%", "pick success in a noisy simulator")]
        for j, (big, small) in enumerate(stats):
            a = ease((t - 0.4 - 0.35 * j) / 0.6)
            x = 60 + j * 450
            d.rounded_rectangle((x, 330, x + 420, 640), radius=24, fill=PANEL, outline=tuple(int(c * a) for c in CYAN), width=2)
            d.text((x + 210, 450), big, font=font(86, "bold"), fill=tuple(int(c * a) for c in WHITE), anchor="mm")
            d.text((x + 210, 560), small, font=font(28), fill=tuple(int(c * a) for c in GRAY), anchor="mm")
        d.text((960, 760), "camera  →  world model (in simulation)  →  real arm", font=font(38), fill=tuple(int(c * ease((t - 2.0) / 0.8)) for c in WHITE), anchor="mm")
        fr.put(img)
    enc.stdin.close()
    enc.wait()
    if poster is not None:
        poster.save(str(out).replace(".mp4", "_poster.png"))
    return fr.n


# ---------------------------------------------------------------- cut B: one-screen pipeline
def cut_B(run, out):
    enc = start_encoder(out)
    fr = Frames(enc)
    PW, PH = 560, 315
    simA, simB = SimRenderer(PW, PH), SimRenderer(PW, PH)
    rd = SeqReader(run.dir / "top_camera.mp4")
    pl = run.prep["placement"]
    idx_frames = list(range(pl["first"], pl["last"] + 1))
    held = hold_poses(pl["detections"])
    clip = len(idx_frames) / pl["fps"]
    m = run.m
    n_steps = run.n_steps
    T_intro, T_clip, T_think = 1.6, clip + 0.6, 5.0
    T_act = n_steps / 20.0 / SPEED
    T_end = 4.5
    total = T_intro + T_clip + T_think + T_act + T_end
    X1, X2, X3 = 40, 680, 1320
    last_real, last_pose, poster = None, None, None
    n = int(total * FPS)
    for i in range(n):
        t = i / FPS
        phase1 = t >= T_intro
        t1 = t - T_intro                                           # time in the SEE phase
        t2 = t1 - T_clip                                           # THINK
        t3 = t2 - T_think                                          # ACT
        t4 = t3 - T_act                                            # end
        img = background()
        d = ImageDraw.Draw(img)
        d.text((40, 28), "Camera  →  world model  →  real arm", font=font(52, "bold"), fill=WHITE)
        d.text((40, 96), "the model only trained in simulation. 3 training runs, 8 days on one laptop GPU", font=font(28, "light"), fill=GRAY)
        stage = 0 if t1 < 0 else (1 if t2 < 0 else (2 if t3 < 0 else (3 if t4 < 0 else 4)))
        # column titles
        chip(img, (X1 + 6, 156), "1  camera", CYAN, active=stage >= 1, f=font(28, "bold"))
        chip(img, (X2 + 6, 156), "2  world model", ORANGE, active=stage >= 2, f=font(28, "bold"))
        chip(img, (X3 + 6, 156), "3  real arm", GREEN, active=stage >= 3, f=font(28, "bold"))

        # ---- column 1: real -> sim
        if stage >= 1:
            if t1 < clip:
                k = int(min(len(idx_frames) - 1, t1 * pl["fps"]))
                f = rd.get(idx_frames[k])
                last_real, pose = f, held[k]
                if pose:
                    last_pose = pose
            else:
                f, pose = last_real, last_pose
            real1 = crop_real(draw_detection(f, pose, t))
            sim1 = sim_cube_twin(simA, run, pose, t)
            paste_bgr(img, real1, (X1, 215), (PW, PH), border=CYAN, label="real", radius=14)
            paste_bgr(img, sim1, (X1, 545), (PW, PH), border=ORANGE, label="simulated copy", radius=14)
            lp = last_pose or run.lock
            d.rounded_rectangle((X1 + 12, 215 + PH - 54, X1 + PW - 12, 215 + PH - 12), radius=12, fill=(9, 13, 22))
            d.text((X1 + PW // 2, 215 + PH - 33), f"x {lp[0]:.1f} cm    y {lp[1]:+.1f} cm    yaw {lp[2]:+.0f}°", font=font(26, "bold"), fill=WHITE, anchor="mm")
        else:
            for yy in (215, 545):
                d.rounded_rectangle((X1, yy, X1 + PW, yy + PH), radius=14, outline=LINE, width=2, fill=PANEL)

        # ---- column 2: world model
        active2 = stage >= 2
        sp3 = (min(n_steps - 1, int(max(0.0, t3) * SPEED * 20)) / run.stretch) if stage >= 3 else 0.0
        tp = 1 + (ease(t2 / T_think) * 70 if 0 <= t2 < T_think else (70 if t2 >= T_think else 0))
        if stage >= 3:
            tp = min(max(sp3, 1.0), len(m["imag_j"]) - 2)                                # keeps imagining in sync with the pick
        d.rounded_rectangle((X2, 215, X2 + PW, 545 + PH), radius=14, fill=PANEL, outline=ORANGE if active2 else LINE, width=3 if active2 else 2)
        if active2:
            draw_network(img, (X2 + 10, 232, X2 + PW - 10, 470), t)
            d.text((X2 + PW // 2, 484), "world model", font=font(32, "bold"), fill=WHITE, anchor="mm")
            chart_imagination(img, (X2 + 14, 512, X2 + PW - 14, 545 + PH - 14), run, tp)
        else:
            d.text((X2 + PW // 2, 380), "waiting for the cube position", font=font(28, "light"), fill=DIM, anchor="mm")

        # ---- column 3: sim -> real
        if stage >= 3:
            ts = max(0.0, t3)
            k = min(n_steps - 1, int(ts * SPEED * 20))
            sp = k / run.stretch
            last_k = k
            f3 = rd.get(int(run.align[k]))
            q = interp_qpos(m["qpos"], sp)
            simimg = simB.render(q, az=134 + 14 * min(1.0, ts / T_act))
            paste_bgr(img, crop_real(f3), (X3, 215), (PW, PH), border=CYAN, label="real arm", radius=14)
            paste_bgr(img, simimg, (X3, 545), (PW, PH), border=ORANGE, label="simulated copy", radius=14)
            phases = [("reach", sp < run.t_close, CYAN), ("grasp", run.t_close <= sp < run.t_lift, ORANGE), ("lift", run.t_lift <= sp < run.t_hold, GREEN), ("hold", sp >= run.t_hold, GREEN)]
            d.rounded_rectangle((X3 + 12, 215 + PH - 54, X3 + PW - 12, 215 + PH - 12), radius=12, fill=(9, 13, 22))
            x = X3 + 34
            for name, act, col in phases:
                b = chip(img, (x, 215 + PH - 46), name, col, active=act, f=font(22, "bold"), pad=(14, 5))
                x = b[2] + 14
            if poster is None and sp >= run.t_hold + 25:
                poster = img.copy()
        else:
            for yy in (215, 545):
                d.rounded_rectangle((X3, yy, X3 + PW, yy + PH), radius=14, outline=LINE, width=2, fill=PANEL)

        # ---- arrows between the columns
        a1 = stage >= 2 and (stage > 2 or t2 < 1.2 or True)
        arrow(img, (X1 + PW + 12, 380), (X2 - 14, 380), CYAN, t=t, active=stage >= 2)
        d.text(((X1 + PW + X2) // 2, 345), "cube\npose", font=font(20), fill=CYAN if stage >= 2 else DIM, anchor="mm", align="center")
        arrow(img, (X2 + PW + 12, 380), (X3 - 14, 380), ORANGE, t=t + 0.3, active=stage >= 3)
        d.text(((X2 + PW + X3) // 2, 345), "joint\ntargets", font=font(20), fill=ORANGE if stage >= 3 else DIM, anchor="mm", align="center")

        # ---- bottom: the output graph
        sp_now = (min(n_steps - 1, int(max(0.0, t3) * SPEED * 20)) / run.stretch) if stage >= 3 else (tp if stage == 2 else 0)
        chart_joints(img, (40, 880, 1880, 1050), run, sp_now if stage >= 2 else 0, upto_only=True)
        if stage == 4:
            a = ease(t4 / 0.7)
            ov = Image.new("RGBA", (W, H), (9, 13, 22, int(255 * a)))
            img.paste(ov, (0, 0), ov)
            d = ImageDraw.Draw(img)
            d.text((960, 380), "Trained only in simulation", font=font(76, "bold"), fill=tuple(int(c * a) for c in WHITE), anchor="mm")
            d.text((960, 500), "3 training runs, 8 days on one laptop GPU, 5M+ simulated steps, 84% pick success in a noisy simulator", font=font(33), fill=tuple(int(c * a) for c in GRAY), anchor="mm")
        fr.put(img)
    enc.stdin.close()
    enc.wait()
    if poster is not None:
        poster.save(str(out).replace(".mp4", "_poster.png"))
    return fr.n



# ---------------------------------------------------------------- cut C: one-screen pipeline, version 1 footage, phone views
PHONE = ROOT / "outputs" / "showcase" / "phone"
PHONE_LAG = 1.55                                                 # phone clip 2 time = replay time + this (measured from the arm's motion)
PH1_CROP, PH2_CROP, PH_CROP_H = 570, 570, 771                    # rows of the 1080x1920 phone clips that show the whole arm (1.4 : 1)


def phone_crop(frame, y0):
    return frame[y0:y0 + PH_CROP_H, :1080]


def pill(img, right_x, y, text, f, fg=WHITE):
    d = ImageDraw.Draw(img)
    w = d.textlength(text, font=f)
    box = (right_x - w - 26, y, right_x, y + f.size + 14)
    d.rounded_rectangle(box, radius=(box[3] - box[1]) // 2, fill=(9, 13, 22))
    d.text((box[0] + 13, y + 6), text, font=f, fill=fg)


def phase_row(img, right_x, y, items):
    f = font(18, "semibold")
    d = ImageDraw.Draw(img)
    widths = [d.textlength(n, font=f) + 26 for n, _, _ in items]
    x = right_x - (sum(widths) + 8 * (len(items) - 1))
    for (name, act, col), w in zip(items, widths):
        box = (x, y, x + w, y + 32)
        if act:
            d.rounded_rectangle(box, radius=16, fill=tuple(int(c * 0.35) for c in col), outline=col, width=2)
            d.text((x + 13, y + 5), name, font=f, fill=col)
        else:
            d.rounded_rectangle(box, radius=16, fill=(9, 13, 22))
            d.text((x + 13, y + 5), name, font=f, fill=DIM)
        x += w + 8


def cut_C(run, out):
    enc = start_encoder(out)
    fr = Frames(enc)
    PW, PHP, PHS = 640, 457, 330                                  # side column: phone panel 640x457 (1.4:1), simulated-copy panel 640x330
    IW, IH = 190, 107                                             # the overhead camera window
    MW = 400                                                      # the narrow middle column
    X1, X2, X3 = 40, 760, 1240
    Y0 = 190
    simA, simB = SimRenderer(PW, PHS), SimRenderer(PW, PHS)
    top = SeqReader(run.dir / "top_camera.mp4")
    ph1, ph2 = SeqReader(PHONE / "proxy_1.mp4"), SeqReader(PHONE / "proxy_2.mp4")
    pl = run.prep["placement"]
    first, last = pl["first"], pl["last"]
    held = hold_poses(pl["detections"])
    m = run.m
    n_steps = run.n_steps
    PH1_LEN, PH2_LEN = 229 / 30.0, 843 / 30.0
    T_title, T_see, T_think, T_end = 3.0, PH1_LEN + 1.0, 5.0, 5.0
    T_act = n_steps / 20.0 / SPEED
    total = T_title + T_see + T_think + T_act + T_end
    n = int(total * FPS)
    last1, poster = None, None
    for i in range(n):
        t = i / FPS
        if STILL_TIMES is not None and min(abs(t - v) for v in STILL_TIMES) > 0.5 / FPS:
            continue
        # ---- title card
        if t < T_title:
            img = background()
            a = ease(t / 0.8)
            ImageDraw.Draw(img).text((960, 420), "A real arm picking up a real cube", font=font(88, "bold"), fill=tuple(int(c * a) for c in WHITE), anchor="mm")
            ImageDraw.Draw(img).text((960, 535), "planned entirely in a simulator", font=font(88, "bold"), fill=tuple(int(c * ease((t - 0.4) / 0.8)) for c in CYAN), anchor="mm")
            ImageDraw.Draw(img).text((960, 690), "The model never trained on the real robot, only in simulation", font=font(38, "light"), fill=tuple(int(c * ease((t - 1.1) / 0.7)) for c in GRAY), anchor="mm")
            fr.put(img)
            continue
        t1 = t - T_title
        t2 = t1 - T_see
        t3 = t2 - T_think
        t4 = t3 - T_act
        stage = 1 if t2 < 0 else (2 if t3 < 0 else (3 if t4 < 0 else 4))
        # ---- end card
        if stage == 4:
            img = background()
            d = ImageDraw.Draw(img)
            d.text((960, 190), "What it took", font=font(72, "bold"), fill=WHITE, anchor="mm")
            stats = [("3", "training runs"), ("~170 h", "of training in the last run"), ("5M+", "simulated steps"), ("84%", "pick success in a noisy simulator")]
            for j, (big, small) in enumerate(stats):
                a = ease((t4 - 0.3 - 0.3 * j) / 0.6)
                x = 60 + j * 450
                d.rounded_rectangle((x, 330, x + 420, 640), radius=24, fill=PANEL, outline=tuple(int(c * a) for c in CYAN), width=2)
                d.text((x + 210, 450), big, font=font(86, "bold"), fill=tuple(int(c * a) for c in WHITE), anchor="mm")
                d.text((x + 210, 560), small, font=font(28), fill=tuple(int(c * a) for c in GRAY), anchor="mm")
            d.text((960, 770), "camera  →  world model (in simulation)  →  real arm", font=font(38), fill=tuple(int(c * ease((t4 - 1.8) / 0.8)) for c in WHITE), anchor="mm")
            fr.put(img)
            continue

        img = background()
        d = ImageDraw.Draw(img)
        d.text((40, 20), "Camera  →  world model  →  real arm", font=font(48, "bold"), fill=WHITE)
        d.text((40, 82), "Trained only in simulation: 5M+ steps, about 170 hours on an RTX 3070 Ti laptop GPU", font=font(24, "light"), fill=GRAY)
        chip(img, (X1 + 6, 140), "1  Camera", CYAN, active=True, f=font(24, "bold"))
        chip(img, (X2 + 6, 140), "2  World model", ORANGE, active=stage >= 2, f=font(24, "bold"))
        chip(img, (X3 + 6, 140), "3  Real arm", GREEN, active=stage >= 3, f=font(24, "bold"))

        # ---- column 1: the phone view of the cube being placed (big), the overhead camera as a small window, and the simulated copy
        if stage == 1:
            tp = min(t1, PH1_LEN - 1.5 / 30)
            phone = ph1.get(int(tp * 30))
            fo = int(round(min(max(456 + (tp - 2.3) / 0.3, first), last)))          # overhead frame matching this moment (anchored on the hand and the cube)
            over = top.get(fo)
            pose = held[fo - first]
            last1 = (phone, over, pose)
        phone, over, pose = last1
        lp = pose or run.lock
        paste_bgr(img, phone_crop(phone, PH1_CROP), (X1, Y0), (PW, PHP), border=CYAN, label="real", radius=16)
        paste_bgr(img, crop_real(draw_detection(over, pose, t)), (X1 + 12, Y0 + PHP - IH - 12), (IW, IH), border=CYAN, radius=10)
        pill(img, X1 + PW - 12, Y0 + PHP - 12 - 36, f"x {lp[0]:.1f} cm   y {lp[1]:+.1f} cm   yaw {lp[2]:+.0f}°", font(18, "semibold"))
        paste_bgr(img, sim_cube_twin(simA, run, pose, t), (X1, Y0 + PHP + 30), (PW, PHS), border=ORANGE, label="simulated copy", radius=16)

        # ---- column 2: world model
        active2 = stage >= 2
        sp3 = 0.0
        if stage >= 3:
            sp3 = min(n_steps - 1, int(max(0.0, t3) * SPEED * 20)) / run.stretch
        tp_m = 1 + (ease(t2 / T_think) * 70 if stage == 2 else 0)
        if stage >= 3:
            tp_m = min(max(sp3, 1.0), len(m["imag_j"]) - 2)
        d.rounded_rectangle((X2, Y0, X2 + MW, Y0 + 817), radius=16, fill=PANEL, outline=ORANGE if active2 else LINE, width=2)
        if active2:
            draw_network(img, (X2 + 10, Y0 + 14, X2 + MW - 10, Y0 + 214), t)
            d.text((X2 + MW // 2, Y0 + 238), "world model", font=font(28, "bold"), fill=WHITE, anchor="mm")
            chart_imagination(img, (X2 + 12, Y0 + 266, X2 + MW - 12, Y0 + 582), run, tp_m)
            chart_joints(img, (X2 + 12, Y0 + 598, X2 + MW - 12, Y0 + 805), run, tp_m if stage == 2 else sp3, upto_only=True, labels=False, title="joint targets")
        else:
            d.text((X2 + MW // 2, Y0 + 408), "waits for the cube position", font=font(24, "light"), fill=DIM, anchor="mm")

        # ---- column 3: the simulated copy on top, then the real arm (phone clip 2 big, overhead camera as a small window)
        if stage >= 3:
            rt = max(0.0, t3) * SPEED
            k = min(n_steps - 1, int(rt * 20))
            sp = k / run.stretch
            simimg = simB.render(interp_qpos(m["qpos"], sp), az=134 + 14 * min(1.0, max(0.0, t3) / T_act), dist=0.52, lookat=(0.17, 0.0, 0.105))
            paste_bgr(img, simimg, (X3, Y0), (PW, PHS), border=ORANGE, label="simulated copy", radius=16)
            pt = min(rt + PHONE_LAG, PH2_LEN - 1.5 / 30)
            py = Y0 + PHS + 30
            paste_bgr(img, phone_crop(ph2.get(int(pt * 30)), PH2_CROP), (X3, py), (PW, PHP), border=CYAN, label="real arm", radius=16)
            paste_bgr(img, crop_real(top.get(int(run.align[k]))), (X3 + 12, py + PHP - IH - 12), (IW, IH), border=CYAN, radius=10)
            phase_row(img, X3 + PW - 12, py + PHP - 12 - 32, [("reach", sp < run.t_close, CYAN), ("grasp", run.t_close <= sp < run.t_lift, ORANGE), ("lift", run.t_lift <= sp < run.t_hold, GREEN), ("hold", sp >= run.t_hold, GREEN)])
            if poster is None and sp >= run.t_hold + 25:
                poster = img.copy()
        else:
            d.rounded_rectangle((X3, Y0, X3 + PW, Y0 + PHS), radius=16, outline=LINE, width=2, fill=PANEL)
            d.rounded_rectangle((X3, Y0 + PHS + 30, X3 + PW, Y0 + PHS + 30 + PHP), radius=16, outline=LINE, width=2, fill=PANEL)

        # ---- arrows between the columns
        ay = Y0 + 408
        arrow(img, (X1 + PW + 14, ay), (X2 - 14, ay), CYAN, t=t, active=stage >= 2, width=4)
        arrow(img, (X2 + MW + 14, ay), (X3 - 14, ay), ORANGE, t=t + 0.3, active=stage >= 3, width=4)
        fr.put(img)
    enc.stdin.close()
    enc.wait()
    if poster is not None:
        poster.save(str(out).replace(".mp4", "_poster.png"))
    return fr.n


if __name__ == "__main__":
    which, runname = sys.argv[1], sys.argv[2]
    run = Run(runname)
    out = ROOT / "outputs" / "showcase" / f"showcase_{which}_{runname.replace('run_', '')}.mp4"
    t0 = time.time()
    n = {"A": cut_A, "B": cut_B, "C": cut_C}[which](run, out)
    print(f"RESULT {out.name}: {n} frames = {n / FPS:.1f} s, {out.stat().st_size / 1e6:.1f} MB, rendered in {time.time() - t0:.0f} s")
