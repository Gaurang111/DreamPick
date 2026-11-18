"""Generates a labeled slideshow GIF from the sparse train_openl TensorBoard
snapshots (truth / model reconstruction-imagination / error - see
WorldModel.video_pred() in models.py and train.py's video_pred_every).

These are logged infrequently by design (video_pred_every, not every step -
we have joint_pos_trace.bin/generate_joint_pos_timelapse_video.py for dense
progression instead), so this is a handful of held frames, not a smooth
timelapse. Each native 64px-per-cell image is upscaled + lightly sharpened
(there's no real extra detail to recover, just presented cleanly at a larger
size) and labeled per row - Truth / Imagination / Error - plus a step
counter, then combined into a looping GIF.

Usage (from src/dreamer/):
    python generate_train_openl_slideshow.py --logdir ../../outputs/dreamer_logdir
"""

import argparse
import io
import pathlib

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from tensorboard.backend.event_processing import event_accumulator

ROW_LABELS = ["Truth", "Dream", "Error"]


def draw_label(draw, pos, text, font):
    x, y = pos
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True, help="Same --logdir the training run used.")
    parser.add_argument("--tag", default="train_openl", choices=["train_openl", "eval_openl"])
    parser.add_argument("--scale", type=int, default=6, help="Upscale factor (native cell is 64x64 per example).")
    parser.add_argument("--seconds_per_frame", type=float, default=2.0)
    parser.add_argument("--out", default=None, help="Output path (default: <logdir>/<tag>_slideshow.gif)")
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir).expanduser()
    ea = event_accumulator.EventAccumulator(str(logdir), size_guidance={event_accumulator.IMAGES: 0})
    ea.Reload()
    events = ea.Images(args.tag)
    if not events:
        raise RuntimeError(f"No '{args.tag}' events found in {logdir}.")
    print(f"Found {len(events)} '{args.tag}' snapshots at steps: {[e.step for e in events]}")

    step_font = ImageFont.load_default(size=32)
    row_font = ImageFont.load_default(size=26)

    frames = []
    for event in events:
        img = Image.open(io.BytesIO(event.encoded_image_string)).convert("RGB")
        native_w, native_h = img.size
        row_h = native_h // 3

        big = img.resize((native_w * args.scale, native_h * args.scale), Image.LANCZOS)
        big = big.filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=2))
        draw = ImageDraw.Draw(big)

        draw_label(draw, (12, 8), f"step {event.step:,}", step_font)
        for i, label in enumerate(ROW_LABELS):
            y = i * row_h * args.scale + (step_font.size + 16 if i == 0 else 10)
            draw_label(draw, (12, y), label, row_font)

        frames.append(big.convert("P", palette=Image.ADAPTIVE, colors=256))

    out_path = pathlib.Path(args.out) if args.out else logdir / f"{args.tag}_slideshow.gif"
    # Saved directly via PIL rather than imageio.mimsave: the latter silently
    # dropped the duration kwarg (every frame came out with duration=None in
    # the saved file, so viewers fell back to their own default ~100ms/frame
    # regardless of what was requested). PIL's own GIF writer takes duration
    # in milliseconds and actually honors it.
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(args.seconds_per_frame * 1000),
        loop=0,
    )
    print(f"saved {out_path} ({len(frames)} frames, {args.seconds_per_frame}s each, {frames[0].size[0]}x{frames[0].size[1]})")


if __name__ == "__main__":
    main()
