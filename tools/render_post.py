#!/usr/bin/env python3
"""Work Out Wars — motion post renderer.
1080x1920, 30fps, ~30s. Raw RGB piped to ffmpeg (no frame files on disk).
Every frame differs from the last: drifting glow, building list, counting numbers.
"""
import sys, math, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 30
DUR = 30.0
NF = int(DUR * FPS)

BG    = (13, 13, 13)
IVORY = (242, 240, 234)
GREY  = (154, 154, 147)
GOLD  = (139, 105, 20)
GOLD_HI = (198, 152, 33)
GREEN = (27, 77, 62)

FD = "/usr/share/fonts/truetype/dejavu/"
def F(name, size): return ImageFont.truetype(FD + name, size)
CB = "DejaVuSansCondensed-Bold.ttf"
SB = "DejaVuSans-Bold.ttf"
SR = "DejaVuSans.ttf"

f_hook   = F(CB, 104)
f_kick   = F(CB, 34)
f_label  = F(CB, 58)
f_num    = F(CB, 92)
f_unit   = F(CB, 40)
f_sub    = F(SR, 38)
f_mark   = F(CB, 44)
f_hand   = F(SR, 30)
f_cap    = F(SB, 46)


_FCACHE = {}
def FF(name, size):
    k = (name, size)
    if k not in _FCACHE:
        _FCACHE[k] = ImageFont.truetype(FD + name, size)
    return _FCACHE[k]

def fit(d, text, name, max_size, max_w, min_size=22):
    """Largest size of `name` at which `text` fits inside max_w."""
    s = max_size
    while s > min_size:
        f = FF(name, s)
        if d.textlength(text, font=f) <= max_w:
            return f
        s -= 2
    return FF(name, min_size)

def ease_out(t):   return 1 - (1 - t) ** 3
def ease_in_out(t): return 3*t*t - 2*t*t*t
def clamp01(x):    return 0.0 if x < 0 else (1.0 if x > 1 else x)

# ---------- background: drifting gold glow over near-black ----------
LW, LH = 108, 192
yy, xx = np.mgrid[0:LH, 0:LW].astype(np.float32)

def background(t):
    # two slow-moving soft blobs
    cx1 = LW * (0.5 + 0.34 * math.sin(t * 0.42))
    cy1 = LH * (0.30 + 0.16 * math.cos(t * 0.31))
    cx2 = LW * (0.5 + 0.30 * math.cos(t * 0.27 + 1.7))
    cy2 = LH * (0.74 + 0.14 * math.sin(t * 0.36 + 0.9))
    d1 = ((xx - cx1) ** 2 + (yy - cy1) ** 2) / (2 * (LW * 0.52) ** 2)
    d2 = ((xx - cx2) ** 2 + (yy - cy2) ** 2) / (2 * (LW * 0.46) ** 2)
    g1 = np.exp(-d1); g2 = np.exp(-d2)
    img = np.zeros((LH, LW, 3), np.float32)
    img[..., 0] = 13 + g1 * 34 + g2 * 8
    img[..., 1] = 13 + g1 * 25 + g2 * 20
    img[..., 2] = 13 + g1 * 6  + g2 * 16
    small = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    return small.resize((W, H), Image.BICUBIC)

# ---------- content ----------
HOOK = ["YOU'RE NOT", "PLATEAUING.", "YOU'RE COUNTING", "THE WRONG THING."]

ITEMS = [
    ("TOTAL VOLUME",     12480, " KG",  "Sets x reps x load. The only number that says you did more."),
    ("SETS TO FAILURE",      9, "",     "Not sets done. Sets that actually cost you something."),
    ("REST DISCIPLINE",     94, "%",    "Rests you held instead of scrolling through."),
    ("DAY STREAK",          31, "",     "Turning up beats every programme you never finished."),
]

# timeline (seconds)
T_HOOK_END = 4.4
SEG = 5.6
T_ITEM0 = T_HOOK_END
T_END   = T_ITEM0 + SEG * 4   # 26.8
# outro runs T_END -> DUR

ROW_H = 215
Y_ACTIVE = 1080


def fmt(n, unit):
    return f"{n:,}{unit}" if n >= 1000 else f"{n}{unit}"


def draw_row(d, i, x_off, alpha, y, prog, active):
    """One list row: rank chip, label, growing bar, counting number. Never overflows."""
    label, target, unit, sub = ITEMS[i]
    A = lambda c: tuple(int(BG[k] + (c[k] - BG[k]) * alpha) for k in range(3))

    L = 92 + x_off
    R = W - 92

    # number first — it owns the right edge, so the label gets what is left
    shown = fmt(int(round(target * prog)), unit)
    widest = fmt(target, unit)
    f_n = fit(d, widest, CB, 92, 400)
    num_w = d.textlength(widest, font=f_n)

    d.rounded_rectangle([L, y, L + 72, y + 72], 10, fill=A(GOLD if active else GREEN))
    d.text((L + 36, y + 34), str(i + 1), font=FF(CB, 50), fill=A(IVORY), anchor="mm")

    f_l = fit(d, label, CB, 58, (R - num_w - 40) - (L + 104))
    d.text((L + 104, y + 36), label, font=f_l,
           fill=A(IVORY if active else GREY), anchor="lm")
    d.text((R, y + 36), shown, font=f_n,
           fill=A(GOLD_HI if active else GOLD), anchor="rm")

    by = y + 96
    d.rounded_rectangle([L, by, R, by + 14], 7, fill=A((30, 30, 30)))
    if prog > 0:
        d.rounded_rectangle([L, by, L + max(14, (R - L) * prog), by + 14], 7,
                            fill=A(GOLD_HI if active else GREEN))

    if active and alpha > 0.9:
        f_s = fit(d, sub, SR, 38, R - L)
        d.text((L, by + 42), sub, font=f_s, fill=A(GREY), anchor="lt")


def frame(n):
    t = n / FPS
    img = background(t)
    d = ImageDraw.Draw(img)

    # persistent top progress bar — guarantees motion in every single frame
    d.rectangle([0, 0, W, 9], fill=(28, 28, 28))
    d.rectangle([0, 0, W * (t / DUR), 9], fill=GOLD_HI)

    # ---- HOOK ----
    if t < T_HOOK_END:
        d.text((92, 300), "WORK OUT WARS", font=f_kick, fill=GOLD)
        for li, line in enumerate(HOOK):
            lt = clamp01((t - 0.25 - li * 0.42) / 0.5)
            if lt <= 0:
                continue
            e = ease_out(lt)
            yb = 470 + li * 128 + (1 - e) * 46
            col = tuple(int(BG[k] + ((GOLD_HI if li >= 2 else IVORY)[k] - BG[k]) * e)
                        for k in range(3))
            d.text((92, yb), line, font=fit(d, line, CB, 104, W - 184), fill=col)
        # exit fade handled by next branch overlap
        if t > T_HOOK_END - 0.5:
            fade = (t - (T_HOOK_END - 0.5)) / 0.5
            ov = Image.new("RGB", (W, H), BG)
            img = Image.blend(img, ov, fade * 0.92)
        return img

    # ---- LIST BUILD ----
    if t < T_END:
        idx = min(3, int((t - T_ITEM0) // SEG))
        local = (t - T_ITEM0) - idx * SEG

        d.text((92, 300), "FOUR NUMBERS THAT ACTUALLY MOVE", font=f_kick, fill=GOLD)

        # rows slide up as the list grows
        shift = (1 - ease_in_out(clamp01(local / 0.55))) * ROW_H if idx > 0 else 0

        for i in range(idx + 1):
            y = Y_ACTIVE + (i - idx) * ROW_H + shift
            active = (i == idx)
            if active:
                a = ease_out(clamp01(local / 0.45))
                xo = (1 - a) * 110
                prog = ease_out(clamp01((local - 0.35) / 2.2))
            else:
                a = 1.0; xo = 0.0; prog = 1.0
            if -260 < y < H:
                draw_row(d, i, xo, a, int(y), prog, active)
        d.text((92, 1600), "WORK OUT WARS", font=f_mark, fill=(64, 64, 61))
        d.text((W - 92, 1610), f"{idx + 1} / 4", font=f_mark, fill=GOLD, anchor="ra")
        return img

    # ---- OUTRO ----
    lt = clamp01((t - T_END) / 0.6)
    e = ease_out(lt)
    d.text((92, 520), "WORK OUT WARS", font=f_kick, fill=GOLD)
    lines = [("EVERY SET EARNS XP.", IVORY), ("EVERY WEEK RANKS YOU", IVORY),
             ("AGAINST YOUR RIVALS.", GOLD_HI)]
    for li, (line, c) in enumerate(lines):
        s = ease_out(clamp01((t - T_END - 0.15 - li * 0.28) / 0.5))
        if s <= 0:
            continue
        col = tuple(int(BG[k] + (c[k] - BG[k]) * s) for k in range(3))
        d.text((92, 640 + li * 122 + (1 - s) * 34), line,
               font=fit(d, line, CB, 104, W - 184), fill=col)

    # pulsing rungs
    for i in range(4):
        p = 0.55 + 0.45 * math.sin(t * 3.0 + i * 0.6)
        bw = 300 + i * 110
        col = tuple(int(GREEN[k] + (GOLD_HI[k] - GREEN[k]) * p) for k in range(3))
        d.rounded_rectangle([92, 1180 + i * 52, 92 + bw, 1180 + i * 52 + 22], 6, fill=col)

    cap = "WHICH ONE ARE YOU ACTUALLY TRACKING?"
    d.text((92, 1560), cap, font=fit(d, cap, SB, 46, W - 184),
           fill=IVORY if int(t * 2) % 2 == 0 else GOLD_HI)
    d.text((92, 1760), "WORK OUT WARS", font=f_mark, fill=IVORY)
    d.text((W - 92, 1760), "@revvion.uk", font=f_hand, fill=GREY, anchor="rt")
    d.text((W - 92, 1800), "@revvion_uk", font=f_hand, fill=GREY, anchor="rt")
    return img


def main(out):
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-an",
           "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    inks = []
    for n in range(NF):
        im = frame(n)
        if n % 90 == 0:
            a = np.asarray(im)
            inks.append((n, float((a.max(axis=2) > 45).mean() * 100)))
        p.stdin.write(im.tobytes())
    p.stdin.close()
    rc = p.wait()
    print("ffmpeg rc", rc)
    for n, v in inks:
        print(f"  frame {n:4d}  ink {v:5.2f}%")
    assert all(v > 2.0 for _, v in inks), "BLANK FRAMES DETECTED"


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/post_motion.mp4")
