#!/usr/bin/env python3
"""Work Out Wars — motion post renderer v2.
1080x1920, 30fps, 15s, seamless loop. Raw RGB piped to ffmpeg; no frame files.

WHAT CHANGED IN v2 (2026-09-07) AND WHY — see project doc 62.

  * SAFE BOX. Critical text now only ever lands in x 72..880, y 260..1400.
    TikTok and Instagram draw their OWN caption, sound name, username and the
    like/comment/share column on top of the video. v1 put the closing question
    at y=1560, the wordmark at y=1600 and the handles at y=1760/1800 — every
    one of those sat underneath the platform's chrome. Trevor reported the
    captions as "a bit small down the bottom"; they were also partly hidden.
    Convention across every source: keep out of the top ~14%, the bottom
    ~25%, and the right ~200px. This file enforces that with assertions.

  * TYPE UP. Sublines 38 -> 62px. Closing question 46 -> 92px.
    BBC Subtitle Guidelines set a 4.5%-of-height floor for 9:16 video = 86px
    on a 1920 canvas. v1's sublines were 38px = 1.98% — under half the floor.
    62px is 3.2%: still below BBC, but it is what fits two lines inside the
    safe box, and it is a 63% increase on v1. The QUESTION clears the floor.

  * 15 SECONDS, LOOPING. CreatorHouse (50 Reels, length the only variable):
    median reach 15s = 11,800 vs 30s = 9,200; completion 78% vs 56%.
    The background is periodic over DUR and the tail cross-dissolves back to
    frame 0, so the loop has no visible seam and rewatch is manufactured.

  * THREE LAYOUTS. LIST / STATEMENT / SPLIT. Every post to date used one
    template, so the profile grid looked like one post published forty times.

  * THE QUESTION IS ON SCREEN AND BIG. Metricool (2.3M posts): a question
    earns +26% comments. v1 buried it in 46px type under the platform UI, so
    that lever was never actually pulled.

USAGE
  Set LAYOUT and the content block below, then:
    python3 render_post.py out.mp4
  A daily run may change: LAYOUT, HOOK, ITEMS, OUTRO, QUESTION. Nothing else.
"""
import sys, math, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 30
DUR = 15.0
NF = int(DUR * FPS)
TAIL = 12                      # frames cross-dissolved back to frame 0

# ---------------- the safe box ----------------
# Everything that must be READ lives inside this rectangle. Nothing else.
SX0, SX1 = 72, 880             # right column: like/comment/share icons
SY0, SY1 = 260, 1400           # top 13.5%, bottom 27%
RX = SX1 - 4                   # fills stop just short, so antialiasing
                               # cannot bleed a pixel under the buttons
SW = SX1 - SX0                 # 808

BG      = (13, 13, 13)         # #0D0D0D
IVORY   = (242, 240, 234)
GREY    = (168, 168, 160)
GOLD    = (139, 105, 20)       # #8B6914
GOLD_HI = (198, 152, 33)
GREEN   = (27, 77, 62)         # #1B4D3E
ORANGE  = (217, 112, 26)       # #D9701A — approved for headline text

FD = "/usr/share/fonts/truetype/dejavu/"
CB = "DejaVuSansCondensed-Bold.ttf"
SB = "DejaVuSans-Bold.ttf"
SR = "DejaVuSans.ttf"

_FC = {}
def FF(name, size):
    k = (name, size)
    if k not in _FC:
        _FC[k] = ImageFont.truetype(FD + name, size)
    return _FC[k]

def fit(d, text, name, max_size, max_w, min_size=30):
    s = max_size
    while s > min_size:
        f = FF(name, s)
        if d.textlength(text, font=f) <= max_w:
            return f
        s -= 2
    return FF(name, min_size)

def wrap(d, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if d.textlength(t, font=font) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur); cur = w_
    if cur:
        lines.append(cur)
    return lines

def ease_out(t):    return 1 - (1 - t) ** 3
def ease_in_out(t): return 3 * t * t - 2 * t * t * t
def cl(x):          return 0.0 if x < 0 else (1.0 if x > 1 else x)
def mix(a, b, f):   return tuple(int(a[i] + (b[i] - a[i]) * f) for i in range(3))

# ---------------- background: periodic over DUR, so the loop is seamless ----------------
LW, LH = 108, 192
yy, xx = np.mgrid[0:LH, 0:LW].astype(np.float32)

def background(t, tint=GOLD_HI):
    # Normalise the tint's brightness so a green-led layout is not three times
    # darker than a gold-led one — raw #1B4D3E glows at ~26/255 and reads black.
    mx = max(tint) or 1
    tint = tuple(c * 200.0 / mx for c in tint)
    p = 2 * math.pi * t / DUR                       # exactly one cycle per loop
    cx1 = LW * (0.5 + 0.34 * math.sin(p))
    cy1 = LH * (0.30 + 0.16 * math.cos(p))
    cx2 = LW * (0.5 + 0.30 * math.cos(p + 1.7))
    cy2 = LH * (0.74 + 0.14 * math.sin(p + 0.9))
    g1 = np.exp(-(((xx - cx1) ** 2 + (yy - cy1) ** 2) / (2 * (LW * 0.52) ** 2)))
    g2 = np.exp(-(((xx - cx2) ** 2 + (yy - cy2) ** 2) / (2 * (LW * 0.46) ** 2)))
    img = np.zeros((LH, LW, 3), np.float32)
    img[..., 0] = 13 + g1 * (tint[0] * 0.17) + g2 * 8
    img[..., 1] = 13 + g1 * (tint[1] * 0.17) + g2 * 20
    img[..., 2] = 13 + g1 * (tint[2] * 0.17) + g2 * 16
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)

# ==================== CONTENT — a daily run edits only this block ====================

LAYOUT = "list"          # "list" | "statement" | "split"

HOOK = ["YOU MISS A", "TRAINING DAY.", "MOST TRACKERS", "RESET YOU TO ZERO."]

ITEMS = [
    ("MISS A DAY",    None, "", "Six days trained. Day seven missed."),
    ("GRACE DAY",     None, "", "Beast Mode freezes the streak instead of wiping it."),
    ("RETRAIN TWICE", None, "", "Train the missed day, then train the next one."),
    ("STREAK INTACT", None, "", "Same streak. Still standing."),
]

OUTRO = ["MISS A DAY.", "DON'T LOSE", "THE STREAK."]

QUESTION = "WHAT'S THE LONGEST STREAK YOU'VE LOST?"

# ====================================================================================

VALUES = [it[1] for it in ITEMS]
MODE = "metrics" if all(isinstance(v, (int, float)) for v in VALUES) else "story"
VMAX = max([v for v in VALUES if isinstance(v, (int, float))], default=0)


def check_copy():
    """Fail loudly rather than render something that misleads or cannot be read."""
    assert LAYOUT in ("list", "statement", "split"), f"unknown layout {LAYOUT}"
    assert len(ITEMS) == 4, f"expected 4 items, got {len(ITEMS)}"
    for label, val, unit, sub in ITEMS:
        assert label.strip(), "empty label"
        assert sub.strip(), f"{label}: every row needs a subline"
        if MODE == "story":
            assert val is None, (
                f"{label}: mixed row types. Either every row is a real measurement "
                f"or none are — a number on one beat and not another reads as broken.")
    if MODE == "metrics":
        assert VMAX > 0, "metrics mode with no positive value — use None values instead"
    q = QUESTION.strip()
    assert q, "QUESTION is required — it is the +26%-comments lever"
    assert q.endswith("?"), f"QUESTION must actually be a question: {q!r}"
    assert len(q) <= 46, f"QUESTION too long to set large ({len(q)} chars, max 46)"
    return MODE

# ---------------- shared pieces ----------------

def kicker(d, text, col=GOLD_HI):
    d.text((SX0, SY0 - 78), text, font=FF(CB, 40), fill=col)


def hook_block(d, t, t_end):
    """The first three seconds. Big type, top-anchored inside the safe box.

    Line 1 is FULLY drawn at t=0 on purpose. Frame 0 is the thumbnail, it is the
    first thing a scroller sees, and it is also the frame the loop returns to —
    v2's first cut opened on a near-black frame (0.26% ink) and looped back into
    it. A blank opening frame is a wasted hook."""
    for li, line in enumerate(HOOK):
        lt = cl((t + 0.40 - li * 0.28) / 0.40)
        if lt <= 0:
            continue
        e = ease_out(lt)
        y = SY0 + 120 + li * 132 + (1 - e) * 44
        base = GOLD_HI if li >= 2 else IVORY
        d.text((SX0, y), line, font=fit(d, line, CB, 108, SW), fill=mix(BG, base, e))


def question_card(d, t, t0, pulse=True):
    """The question, set large, inside the safe box. This is the comments lever."""
    f = fit(d, QUESTION, SB, 92, SW, min_size=54)
    lines = wrap(d, QUESTION, f, SW)
    e = ease_out(cl((t - t0) / 0.45))
    col = GOLD_HI if (pulse and int(t * 2) % 2 == 0) else IVORY
    y = SY1 - 40 - len(lines) * (f.size + 14)
    d.rounded_rectangle([SX0 - 26, y - 34, RX, SY1 - 6], 18,
                        fill=mix(BG, (26, 24, 20), e))
    for i, ln in enumerate(lines):
        d.text((SX0, y + i * (f.size + 14) + (1 - e) * 26), ln, font=f,
               fill=mix(BG, col, e))

# ---------------- LAYOUT: list ----------------
ROW_H = 232

def draw_row(d, i, x_off, alpha, y, prog, active):
    label, target, unit, sub = ITEMS[i]
    A = lambda c: mix(BG, c, alpha)
    L, R = SX0 + x_off, RX

    d.rounded_rectangle([L, y, L + 76, y + 76], 12, fill=A(GOLD if active else GREEN))
    d.text((L + 38, y + 36), str(i + 1), font=FF(CB, 52), fill=A(IVORY), anchor="mm")

    if MODE == "metrics":
        widest = f"{target:,}{unit}" if target >= 1000 else f"{target}{unit}"
        shown = f"{int(round(target*prog)):,}{unit}" if target >= 1000 else f"{int(round(target*prog))}{unit}"
        f_n = fit(d, widest, CB, 96, 340)
        nw = d.textlength(widest, font=f_n)
        f_l = fit(d, label, CB, 72, (R - nw - 34) - (L + 108))
        d.text((L + 108, y + 38), label, font=f_l, fill=A(IVORY if active else GREY), anchor="lm")
        d.text((R, y + 38), shown, font=f_n, fill=A(GOLD_HI if active else GOLD), anchor="rm")
        by = y + 100
        d.rounded_rectangle([L, by, R, by + 16], 8, fill=A((32, 32, 32)))
        share = (target / VMAX) if VMAX else 0.0
        filled = (R - L) * share * prog
        if filled >= 3:
            d.rounded_rectangle([L, by, L + filled, by + 16], 8,
                                fill=A(GOLD_HI if active else GREEN))
        sub_y = by + 34
    else:
        f_l = fit(d, label, CB, 72, R - (L + 108))
        d.text((L + 108, y + 38), label, font=f_l, fill=A(IVORY if active else GREY), anchor="lm")
        d.rounded_rectangle([L, y + 92, L + 120 + (R - L - 120) * prog, y + 97], 3,
                            fill=A(GOLD_HI if active else GREEN))
        sub_y = y + 116

    if active and alpha > 0.85:
        f_s = FF(SR, 62)
        for j, ln in enumerate(wrap(d, sub, f_s, R - L)[:2]):
            d.text((L, sub_y + j * 72), ln, font=f_s, fill=A(GREY))


def layout_list(d, t, T_HOOK, T_BODY, seg):
    if t < T_HOOK:
        kicker(d, "WORK OUT WARS")
        hook_block(d, t, T_HOOK)
        return
    if t < T_BODY:
        idx = min(3, int((t - T_HOOK) // seg))
        local = (t - T_HOOK) - idx * seg
        kicker(d, "WORK OUT WARS")
        shift = (1 - ease_in_out(cl(local / 0.42))) * ROW_H if idx > 0 else 0
        for i in range(idx + 1):
            y = (SY1 - 300) + (i - idx) * ROW_H + shift
            active = (i == idx)
            if active:
                a = ease_out(cl(local / 0.38)); xo = (1 - a) * 90
                prog = ease_out(cl((local - 0.25) / 1.15))
            else:
                a, xo, prog = 1.0, 0.0, 1.0
            if SY0 - 300 < y < SY1:
                draw_row(d, i, xo, a, int(y), prog, active)
        d.text((RX, SY0 - 78), f"{idx+1}/4", font=FF(CB, 40), fill=GOLD, anchor="ra")
        return
    kicker(d, "WORK OUT WARS")
    for li, line in enumerate(OUTRO):
        e = ease_out(cl((t - T_BODY - 0.1 - li * 0.22) / 0.42))
        if e <= 0:
            continue
        col = GOLD_HI if li == len(OUTRO) - 1 else IVORY
        d.text((SX0, SY0 + 110 + li * 128 + (1 - e) * 30), line,
               font=fit(d, line, CB, 108, SW), fill=mix(BG, col, e))
    question_card(d, t, T_BODY + 0.5)


# ---------------- LAYOUT: statement ----------------
def layout_statement(d, t, T_HOOK, T_BODY, seg):
    """One idea on screen at a time, set as large as it will go. No rows, no chrome.
    Produces a completely different thumbnail from the list layout."""
    if t < T_HOOK:
        kicker(d, "WORK OUT WARS", ORANGE)
        hook_block(d, t, T_HOOK)
        return
    if t < T_BODY:
        idx = min(3, int((t - T_HOOK) // seg))
        local = (t - T_HOOK) - idx * seg
        label, _, _, sub = ITEMS[idx]
        kicker(d, f"{idx+1} / 4", ORANGE)
        e = ease_out(cl(local / 0.40))
        out = ease_out(cl((local - (seg - 0.32)) / 0.32))
        a = e * (1 - out)
        f_l = fit(d, label, CB, 132, SW, min_size=64)
        d.text((SX0, SY0 + 90 + (1 - e) * 40), label, font=f_l,
               fill=mix(BG, GOLD_HI if idx % 2 else IVORY, a))
        f_s = FF(SR, 68)
        for j, ln in enumerate(wrap(d, sub, f_s, SW)[:4]):
            de = ease_out(cl((local - 0.22 - j * 0.09) / 0.40)) * (1 - out)
            d.text((SX0, SY0 + 300 + j * 84 + (1 - de) * 22), ln, font=f_s,
                   fill=mix(BG, IVORY, de))
        bw = SW * ((idx + ease_out(cl(local / seg))) / 4.0)
        d.rounded_rectangle([SX0, SY1 - 18, SX0 + bw, SY1 - 6], 6, fill=ORANGE)
        return
    for li, line in enumerate(OUTRO):
        e = ease_out(cl((t - T_BODY - 0.1 - li * 0.22) / 0.42))
        if e <= 0:
            continue
        col = ORANGE if li == len(OUTRO) - 1 else IVORY
        d.text((SX0, SY0 + 60 + li * 140 + (1 - e) * 30), line,
               font=fit(d, line, CB, 124, SW), fill=mix(BG, col, e))
    question_card(d, t, T_BODY + 0.5)


# ---------------- LAYOUT: split ----------------
def layout_split(d, t, T_HOOK, T_BODY, seg):
    """Two stacked panels, items 1-2 above and 3-4 below, divided by a hard rule.
    Reads as a comparison — a third distinct shape in the grid."""
    if t < T_HOOK:
        kicker(d, "WORK OUT WARS", GREEN if False else GOLD_HI)
        hook_block(d, t, T_HOOK)
        return
    if t < T_BODY:
        half = (T_BODY - T_HOOK) / 2.0
        panel = 0 if (t - T_HOOK) < half else 1
        local = (t - T_HOOK) - panel * half
        mid = (SY0 + SY1) // 2
        kicker(d, "WORK OUT WARS")
        for p in (0, 1):
            y0 = (SY0 - 20) if p == 0 else (mid + 16)
            y1 = (mid - 16) if p == 0 else (SY1)
            on = (p == panel)
            d.rounded_rectangle([SX0 - 26, y0, RX, y1], 20,
                                fill=(34, 33, 30) if on else (19, 19, 18))
            if on:                       # gold edge marks the live panel
                d.rounded_rectangle([SX0 - 26, y0, SX0 - 14, y1], 6, fill=GOLD_HI)
        d.rounded_rectangle([SX0 - 26, mid - 7, RX, mid + 7], 7, fill=GOLD)
        for p in (0, 1):
            for k in (0, 1):
                i = p * 2 + k
                label, _, _, sub = ITEMS[i]
                top = (SY0 + 24) if p == 0 else (mid + 56)
                y = top + k * 268
                if p > panel:
                    continue
                lt = local if p == panel else half
                e = ease_out(cl((lt - k * (half / 2)) / 0.40))
                if e <= 0:
                    continue
                on = (p == panel)
                f_l = fit(d, label, CB, 76, SW - 20)
                d.text((SX0, y + (1 - e) * 24), label, font=f_l,
                       fill=mix(BG, GOLD_HI if on else GREY, e))
                f_s = FF(SR, 58)
                for j, ln in enumerate(wrap(d, sub, f_s, SW - 20)[:2]):
                    d.text((SX0, y + 92 + j * 68), ln, font=f_s,
                           fill=mix(BG, IVORY if on else (96, 96, 92), e))
        return
    for li, line in enumerate(OUTRO):
        e = ease_out(cl((t - T_BODY - 0.1 - li * 0.22) / 0.42))
        if e <= 0:
            continue
        col = GOLD_HI if li == len(OUTRO) - 1 else IVORY
        d.text((SX0, SY0 + 110 + li * 128 + (1 - e) * 30), line,
               font=fit(d, line, CB, 108, SW), fill=mix(BG, col, e))
    question_card(d, t, T_BODY + 0.5)


# ---------------- timeline ----------------
T_HOOK = 3.2
T_BODY = 11.4
SEG = (T_BODY - T_HOOK) / 4.0
TINT = {"list": GOLD_HI, "statement": ORANGE, "split": GREEN}[LAYOUT]
DRAW = {"list": layout_list, "statement": layout_statement, "split": layout_split}[LAYOUT]


# Alpha mask applied to every content layer. Outside the safe box the content
# simply does not exist — a layout cannot accidentally animate a row down into
# the platform's caption bar, because there is nothing there to draw into.
# The last 70px at top and bottom ramp instead of cutting, so a row sliding in
# fades at the edge rather than being sliced through the middle of a glyph.
def _mask():
    m = np.zeros((H, W), np.float32)
    m[SY0 - 90:SY1 + 1, :SX1] = 1.0
    ramp = np.linspace(0, 1, 70, dtype=np.float32)
    m[SY1 - 69:SY1 + 1, :SX1] *= ramp[::-1][:, None]
    m[SY0 - 90:SY0 - 20, :SX1] *= ramp[:, None]
    m[:, SX1 - 24:SX1] *= np.linspace(1, 0, 24, dtype=np.float32)[None, :]
    return m

MASK = _mask()


def frame(n):
    t = n / FPS
    img = background(t, TINT).convert("RGBA")
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    DRAW(ImageDraw.Draw(layer), t, T_HOOK, T_BODY, SEG)
    a = np.asarray(layer).astype(np.float32)
    a[..., 3] *= MASK
    layer = Image.fromarray(a.astype(np.uint8), "RGBA")
    return Image.alpha_composite(img, layer).convert("RGB")


def assert_in_safe_box(im, n):
    """No lit pixel outside the box the viewer can actually see.

    This is the gate v1 never had. Every check in v1 was numerical — ink
    coverage, byte size, ffprobe — and not one of them could tell that the
    closing question was sitting underneath TikTok's own caption. Text is
    drawn at 150-242 brightness; the background glow peaks around 47, so a
    threshold of 120 separates them cleanly.
    """
    a = np.asarray(im).max(axis=2)
    lit = a > 120
    bad_r = int(lit[:, SX1:].sum())
    bad_t = int(lit[:SY0 - 90, :].sum())
    bad_b = int(lit[SY1 + 4:, :].sum())
    tot = bad_r + bad_t + bad_b
    assert tot == 0, (
        f"frame {n}: {tot} lit pixels outside the safe box "
        f"(right {bad_r}, top {bad_t}, bottom {bad_b}). "
        f"That content is underneath the platform's own UI and cannot be read.")


def main(out):
    mode = check_copy()
    print(f"layout: {LAYOUT}   mode: {mode}   {DUR}s   {NF} frames")
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    first = frame(0)
    inks = []
    for n in range(NF):
        im = frame(n) if n else first
        if n >= NF - TAIL:                      # dissolve back to frame 0: seamless loop
            im = Image.blend(im, first, (n - (NF - TAIL) + 1) / float(TAIL))
        if n % 15 == 0:
            assert_in_safe_box(im, n)
        if n % 45 == 0:
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
