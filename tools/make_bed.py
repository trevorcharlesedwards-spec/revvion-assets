#!/usr/bin/env python3
"""Work Out Wars — original royalty-free audio beds. Offline, numpy/scipy only.

    make_bed.py <out.wav> [seconds] [style] [seed]

Styles: ragga · ragga_chill · ragga_up · dnb · hybrid · auto (default)

The house sound is **ragga jungle / reggae drum & bass** — Trevor's own listening
(Benny Page mashups, D&B reggae bootlegs). Chopped break at 168-180, warm long
sub, and the offbeat skank chord soaked in dub delay, which is the bit that makes
it read as reggae rather than plain D&B.

`ragga_chill` and `ragga_up` are the same genre at different energies, so a bed
can suit what the post actually says: a "you missed a day" post wants chill, a
merch or PB post wants up.

Called with no arguments it picks style and seed freshly at random, so three
posts in one run get three different tracks. Pass `auto "YYYY-MM-DD#slot"` for
the deterministic version, which guarantees three different styles per day.

Nothing here is sampled or licensed — every sound is synthesised from scratch, so
there is no copyright question, no Content ID fingerprint to match, and no music
library to keep in sync.
"""
import sys, hashlib, datetime
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, fftconvolve

SR = 44100


# ---------------------------------------------------------------- helpers
def lp(x, hz, order=2):
    b, a = butter(order, min(hz / (SR / 2), 0.99), btype="low")
    return lfilter(b, a, x)


def hp(x, hz, order=2):
    b, a = butter(order, min(hz / (SR / 2), 0.99), btype="high")
    return lfilter(b, a, x)


def place(buf, sig, at, gain=1.0):
    i = int(at * SR)
    if i >= len(buf) or i < 0:
        return
    e = min(len(buf), i + len(sig))
    buf[i:e] += sig[:e - i] * gain


def adsr(n, a=0.01, d=0.1, s=0.7, r=0.2):
    at, dt, rt = int(a * SR), int(d * SR), int(r * SR)
    st = max(1, n - at - dt - rt)
    env = np.concatenate([np.linspace(0, 1, max(1, at)),
                          np.linspace(1, s, max(1, dt)),
                          np.full(st, s),
                          np.linspace(s, 0, max(1, rt))])
    return env[:n] if len(env) >= n else np.pad(env, (0, n - len(env)))


def reverb(x, seconds=1.4, mix=0.30, rng=None, damp=3800):
    """Cheap hall: convolve with exponentially-decaying filtered noise."""
    n = int(seconds * SR)
    ir = (rng.standard_normal(n) * np.exp(-np.arange(n) / (SR * seconds / 5.5)))
    ir[: int(0.008 * SR)] = 0            # small pre-delay
    ir = lp(ir, damp)
    ir /= np.abs(ir).max() + 1e-9
    wet = fftconvolve(x, ir)[: len(x)]
    wet /= np.abs(wet).max() + 1e-9
    return (1 - mix) * x + mix * wet * np.abs(x).max()


# ---------------------------------------------------------------- voices
def pluck(rng, freq, dur, decay=0.995, bright=0.5):
    """Karplus-Strong — reads as a plucked gut string. The lyre/tagelharpa sound."""
    N = max(2, int(SR / freq))
    buf = rng.standard_normal(N)
    buf = lp(buf, 200 + 6000 * bright)
    n = int(dur * SR)
    out = np.empty(n)
    idx = 0
    for i in range(n):
        out[i] = buf[idx]
        nxt = (idx + 1) % N
        buf[idx] = decay * 0.5 * (buf[idx] + buf[nxt])
        idx = nxt
    return out * adsr(n, 0.001, 0.05, 0.85, 0.25)


def frame_drum(rng, dur=0.55, pitch=58.0, tone=1.0):
    """Big hide drum: pitch-dropping body plus a dry skin slap."""
    n = int(dur * SR)
    x = np.arange(n) / SR
    f = pitch * (1 + 1.7 * np.exp(-x * 30))
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-x * 7.5)
    skin = lp(hp(rng.standard_normal(n), 400), 2600) * np.exp(-x * 34) * 0.55 * tone
    return (body * 1.0 + skin) * 0.9


def horn(freq, dur, rng):
    """Slow low swell — the 'here comes the drop' brass bed."""
    n = int(dur * SR)
    x = np.arange(n) / SR
    saw = sum(np.sin(2 * np.pi * freq * k * x + rng.random()) / k for k in range(1, 9))
    env = np.sin(np.pi * np.linspace(0, 1, n)) ** 1.6
    return lp(saw, 900) * env * 0.30


def drone(freq, n, rng, detune=0.004):
    x = np.arange(n) / SR
    a = np.sin(2 * np.pi * freq * x)
    b = np.sin(2 * np.pi * freq * (1 + detune) * x + 1.1)
    c = np.sin(2 * np.pi * freq * 1.5 * x + 0.4) * 0.55      # the fifth
    swell = 0.75 + 0.25 * np.sin(2 * np.pi * 0.07 * x)
    return lp((a + b + c) / 3.0, 700) * swell


def saw(f, n, phase=0.0):
    x = np.arange(n) / SR
    return 2.0 * ((f * x + phase) % 1.0) - 1.0


# ---------------------------------------------------------------- styles
def build_nordic(dur, rng):
    """Slow, modal, big room. Frame drum, drone fifth, plucked lyre, horn swells."""
    N = int(SR * dur)
    out = np.zeros(N)
    BPM = float(rng.choice([78, 84, 90]))
    beat = 60.0 / BPM
    bar = beat * 4

    root = float(rng.choice([55.00, 58.27, 61.74]))          # A1 / Bb1 / B1
    out += drone(root, N, rng) * 0.30

    # Aeolian degrees, pentatonic-leaning — the folk sound without the cliché
    scale = np.array([0, 2, 3, 5, 7, 8, 10]) / 12.0
    deg = rng.choice([0, 2, 3, 4, 6], size=64)
    octs = rng.choice([4, 4, 8], size=64)

    nb = int(dur / bar) + 2
    for b in range(nb):
        b0 = b * bar
        place(out, frame_drum(rng, 0.6, root * 1.05), b0, 0.95)
        place(out, frame_drum(rng, 0.45, root * 1.05, 0.7), b0 + beat * 2, 0.72)
        if b % 2 == 1:
            place(out, frame_drum(rng, 0.3, root * 1.2, 0.5), b0 + beat * 3.5, 0.5)
        # a light pulse so it never sits still
        for k in (1.5, 2.5, 3.5):
            place(out, frame_drum(rng, 0.16, root * 2.4, 0.35), b0 + beat * k, 0.22)

        # lyre phrase: three notes a bar, never on the downbeat
        for j, off in enumerate((0.5, 1.75, 3.0)):
            i = (b * 3 + j) % len(deg)
            f = root * octs[i] * (2 ** scale[deg[i]])
            place(out, pluck(rng, f, beat * 1.5, 0.9955, 0.42), b0 + beat * off, 0.34)

        if b % 4 == 0:
            place(out, horn(root * 2, bar * 2.0, rng), b0, 0.9)

    out = reverb(out, 1.6, 0.34, rng, 3600)
    return out, BPM


def build_dnb(dur, rng):
    """174 BPM two-step: kick/snare break, Reese bass, 16th rides."""
    N = int(SR * dur)
    out = np.zeros(N)
    BPM = 174.0
    beat = 60.0 / BPM
    bar = beat * 4

    def kick():
        n = int(0.30 * SR); x = np.arange(n) / SR
        f = 132 * np.exp(-x * 34) + 47
        body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-x / 0.055)
        click = hp(rng.standard_normal(n), 1800) * np.exp(-x * 420)
        return body * 0.72 + click * 0.5

    def snare():
        n = int(0.24 * SR); x = np.arange(n) / SR
        noise = hp(rng.standard_normal(n), 1400) * np.exp(-x / 0.045)
        tone = (np.sin(2 * np.pi * 188 * x) + np.sin(2 * np.pi * 331 * x)) * np.exp(-x / 0.028)
        return (noise * 0.95 + tone * 0.45) * 1.25

    def ride():
        n = int(0.07 * SR)
        return hp(rng.standard_normal(n), 7000) * np.exp(-np.arange(n) / (SR * 0.014)) * 0.34

    def reese(note, d):
        n = int(d * SR)
        s = (saw(note, n) + saw(note * 1.0055, n, 0.31) + saw(note * 0.9946, n, 0.67)) / 3.0
        o = np.zeros(n)
        for k in range(10):
            a, z = int(n * k / 10), int(n * (k + 1) / 10)
            cut = 420 + 1780 * (0.5 - 0.5 * np.cos(np.pi * 2 * k / 10))
            o[a:z] = lp(s[a:z], cut)
        return o * adsr(n, 0.008, 0.05, 0.9, 0.05) * 0.34

    base = float(rng.choice([43.65, 46.25, 49.00]))
    roots = [base, base, base * 1.189, base * 1.498, base, base, base * 1.587, base * 1.189]
    nb = int(dur / bar) + 2
    for b in range(nb):
        b0 = b * bar
        r = roots[b % len(roots)]
        place(out, kick(), b0)
        place(out, snare(), b0 + beat * 2)
        if b % 2 == 1:
            place(out, kick(), b0 + beat * 2.75, 0.7)
        if b % 4 == 3:
            place(out, snare(), b0 + beat * 3.5, 0.75)
        for k in range(16):
            place(out, ride(), b0 + k * beat / 4, 1.0 if k % 4 == 2 else 0.5)
        place(out, reese(r * 2, bar * 0.55), b0)
        place(out, reese(r * 2, bar * 0.32), b0 + bar * 0.62)
    out = reverb(out, 0.7, 0.14, rng, 5200)
    return out, BPM


def build_hybrid(dur, rng):
    """Nordic drum and drone under a modern half-time sub. Gym-trailer territory."""
    nordic, _ = build_nordic(dur, rng)
    N = len(nordic)
    out = nordic * 0.82
    BPM = 88.0
    beat = 60.0 / BPM
    bar = beat * 4
    root = 46.25
    nb = int(dur / bar) + 2
    for b in range(nb):
        b0 = b * bar
        n = int(beat * 1.6 * SR)
        x = np.arange(n) / SR
        subv = np.sin(2 * np.pi * root * x) * adsr(n, 0.012, 0.06, 0.8, 0.12) * 0.34
        place(out, subv, b0)
        place(out, subv[: int(len(subv) * 0.6)], b0 + beat * 2.5, 0.8)
    return out, BPM



def build_ragga(dur, rng, mood=None):
    """Ragga jungle / reggae D&B — the Benny Page lane.

    `mood` drives energy so the bed can suit what the post actually says:
      chill  — slower, half-time, sparse, long sub, drowning in delay
      mid    — the middle
      up     — faster, busier break, brighter skank, punchier sub

    Almost every other choice is randomised per render — progression, skank
    rhythm, break pattern, filter colour, whether a melodica line appears — so
    two beds in the same mood still don't sound like the same track.
    """
    if mood is None:
        mood = str(rng.choice(["chill", "mid", "up"]))
    N = int(SR * dur)
    out = np.zeros(N)
    skank_bus = np.zeros(N)

    cfg = {
        "chill": dict(bpm=[166., 168., 170.], ghosts=1, extra=False, bright=0.72,
                      subdur=2.9, hats=0.5, verb=0.30, dly=0.70, skank=1.20, lead=0.55),
        "mid":   dict(bpm=[172., 174.],       ghosts=3, extra=True,  bright=1.0,
                      subdur=2.1, hats=0.8, verb=0.20, dly=0.62, skank=1.45, lead=0.30),
        "up":    dict(bpm=[176., 178., 180.], ghosts=5, extra=True,  bright=1.28,
                      subdur=1.4, hats=1.0, verb=0.13, dly=0.48, skank=1.60, lead=0.15),
    }[mood]

    BPM = float(rng.choice(cfg["bpm"]))
    beat = 60.0 / BPM
    bar = beat * 4

    def kick():
        n = int(0.32 * SR); x = np.arange(n) / SR
        f = 124 * np.exp(-x * 30) + 46
        body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-x / 0.062)
        return body * 0.95 + hp(rng.standard_normal(n), 2000) * np.exp(-x * 500) * 0.35

    def snare(b=1.0, lvl=1.0):
        n = int(0.22 * SR); x = np.arange(n) / SR
        noise = hp(rng.standard_normal(n), 1200 * b * cfg["bright"]) * np.exp(-x / 0.048)
        tone = (np.sin(2 * np.pi * 196 * x) + np.sin(2 * np.pi * 342 * x)) * np.exp(-x / 0.03)
        return (noise * 0.95 + tone * 0.4) * 1.15 * lvl

    def ghost():
        n = int(0.06 * SR); x = np.arange(n) / SR
        return hp(rng.standard_normal(n), 2500) * np.exp(-x / 0.012) * 0.30

    def hat(lvl=1.0):
        n = int(0.05 * SR)
        return hp(rng.standard_normal(n), 8000) * np.exp(-np.arange(n) / (SR * 0.009)) * 0.26 * lvl

    def sub(freq, d):
        n = int(d * SR); x = np.arange(n) / SR
        sig = np.sin(2 * np.pi * freq * x) + 0.13 * np.sin(4 * np.pi * freq * x)
        return sig * adsr(n, 0.014, 0.05, 0.92, 0.09) * 0.30

    def skank(root, d=0.085):
        n = int(d * SR); x = np.arange(n) / SR
        ch = sum(saw(root * r, n, rng.random()) for r in (1.0, 1.1892, 1.4983)) / 3.0
        ch = lp(hp(ch, 420), 3100 * cfg["bright"])
        ch = ch + 0.35 * hp(ch, 1800)
        return ch * np.exp(-x / 0.034) * cfg["skank"]

    def melodica(freq, d):
        n = int(d * SR); x = np.arange(n) / SR
        vib = 1 + 0.006 * np.sin(2 * np.pi * 5.2 * x)
        tone = (np.sin(2 * np.pi * freq * x * vib)
                + 0.34 * np.sin(4 * np.pi * freq * x * vib)
                + 0.16 * np.sin(6 * np.pi * freq * x * vib))
        return lp(tone, 2600) * adsr(n, 0.05, 0.12, 0.7, 0.3) * 0.20

    PROGS = [[1.0, 1.0, .8909, .8909, .8409, .8409, .8909, 1.0],
             [1.0, 1.0, 1.0, .8409, .8909, .8909, 1.0, 1.0],
             [1.0, .9439, .8909, .8409, 1.0, .9439, .8909, .7937]]
    SKANKS = [(0.5, 1.5, 2.5, 3.5), (0.5, 1.5, 2.25, 2.5, 3.5), (0.5, 2.5), (0.75, 1.5, 2.75, 3.5)]
    prog = PROGS[int(rng.integers(len(PROGS)))]
    skank_at = SKANKS[int(rng.integers(len(SKANKS)))]
    base = float(rng.choice([46.249, 48.999, 51.913, 55.000]))
    use_lead = rng.random() < cfg["lead"]
    scale = np.array([0, 3, 5, 7, 10]) / 12.0

    nb = int(dur / bar) + 2
    for b in range(nb):
        b0 = b * bar
        r = base * prog[b % len(prog)]

        place(out, kick(), b0)
        place(out, snare(), b0 + beat * 2)
        if cfg["extra"]:
            place(out, kick(), b0 + beat * 2.5, 0.72)
            if b % 2 == 1:
                place(out, kick(), b0 + beat * 3.25, 0.6)
        if cfg["extra"] and b % 4 == 3:
            place(out, snare(1.15, 0.8), b0 + beat * 3.5)
            place(out, snare(1.3, 0.6), b0 + beat * 3.75)
        gpos = rng.choice(range(16), size=cfg["ghosts"], replace=False)
        for k in range(16):
            t = b0 + k * beat / 4
            if k in gpos:
                place(out, ghost(), t, 0.9)
            place(out, hat(cfg["hats"] * (1.0 if k % 4 == 2 else 0.45)), t)

        place(out, sub(r, beat * cfg["subdur"]), b0)
        if mood != "chill":
            place(out, sub(r, beat * 1.3), b0 + beat * 2.5)

        for k, off in enumerate(skank_at):
            place(skank_bus, skank(r * 4), b0 + beat * off, 1.0 if k % 2 else 0.72)

        if use_lead and b % 4 == 1:
            deg = scale[int(rng.integers(len(scale)))]
            place(out, melodica(r * 8 * (2 ** deg), beat * 2.2), b0 + beat * 1.5, 0.9)

    delayed = np.zeros(N)
    tap = skank_bus.copy()
    for rep in range(4):
        shift = int(beat * 1.5 * SR * (rep + 1))
        if shift >= N:
            break
        tap = lp(tap, 2600 - rep * 400) * cfg["dly"]
        delayed[shift:] += tap[: N - shift]
    out += skank_bus + delayed

    out = reverb(out, 1.0 + cfg["verb"], cfg["verb"], rng, 4200)
    return out, BPM


def build_ragga_chill(dur, rng): return build_ragga(dur, rng, "chill")
def build_ragga_up(dur, rng):    return build_ragga(dur, rng, "up")


STYLES = {"ragga": build_ragga, "ragga_chill": build_ragga_chill,
          "ragga_up": build_ragga_up, "dnb": build_dnb, "hybrid": build_hybrid}
SLOTS = ["app", "audience", "merch"]


def auto_pick(tag=None):
    """Deterministic, but never the same style twice in one day.

    `tag` is "YYYY-MM-DD#slot". The date shuffles the three styles; the slot
    picks one of them. So the three posts on a day are always three different
    styles, and the assignment moves around from day to day.
    """
    # No tag at all: pick freshly at random, so a run that calls this three times
    # gets three different tracks without having to know about slots.
    if not tag:
        r = np.random.default_rng()
        order = sorted(STYLES)
        return order[int(r.integers(len(order)))], int(r.integers(2 ** 31))
    day, _, slot = tag.partition("#")
    order = sorted(STYLES)
    day_rng = np.random.default_rng(
        int(hashlib.sha256(day.encode()).hexdigest()[:12], 16) % (2 ** 31))
    day_rng.shuffle(order)
    i = SLOTS.index(slot) if slot in SLOTS else (
        int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16) % len(order))
    seed = int(hashlib.sha256(tag.encode()).hexdigest()[:12], 16) % (2 ** 31)
    return order[i % len(order)], seed


# ---------------------------------------------------------------- main
def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "bed.wav"
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 30.6
    style = sys.argv[3] if len(sys.argv) > 3 else "auto"
    seed_arg = sys.argv[4] if len(sys.argv) > 4 else None

    if style == "auto":
        style, seed = auto_pick(seed_arg)
    else:
        seed = int(hashlib.sha256((seed_arg or style).encode()).hexdigest()[:8], 16)
    if style not in STYLES:
        raise SystemExit(f"unknown style {style!r}; choose from {sorted(STYLES)} or auto")

    rng = np.random.default_rng(seed)
    out, bpm = STYLES[style](dur, rng)
    out = out[: int(SR * dur)]

    # master: clear the sub a phone cannot reproduce, add presence
    out = out - out.mean()
    out = hp(out, 48, order=2)
    out = out + 0.22 * hp(out, 2500)

    # Match on RMS, not peak. Peak-normalising made the transient-heavy D&B bed
    # come out at a third the loudness of the sustained Nordic one.
    rms = np.sqrt((out ** 2).mean()) + 1e-9
    out = np.tanh(out * (0.15 / rms) * 1.10)
    out = out / (np.abs(out).max() + 1e-9) * 0.82

    fi, fo = int(0.4 * SR), int(1.2 * SR)
    out[:fi] *= np.linspace(0, 1, fi)
    out[-fo:] *= np.linspace(1, 0, fo)

    stereo = np.stack([out, np.roll(out, 80)], axis=1)
    wavfile.write(out_path, SR, (stereo * 32767).astype(np.int16))
    print(f"{out_path}  style={style}  seed={seed}  {bpm:.0f} BPM  "
          f"{dur:.1f}s  rms={np.sqrt((out ** 2).mean()):.4f}")


if __name__ == "__main__":
    main()
