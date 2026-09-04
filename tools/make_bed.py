#!/usr/bin/env python3
"""Work Out Wars — original royalty-free audio bed. Offline, numpy only.
174 BPM drum & bass: two-step break, Reese bass, sub, filtered rides.
Trevor's set direction is D&B / jungle; the earlier 100 BPM version read as dated.
"""
import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter

SR = 44100
BPM = 174.0
BEAT = 60.0 / BPM              # 0.3448 s
BAR = BEAT * 4
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
N = int(SR * DUR)
out = np.zeros(N)
rng = np.random.default_rng(11)


def place(sig, at, buf=None, gain=1.0):
    buf = out if buf is None else buf
    i = int(at * SR)
    if i >= len(buf) or i < 0:
        return
    e = min(len(buf), i + len(sig))
    buf[i:e] += sig[:e - i] * gain


def decay(n, tau, a=0.0015):
    at = max(1, int(a * SR))
    return np.concatenate([np.linspace(0, 1, at),
                           np.exp(-np.arange(n - at) / (SR * tau))])[:n]


def lp(x, hz, order=2):
    b, a = butter(order, min(hz / (SR / 2), 0.99), btype="low")
    return lfilter(b, a, x)


def hp(x, hz, order=2):
    b, a = butter(order, min(hz / (SR / 2), 0.99), btype="high")
    return lfilter(b, a, x)


# ---------------- drums ----------------
def kick():
    n = int(0.30 * SR); x = np.arange(n) / SR
    f = 132 * np.exp(-x * 34) + 47
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * decay(n, 0.055)
    click = hp(rng.standard_normal(n), 1800) * np.exp(-x * 420) * 0.35
    return (body * 0.72 + click * 1.4) * 1.0


def snare():
    n = int(0.24 * SR); x = np.arange(n) / SR
    noise = hp(rng.standard_normal(n), 1400) * decay(n, 0.045)
    tone = (np.sin(2 * np.pi * 188 * x) + np.sin(2 * np.pi * 331 * x)) * decay(n, 0.028)
    return (noise * 0.95 + tone * 0.45) * 1.25


def ride(lvl=1.0, tau=0.014):
    n = int(0.07 * SR)
    return hp(rng.standard_normal(n), 7000) * np.exp(-np.arange(n) / (SR * tau)) * 0.34 * lvl


# ---------------- bass ----------------
def saw(f, n, phase=0.0):
    x = np.arange(n) / SR
    return 2.0 * ((f * x + phase) % 1.0) - 1.0


def reese(note, dur, cutoff0=420, cutoff1=2200):
    """Detuned saws, slow phase beat, sweeping low-pass — the D&B signature."""
    n = int(dur * SR)
    s = (saw(note, n) + saw(note * 1.0055, n, 0.31) + saw(note * 0.9946, n, 0.67)) / 3.0
    # sweep the filter across the note in a few blocks (cheap but audible)
    blocks = 10
    outb = np.zeros(n)
    for b in range(blocks):
        a, z = int(n * b / blocks), int(n * (b + 1) / blocks)
        cut = cutoff0 + (cutoff1 - cutoff0) * (0.5 - 0.5 * np.cos(np.pi * 2 * b / blocks))
        outb[a:z] = lp(s[a:z], cut)
    env = np.concatenate([np.linspace(0, 1, int(0.008 * SR)),
                          np.ones(max(1, n - int(0.05 * SR))),
                          np.linspace(1, 0, int(0.042 * SR))])[:n]
    if len(env) < n:
        env = np.pad(env, (0, n - len(env)), constant_values=0)
    return outb * env * 0.34


def sub(note, dur):
    n = int(dur * SR); x = np.arange(n) / SR
    env = np.concatenate([np.linspace(0, 1, int(0.01 * SR)),
                          np.ones(max(1, n - int(0.06 * SR))),
                          np.linspace(1, 0, int(0.05 * SR))])[:n]
    if len(env) < n:
        env = np.pad(env, (0, n - len(env)), constant_values=0)
    return np.sin(2 * np.pi * note * x) * env * 0.16


# F minor: F1 43.65, Ab1 51.91, C2 65.41, Db2 69.30
ROOTS = [43.65, 43.65, 51.91, 65.41, 43.65, 43.65, 69.30, 51.91]

bars = int(DUR / BAR) + 2
for b in range(bars):
    b0 = b * BAR
    root = ROOTS[b % len(ROOTS)]

    # two-step: kick 1, snare 3 — the backbone of the genre
    place(kick(), b0 + 0.0 * BEAT)
    place(snare(), b0 + 2.0 * BEAT)
    if b % 2 == 1:                      # ghost kick before the turnaround
        place(kick(), b0 + 2.75 * BEAT, gain=0.7)
    if b % 4 == 3:                      # extra snare on the bar-4 fill
        place(snare(), b0 + 3.5 * BEAT, gain=0.75)

    # 16th rides, accented on the offbeats
    for k in range(16):
        place(ride(1.0 if k % 4 == 2 else 0.5), b0 + k * BEAT / 4)

    # bass: long root under the bar, stab on the second half
    place(reese(root * 2, BAR * 0.55), b0)
    place(reese(root * 2, BAR * 0.32), b0 + BAR * 0.62)
    place(sub(root, BAR * 0.5), b0)
    place(sub(root, BAR * 0.3), b0 + BAR * 0.62)

out = out[:N]

# ---------------- master ----------------
out = out - out.mean()
out = hp(out, 55, order=2)          # phones can't play below this; it only eats headroom
out = out + 0.30 * hp(out, 2500)     # presence, so it reads on a handset speaker
out = np.tanh(out * 1.35)
fi, fo = int(0.35 * SR), int(1.0 * SR)
out[:fi] *= np.linspace(0, 1, fi)
out[-fo:] *= np.linspace(1, 0, fo)
out = out / (np.abs(out).max() + 1e-9) * 0.78

stereo = np.stack([out, np.roll(out, 70)], axis=1)
wavfile.write(sys.argv[1], SR, (stereo * 32767).astype(np.int16))
print(f"wrote {sys.argv[1]}  {DUR}s  {BPM:.0f} BPM  peak {np.abs(stereo).max():.3f}  rms {np.sqrt((out**2).mean()):.4f}")
