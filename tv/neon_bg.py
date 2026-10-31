"""Render a seamless looping neon glow background (1080p MP4) for Projectivy."""
import subprocess
import sys

import numpy as np

OUT = sys.argv[1]
W, H = 640, 360          # render size; ffmpeg upscales (content is all soft gradients)
FPS, SECONDS = 30, 20
N = FPS * SECONDS
TAU = 2 * np.pi

ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
x = xs / H                # 0 .. 16/9
y = ys / H                # 0 .. 1

BASE = np.array([0.015, 0.008, 0.045], np.float32)

# Glow blobs: colour, radius, centre, orbit size, integer orbit speeds, phase
BLOBS = [
    ((0.55, 0.15, 1.00), 0.42, (0.35, 0.35), (0.18, 0.10), (1, 2), 0.0),   # violet
    ((0.05, 0.35, 1.00), 0.48, (1.35, 0.60), (0.20, 0.12), (1, 1), 1.7),   # blue
    ((0.00, 0.85, 0.95), 0.30, (0.95, 0.25), (0.25, 0.08), (2, 1), 3.1),   # cyan
    ((1.00, 0.15, 0.65), 0.32, (1.55, 0.20), (0.12, 0.10), (1, 3), 4.4),   # magenta
    ((0.35, 0.10, 0.90), 0.38, (0.70, 0.85), (0.22, 0.08), (2, 1), 5.2),   # deep purple
]

# Aurora ribbons: colour, base height, width, strength, phase
RIBBONS = [
    ((0.20, 0.60, 1.00), 0.62, 0.035, 0.55, 0.0),
    ((0.80, 0.25, 1.00), 0.70, 0.025, 0.45, 2.1),
    ((0.00, 0.90, 0.90), 0.55, 0.020, 0.30, 4.0),
]

vignette = 1.0 - 0.55 * (((x - 8 / 9) / 1.1) ** 2 + ((y - 0.5) / 0.75) ** 2)
vignette = np.clip(vignette, 0.25, 1.0)[..., None]


def frame(i):
    p = TAU * i / N       # every motion uses integer multiples of p -> perfect loop
    img = np.zeros((H, W, 3), np.float32)

    for col, r, (cx, cy), (ax, ay), (kx, ky), ph in BLOBS:
        bx = cx + ax * np.sin(kx * p + ph)
        by = cy + ay * np.cos(ky * p + ph)
        pulse = 0.8 + 0.2 * np.sin(p + ph)
        g = np.exp(-((x - bx) ** 2 + (y - by) ** 2) / (r * r)) * pulse
        img += g[..., None] * np.array(col, np.float32) * 0.55

    for col, cy, w, s, ph in RIBBONS:
        y0 = (cy + 0.07 * np.sin(2.2 * x + p + ph)
              + 0.035 * np.sin(4.7 * x - 2 * p + ph))
        shimmer = 0.55 + 0.45 * np.sin(3.0 * x - p + ph)
        d = (y - y0) / w
        core = np.exp(-d * d)
        halo = np.exp(-d * d / 16) * 0.35
        img += ((core + halo) * shimmer * s)[..., None] * np.array(col, np.float32)

    img = BASE + img * vignette
    img = 1.0 - np.exp(-1.6 * img)          # soft tonemap, keeps highlights glowing
    img *= 0.78                              # stay dark enough for launcher text/tiles
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


ff = subprocess.Popen([
    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
    "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
    "-vf", "scale=1920:1080:flags=lanczos,noise=alls=4:allf=t,format=yuv420p",
    "-c:v", "libx264", "-preset", "slow", "-crf", "20",
    "-profile:v", "high", "-level", "4.1", "-movflags", "+faststart", "-an", OUT,
], stdin=subprocess.PIPE)

for i in range(N):
    ff.stdin.write(frame(i).tobytes())
ff.stdin.close()
sys.exit(ff.wait())
