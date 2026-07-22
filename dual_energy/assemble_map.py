import glob
import os
import re

import cv2
import numpy as np

PATCH, STEP = 600, 4
PW = PATCH // STEP  # 150
W, H = 25706, 2491
GW, GH = (W // PATCH) * PW, (H // PATCH + 1) * PW

canvas_r = np.zeros((GH, GW), np.float32)   # log-ratio 53/70
canvas_z = np.zeros((GH, GW), np.float32)   # high-Z z-score
canvas_i = np.zeros((GH, GW), np.float32)   # 53keV intensity
covered = np.zeros((GH, GW), bool)

files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan", "patch_*.npz")))
used = 0
vals_all = []
for f in files:
    m = re.search(r"patch_v(\d+)_u(\d+)", f)
    v0, u0 = int(m.group(1)), int(m.group(2))
    d = np.load(f)
    if "empty" in d:
        continue
    a = d["v53"].astype(np.float32)  # (3, ph, pw)
    b = d["v70"].astype(np.float32)
    valid = d["valid"]
    sa, sb = a.mean(0), b.mean(0)
    logr = np.log((sa + 1) / (sb + 1))
    logr[~valid] = np.nan
    gy, gx = v0 // PATCH * PW, u0 // PATCH * PW
    ph, pw = logr.shape
    canvas_r[gy:gy + ph, gx:gx + pw] = np.nan_to_num(logr)
    canvas_i[gy:gy + ph, gx:gx + pw] = sa * valid
    covered[gy:gy + ph, gx:gx + pw] = valid
    vals_all.append(logr[valid & np.isfinite(logr)])
    used += 1

vals = np.concatenate(vals_all)
mu, sd = vals.mean(), vals.std()
canvas_z = np.where(covered, (canvas_r - mu) / sd, 0)
print(f"patches used: {used}, global logr mu={mu:.4f} sd={sd:.4f}")
print(f"high-Z px (z>3): {(canvas_z > 3).sum()}, (z>4): {(canvas_z > 4).sum()}")

# render: intensity gray + high-Z red overlay
inten = canvas_i.copy()
p2, p98 = np.percentile(inten[covered], [2, 98]) if covered.any() else (0, 1)
g = np.clip((inten - p2) / max(1e-6, p98 - p2), 0, 1)
img = (np.dstack([g, g, g]) * 255).astype(np.uint8)
img[canvas_z > 3] = [0, 0, 255]
img[~covered] //= 3
cv2.imwrite("metal_map_overlay.png", img)

# smoothed high-Z density (letters = clusters, dust = isolated)
dens = cv2.GaussianBlur((canvas_z > 3).astype(np.float32), (0, 0), 3)
dn = dens / max(1e-6, dens.max())
cv2.imwrite("metal_map_density.png", (dn * 255).astype(np.uint8))
print("saved metal_map_overlay.png + metal_map_density.png", flush=True)
