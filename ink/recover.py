"""Does re-centring the stack recover ink lost to a loose surface? Measured against truth.

An automatically grown surface carries a 192 um grid; at the ink model's 2.4 um that is 80
pixels of interpolation between nodes, and the sheet wanders inside them. On an unread
scroll the cost is unmeasurable -- there is nothing to compare against.

PHerc0139 has both a published 109-layer surface volume and the team's own ink map for it.
Displacing the model's 62-layer window by a smooth field with the correlation length of a
192 um grid reproduces what a loose surface does. Re-centring on the sheet, in the stack's
own 2.4 um depth axis, is the proposed fix. Agreement with the published map says whether
it works.

Three numbers per amplitude: truth, damaged, repaired.
"""
from __future__ import annotations

import io
import json
import os
import sys

import cv2
cv2.setNumThreads(1)
import numpy as np
import requests
import tifffile
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reproduce import LAYERS, INK_MAP, fetch_surface_window, fetch_checkpoint  # noqa: E402
from flatten_stack import sheet_offset, gather  # noqa: E402
from seat_mesh import sheet_contrast  # noqa: E402
from canonical_ink import predict  # noqa: E402

WINDOW = int(os.environ.get("WINDOW", "1024"))
GRID_UM = float(os.environ.get("GRID_UM", "192"))      # the trace grid step being simulated
VOXEL_UM = 2.399
AMPS = [float(a) for a in os.environ.get("AMPS", "1,2,4,6").split(",")]   # layers of wobble


def wobble(shape, amp, seed):
    corr_px = GRID_UM / VOXEL_UM / 2.0     # errors correlate over about half a grid cell
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.standard_normal(shape).astype(np.float32), corr_px)
    return f * (amp / f.std())


def take(stack, off, n_out):
    C = stack.shape[0]
    lo = (C - n_out) / 2.0
    base = np.arange(n_out, dtype=np.float32)[:, None, None] + lo
    idx = np.clip(np.rint(base + off[None]), 0, C - 1).astype(np.int16)
    return np.take_along_axis(stack, idx, axis=0)


def main():
    from model_resnet3d_3d_decoder import load_model
    ckpt = fetch_checkpoint()
    published = tifffile.imread(io.BytesIO(requests.get(INK_MAP, timeout=1800).content))
    best = None
    for y in range(0, published.shape[0] - WINDOW, 512):
        for x in range(0, published.shape[1] - WINDOW, 512):
            share = (published[y:y + WINDOW, x:x + WINDOW] > 200).mean()
            if 0.10 < share < 0.35 and (best is None or share > best[0]):
                best = (share, y, x)
    share, y0, x0 = best
    print(f"окно y={y0} x={x0} {WINDOW}px, чернил по карте команды {share * 100:.1f}%", flush=True)
    stack = fetch_surface_window(y0, x0, WINDOW)
    truth = published[y0:y0 + WINDOW, x0:x0 + WINDOW].astype(np.float32) / 255.0
    n_out = LAYERS[1] - LAYERS[0]
    print(f"поверхностный объём {stack.shape}, окно модели {n_out} слоёв\n", flush=True)

    clean = take(stack, np.zeros((WINDOW, WINDOW), np.float32), n_out)
    p0 = predict(clean, ckpt, load_model, reverse=False)
    r0 = float(np.corrcoef(p0.ravel(), truth.ravel())[0, 1])
    c0 = sheet_contrast(clean)
    print(f"истина: контраст {c0:+.2f}  r {r0:+.3f}\n", flush=True)
    deep0 = take(stack, np.zeros((WINDOW, WINDOW), np.float32), min(stack.shape[0], n_out + 46))
    fix0 = gather(deep0, sheet_offset(deep0, 20, 4, 20), n_out)
    pc = predict(fix0, ckpt, load_model, reverse=False)
    rc = float(np.corrcoef(pc.ravel(), truth.ravel())[0, 1])
    print(f"КОНТРОЛЬ ВРЕДА: выравнивание хорошей поверхности -> контраст {sheet_contrast(fix0):+.2f} "
          f"r {rc:+.3f} (было {r0:+.3f})\n", flush=True)
    del deep0, fix0, pc
    print(f"{'дрожание мкм':>13} {'контраст':>9} {'r повреждён':>12} | "
          f"{'контраст':>9} {'r починен':>10} {'вернули':>9}", flush=True)

    rows = []
    for amp in AMPS:
        field = wobble((WINDOW, WINDOW), amp, seed=int(amp * 100) + 11)
        dam = take(stack, field, n_out)
        cd = sheet_contrast(dam)
        pd = predict(dam, ckpt, load_model, reverse=False)
        rd = float(np.corrcoef(pd.ravel(), truth.ravel())[0, 1])

        deep = take(stack, field, min(stack.shape[0], n_out + 46))
        fix = gather(deep, sheet_offset(deep, 20, 4, 20), n_out)
        cf = sheet_contrast(fix)
        pf = predict(fix, ckpt, load_model, reverse=False)
        rf = float(np.corrcoef(pf.ravel(), truth.ravel())[0, 1])

        got = (rf - rd) / max(r0 - rd, 1e-6) * 100
        rows.append({"amp_um": round(amp * VOXEL_UM, 2), "contrast_damaged": round(cd, 2),
                     "r_damaged": round(rd, 3), "contrast_fixed": round(cf, 2),
                     "r_fixed": round(rf, 3), "recovered_pct": round(got, 1)})
        print(f"{amp * VOXEL_UM:13.1f} {cd:9.2f} {rd:+12.3f} | {cf:9.2f} {rf:+10.3f} {got:8.0f}%",
              flush=True)
        del dam, deep, fix, pd, pf

    json.dump({"window": [int(y0), int(x0), WINDOW], "grid_um": GRID_UM,
               "clean": {"contrast": round(c0, 2), "r": round(r0, 3)}, "rows": rows},
              open("recover.json", "w"), indent=1)
    print("\nзаписано recover.json", flush=True)
    return 0


if __name__ == "__main__":
    from inference_env import ensure
    ensure()
    sys.exit(main())
