"""Calibrate the depth window against ink, not against sheet contrast.

Re-centring a stack on the brightest depth maximises sheet contrast and *costs* agreement
with the team's published ink map (+0.877 -> +0.620 on an already-correct surface). Ink
sits on the sheet's face, not in its bright middle, so the brightest-depth criterion moves
the model's window off the layer the ink is in.

This sweeps what the depth window actually does -- a constant shift, and re-centring damped
by a factor between none and full -- and scores every setting by agreement with the
published map. Whatever wins is the criterion worth applying to an unread scroll.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reproduce import LAYERS, INK_MAP, fetch_surface_window, fetch_checkpoint  # noqa: E402
from flatten_stack import sheet_offset  # noqa: E402
from seat_mesh import sheet_contrast  # noqa: E402
from canonical_ink import predict  # noqa: E402

WINDOW = int(os.environ.get("WINDOW", "1024"))
SHIFTS = [float(s) for s in os.environ.get("SHIFTS", "-8,-4,0,4,8").split(",")]
DAMPS = [float(d) for d in os.environ.get("DAMPS", "0,0.25,0.5,1.0").split(",")]


def take(stack, off, n_out):
    C = stack.shape[0]
    lo = (C - n_out) / 2.0
    base = np.arange(n_out, dtype=np.float32)[:, None, None] + lo
    idx = np.clip(np.rint(base + off[None]), 0, C - 1).astype(np.int16)
    return np.take_along_axis(stack, idx, axis=0)


def main():
    from model_resnet3d_3d_decoder import load_model
    ckpt = fetch_checkpoint()
    pub = tifffile.imread(io.BytesIO(requests.get(INK_MAP, timeout=1800).content))
    best = None
    for y in range(0, pub.shape[0] - WINDOW, 512):
        for x in range(0, pub.shape[1] - WINDOW, 512):
            share = (pub[y:y + WINDOW, x:x + WINDOW] > 200).mean()
            if 0.10 < share < 0.35 and (best is None or share > best[0]):
                best = (share, y, x)
    share, y0, x0 = best
    stack = fetch_surface_window(y0, x0, WINDOW)
    truth = pub[y0:y0 + WINDOW, x0:x0 + WINDOW].astype(np.float32) / 255.0
    n_out = LAYERS[1] - LAYERS[0]
    print(f"окно y={y0} x={x0} {WINDOW}px, чернил по карте команды {share * 100:.1f}%", flush=True)
    print(f"объём {stack.shape}\n", flush=True)

    field = sheet_offset(take(stack, np.zeros((WINDOW, WINDOW), np.float32),
                              min(stack.shape[0], n_out + 46)), 20, 4, 20)
    print(f"{'сдвиг сл':>9} {'мкм':>7} {'демпф':>7} {'контраст':>9} {'r с истиной':>12}", flush=True)
    rows = []
    for damp in DAMPS:
        for sh in SHIFTS:
            off = field * damp + sh
            s = take(stack, off.astype(np.float32), n_out)
            p = predict(s, ckpt, load_model, reverse=False)
            r = float(np.corrcoef(p.ravel(), truth.ravel())[0, 1])
            c = sheet_contrast(s)
            rows.append({"shift": sh, "damp": damp, "contrast": round(c, 2), "r": round(r, 3),
                         "ink_pct": round(float((p > 0.5).mean()) * 100, 2)})
            print(f"{sh:9.1f} {sh * 2.399:7.1f} {damp:7.2f} {c:9.2f} {r:+12.3f}", flush=True)
            del s, p
    rows.sort(key=lambda t: -t["r"])
    print(f"\nлучшее по чернилам: сдвиг {rows[0]['shift']} сл, демпф {rows[0]['damp']}, "
          f"r {rows[0]['r']:+.3f}, контраст {rows[0]['contrast']:+.2f}", flush=True)
    top_c = max(rows, key=lambda t: t["contrast"])
    print(f"лучшее по контрасту: сдвиг {top_c['shift']} сл, демпф {top_c['damp']}, "
          f"r {top_c['r']:+.3f}, контраст {top_c['contrast']:+.2f}", flush=True)
    json.dump({"window": [int(y0), int(x0), WINDOW], "rows": rows}, open("calib_depth.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    from inference_env import ensure
    ensure()
    sys.exit(main())
