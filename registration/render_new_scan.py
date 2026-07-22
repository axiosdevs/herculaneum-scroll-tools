"""Re-render a legacy Scroll 3 segment window from the new 2.4um scan.

Chain: PPM (7.91um frame) -> global registration -> snap to team surface-prediction
sheet midpoints -> sample intensity layers along normals at 4.8um (L1).
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import zarr
from scipy.ndimage import median_filter, uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "dual_energy"))
from map_to_new import p791L0_to_p24L0
from sample_patch import read_ppm_patch

SURF_URL = "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0332/representations/predictions/surfaces/20251211183505-surface-20260413222639-surface-m7-L2-th0.2.zarr"
INT_URL = "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr"


def read_box(arr, z0, z1, y0, y1, x0, x1, tries=6):
    for i in range(tries):
        try:
            return np.asarray(arr[z0:z1, y0:y1, x0:x1])
        except Exception as e:
            print(f"read retry {i+1}: {type(e).__name__}", flush=True)
            time.sleep(min(60, 5 * 2 ** i))
    raise RuntimeError("box read failed after retries")


def snap_offsets(S, pts96, n96, reach=40):
    offs = np.arange(-reach, reach + 1)
    allp = np.concatenate([pts96 + n96 * t for t in offs])
    zi = np.clip(np.rint(allp[:, 2]).astype(int), 0, S.shape[0] - 1)
    yi = np.clip(np.rint(allp[:, 1]).astype(int), 0, S.shape[1] - 1)
    xi = np.clip(np.rint(allp[:, 0]).astype(int), 0, S.shape[2] - 1)
    z0, z1, y0, y1, x0, x1 = zi.min(), zi.max() + 1, yi.min(), yi.max() + 1, xi.min(), xi.max() + 1
    box = read_box(S, z0, z1, y0, y1, x0, x1) > 127
    vals = box[zi - z0, yi - y0, xi - x0].reshape(len(offs), -1)
    n = vals.shape[1]
    snap = np.full(n, np.nan, np.float32)
    mid = len(offs) // 2
    v = vals.astype(np.int8)
    for i in range(n):
        col = v[:, i]
        if not col.any():
            continue
        d = np.diff(col)
        starts = np.where(d == 1)[0] + 1
        ends = np.where(d == -1)[0] + 1
        s = ([0] if col[0] else []) + list(starts)
        e = list(ends) + ([len(col)] if col[-1] else [])
        mids = [(a + b - 1) / 2 for a, b in zip(s, e)]
        snap[i] = min(mids, key=lambda m: abs(m - mid)) - mid
    return snap  # in 9.6um voxels along normal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--u0", type=int, required=True)
    ap.add_argument("--v0", type=int, required=True)
    ap.add_argument("--w", type=int, default=1500)
    ap.add_argument("--h", type=int, default=1245)
    ap.add_argument("--step", type=int, default=2, help="PPM px step; 2 -> 4.8um output grid")
    ap.add_argument("--layers", type=int, default=12, help="+- layers at 4.8um spacing")
    ap.add_argument("--out", default="render_center")
    args = ap.parse_args()

    ppm = os.path.join(HERE, "..", "dual_energy", "20240618142020.ppm")
    xyz, nrm, _ = read_ppm_patch(ppm, args.v0, args.u0, 10 ** 9, args.step)
    xyz = xyz[: args.h // args.step, : args.w // args.step]
    nrm = nrm[: args.h // args.step, : args.w // args.step]
    ph, pw = xyz.shape[:2]
    print(f"render grid {ph}x{pw}", flush=True)

    P = xyz.reshape(-1, 3)
    valid = np.linalg.norm(P, axis=1) > 0
    P24 = np.zeros((len(P), 3))
    N24 = np.zeros((len(P), 3))
    P24[valid] = p791L0_to_p24L0(P[valid])
    tip = p791L0_to_p24L0(P[valid] + nrm.reshape(-1, 3)[valid] * 10)
    nv = (tip - P24[valid]) / 10
    nv /= np.clip(np.linalg.norm(nv, axis=1, keepdims=True), 1e-9, None)
    N24[valid] = nv

    Sg = zarr.open_group(SURF_URL, mode="r")
    S = Sg["0"]
    t0 = time.time()
    snap = np.full(len(P), np.nan, np.float32)
    CH = 200000
    vidx = np.where(valid)[0]
    for c in range(0, len(vidx), CH):
        sel = vidx[c:c + CH]
        snap[sel] = snap_offsets(S, P24[sel] / 4.0, N24[sel])
        print(f"snap {c + len(sel)}/{len(vidx)} {time.time()-t0:.0f}s", flush=True)
    F = snap.reshape(ph, pw)
    F = median_filter(np.nan_to_num(F, nan=0.0), 5)
    print(f"snap field: median {np.nanmedian(F)*9.6:+.0f}um", flush=True)

    Ig = zarr.open_group(INT_URL, mode="r")
    I = Ig["1"]  # 4.8um
    # F is in 9.6um voxels; L1 coords = L0/2 -> offset in L1 vox = F*2
    base = (P24 / 2.0 + N24 * (F.reshape(-1) * 2)[:, None]).reshape(ph, pw, 3)
    N24g = N24.reshape(ph, pw, 3)
    validg = valid.reshape(ph, pw)
    layers = np.arange(-args.layers, args.layers + 1)
    stack = np.zeros((len(layers), ph, pw), np.uint8)
    t0 = time.time()
    TS = 100
    tiles = [(r, c) for r in range(0, ph, TS) for c in range(0, pw, TS)]
    for ti, (r, c) in enumerate(tiles):
        b = base[r:r + TS, c:c + TS].reshape(-1, 3)
        n = N24g[r:r + TS, c:c + TS].reshape(-1, 3)
        v = validg[r:r + TS, c:c + TS].reshape(-1)
        if not v.any():
            continue
        pts = np.concatenate([b + n * t for t in layers])
        vv = np.tile(v, len(layers))
        zi = np.clip(np.rint(pts[:, 2]).astype(int), 0, I.shape[0] - 1)
        yi = np.clip(np.rint(pts[:, 1]).astype(int), 0, I.shape[1] - 1)
        xi = np.clip(np.rint(pts[:, 0]).astype(int), 0, I.shape[2] - 1)
        z0, z1 = zi[vv].min(), zi[vv].max() + 1
        y0, y1 = yi[vv].min(), yi[vv].max() + 1
        x0, x1 = xi[vv].min(), xi[vv].max() + 1
        box = read_box(I, z0, z1, y0, y1, x0, x1)
        out = np.zeros(len(pts), np.uint8)
        out[vv] = box[zi[vv] - z0, yi[vv] - y0, xi[vv] - x0]
        th, tw = base[r:r + TS, c:c + TS].shape[:2]
        stack[:, r:r + th, c:c + tw] = out.reshape(len(layers), th, tw)
        if ti % 5 == 0:
            print(f"tile {ti+1}/{len(tiles)} {time.time()-t0:.0f}s", flush=True)
    np.save(f"{args.out}_stack.npy", stack)
    mx = stack.max(axis=0)
    cv2.imwrite(f"{args.out}_max.png", mx)
    cv2.imwrite(f"{args.out}_mid.png", stack[args.layers])
    print(f"RENDER DONE {args.out}", flush=True)


if __name__ == "__main__":
    main()
