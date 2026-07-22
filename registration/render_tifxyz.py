"""Render surface layers from a tifxyz auto-grown segment + scroll volume zarr.

Output: contrasted u8 layer TIFFs in the winners' chunk layout, ready for inference.py.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import tifffile
import zarr


def read_box(arr, z0, z1, y0, y1, x0, x1, tries=6):
    for i in range(tries):
        try:
            return np.asarray(arr[z0:z1, y0:y1, x0:x1])
        except Exception as e:
            print(f"read retry {i+1}: {type(e).__name__}", flush=True)
            time.sleep(min(60, 5 * 2 ** i))
    raise RuntimeError("box read failed")


def contrast_lut():
    # winners' control points (0,0),(128,64),(255,255) — piecewise linear, u8
    xs = np.array([0, 128, 255], np.float32)
    ys = np.array([0, 64, 255], np.float32)
    return np.interp(np.arange(256), xs, ys).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_url", required=True, help="S3 https prefix of tifxyz_original dir")
    ap.add_argument("--vol_url", required=True, help="volume zarr https url")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--layers", type=int, default=10, help="+- layers, 1 voxel spacing")
    ap.add_argument("--upsample", type=int, default=20, help="mesh grid -> voxel grid factor")
    args = ap.parse_args()

    os.makedirs(f"{args.out_dir}/layers", exist_ok=True)
    import urllib.request
    maps = {}
    for c in "xyz":
        p = f"/tmp/{c}.tif"
        urllib.request.urlretrieve(f"{args.seg_url}/{c}.tif", p)
        maps[c] = tifffile.imread(p).astype(np.float32)
    X, Y, Z = maps["x"], maps["y"], maps["z"]
    gh, gw = X.shape
    H, W = gh * args.upsample, gw * args.upsample
    print(f"mesh grid {gh}x{gw} -> render {H}x{W}", flush=True)
    up = lambda M: cv2.resize(M, (W, H), interpolation=cv2.INTER_LINEAR)
    Xf, Yf, Zf = up(X), up(Y), up(Z)
    valid = (Xf != 0) | (Yf != 0) | (Zf != 0)

    # normals from surface tangents
    dx_u = cv2.Sobel(Xf, cv2.CV_32F, 1, 0, ksize=5)
    dy_u = cv2.Sobel(Yf, cv2.CV_32F, 1, 0, ksize=5)
    dz_u = cv2.Sobel(Zf, cv2.CV_32F, 1, 0, ksize=5)
    dx_v = cv2.Sobel(Xf, cv2.CV_32F, 0, 1, ksize=5)
    dy_v = cv2.Sobel(Yf, cv2.CV_32F, 0, 1, ksize=5)
    dz_v = cv2.Sobel(Zf, cv2.CV_32F, 0, 1, ksize=5)
    N = np.cross(np.dstack([dx_u, dy_u, dz_u]), np.dstack([dx_v, dy_v, dz_v]))
    N /= np.clip(np.linalg.norm(N, axis=2, keepdims=True), 1e-9, None)

    g = zarr.open_group(args.vol_url, mode="r")
    V = g["0"]
    lut = contrast_lut()
    offsets = list(range(-args.layers, args.layers + 1))
    TS = 150
    stack_paths = {k: f"{args.out_dir}/layers/{k + args.layers:02}.tif" for k in offsets}
    outs = {k: np.zeros((H, W), np.uint8) for k in offsets}
    t0 = time.time()
    tiles = [(r, c) for r in range(0, H, TS) for c in range(0, W, TS)]
    for ti, (r, c) in enumerate(tiles):
        v = valid[r:r + TS, c:c + TS]
        if not v.any():
            continue
        P = np.dstack([Xf[r:r + TS, c:c + TS], Yf[r:r + TS, c:c + TS], Zf[r:r + TS, c:c + TS]])
        n = N[r:r + TS, c:c + TS]
        th, tw = v.shape
        pts = np.concatenate([(P + n * k).reshape(-1, 3) for k in offsets])
        vv = np.tile(v.reshape(-1), len(offsets))
        xi = np.clip(np.rint(pts[:, 0]).astype(int), 0, V.shape[2] - 1)
        yi = np.clip(np.rint(pts[:, 1]).astype(int), 0, V.shape[1] - 1)
        zi = np.clip(np.rint(pts[:, 2]).astype(int), 0, V.shape[0] - 1)
        z0, z1 = zi[vv].min(), zi[vv].max() + 1
        y0, y1 = yi[vv].min(), yi[vv].max() + 1
        x0, x1 = xi[vv].min(), xi[vv].max() + 1
        if (z1 - z0) * (y1 - y0) * (x1 - x0) > 1.5e9:
            print(f"tile {ti}: bbox too big, skipping", flush=True)
            continue
        box = read_box(V, z0, z1, y0, y1, x0, x1)
        vals = np.zeros(len(pts), np.uint8)
        vals[vv] = box[zi[vv] - z0, yi[vv] - y0, xi[vv] - x0]
        vals = vals.reshape(len(offsets), th, tw)
        for j, k in enumerate(offsets):
            outs[k][r:r + th, c:c + tw] = vals[j]
        if ti % 20 == 0:
            print(f"tile {ti+1}/{len(tiles)} {time.time()-t0:.0f}s", flush=True)
    for k in offsets:
        tifffile.imwrite(stack_paths[k], lut[outs[k]])
    cv2.imwrite(f"{args.out_dir}/mask.png", (valid * 255).astype(np.uint8))
    # quick-look mid layer
    cv2.imwrite(f"{args.out_dir}/preview_mid.png", lut[outs[0]])
    print("RENDER_TIFXYZ_DONE", flush=True)


if __name__ == "__main__":
    main()
