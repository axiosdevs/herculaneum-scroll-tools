"""Which volume's voxel grid is this mesh written in? Ask the store, don't guess.

A published tifxyz carries bare coordinates. Nothing in it says which volume they index,
and a scroll usually has several at different resolutions -- so a segment grown in a 9.36 um
scan is routinely handed to a renderer pointed at a 2.4 um one. The render then samples air
and returns a clean, empty, entirely convincing result. On PHerc0009B that mistake cost a
day and turned an agreement of 0.395 with the team's published ink map into 0.068.

Two independent answers are available, and this module gives both.

**From the mesh's own metadata.** `area_cm2 / area_vx2` in meta.json is the area of one
voxel, so its square root is the voxel size of the frame the mesh was laid out in. The
scale to any target volume is then the ratio of voxel sizes -- 9.363 / 2.401 = 3.9 for the
PHerc0009B segments. This costs nothing and needs no network, but it is only as good as the
metadata.

**From the data.** Convert sample points at each candidate scale and ask the store whether
the chunks they land in exist. A sparse OME-Zarr holds nothing outside the scroll, so a
wrong scale misses: measured on a PHerc0009B segment, scale 3.55 hit 16 of 16 sampled
chunks while its neighbours hit 0 of 10. This is slower and it cannot resolve better than
a chunk -- 128 voxels -- but it answers even when the metadata is absent or wrong.

The chunk test's resolution is its limit, and the limit matters: 3.55 and 3.90 both "hit",
and at 70 px per grid cell the difference between them is 175 px of displacement, which
reads as no correlation at all. When a reference map of the same surface exists, finish the
job with `residual_shift`: one FFT cross-correlation recovers the leftover offset over every
shift at once instead of searching.

    python ink/resolve_frame.py <mesh_dir> <volume_url> [--meta]

A caution worth stating: a scale that lands on stored chunks is not proof the two volumes
share a coordinate space. On PHerc1203 the team's meshes hit at scale 1.00 against the
9.362 um volume and 0 of 10 against the 2.403 um one at every scale tried -- those two scans
are separate acquisitions with their own origins, and no single number maps between them.
`resolve()` reports that as a failure rather than returning the least-bad scale.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import tifffile

DEFAULT_SCALES = (0.25, 0.5, 1.0, 2.0, 2.4, 3.0, 3.55, 3.6, 3.9, 4.0, 4.8, 8.0)
MIN_HIT_RATE = 0.6


def voxel_size_from_meta(mesh_dir):
    """The voxel size of the frame this mesh was laid out in, from its own metadata."""
    path = os.path.join(mesh_dir, "meta.json")
    if not os.path.exists(path):
        return None
    try:
        meta = json.load(open(path))
        area_cm2, area_vx2 = float(meta["area_cm2"]), float(meta["area_vx2"])
    except Exception:
        return None
    if area_vx2 <= 0:
        return None
    mm2_per_voxel = area_cm2 * 100.0 / area_vx2          # cm^2 -> mm^2
    return float(np.sqrt(mm2_per_voxel) * 1000.0)        # mm -> um


def load_points(mesh_dir, count=16, seed=11):
    X, Y, Z = (tifffile.imread(os.path.join(mesh_dir, f"{a}.tif")).astype(np.float64) for a in "xyz")
    valid = (X > 0) & (Y > 0) & (Z > 0)
    rows, cols = np.nonzero(valid)
    if len(rows) == 0:
        return None
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(rows), min(count, len(rows)), replace=False)
    return np.stack([Z[rows[pick], cols[pick]],
                     Y[rows[pick], cols[pick]],
                     X[rows[pick], cols[pick]]], axis=1)


def chunk_hits(points, volume_url, scale, session=None, timeout=30):
    """How many of these points, at this scale, land in chunks the store actually holds."""
    import requests
    session = session or requests.Session()
    base = volume_url.rstrip("/") + "/"
    meta = session.get(base + ".zarray", timeout=timeout).json()
    chunks = np.array(meta["chunks"], dtype=np.int64)
    shape = np.array(meta["shape"], dtype=np.int64)
    sep = meta.get("dimension_separator", "/")
    hits = 0
    for p in points:
        idx = (p * scale).astype(np.int64)
        if np.any(idx < 0) or np.any(idx >= shape):
            continue
        key = sep.join(str(int(v)) for v in (idx // chunks))
        try:
            if session.get(base + key, timeout=timeout).status_code == 200:
                hits += 1
        except Exception:
            pass
    return hits


def resolve(mesh_dir, volume_url, scales=DEFAULT_SCALES, count=16):
    """Best scale from the mesh to this volume, or None when nothing lands.

    Returns (scale, hits, tried) -- scale is None if no candidate cleared MIN_HIT_RATE,
    which is the honest answer when the two volumes do not share a coordinate space.
    """
    import requests
    points = load_points(mesh_dir, count)
    if points is None:
        return None, 0, 0
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=16, max_retries=2))
    best = (0, None)
    for s in scales:
        h = chunk_hits(points, volume_url, s, session)
        if h > best[0]:
            best = (h, s)
    hits, scale = best
    return (scale if hits >= len(points) * MIN_HIT_RATE else None), hits, len(points)


def residual_shift(ours, reference):
    """Offset of `ours` from `reference` and the correlation there, over all shifts at once.

    A chunk test cannot see inside its own 128-voxel block, so a scale it accepts can still
    be a percent off -- which at 70 px per cell is a displacement large enough to read as no
    correlation. One cross-correlation settles the remainder without a search.
    """
    a = np.asarray(ours, np.float64)
    b = np.asarray(reference, np.float64)
    if a.shape != b.shape:
        import cv2
        b = cv2.resize(b.astype(np.float32), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    c = np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=a.shape) / a.size
    flat = int(np.argmax(c))
    dy, dx = divmod(flat, c.shape[1])
    if dy > a.shape[0] // 2:
        dy -= a.shape[0]
    if dx > a.shape[1] // 2:
        dx -= a.shape[1]
    return float(c.max()), int(dy), int(dx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh_dir")
    ap.add_argument("volume_url")
    ap.add_argument("--target-um", type=float, default=None,
                    help="voxel size of the target volume, for the metadata estimate")
    ap.add_argument("--points", type=int, default=16)
    args = ap.parse_args()

    um = voxel_size_from_meta(args.mesh_dir)
    if um:
        print(f"по метаданным сетка в кадре {um:.3f} мкм", end="")
        if args.target_um:
            print(f" -> масштаб к тому x{um / args.target_um:.3f}")
        else:
            print()
    else:
        print("метаданные не дают размер вокселя")

    scale, hits, tried = resolve(args.mesh_dir, args.volume_url, count=args.points)
    if scale is None:
        print(f"по данным: ни один масштаб не попал ({hits} из {tried}) — "
              f"вероятно, это разные съёмки со своими началами координат")
        return 1
    print(f"по данным: масштаб x{scale} ({hits} попаданий из {tried})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
