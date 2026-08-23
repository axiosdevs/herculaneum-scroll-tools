"""Decide whether a tifxyz surface is seated on a sheet of a given volume — and at what scale.

Meshes in this corpus are published in whatever frame produced them: sometimes the volume's own
voxels, sometimes a 2x or 4x binning of it, sometimes an older scan entirely. Nothing in the file
says which. Rendering a mis-seated mesh yields a cross-section through the windings that looks
plausible until you inspect it.

The test used here is not "are we inside material" — a cut across the windings is inside material
too. It is "does the sheet run along us": sample each surface point at 0 and at +-`gap` microns
along its own normal, and ask how much brighter the surface is than its own surroundings. On a
seated sheet the centre sits in papyrus and both sides fall into the gaps between windings.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_surface import ChunkedVolume, grid_normals  # noqa: E402


def sheet_contrast(stack, span=25):
    """The same question asked of an already-rendered stack: is the middle brighter than
    +-`span` layers out? Seated surfaces give +30 to +39; a cut across the windings gives 0."""
    import numpy as np
    stack = np.asarray(stack, np.float32)
    inside = (stack > 0).all(0)
    if inside.sum() < 1000:
        return float("nan")
    profile = np.array([stack[i][inside].mean() for i in range(stack.shape[0])])
    mid = len(profile) // 2
    lo = max(0, mid - span)
    hi = min(len(profile) - 1, mid + span)
    return float(profile[mid] - 0.5 * (profile[lo] + profile[hi]))


def sample_points(mesh_dir, count=800, seed=0):
    x, y, z = (tifffile.imread(f"{mesh_dir}/{a}.tif").astype(np.float64) for a in "xyz")
    valid = (x > 0) & (y > 0) & (z > 0)
    nx, ny, nz = grid_normals(x, y, z)
    idx = np.argwhere(valid)
    if not len(idx):
        raise ValueError("сетка пуста")
    rng = np.random.default_rng(seed)
    idx = idx[rng.choice(len(idx), size=min(count, len(idx)), replace=False)]
    r, c = idx[:, 0], idx[:, 1]
    return (np.stack([x[r, c], y[r, c], z[r, c]], 1),
            np.stack([nx[r, c], ny[r, c], nz[r, c]], 1))


def seating_score(points, normals, volume, scale, offset=(0.0, 0.0, 0.0), level=0,
                  gap_voxels=25.0):
    """Higher is better. Returns (score, mean intensity on the surface, coverage)."""
    div = 2.0 ** level
    px = (points[:, 0] * scale + offset[0]) / div
    py = (points[:, 1] * scale + offset[1]) / div
    pz = (points[:, 2] * scale + offset[2]) / div
    gap = gap_voxels / div
    read = lambda t: volume.at(
        np.rint(pz + normals[:, 2] * t).astype(np.int64),
        np.rint(py + normals[:, 1] * t).astype(np.int64),
        np.rint(px + normals[:, 0] * t).astype(np.int64))
    centre, before, after = read(0.0), read(-gap), read(gap)
    on = centre > 40
    coverage = float(on.mean())
    if coverage < 0.25:
        return -1.0, float(centre.mean()), coverage
    contrast = float((centre[on] - 0.5 * (before[on] + after[on])).mean())
    return contrast * coverage, float(centre[on].mean()), coverage


def search(mesh_dir, volume_url, scales=(1.0, 2.0, 4.0, 8.0, 0.5, 0.25), level=2,
           offsets=((0.0, 0.0, 0.0),), count=800):
    volume = ChunkedVolume(volume_url, threads=10)
    points, normals = sample_points(mesh_dir, count)
    rows = []
    for scale in scales:
        for offset in offsets:
            score, mean, coverage = seating_score(points, normals, volume, scale, offset, level)
            rows.append({"scale": scale, "offset": offset, "score": round(score, 2),
                         "surface_mean": round(mean, 1), "coverage": round(coverage, 3)})
    rows.sort(key=lambda r: -r["score"])
    return rows


if __name__ == "__main__":
    mesh_dir, url = sys.argv[1], sys.argv[2]
    level = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    for row in search(mesh_dir, url, level=level):
        print(f"  масштаб {row['scale']:>5}: показатель {row['score']:8.2f} "
              f"(яркость на поверхности {row['surface_mean']:5.1f}, покрытие {row['coverage']*100:3.0f}%)")
