"""Fit a rigid map from a mesh's own frame into a target volume by maximising seating.

Correlating slices between two scans recovers a translation but says nothing about whether the
result puts a surface *on a sheet* — on PHerc0009B a translation-with-tilt fit scored 0.86 on
slice correlation and −0.2 on sheet contrast, i.e. it was wrong. Optimising the seating score
directly avoids that: the objective is the thing we actually need.

Search is coarse-to-fine over a rotation about z plus a translation. Rotation about z is the
degree of freedom that matters when a scroll is re-mounted between scans; the tilt terms are
absorbed by re-fitting the translation per stage.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_surface import ChunkedVolume          # noqa: E402
from seat_mesh import sample_points, seating_score  # noqa: E402


def rotate(points, normals, degrees, centre):
    a = np.radians(degrees)
    ca, sa = np.cos(a), np.sin(a)
    px = centre[0] + (points[:, 0] - centre[0]) * ca - (points[:, 1] - centre[1]) * sa
    py = centre[1] + (points[:, 0] - centre[0]) * sa + (points[:, 1] - centre[1]) * ca
    out = points.copy()
    out[:, 0], out[:, 1] = px, py
    nx = normals[:, 0] * ca - normals[:, 1] * sa
    ny = normals[:, 0] * sa + normals[:, 1] * ca
    rot = normals.copy()
    rot[:, 0], rot[:, 1] = nx, ny
    return out, rot


def fit(mesh_dir, volume_url, scale=1.0, level=3, count=500,
        angles=range(0, 360, 30), span=24000.0, step=8000.0, rounds=3, voxel_um=2.4):
    """Returns the best (score, degrees, offset in mesh units)."""
    volume = ChunkedVolume(volume_url, threads=10)
    points, normals = sample_points(mesh_dir, count)
    centre = (float(np.median(points[:, 0])), float(np.median(points[:, 1])))
    best = (-1e9, 0.0, np.zeros(3))
    for stage in range(rounds):
        candidates = angles if stage == 0 else [best[1] + d for d in (-10, -5, 0, 5, 10)]
        grid = np.arange(-span, span + 1e-6, step)
        for degrees in candidates:
            rp, rn = rotate(points, normals, degrees, centre)
            for dz in grid:
                for dy in grid:
                    for dx in grid:
                        offset = best[2] + np.array([dx, dy, dz]) / voxel_um
                        score, _, coverage = seating_score(rp, rn, volume, scale, offset, level)
                        if score > best[0]:
                            best = (score, degrees, offset)
        print(f"  этап {stage+1}: показатель {best[0]:7.2f}, угол {best[1]:6.1f}°, "
              f"сдвиг {np.round(best[2]*voxel_um).astype(int)} мкм", flush=True)
        span, step = step, step / 3.0
    return best


if __name__ == "__main__":
    mesh_dir, url = sys.argv[1], sys.argv[2]
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    level = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    score, degrees, offset = fit(mesh_dir, url, scale=scale, level=level,
                                 span=float(os.environ.get("SPAN", "24000")),
                                 step=float(os.environ.get("STEP", "8000")),
                                 rounds=int(os.environ.get("ROUNDS", "3")))
    print(f"ЛУЧШЕЕ: показатель {score:.2f} (>=15 значит село на лист), поворот {degrees:.1f}°, "
          f"сдвиг {np.round(offset*2.4).astype(int)} мкм")
