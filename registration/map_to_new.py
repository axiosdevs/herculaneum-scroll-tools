"""Coordinate chain: 7.91um L0 voxel coords -> 2.4um scan L0 voxel coords.

Coarse transform estimated at 76.77um: v24 slice (flipped in x, rotated ANG about
frame center, shifted by (dx(z), dy(z))) matches r791 slice; z791 = Z0 - z24.
Inverting: p791 -> p24.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(HERE, "coarse_transform.json")))
ANG, Z0 = T["angle"], T["z0"]
# per-pair shifts showed linear drift in z; fit from measured pairs
Z24_PAIRS = np.array([315, 420, 525, 630, 735])
DX_PAIRS = np.array([-65, -65, -65, -65, -73])
DY_PAIRS = np.array([-63, -58, -55, -51, -51])
DX_FIT = np.polyfit(Z24_PAIRS, DX_PAIRS, 1)
DY_FIT = np.polyfit(Z24_PAIRS, DY_PAIRS, 1)

S791 = 7.91 / 76.77          # 7.91um L0 voxel -> 76um grid
S24 = 76.77 / 2.399 / 32     # v24 L5 voxel -> ... (L5 = 32x L0 binning of 2.399um)
SHAPE24_L5 = (1050, 493, 493)  # z, y, x


def p791L0_to_p24L0(pts):
    """pts: (N,3) columns (x, y, z) in 7.91um L0 voxel coords -> 2.4um L0 (x, y, z)."""
    x, y, z = (pts[:, 0] * S791, pts[:, 1] * S791, pts[:, 2] * S791)
    # invert z mapping: z791 = Z0 - z24
    z24 = Z0 - z
    dx = np.polyval(DX_FIT, z24)
    dy = np.polyval(DY_FIT, z24)
    # invert in-plane: forward was flip_x -> rotate(ANG about center c) -> translate(d)
    h, w = SHAPE24_L5[1], SHAPE24_L5[2]
    cx, cy = w / 2, h / 2
    a = np.deg2rad(ANG)
    # forward: q = R @ (p_flipped - c) + c + d  with R = [[cos, sin], [-sin, cos]] (cv2 convention, y down)
    # invert:  p_flipped = R^T @ (q - c - d) + c ; then unflip x' = w-1-x
    qx = x - cx - dx
    qy = y - cy - dy
    ca, sa = np.cos(a), np.sin(a)
    fx = ca * qx - sa * qy + cx
    fy = sa * qx + ca * qy + cy
    x24 = (w - 1) - fx
    y24 = fy
    # to L0 of 2.4um scan
    return np.c_[x24 * 32, y24 * 32, z24 * 32]


if __name__ == "__main__":
    # sanity: map the center voxel of 7.91 volume
    test = np.array([[1700.0, 1775.0, 4889.0]])  # x, y, z mid of 3400x3550x9778
    print(p791L0_to_p24L0(test))
