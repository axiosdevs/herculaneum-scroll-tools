"""Centre a rendered stack on the sheet, at the render's own resolution.

Snapping a surface against level-2 CT cannot do better than its 9.6 um voxel, and ink sits
in a layer a couple of voxels thick at 2.4 um. But a stack rendered deeper than the model
needs already contains the sheet at full resolution: the depth of peak brightness, per
pixel, is the sheet centre. Re-gathering the model's 62 layers around that surface removes
the residual wobble that no geometric refinement at level 2 can reach.

The offset field is smoothed before use -- a sheet undulates smoothly, per-pixel argmax
does not -- and clipped so no pixel can walk onto a neighbouring winding.

    python flatten_stack.py <stack_dir> <out_dir> [n_out]
"""
from __future__ import annotations

import glob
import os
import re
import sys

import cv2
import numpy as np
import tifffile

NOUT = 62
SMOOTH_PX = float(os.environ.get("SMOOTH_PX", "24"))   # ~58 um at 2.4 um/px
MAX_SHIFT = float(os.environ.get("MAX_SHIFT", "12"))   # layers; ~29 um leash
SEARCH = float(os.environ.get("SEARCH", "12"))         # half-band the centre is sought in


def sheet_offset(stack, max_shift=MAX_SHIFT, smooth_px=SMOOTH_PX, search=SEARCH):
    """Per-pixel depth of the sheet centre, relative to the stack's middle.

    The centre of mass is taken over a band around the middle, not the whole stack: the
    windings on either side are only ~150-300 um away, and a wide band drags the estimate
    onto a neighbour -- measured, it costs more contrast than it recovers.
    """
    C = stack.shape[0]
    mid = (C - 1) / 2.0
    lo = int(max(0, round(mid - search)))
    hi = int(min(C, round(mid + search) + 1))
    band = stack[lo:hi].astype(np.float32)
    depth = np.arange(lo, hi, dtype=np.float32)[:, None, None]
    base = np.percentile(band, 20, axis=0, keepdims=True)
    w = np.clip(band - base, 0, None)
    tot = w.sum(0)
    centre = np.where(tot > 1e-6, (w * depth).sum(0) / np.maximum(tot, 1e-6), mid)
    off = np.clip(centre - mid, -max_shift, max_shift)
    k = int(max(3, round(smooth_px) * 2 + 1))
    off = cv2.GaussianBlur(off.astype(np.float32), (k, k), smooth_px)
    return off


def gather(stack, off, n_out=NOUT):
    C, H, W = stack.shape
    lo = (C - n_out) / 2.0
    base = (np.arange(n_out, dtype=np.float32)[:, None, None] + lo)
    idx = np.clip(np.rint(base + off[None]), 0, C - 1).astype(np.int16)
    return np.take_along_axis(stack, idx, axis=0)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    n_out = int(sys.argv[3]) if len(sys.argv) > 3 else NOUT
    files = sorted(glob.glob(os.path.join(src, "*.tif")),
                   key=lambda f: int(re.findall(r"\d+", os.path.basename(f))[-1]))
    stack = np.stack([tifffile.imread(f) for f in files])
    off = sheet_offset(stack)
    flat = gather(stack, off, n_out)
    os.makedirs(dst, exist_ok=True)
    for i in range(flat.shape[0]):
        tifffile.imwrite(os.path.join(dst, f"{i:03d}.tif"), flat[i])

        from seat_mesh import sheet_contrast
    mid = (stack.shape[0] - n_out) // 2
    before = stack[mid:mid + n_out]
    print(f"стек {stack.shape} -> {flat.shape}", flush=True)
    print(f"сдвиг: средний |{np.abs(off).mean():.1f}| слоёв ({np.abs(off).mean() * 2.401:.1f} мкм), "
          f"максимум {np.abs(off).max():.1f}", flush=True)
    print(f"контраст листа до {sheet_contrast(before):+.2f}  после {sheet_contrast(flat):+.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
