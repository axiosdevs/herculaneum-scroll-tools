"""Assemble per-window ink maps into one continuous map per segment.

The sweep writes a map per window, and the windows tile the mesh grid contiguously — window
(r, c) covers cells [r, r+N) x [c, c+N) and the next starts at r+N. So the pieces join without
gaps, and a line of script crossing a window boundary is only broken by how they were stored,
not by the geometry. Stitching restores it, which matters because reading text needs several
lines in one frame, not a 4.9 mm tile.
"""
from __future__ import annotations

import glob
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_score import text_score  # noqa: E402


def stitch(map_dir, segment, upsample=20, pattern=None):
    pattern = pattern or f"{map_dir}/{segment}_*.npy"
    pieces = []
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"_(\d+)_(\d+)\.npy$", path)
        if not m:
            continue
        pieces.append((int(m.group(1)), int(m.group(2)), path))
    if not pieces:
        return None, []
    tile = np.load(pieces[0][2]).shape[0]
    rows = max(r for r, _, _ in pieces) * upsample + tile
    cols = max(c for _, c, _ in pieces) * upsample + tile
    canvas = np.zeros((rows, cols), np.float32)
    for r, c, path in pieces:
        piece = np.load(path)
        y, x = r * upsample, c * upsample
        canvas[y:y + piece.shape[0], x:x + piece.shape[1]] = piece
    return canvas, pieces


def main():
    map_dir, out_dir = sys.argv[1], sys.argv[2]
    upsample = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    os.makedirs(out_dir, exist_ok=True)
    segments = sorted({re.sub(r"_\d+_\d+\.npy$", "", os.path.basename(p))
                       for p in glob.glob(f"{map_dir}/*.npy")})
    results = []
    for segment in segments:
        canvas, pieces = stitch(map_dir, segment, upsample)
        if canvas is None:
            continue
        score, period = text_score(canvas)
        span_mm = canvas.shape[0] * 2.399 / 1000, canvas.shape[1] * 2.399 / 1000
        results.append((score, period, segment, canvas.shape, len(pieces), span_mm))
        np.save(f"{out_dir}/{segment}.npy", canvas)
        cv2.imwrite(f"{out_dir}/{segment}.png",
                    (np.clip(canvas, 0, 1) * 255).astype(np.uint8)[::2, ::2])
    results.sort(reverse=True)
    print("признак | шаг мм | кусков |    размер, мм | сегмент")
    for score, period, segment, shape, n, span in results:
        print(f"  {score:.3f} | {period:6.2f} | {n:6d} | {span[0]:5.1f}x{span[1]:5.1f} | {segment[:40]}")


if __name__ == "__main__":
    main()
