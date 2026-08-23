"""Render layer stacks from a tifxyz surface, matching the convention the published
ink checkpoints were trained on.

Four details decide whether the result is usable; each is verified in test_ink.py:

* the normal is ``d/dcol x d/drow`` of central differences on the grid,
* the volume is sampled trilinearly,
* grid coordinate for output pixel ``p`` is ``p / upsample`` (cell corner, not centre),
* depth runs opposite to the geometric normal in the published surface volumes.

Chunks are fetched over plain HTTPS with ranged reads, so nothing is stored locally.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests
import tifffile


class ChunkedVolume:
    """Read-only view of a zarr v2 uint8 volume served over HTTPS."""

    def __init__(self, base: str, threads: int = 12):
        self.base = base.rstrip("/") + "/"
        self.threads = threads
        self.session = requests.Session()
        self.session.mount("https://", requests.adapters.HTTPAdapter(
            pool_maxsize=threads * 2, max_retries=3))
        meta = self.session.get(self.base + ".zarray", timeout=60).json()
        if meta["dtype"] != "|u1":
            raise ValueError(f"expected uint8, got {meta['dtype']}")
        self.shape = np.array(meta["shape"])
        self.chunks = np.array(meta["chunks"])
        self.sep = meta.get("dimension_separator", ".")
        self.codec = None
        if meta["compressor"]:
            import numcodecs
            self.codec = numcodecs.get_codec(meta["compressor"])
        self._cache: dict[tuple, np.ndarray | None] = {}

    def _load(self, keys) -> None:
        todo = [k for k in keys if k not in self._cache]
        if not todo:
            return

        def get(key):
            url = self.base + self.sep.join(str(int(t)) for t in key)
            for _ in range(4):
                try:
                    r = self.session.get(url, timeout=120)
                    if r.status_code != 200:
                        self._cache[key] = None
                        return
                    raw = self.codec.decode(r.content) if self.codec else r.content
                    buf = np.frombuffer(raw, np.uint8)
                    self._cache[key] = (buf.reshape(tuple(self.chunks))
                                        if buf.size == int(np.prod(self.chunks)) else None)
                    return
                except Exception:
                    continue
            self._cache[key] = None

        with ThreadPoolExecutor(self.threads) as pool:
            list(pool.map(get, todo))

    def at(self, pz, py, px) -> np.ndarray:
        """Nearest-voxel read at integer coordinates; out-of-bounds reads as 0."""
        out = np.zeros(len(pz), np.float32)
        inside = ((pz >= 0) & (pz < self.shape[0]) & (py >= 0) & (py < self.shape[1])
                  & (px >= 0) & (px < self.shape[2]))
        if not inside.any():
            return out
        z, y, x = pz[inside], py[inside], px[inside]
        keys = np.stack([z // self.chunks[0], y // self.chunks[1], x // self.chunks[2]], 1)
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        keys_s = keys[order]
        edges = np.flatnonzero(np.any(np.diff(keys_s, axis=0) != 0, axis=1)) + 1
        starts, ends = np.r_[0, edges], np.r_[edges, len(keys_s)]
        unique = [tuple(int(t) for t in keys_s[s]) for s in starts]
        self._load(unique)
        vals = np.zeros(len(z), np.float32)
        zs, ys, xs = z[order], y[order], x[order]
        for (s, e), key in zip(zip(starts, ends), unique):
            block = self._cache.get(key)
            if block is None:
                continue
            vals[s:e] = block[zs[s:e] - key[0] * self.chunks[0],
                              ys[s:e] - key[1] * self.chunks[1],
                              xs[s:e] - key[2] * self.chunks[2]]
        inverse = np.empty(len(order), np.int64)
        inverse[order] = np.arange(len(order))
        out[inside] = vals[inverse]
        return out

    def trilinear(self, fz, fy, fx) -> np.ndarray:
        z0 = np.floor(fz).astype(np.int64)
        y0 = np.floor(fy).astype(np.int64)
        x0 = np.floor(fx).astype(np.int64)
        tz = (fz - z0).astype(np.float32)
        ty = (fy - y0).astype(np.float32)
        tx = (fx - x0).astype(np.float32)
        acc = np.zeros(len(fz), np.float32)
        for dz in (0, 1):
            wz = tz if dz else 1 - tz
            for dy in (0, 1):
                wy = ty if dy else 1 - ty
                for dx in (0, 1):
                    w = wz * wy * (tx if dx else 1 - tx)
                    sel = w > 1e-4
                    if not sel.any():
                        continue
                    acc[sel] += w[sel] * self.at(z0[sel] + dz, y0[sel] + dy, x0[sel] + dx)
        return acc

    def drop_cache(self) -> None:
        self._cache.clear()


def grid_normals(x, y, z):
    """Unit normals from central differences, matching volume-cartographer's grid_normal."""
    def central(m, axis):
        g = np.zeros_like(m)
        if axis == 1:
            g[:, 1:-1] = m[:, 2:] - m[:, :-2]
        else:
            g[1:-1] = m[2:] - m[:-2]
        return g

    ux, uy, uz = central(x, 1), central(y, 1), central(z, 1)
    vx, vy, vz = central(x, 0), central(y, 0), central(z, 0)
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    return nx / norm, ny / norm, nz / norm


def upsample_grid(m, upsample, height, width):
    """Cell-corner resampling: output pixel p reads grid coordinate p / upsample."""
    if upsample == 1:
        return m
    rows = (np.arange(height, dtype=np.float32) / upsample)
    cols = (np.arange(width, dtype=np.float32) / upsample)
    map_y, map_x = np.meshgrid(rows, cols, indexing="ij")
    return cv2.remap(m, map_x.astype(np.float32), map_y.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def render(mesh_dir, volume, row0, col0, rows, cols, upsample=20, layers=62,
           level=0, z_offset=0.0, band=128, progress=None):
    """Return a (layers, rows*upsample, cols*upsample) uint8 stack."""
    sl = (slice(row0, row0 + rows), slice(col0, col0 + cols))
    gx, gy, gz = (tifffile.imread(f"{mesh_dir}/{a}.tif")[sl].astype(np.float32) for a in "xyz")
    valid_cells = (gx > 0) & (gy > 0) & (gz > 0)
    div = float(2 ** level)
    gx, gy, gz = gx / div, gy / div, gz / div
    ncx, ncy, ncz = grid_normals(gx, gy, gz)

    height, width = rows * upsample, cols * upsample
    fx = upsample_grid(gx, upsample, height, width)
    fy = upsample_grid(gy, upsample, height, width)
    fz = upsample_grid(gz, upsample, height, width)
    nx = upsample_grid(ncx, upsample, height, width)
    ny = upsample_grid(ncy, upsample, height, width)
    nz = upsample_grid(ncz, upsample, height, width)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    valid = upsample_grid(valid_cells.astype(np.float32), upsample, height, width) > 0.99

    offsets = np.arange(-(layers // 2), layers // 2 + (layers % 2)).astype(np.float32) + z_offset
    stack = np.zeros((layers, height, width), np.uint8)
    started = time.time()
    for top in range(0, height, band):
        bottom = min(top + band, height)
        mask = valid[top:bottom]
        rr, cc = np.nonzero(mask)
        if not len(rr):
            continue
        bx, by, bz = fx[top:bottom][mask], fy[top:bottom][mask], fz[top:bottom][mask]
        dx, dy, dz = nx[top:bottom][mask], ny[top:bottom][mask], nz[top:bottom][mask]
        for index, off in enumerate(offsets):
            values = volume.trilinear(bz + dz * off, by + dy * off, bx + dx * off)
            stack[index, top + rr, cc] = np.clip(values, 0, 255).astype(np.uint8)
        if progress:
            progress(bottom, height, time.time() - started)
        if len(volume._cache) > 4000:
            volume.drop_cache()
    return stack


def main(argv):
    mesh_dir, base, out_dir = argv[1], argv[2], argv[3]
    row0, col0, rows, cols, upsample, layers, level = (int(v) for v in argv[4:11])
    volume = ChunkedVolume(base, threads=int(os.environ.get("THREADS", "12")))
    def show(done, total, elapsed):
        print(f"  {done}/{total} строк, {elapsed/60:.1f} мин", flush=True)
    stack = render(mesh_dir, volume, row0, col0, rows, cols, upsample, layers, level,
                   z_offset=float(os.environ.get("ZOFF", "0")), progress=show)
    os.makedirs(f"{out_dir}/layers", exist_ok=True)
    for i in range(stack.shape[0]):
        tifffile.imwrite(f"{out_dir}/layers/{i:02d}.tif", stack[i])
    print(f"готово: {stack.shape}", flush=True)


if __name__ == "__main__":
    main(sys.argv)
