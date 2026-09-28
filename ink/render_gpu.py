"""Surface rendering on the GPU: same output as render_tri.py, same arguments.

The CPU renderer spends about ninety minutes on a 14.4 mm canvas, nearly all of it in the
trilinear gather -- eight corner lookups per sample through a Python dict of chunks. That
gather is exactly what `grid_sample` does, so the volume goes to the card once per tile and
all sixty-two layers are sampled in one call.

Validated against render_tri.py on the same mesh from the same volume: **r = 0.999994, largest
disagreement one grey level**, and 0.1 minutes against 7.7 for a 640x640x62 patch.

Two things had to be right for that. `grid_sample` takes its grid as (x, y, z) and normalises
each against (W, H, D), so the extents must be handed over in that order and not in the
array's own (z, y, x) -- getting it backwards renders something plausible that correlates at
0.06 with the truth. And the grid must be float32; a float64 grid is simply refused. Everything outside the gather -- normals, the corner
-origin remap, the layer offsets -- is the same code path as before.
"""
import glob, os, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np, requests, numcodecs, tifffile, cv2
import torch
import torch.nn.functional as F

mesh_dir, base, out_dir = sys.argv[1], sys.argv[2].rstrip('/') + '/', sys.argv[3]
cy, cx, rows_n, cols_n, up, nlay, level = (int(v) for v in sys.argv[4:11])
THREADS = int(os.environ.get("THREADS", "24"))
ZOFF = float(os.environ.get("ZOFF", "0"))
TILE = int(os.environ.get("TILE", "384"))
DEV = os.environ.get("DEV", "cuda")

sess = requests.Session()
sess.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=THREADS * 2, max_retries=3))
meta = sess.get(base + ".zarray", timeout=60).json()
shp = np.array(meta["shape"]); C = np.array(meta["chunks"])
SEP = meta.get("dimension_separator", "/")
codec = numcodecs.get_codec(meta["compressor"]) if meta["compressor"] else None
print(f"том {tuple(shp)} блоки {tuple(C)} устройство {DEV}", flush=True)

sl = (slice(cy, cy + rows_n), slice(cx, cx + cols_n))
X, Y, Z = (tifffile.imread(f"{mesh_dir}/{a}.tif")[sl].astype(np.float32) for a in "xyz")
D = float(2 ** level)
Xc, Yc, Zc = X / D, Y / D, Z / D


def cd(M, axis):
    g = np.zeros_like(M)
    if axis == 1:
        g[:, 1:-1] = M[:, 2:] - M[:, :-2]
    else:
        g[1:-1] = M[2:] - M[:-2]
    return g


ux, uy, uz = cd(Xc, 1), cd(Yc, 1), cd(Zc, 1)
vx, vy, vz = cd(Xc, 0), cd(Yc, 0), cd(Zc, 0)
ncx, ncy, ncz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
nc = np.sqrt(ncx ** 2 + ncy ** 2 + ncz ** 2) + 1e-9
ncx, ncy, ncz = ncx / nc, ncy / nc, ncz / nc

H, W = rows_n * up, cols_n * up
rs = np.arange(H, dtype=np.float32) / up
cs = np.arange(W, dtype=np.float32) / up
mapy, mapx = np.meshgrid(rs, cs, indexing="ij")
mapy = mapy.astype(np.float32); mapx = mapx.astype(np.float32)
rm = lambda M: cv2.remap(M, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
Xf, Yf, Zf = rm(Xc), rm(Yc), rm(Zc)
nx, ny, nz = rm(ncx), rm(ncy), rm(ncz)
nn = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
nx, ny, nz = nx / nn, ny / nn, nz / nn
valid = rm((X > 0).astype(np.float32)) > 0.99
print(f"итог {H}x{W}, валидных {valid.mean()*100:.1f}%", flush=True)

cache = {}


def fetch_chunks(keys):
    todo = [k for k in keys if k not in cache]
    if not todo:
        return

    def get(k):
        for _ in range(4):
            try:
                r = sess.get(base + SEP.join(str(int(t)) for t in k), timeout=120)
                if r.status_code != 200:
                    cache[k] = None
                    return
                b = np.frombuffer(codec.decode(r.content) if codec else r.content, np.uint8)
                cache[k] = b.reshape(tuple(C)) if b.size == int(np.prod(C)) else None
                return
            except Exception:
                continue
        cache[k] = None

    with ThreadPoolExecutor(THREADS) as ex:
        list(ex.map(get, todo))


def subvolume(lo, hi):
    """Assemble the voxel box [lo, hi) from the store. Missing chunks read as zero."""
    size = (hi - lo).astype(np.int64)
    vol = np.zeros(tuple(size), np.uint8)
    k0 = lo // C
    k1 = (hi - 1) // C
    keys = [(a, b, c)
            for a in range(k0[0], k1[0] + 1)
            for b in range(k0[1], k1[1] + 1)
            for c in range(k0[2], k1[2] + 1)]
    fetch_chunks(keys)
    for k in keys:
        blk = cache.get(k)
        if blk is None:
            continue
        o = np.array(k, np.int64) * C
        s0 = np.maximum(lo, o)
        s1 = np.minimum(hi, o + C)
        if np.any(s1 <= s0):
            continue
        vol[s0[0] - lo[0]:s1[0] - lo[0],
            s0[1] - lo[1]:s1[1] - lo[1],
            s0[2] - lo[2]:s1[2] - lo[2]] = \
            blk[s0[0] - o[0]:s1[0] - o[0],
                s0[1] - o[1]:s1[1] - o[1],
                s0[2] - o[2]:s1[2] - o[2]]
    return vol


offs = np.arange(-(nlay // 2), nlay // 2 + (nlay % 2)).astype(np.float32) + ZOFF
stack = np.zeros((nlay, H, W), np.uint8)
dev = torch.device(DEV)
t0 = time.time()
tiles = 0

for r0 in range(0, H, TILE):
    r1 = min(r0 + TILE, H)
    for c0 in range(0, W, TILE):
        c1 = min(c0 + TILE, W)
        m = valid[r0:r1, c0:c1]
        if not m.any():
            continue
        bz, by, bx = Zf[r0:r1, c0:c1], Yf[r0:r1, c0:c1], Xf[r0:r1, c0:c1]
        dz, dy, dx = nz[r0:r1, c0:c1], ny[r0:r1, c0:c1], nx[r0:r1, c0:c1]
        o = offs[:, None, None]
        fz = bz[None] + dz[None] * o
        fy = by[None] + dy[None] * o
        fx = bx[None] + dx[None] * o
        mm = np.broadcast_to(m[None], fz.shape)
        lo = np.array([np.floor(fz[mm].min()) - 2, np.floor(fy[mm].min()) - 2,
                       np.floor(fx[mm].min()) - 2], np.int64)
        hi = np.array([np.ceil(fz[mm].max()) + 3, np.ceil(fy[mm].max()) + 3,
                       np.ceil(fx[mm].max()) + 3], np.int64)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, shp)
        if np.any(hi <= lo):
            continue
        vol = subvolume(lo, hi)
        vt = torch.from_numpy(vol).to(dev, torch.float32)[None, None]
        # grid_sample normalises x against W, y against H and z against D, so the
        # extents have to be given in that order, not in the array's own (z, y, x)
        size = torch.tensor((hi - lo - 1)[::-1].copy().astype(np.float32), device=dev)
        g = torch.from_numpy(np.stack([fx - lo[2], fy - lo[1], fz - lo[0]],
                                      -1).astype(np.float32)).to(dev)
        # grid_sample wants x,y,z last and coordinates in [-1, 1]
        g = (g / size.clamp(min=1) * 2.0 - 1.0)[None]
        vals = F.grid_sample(vt, g, mode="bilinear", align_corners=True,
                             padding_mode="zeros")[0, 0]
        out = vals.clamp(0, 255).to(torch.uint8).cpu().numpy()
        out[~mm] = 0
        stack[:, r0:r1, c0:c1] = out
        del vt, g, vals
        # Several renderers share one card, and each holding on to its allocator's cache grew
        # twelve of them to 20 GB of a 24 GB card with nothing in use. Give it back per tile.
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        tiles += 1
        if len(cache) > 3000:
            cache.clear()
    print(f"  строки {r1}/{H}: {(time.time()-t0)/60:.1f} мин, плиток {tiles}", flush=True)

os.makedirs(f"{out_dir}/layers", exist_ok=True)
for li in range(nlay):
    tifffile.imwrite(f"{out_dir}/layers/{li:04d}.tif", stack[li])
print(f"RENDER_DONE за {(time.time()-t0)/60:.1f} мин", flush=True)
