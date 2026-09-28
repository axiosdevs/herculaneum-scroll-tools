"""Render surface layers with trilinear sampling, matching vc_render_tifxyz."""
import sys, os, time, numpy as np, tifffile, cv2, requests, numcodecs
from concurrent.futures import ThreadPoolExecutor

mesh_dir, base, out_dir = sys.argv[1], sys.argv[2].rstrip('/') + '/', sys.argv[3]
cy, cx, rows_n, cols_n, up, nlay, level = (int(v) for v in sys.argv[4:11])
THREADS = int(os.environ.get("THREADS", "12")); ZOFF = float(os.environ.get("ZOFF", "0"))
BAND = int(os.environ.get("BAND", "128"))

sess = requests.Session()
sess.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=THREADS*2, max_retries=3))
meta = sess.get(base + ".zarray", timeout=60).json()
shp = np.array(meta["shape"]); C = np.array(meta["chunks"]); SEP = meta.get("dimension_separator", "/")
codec = numcodecs.get_codec(meta["compressor"]) if meta["compressor"] else None
print(f"том {tuple(shp)} блоки {tuple(C)}", flush=True)

sl = (slice(cy, cy+rows_n), slice(cx, cx+cols_n))
X, Y, Z = (tifffile.imread(f"{mesh_dir}/{a}.tif")[sl].astype(np.float32) for a in "xyz")
D = float(2**level)
Xc, Yc, Zc = X/D, Y/D, Z/D
# нормаль: центральная разность и векторное произведение — как в grid_normal
def cd(M, axis):
    g = np.zeros_like(M)
    if axis == 1: g[:, 1:-1] = M[:, 2:] - M[:, :-2]
    else:         g[1:-1] = M[2:] - M[:-2]
    return g
ux, uy, uz = cd(Xc,1), cd(Yc,1), cd(Zc,1)
vx, vy, vz = cd(Xc,0), cd(Yc,0), cd(Zc,0)
ncx, ncy, ncz = uy*vz - uz*vy, uz*vx - ux*vz, ux*vy - uy*vx
nc = np.sqrt(ncx**2 + ncy**2 + ncz**2) + 1e-9
ncx, ncy, ncz = ncx/nc, ncy/nc, ncz/nc

H, W = rows_n*up, cols_n*up
# их соглашение: пиксель r соответствует координате сетки r/up (угол ячейки), не центру
rs = np.arange(H, dtype=np.float32)/up
cs = np.arange(W, dtype=np.float32)/up
mapy, mapx = np.meshgrid(rs, cs, indexing="ij")
mapy = mapy.astype(np.float32); mapx = mapx.astype(np.float32)
rm = lambda M: cv2.remap(M, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
Xf, Yf, Zf = rm(Xc), rm(Yc), rm(Zc)
nx, ny, nz = rm(ncx), rm(ncy), rm(ncz)
nn = np.sqrt(nx*nx + ny*ny + nz*nz) + 1e-9
nx, ny, nz = nx/nn, ny/nn, nz/nn
valid = rm((X > 0).astype(np.float32)) > 0.99
print(f"итог {H}x{W}, валидных {valid.mean()*100:.1f}%", flush=True)

cache = {}
def fetch_chunks(keys):
    todo = [k for k in keys if k not in cache]
    if not todo: return
    def get(k):
        for attempt in range(4):
            try:
                r = sess.get(base + SEP.join(str(int(t)) for t in k), timeout=120)
                if r.status_code != 200: cache[k] = None; return
                b = np.frombuffer(codec.decode(r.content) if codec else r.content, np.uint8)
                cache[k] = b.reshape(tuple(C)) if b.size == int(np.prod(C)) else None
                return
            except Exception:
                continue
        cache[k] = None                      # блок не дался — считаем пустым, прогон не рушим
    with ThreadPoolExecutor(THREADS) as ex: list(ex.map(get, todo))

def gather(pz, py, px):
    """values at integer voxel coords, 0 outside"""
    out = np.zeros(len(pz), np.float32)
    ok = (pz >= 0) & (pz < shp[0]) & (py >= 0) & (py < shp[1]) & (px >= 0) & (px < shp[2])
    if not ok.any(): return out
    zi, yi, xi = pz[ok], py[ok], px[ok]
    keys = np.stack([zi//C[0], yi//C[1], xi//C[2]], 1)
    order = np.lexsort((keys[:,2], keys[:,1], keys[:,0]))
    keys_s = keys[order]
    bnd = np.flatnonzero(np.any(np.diff(keys_s, axis=0) != 0, axis=1)) + 1
    st, en = np.r_[0, bnd], np.r_[bnd, len(keys_s)]
    uniq = [tuple(int(t) for t in keys_s[s]) for s in st]
    fetch_chunks(uniq)
    vals = np.zeros(len(zi), np.float32)
    zs, ys, xs = zi[order], yi[order], xi[order]
    for (s, e), k in zip(zip(st, en), uniq):
        blk = cache.get(k)
        if blk is None: continue
        vals[s:e] = blk[zs[s:e]-k[0]*C[0], ys[s:e]-k[1]*C[1], xs[s:e]-k[2]*C[2]]
    inv = np.empty(len(order), np.int64); inv[order] = np.arange(len(order))
    out[ok] = vals[inv]
    return out

SNAP = int(os.environ.get("SNAP", "0"))       # ± вокселей поиска настоящего листа вдоль нормали
def snap_offsets(bx, by, bz, dx, dy, dz, rr, cc, H, W, rad, sub=16):
    """для разреженной решётки точек ищем середину листа по профилю вдоль нормали"""
    sel = np.flatnonzero((rr % sub == 0) & (cc % sub == 0))
    if not len(sel): return np.zeros((H, W), np.float32)
    steps = np.arange(-rad, rad+1, 2, dtype=np.float32)
    prof = np.zeros((len(sel), len(steps)), np.float32)
    for si, t in enumerate(steps):
        fz, fy, fx = bz[sel]+dz[sel]*t, by[sel]+dy[sel]*t, bx[sel]+dx[sel]*t
        z0 = np.floor(fz).astype(np.int64); y0 = np.floor(fy).astype(np.int64); x0 = np.floor(fx).astype(np.int64)
        prof[:, si] = gather(z0, y0, x0)
    k = np.ones(11, np.float32)/11
    sm = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), 1, prof)
    # центр тяжести профиля: устойчив к волокнам внутри листа, не прыгает на соседний виток
    base = np.percentile(sm, 20, axis=1, keepdims=True)
    w = np.clip(sm - base, 0, None)
    tot = w.sum(1)
    best = np.where(tot > 1e-6, (w*steps).sum(1)/np.maximum(tot, 1e-6), 0.0).astype(np.float32)
    best[sm.max(1) < 20] = 0.0
    best = np.clip(best, -rad*0.6, rad*0.6)
    field = np.zeros((H, W), np.float32); wgt = np.zeros((H, W), np.float32)
    field[rr[sel], cc[sel]] = best; wgt[rr[sel], cc[sel]] = 1.0
    ks = sub*4+1
    field = cv2.GaussianBlur(field, (ks, ks), 0); wgt = cv2.GaussianBlur(wgt, (ks, ks), 0)
    return np.divide(field, wgt, out=np.zeros_like(field), where=wgt > 1e-6)

offs = np.arange(-(nlay//2), nlay//2 + (nlay % 2)).astype(np.float32) + ZOFF
stack = np.zeros((nlay, H, W), np.uint8)
t0 = time.time()
for b0 in range(0, H, BAND):
    b1 = min(b0+BAND, H)
    m = valid[b0:b1]
    rr, cc = np.nonzero(m)
    if not len(rr): continue
    bx, by, bz = Xf[b0:b1][m], Yf[b0:b1][m], Zf[b0:b1][m]
    dx, dy, dz = nx[b0:b1][m], ny[b0:b1][m], nz[b0:b1][m]
    shift = np.zeros(len(rr), np.float32)
    if SNAP:
        fld = snap_offsets(bx, by, bz, dx, dy, dz, rr, cc, H, W, SNAP)
        shift = fld[rr, cc]
    for li, off in enumerate(offs):
        o = off + shift
        fz, fy, fx = bz + dz*o, by + dy*o, bx + dx*o
        z0, y0, x0 = np.floor(fz).astype(np.int64), np.floor(fy).astype(np.int64), np.floor(fx).astype(np.int64)
        tz, ty, tx = (fz-z0).astype(np.float32), (fy-y0).astype(np.float32), (fx-x0).astype(np.float32)
        acc = np.zeros(len(rr), np.float32)
        for dzi in (0, 1):
            wz = tz if dzi else 1-tz
            for dyi in (0, 1):
                wy = ty if dyi else 1-ty
                for dxi in (0, 1):
                    wx = tx if dxi else 1-tx
                    w = wz*wy*wx
                    sel = w > 1e-4
                    if not sel.any(): continue
                    v = gather(z0[sel]+dzi, y0[sel]+dyi, x0[sel]+dxi)
                    acc[sel] += w[sel]*v
        stack[li, b0+rr, cc] = np.clip(acc, 0, 255).astype(np.uint8)
    el = time.time()-t0
    print(f"  полоса {b0//BAND+1}/{(H+BAND-1)//BAND}: {el/60:.1f} мин, блоков в кэше {len(cache)}", flush=True)
    if len(cache) > 1500: cache.clear()
os.makedirs(f"{out_dir}/layers", exist_ok=True)
for li in range(nlay): tifffile.imwrite(f"{out_dir}/layers/{li:04d}.tif", stack[li])
print(f"RENDER_DONE за {(time.time()-t0)/60:.1f} мин", flush=True)
