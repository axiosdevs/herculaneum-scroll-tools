"""Snap a traced surface to the CT sheet centre-of-mass with a short leash, then re-seat.

Earlier experiments rejected snapping outright: with a wide search radius it drags an
already-correct mesh onto neighbouring windings (r 0.34-0.71 against the unsnapped truth).
The version that works keeps a short leash (default 8 grid voxels), smooths the shift field,
and is applied as a measured hill-climb — evaluate seating after every pass, keep the best.
Measured on native PHerc0009B traces: 18.06 -> 24.57 over two passes on one surface,
13.10 -> 16.53 on another (crossing the seated threshold), while a third got worse and its
original was kept. The gate decides, not the pass count."""
import sys, os, json
import numpy as np, tifffile
sys.path.insert(0, '/Users/pc/defi/vesuvius/scroll-tools/ink')
from render_surface import ChunkedVolume, grid_normals
from seat_mesh import sample_points, seating_score

src, dst = sys.argv[1], sys.argv[2]
RAD = int(os.environ.get("RAD", "8"))          # поводок, в вокселях КТ-уровня-2 (= вокселях сетки)
CT2 = "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0009B/volumes/20260319104112-2.401um-0.3m-77keV-masked.zarr/2/"
V = ChunkedVolume(CT2, threads=10)

X, Y, Z = (tifffile.imread(f"{src}/{a}.tif").astype(np.float64) for a in "xyz")
valid = (X > 0) & (Y > 0) & (Z > 0)
nx, ny, nz = grid_normals(X, Y, Z)
steps = np.arange(-RAD, RAD + 1, 1.0)
rr, cc = np.nonzero(valid)
px, py, pz = X[rr, cc], Y[rr, cc], Z[rr, cc]     # координаты сетки = воксели КТ L2 (после x4 масштаба... сетка в пред-вокселях = КТ L2)
ux, uy, uz = nx[rr, cc], ny[rr, cc], nz[rr, cc]
prof = np.zeros((len(rr), len(steps)), np.float32)
for i, t in enumerate(steps):
    prof[:, i] = V.at(np.rint(pz + uz*t).astype(np.int64),
                      np.rint(py + uy*t).astype(np.int64),
                      np.rint(px + ux*t).astype(np.int64))
base = np.percentile(prof, 20, axis=1, keepdims=True)
w = np.clip(prof - base, 0, None)
tot = w.sum(1)
shift = np.where(tot > 1e-6, (w*steps).sum(1)/np.maximum(tot, 1e-6), 0.0)
shift[prof.max(1) < 40] = 0.0
S = np.zeros_like(X); S[rr, cc] = shift
import cv2
Sm = cv2.GaussianBlur(S.astype(np.float32), (9, 9), 0)   # гладкое поле сдвигов
Xn = np.where(valid, X + nx*Sm, 0).astype(np.float32)
Yn = np.where(valid, Y + ny*Sm, 0).astype(np.float32)
Zn = np.where(valid, Z + nz*Sm, 0).astype(np.float32)
os.makedirs(dst, exist_ok=True)
for a, M in zip("xyz", (Xn, Yn, Zn)):
    tifffile.imwrite(f"{dst}/{a}.tif", M)
json.dump(json.load(open(f"{src}/meta.json")), open(f"{dst}/meta.json", "w"))
print("средний |сдвиг| %.2f вкс, максимум %.1f" % (np.abs(shift).mean(), np.abs(shift).max()))
pts, nrm = sample_points(dst, 600)
sc, mean, cov = seating_score(pts, nrm, V, 4.0, level=2)
print("посадка после притяжки: %.2f" % sc)
