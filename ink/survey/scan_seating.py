"""Rank every grown surface by how flat it lies on its sheet, cheaply.

A full canvas costs about ninety minutes; a 12x12-cell patch at the same scale costs two or
three, and stroke-scale scatter is measurable on it. Rendering the team's own PHerc0139 mesh
with this renderer gives 26.4, so that is the reference a well-seated surface should approach;
our PHerc1451 surfaces measured so far sit near 42.6, which is material mixed with gap.

Writes one line per surface to seating_scan_<shard>.json so the full survey can be pointed at
the flattest ones instead of at all of them.
"""
import glob, json, os, subprocess, sys, time
import numpy as np, tifffile
sys.path.insert(0, "/workspace/ink")
from scan_cnr import stroke_noise, sheet_cnr
from center_window import depth_profile, window_verdict

HERE = "/workspace"
VOL = os.environ["VOLURL"]
SHARD = int(os.environ.get("SHARD", "0"))
NSHARD = int(os.environ.get("NSHARD", "1"))
THREADS = os.environ.get("THREADS", "10")
W_ID = os.environ.get("WORKTAG", "") + str(SHARD)   # work directories, per survey
VOXEL_UM = float(os.environ.get("VOXEL_UM", "2.399"))   # the scan's level-0 voxel
STEP_UM = VOXEL_UM * 4                                  # surfaces are grown at level 2
CELLS, NLAY = 12, 62
LEDGER = f"{HERE}/{os.environ.get('SCAN', 'seating_scan')}_{SHARD}.json"


def surfaces():
    """The same list the full survey walks, ordered the same way."""
    rows = json.load(open(os.environ.get("PICKS", f"{HERE}/picks1451.json")))
    rows = [r for r in rows if os.path.exists(os.path.join(HERE, r["dir"], "x.tif"))]
    rows.sort(key=lambda r: (-r["seating"], -r["area_cm2"]))
    return rows[SHARD::NSHARD]


done = {}
if os.path.exists(LEDGER):
    try:
        done = {(r["dir"], r.get("y0"), r.get("x0")): r for r in json.load(open(LEDGER))}
    except Exception:
        done = {}
rows = list(done.values())
todo = surfaces()
print(f"шард {SHARD}: поверхностей {len(todo)}, уже посчитано {len(done)}", flush=True)

for s in todo:
    if (s["dir"], s.get("y0"), s.get("x0")) in done:
        continue
    src = s["dir"] if s["dir"].startswith("/") else os.path.join(HERE, s["dir"])
    try:
        X, Y, Z = (tifffile.imread(os.path.join(src, f"{a}.tif")).astype(np.float32)
                   for a in "xyz")
    except Exception as exc:
        print(f"  {s['dir']}: сетка не читается ({type(exc).__name__})", flush=True)
        continue
    v = (X > 0) & (Y > 0) & (Z > 0)
    H, W = X.shape
    if H < CELLS or W < CELLS:
        continue
    P = np.stack([X, Y, Z], -1).astype(np.float64)
    g = np.linalg.norm(P[:, 1:] - P[:, :-1], axis=-1)[v[:, 1:] & v[:, :-1]]
    if not g.size:
        continue
    up = max(1, int(round(float(np.median(g)) * STEP_UM / VOXEL_UM)))
    work = f"{HERE}/pm_{W_ID}"
    os.makedirs(work, exist_ok=True)
    for a, M in zip("xyz", (X, Y, Z)):
        tifffile.imwrite(os.path.join(work, f"{a}.tif"),
                         np.where(M > 0, M * 4.0, 0.0).astype(np.float32))
    # the most complete patch, not the geometric middle -- inside the window, for a window pick
    ry0, rx0 = s.get("y0", 0), s.get("x0", 0)
    ry1 = min(H, ry0 + s.get("cells", 75)) if "y0" in s else H
    rx1 = min(W, rx0 + s.get("cells", 75)) if "x0" in s else W
    best = None
    for r in range(ry0, max(ry0 + 1, ry1 - CELLS), 3):
        for c in range(rx0, max(rx0 + 1, rx1 - CELLS), 3):
            sc = v[r:r + CELLS, c:c + CELLS].mean()
            if best is None or sc > best[0]:
                best = (sc, r, c)
    cover, y0, x0 = best
    out = f"{HERE}/pt_{W_ID}"
    subprocess.run(["rm", "-rf", out])
    t0 = time.time()
    subprocess.run([sys.executable, f"{HERE}/" + os.environ.get("RENDERER", "render_tri.py"), work, VOL, out,
                    str(y0), str(x0), str(CELLS), str(CELLS), str(up), str(NLAY), "0"],
                   env=dict(os.environ, THREADS=THREADS, BAND="96", ZOFF="0"),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    files = sorted(glob.glob(os.path.join(out, "layers", "*.tif")))
    if len(files) != NLAY:
        subprocess.run(["rm", "-rf", out])
        print(f"  {s['dir']}: заплата не отрендерилась", flush=True)
        continue
    stack = np.stack([tifffile.imread(f) for f in files])
    subprocess.run(["rm", "-rf", out])
    prof = depth_profile(stack, sub=4)
    q = sheet_cnr(stack)
    row = {"dir": s["dir"], "y0": s.get("y0"), "x0": s.get("x0"),
           "seating": s.get("seating"), "area_cm2": s.get("area_cm2"), "up": up,
           "cover": round(float(cover), 3),
           "noise": q["noise"], "sheet_cnr": q["sheet_cnr"],
           "window": window_verdict(prof)[0],
           "minutes": round((time.time() - t0) / 60, 1)}
    rows.append(row)
    json.dump(rows, open(LEDGER, "w"), ensure_ascii=False, indent=1)
    print(f"  {s['dir'][:34]:36s} шум {row['noise']:6.2f}  окно {row['window']:8s} "
          f"лист/шум {row['sheet_cnr']:5.2f}  {row['minutes']:.1f} мин", flush=True)
print("шард завершён", flush=True)
