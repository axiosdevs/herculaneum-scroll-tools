"""Render a whole surface in one pass instead of nine windows, then hand it to inference.

Nine separate renders of one surface each re-fetch the chunks their neighbours already
pulled: the cache lives inside a render process and dies with it. One 75x75-cell render
shares that cache across the whole patch -- measured on the previous box at ~900 MB per
window, this roughly halves the traffic for the same pixels, and the canvas comes out
without seams because it was never cut.
"""
import glob, json, os, re, subprocess, sys, time
sys.path.insert(0, "/workspace/ink")
from center_window import sheet_offset
import numpy as np, tifffile

HERE = "/workspace"
QUEUE = os.environ.get("QUEUE", "/workspace/queue")   # one queue per layer order
VOL = os.environ.get("VOLURL", "")
VOXEL_UM, STEP_UM, NLAY = 2.399, 9.596, 62
CELLS = 75                      # 3x3 blocks of 25, rendered as one
SHARD = int(os.environ.get("SHARD", "0"))
NSHARD = int(os.environ.get("NSHARD", "1"))
THREADS = os.environ.get("THREADS", "32")
MAXQ = int(os.environ.get("MAXQ", "6"))


def surfaces():
    rows = json.load(open(os.environ.get("PICKS", f"{HERE}/picks1451.json")))
    rows = [r for r in rows if os.path.exists(os.path.join(HERE, r["dir"], "x.tif"))]
    rows.sort(key=lambda r: (-r["seating"], -r["area_cm2"]))
    return rows[SHARD::NSHARD]



PROBE_CELLS, PROBE_UP, PROBE_LAYERS = 8, 8, 301   # +-360 um: one full winding either side


def find_zoff(work, y0, x0, up, shard):
    """Where the sheet is, relative to the window this render would otherwise use.

    A tracer follows the winding; it is not obliged to land on the papyrus. On PHerc1451 a
    surface's sheet sat 149 um from the mesh -- twice the 62-layer window's half-width --
    so the render was of the gap, and every ink map over it was a picture of air. This runs
    one coarse, deep probe first and returns the offset that puts the sheet in the middle.
    """
    out = os.path.join(HERE, f"wp_{shard}")
    subprocess.run(["rm", "-rf", out])
    cy, cx = y0 + (CELLS - PROBE_CELLS) // 2, x0 + (CELLS - PROBE_CELLS) // 2
    r = subprocess.run([sys.executable, f"{HERE}/" + os.environ.get("RENDERER", "render_tri.py"), work, VOL, out,
                        str(cy), str(cx), str(PROBE_CELLS), str(PROBE_CELLS),
                        str(PROBE_UP), str(PROBE_LAYERS), "0"],
                       env=dict(os.environ, THREADS=THREADS, BAND="64", ZOFF="0"),
                       stdout=subprocess.DEVNULL, stderr=open(f"{HERE}/rerr_{SHARD}.log", "ab"))
    files = sorted(glob.glob(os.path.join(out, "layers", "*.tif")))
    if r.returncode != 0 or len(files) != PROBE_LAYERS:
        subprocess.run(["rm", "-rf", out])
        return 0, 0.0
    prof = np.array([tifffile.imread(f).astype(np.float32).mean() for f in files])
    subprocess.run(["rm", "-rf", out])
    if not np.isfinite(prof).all() or prof.max() <= 0:
        return 0, 0.0
    return sheet_offset(prof)


def main():
    os.makedirs(QUEUE, exist_ok=True)
    todo = surfaces()
    print(f"шард {SHARD}: поверхностей {len(todo)}", flush=True)
    for s in todo:
        src = os.path.join(HERE, s["dir"])
        tag = s["dir"].replace("/", "_") + "_whole"
        if os.path.exists(os.path.join(QUEUE, tag + ".done")):
            continue
        X, Y, Z = (tifffile.imread(os.path.join(src, f"{a}.tif")).astype(np.float32) for a in "xyz")
        v = (X > 0) & (Y > 0) & (Z > 0)
        H, W = X.shape
        if H < CELLS or W < CELLS:
            continue
        P = np.stack([X, Y, Z], -1).astype(np.float64)
        g = np.linalg.norm(P[:, 1:] - P[:, :-1], axis=-1)[v[:, 1:] & v[:, :-1]]
        if not g.size:
            continue
        up = max(1, int(round(float(np.median(g)) * STEP_UM / VOXEL_UM)))
        work = os.path.join(HERE, f"wm_{SHARD}")
        os.makedirs(work, exist_ok=True)
        for a, M in zip("xyz", (X, Y, Z)):
            tifffile.imwrite(os.path.join(work, f"{a}.tif"),
                             np.where(M > 0, M * 4.0, 0.0).astype(np.float32))
        y0, x0 = (H - CELLS) // 2, (W - CELLS) // 2
        while len(glob.glob(os.path.join(QUEUE, "*.npy"))) >= MAXQ:
            time.sleep(20)
        # the deep probe costs as much as the render on a slow line, and the clamp below
        # zeroes its answer on almost every surface -- skippable when re-reading a known set
        if os.environ.get("SKIP_PROBE") == "1":
            zoff, sheet_contrast = 0, 0.0
        else:
            zoff, sheet_contrast = find_zoff(work, y0, x0, up, SHARD)
        # Only move a window that is genuinely somewhere else. Nudging one that already
        # holds the sheet measured *harmful* on PHerc0139: sheet contrast anti-correlates
        # with ink readability at r = -0.900, and centring on the sheet's bright middle
        # took one PHerc1451 window from a planted-ink lift of +0.027 to +0.000.
        if abs(zoff) < 15 or sheet_contrast < 0.20:
            zoff = 0
        print(f"  {s['dir']}: лист на {zoff} слоёв ({zoff*VOXEL_UM:+.0f} мкм), "
              f"контраст {sheet_contrast:.2f}", flush=True)
        out = os.path.join(HERE, f"wt_{SHARD}")
        subprocess.run(["rm", "-rf", out])
        t0 = time.time()
        subprocess.run([sys.executable, f"{HERE}/" + os.environ.get("RENDERER", "render_tri.py"), work, VOL, out,
                        str(y0), str(x0), str(CELLS), str(CELLS), str(up), str(NLAY), "0"],
                       env=dict(os.environ, THREADS=THREADS, BAND="192", ZOFF=str(zoff)),
                       stdout=subprocess.DEVNULL, stderr=open(f"{HERE}/rerr_{SHARD}.log", "ab"))
        files = sorted(glob.glob(os.path.join(out, "layers", "*.tif")),
                       key=lambda f: int(re.findall(r"\d+", os.path.basename(f))[-1]))
        if len(files) < NLAY:
            print(f"  {tag}: рендер не удался ({len(files)})", flush=True)
            print(f"  {s['dir']}: рендер не дал слоёв — см. rerr_{SHARD}.log", flush=True)
            print(f"  {s['dir']}: основной рендер пуст — см. rerr_{SHARD}.log", flush=True)
            subprocess.run(["rm", "-rf", out]); continue
        stack = np.stack([tifffile.imread(f) for f in files])
        tmp = os.path.join(QUEUE, tag + ".part")
        np.save(tmp, stack)
        os.rename(tmp + ".npy", os.path.join(QUEUE, tag + ".npy"))
        json.dump({"dir": s["dir"], "y": y0, "x": x0, "seating": s["seating"],
                   "area_cm2": s["area_cm2"], "up": up, "whole": True,
                   "zoff": int(zoff), "sheet_contrast": round(float(sheet_contrast), 3)},
                  open(os.path.join(QUEUE, tag + ".json"), "w"))
        subprocess.run(["rm", "-rf", out])
        print(f"  отрендерено {tag} ({stack.shape[1]}px, {(time.time()-t0)/60:.1f} мин)", flush=True)
    print(f"шард {SHARD}: ГОТОВО", flush=True)


if __name__ == "__main__":
    main()
