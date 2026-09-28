"""Cut grown surfaces into canvas-sized windows, so a large surface is read all over.

The survey reads one 75-cell canvas per surface, from its middle. On PHerc1451 the surfaces
were about that size, 2.2 cm² at the median; on PHerc0846A a single seed grows 10-70 cm², and
the middle canvas would leave most of it unread. Each window here is a pick of its own, with
its grid origin, and is ranked for seating and read as a separate canvas.

    python tile_picks.py LEDGER.json [...] --out picks_windows.json [--min-seating 6]

Windows already listed in any `--seen` file are left out, so it can be run again as the
tracers add surfaces and only the new windows come back.
"""
import argparse, glob, json, os

import numpy as np, tifffile

HERE = "/workspace"
CELLS = 75


def windows(surface_dir, cells=CELLS, min_cover=0.6):
    """Non-overlapping cells x cells windows of the tifxyz grid, well enough covered."""
    X = tifffile.imread(os.path.join(surface_dir, "x.tif"))
    v = X > 0
    H, W = v.shape
    out = []
    for y0 in range(0, max(1, H - cells + 1), cells):
        for x0 in range(0, max(1, W - cells + 1), cells):
            cover = float(v[y0:y0 + cells, x0:x0 + cells].mean())
            if H >= cells and W >= cells and cover >= min_cover:
                out.append((y0, x0, cover))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ledgers", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seen", nargs="*", default=[])
    ap.add_argument("--min-seating", type=float, default=6.0)
    ap.add_argument("--min-cover", type=float, default=0.6)
    args = ap.parse_args()

    seen = set()
    for pattern in args.seen:
        for f in glob.glob(pattern):
            try:
                seen |= {(r["dir"], r.get("y0"), r.get("x0")) for r in json.load(open(f))}
            except Exception:
                pass
    picks, dirs = [], set()
    for f in args.ledgers:
        try:
            rows = json.load(open(f))
        except Exception:
            continue
        for r in rows:
            s = max(r.get("refined_seating", -1), r.get("seating", -1))
            d = r.get("refined_dir") if r.get("refined_seating", -1) >= r.get("seating", -1) else r.get("dir")
            d = d or r.get("dir")
            if s < args.min_seating or not d or not os.path.exists(os.path.join(d, "x.tif")) or d in dirs:
                continue
            dirs.add(d)
            rel = os.path.relpath(d, HERE)
            for y0, x0, cover in windows(d, min_cover=args.min_cover):
                if (rel, y0, x0) in seen:
                    continue
                picks.append({"dir": rel, "y0": y0, "x0": x0, "cover": round(cover, 3),
                              "seating": round(s, 2), "area_cm2": r.get("area_cm2", 0.0)})
    picks.sort(key=lambda p: (-p["seating"], -p["cover"]))
    json.dump(picks, open(args.out, "w"), indent=1)
    print(f"поверхностей {len(dirs)}, новых окон {len(picks)}", flush=True)


if __name__ == "__main__":
    main()
