#!/usr/bin/env python3
"""ct-support: CT-consistency QA and cleaning for scroll surface-prediction volumes.

Surface-prediction zarrs can contain "phantom" positives sitting where the masked CT
volume reads exactly 0 (outside the scroll: halo rings, end caps). See
https://github.com/ScrollPrize/villa/issues/1114 — on Scroll 3 (PHerc0332) ~70% of
positive voxels in the published m7 predictions are phantoms, and seeded growers that
do not consult the CT will happily ride the phantom shell.

This tool streams both zarrs (no local copy needed) and provides:

  survey   per-plane phantom/support statistics over a z-stride, JSON report + CSV
  chunks   per-chunk (cube) support map over a z-window -> npz keep/drop mask
  clean    write a cleaned copy (preds AND ct>0) of a z-window to a local zarr

All modes are CPU-only, resumable, and laptop-friendly.

Usage examples
--------------
  # quick spot-check, 3 planes:
  python ct_support.py survey --preds $PRED --ct $CT --planes 2000,4224,6000

  # strided survey of the whole height (every 100th plane), with report:
  python ct_support.py survey --preds $PRED --ct $CT --stride 100 --out report.json

  # per-cube support map for cube-level filtering (bimodal: keep/drop is clean):
  python ct_support.py chunks --preds $PRED --ct $CT --z0 4000 --z1 4400 --out cubes.npz

  # cleaned volume for a window:
  python ct_support.py clean --preds $PRED --ct $CT --z0 4000 --z1 4400 --out cleaned.zarr

$PRED/$CT are https URLs of the prediction zarr (binary, level 0) and the masked CT
zarr level matching the prediction grid (for the PHerc0332 m7 preds that is CT level 2:
both are [8398, 3941, 3941] at 9.596 um).
"""
import argparse
import csv
import json
import sys
import time

import numpy as np
import zarr


def open_arrays(preds_url, ct_url, preds_level="0", ct_level="2"):
    P = zarr.open_group(preds_url, mode="r")[preds_level]
    C = zarr.open_group(ct_url, mode="r")[ct_level]
    if tuple(P.shape) != tuple(C.shape):
        sys.exit(f"grid mismatch: preds {P.shape} vs ct {C.shape} — "
                 f"pick the CT level that matches the prediction grid (--ct-level)")
    return P, C


def plane_stats(P, C, z, thr):
    p = np.asarray(P[z]) > thr
    c = np.asarray(C[z]) > 0
    pos = int(p.sum())
    phant = int((p & ~c).sum())
    return {"z": int(z), "positives": pos, "phantom": phant,
            "phantom_frac": (phant / pos) if pos else 0.0}


def cmd_survey(a):
    P, C = open_arrays(a.preds, a.ct, a.preds_level, a.ct_level)
    Z = P.shape[0]
    if a.planes:
        zs = [int(z) for z in a.planes.split(",")]
    else:
        zs = list(range(0, Z, a.stride))
    rows, t0 = [], time.time()
    tot_pos = tot_ph = 0
    for i, z in enumerate(zs):
        r = plane_stats(P, C, z, a.thr)
        rows.append(r)
        tot_pos += r["positives"]; tot_ph += r["phantom"]
        print(f"[{i+1}/{len(zs)}] z={z} positives={r['positives']:,} "
              f"phantom_frac={r['phantom_frac']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    report = {
        "preds": a.preds, "ct": a.ct, "threshold": a.thr,
        "planes_sampled": len(zs), "stride": a.stride if not a.planes else None,
        "sampled_positives": tot_pos, "sampled_phantom": tot_ph,
        "sampled_phantom_frac": (tot_ph / tot_pos) if tot_pos else 0.0,
        "sampled_support_frac": 1.0 - ((tot_ph / tot_pos) if tot_pos else 0.0),
        "per_plane": rows,
    }
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)
        with open(a.out.replace(".json", "") + "_planes.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["z", "positives", "phantom", "phantom_frac"])
            w.writeheader(); w.writerows(rows)
        print(f"wrote {a.out}")
    print(f"SAMPLED phantom fraction: {report['sampled_phantom_frac']:.4f} "
          f"(support {report['sampled_support_frac']:.4f})")


def iter_chunks(shape, cz, z0, z1):
    for zc in range(z0, z1, cz):
        for yc in range(0, shape[1], cz):
            for xc in range(0, shape[2], cz):
                yield zc, yc, xc


def cmd_chunks(a):
    P, C = open_arrays(a.preds, a.ct, a.preds_level, a.ct_level)
    cz = a.cube
    z0, z1 = a.z0, min(a.z1, P.shape[0])
    keys, sup, pos = [], [], []
    t0 = time.time()
    n = 0
    for zc, yc, xc in iter_chunks(P.shape, cz, z0, z1):
        p = np.asarray(P[zc:zc+cz, yc:yc+cz, xc:xc+cz]) > a.thr
        np_pos = int(p.sum())
        if np_pos == 0:
            continue
        c = np.asarray(C[zc:zc+cz, yc:yc+cz, xc:xc+cz]) > 0
        s = int((p & c).sum()) / np_pos
        keys.append((zc, yc, xc)); sup.append(s); pos.append(np_pos)
        n += 1
        if n % 50 == 0:
            print(f"{n} cubes, {time.time()-t0:.0f}s", flush=True)
    keys = np.array(keys); sup = np.array(sup); pos = np.array(pos)
    np.savez_compressed(a.out, keys=keys, support=sup, positives=pos,
                        cube=cz, threshold=a.thr, keep_threshold=a.keep)
    kept = int((sup >= a.keep).sum())
    print(f"cubes with positives: {len(sup)} | keep(support>={a.keep}): {kept} "
          f"| drop: {len(sup)-kept}")
    print(f"wrote {a.out}")


def cmd_clean(a):
    P, C = open_arrays(a.preds, a.ct, a.preds_level, a.ct_level)
    z0, z1 = a.z0, min(a.z1, P.shape[0])
    out = zarr.open_array(a.out, mode="w", shape=(z1 - z0, P.shape[1], P.shape[2]),
                          chunks=(32, 512, 512), dtype="u1")
    t0 = time.time()
    for i, z in enumerate(range(z0, z1)):
        p = np.asarray(P[z])
        c = np.asarray(C[z]) > 0
        out[i] = np.where(c, p, 0)
        if i % 20 == 0:
            print(f"{i}/{z1-z0} planes, {time.time()-t0:.0f}s", flush=True)
    print(f"wrote cleaned window to {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--preds", required=True)
        p.add_argument("--ct", required=True)
        p.add_argument("--preds-level", default="0")
        p.add_argument("--ct-level", default="2",
                       help="CT pyramid level matching the prediction grid")
        p.add_argument("--thr", type=int, default=127,
                       help="prediction positive threshold (default 127)")

    s = sub.add_parser("survey"); common(s)
    s.add_argument("--planes", help="comma-separated z planes (overrides --stride)")
    s.add_argument("--stride", type=int, default=100)
    s.add_argument("--out", help="write JSON report + CSV")
    s.set_defaults(f=cmd_survey)

    c = sub.add_parser("chunks"); common(c)
    c.add_argument("--z0", type=int, required=True)
    c.add_argument("--z1", type=int, required=True)
    c.add_argument("--cube", type=int, default=128)
    c.add_argument("--keep", type=float, default=0.5,
                   help="keep threshold on support fraction (default 0.5)")
    c.add_argument("--out", default="cubes.npz")
    c.set_defaults(f=cmd_chunks)

    k = sub.add_parser("clean"); common(k)
    k.add_argument("--z0", type=int, required=True)
    k.add_argument("--z1", type=int, required=True)
    k.add_argument("--out", default="cleaned.zarr")
    k.set_defaults(f=cmd_clean)

    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
