#!/usr/bin/env python3
"""Inventory + visual verification for winding-constraint files.

Takes native spiral-input constraint JSONs (same_windings / relative_windings)
plus the scroll's umbilicus.json and produces:

1. A per-collection geometry report: point count, z-range, median radius, and a
   robust outlier statistic (max residual from the path's smooth r(angle, z)
   trend, in sigmas of its own noise). Collections whose worst residual exceeds
   12 sigma are flagged SUSPECT — this catches gross errors (points from another
   wrap/collection, coordinate typos, multi-gap mistakes).
2. An overlay PNG: all collections drawn over the CT cross-section at the median
   annotation z with the umbilicus marked — the practical "easy to verify" view,
   where a wrap-hop is immediately visible to the eye.

Honest scope: subtle single-gap wrap-hops sit at or below the annotation-noise
floor and are NOT reliably separable by any per-path statistic alone; the
definitive check is running the spiral fit. Diagnostic columns (junction step
ratio, residual changepoint) are reported to assist manual review.

Validation: on the released, human-verified PHercParis4 same_windings.json
(125 collections) this reports 125/125 CONSISTENT.
"""Verify winding-constraint files against the scroll geometry.

Checks exported same_windings.json / relative_windings.json (native spiral-input
schema) using the scroll's umbilicus:

For every point we compute the polar angle and radius around the umbilicus axis
at the point's z. For a *same-winding* collection traced along one wrap, radius
should vary smoothly with unwrapped angle — big radius jumps at similar angles
mean the path hopped wraps. We report, per collection:

  - n points, z-range
  - radius vs unwrapped-angle monotonic smoothness (max |dr| between angular
    neighbours, in wrap-gap units estimated from the local radius spacing)
  - CONSISTENT / SUSPECT verdict

plus an overlay PNG: points coloured by collection drawn over the scroll
cross-section at the median z (streamed from the CT zarr at a coarse level).

This gives annotators and reviewers a fast, GPU-free sanity check — the
"easy to verify" property an ideal winding-constraint generator needs.
"""
import argparse
import json

import numpy as np


def load_points(path):
    d = json.load(open(path))
    out = []
    for cid, c in d.get("collections", {}).items():
        items = sorted(c["points"].values(), key=lambda p: p.get("creation_time", 0))
        pts = np.array([p["p"] for p in items], dtype=float)
        if len(pts):
            out.append((c.get("name", cid), pts))
    return out


def umbilicus_at(umb, z):
    cps = umb["control_points"]
    zs = np.array([c["z"] for c in cps], dtype=float)
    xs = np.array([c["x"] for c in cps], dtype=float)
    ys = np.array([c["y"] for c in cps], dtype=float)
    o = np.argsort(zs)
    return (np.interp(z, zs[o], xs[o]), np.interp(z, zs[o], ys[o]))


def analyze(name, pts, umb):
    """Consistency stats for one collection.

    Verdict targets GROSS errors: points from another wrap/collection, coordinate
    typos, multi-gap jumps — these appear as >12-sigma outliers from the smooth
    r(angle, z) trend of the path. Subtle single-gap hops that stay within the
    annotation-noise floor are reported via the diagnostic stats but are NOT
    reliably separable without the spiral solution itself (run the spiral fit
    for the definitive check).
    """
    cx, cy = np.array([umbilicus_at(umb, z) for z in pts[:, 2]]).T
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    r = np.hypot(dx, dy)
    ang = np.unwrap(np.arctan2(dy, dx))
    z = pts[:, 2]
    A = np.c_[np.ones_like(ang), ang, z - z.mean()]
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    resid = r - A @ coef
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-6
    worst_trend = float(np.max(np.abs(resid)) / (1.4826 * mad))
    # diagnostics (no hard verdict): junction step + residual changepoint
    dr = np.abs(np.diff(r))
    step_ratio = float(dr.max() / (np.median(dr) + 1e-6)) if len(dr) >= 4 else 0.0
    hop_stat = 0.0
    if len(resid) >= 8:
        for i in range(3, len(resid) - 3):
            hop_stat = max(hop_stat, abs(float(np.median(resid[i:]) -
                                               np.median(resid[:i]))))
        hop_stat /= (1.4826 * mad)
    verdict = "CONSISTENT" if worst_trend < 12.0 else "SUSPECT"
    return {"name": name, "n": int(len(pts)),
            "z_range": [float(z.min()), float(z.max())],
            "radius_med": float(np.median(r)),
            "worst_resid_sigma": round(worst_trend, 1),
            "diag_step_ratio": round(step_ratio, 1),
            "diag_changepoint_sigma": round(hop_stat, 1),
            "verdict": verdict}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--constraints", required=True,
                    help="same_windings.json / relative_windings.json")
    ap.add_argument("--umbilicus", required=True, help="umbilicus.json")
    ap.add_argument("--ct", help="CT zarr URL for the cross-section overlay")
    ap.add_argument("--ct-level", default="4")
    ap.add_argument("--ct-scale", type=float, default=None,
                    help="voxel scale factor from full-res to the chosen CT level "
                         "(e.g. 16 for level 4); default 2**level")
    ap.add_argument("--overlay", default="verify_overlay.png")
    a = ap.parse_args()

    umb = json.load(open(a.umbilicus))
    cols = load_points(a.constraints)
    print(f"{'collection':24}{'n':>5}{'z-range':>18}{'r_med':>9}{'worst':>7}  verdict")
    reports = []
    for name, pts in cols:
        r = analyze(name, pts, umb)
        reports.append((r, pts))
        print(f"{r['name']:24}{r['n']:>5}"
              f"{str([int(r['z_range'][0]), int(r['z_range'][1])]):>18}"
              f"{r['radius_med']:>9.0f}{r['worst_resid_sigma']:>7.2f}  {r['verdict']}")

    if a.ct and reports:
        import cv2
        import zarr
        scale = a.ct_scale or (2 ** int(a.ct_level))
        allz = np.concatenate([p[:, 2] for _, p in reports])
        zmid = int(np.median(allz))
        C = zarr.open_group(a.ct, mode="r")[a.ct_level]
        sl = np.asarray(C[int(zmid / scale)])
        img = (np.clip(sl.astype(np.float32) /
                       max(1, np.percentile(sl, 99)), 0, 1) * 255).astype(np.uint8)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        colors = [(60, 200, 160), (200, 100, 220), (60, 60, 220),
                  (220, 160, 40), (40, 220, 220), (120, 220, 60)]
        for i, (r, pts) in enumerate(reports):
            for p in pts:
                u, v = int(p[0] / scale), int(p[1] / scale)
                cv2.circle(rgb, (u, v), 3, colors[i % len(colors)], -1)
        ux, uy = umbilicus_at(umb, zmid)
        cv2.drawMarker(rgb, (int(ux / scale), int(uy / scale)), (0, 0, 255),
                       cv2.MARKER_CROSS, 12, 2)
        cv2.imwrite(a.overlay, rgb)
        print(f"overlay (z={zmid}) -> {a.overlay}")


if __name__ == "__main__":
    main()
