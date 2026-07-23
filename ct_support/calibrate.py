#!/usr/bin/env python3
"""Calibrate the chunk-level -> voxel-level phantom mapping and predict
voxel-level contamination for every sample in the 2026-04-13 m7 batch.

Inputs: chunk-level certain-phantom shares for all 36 samples (from
github.com/Schurkai/vesuvius-phantom-audit) and exact voxel-level phantom
fractions measured with ct_support.py `survey` on anchor samples.

Model: the mapping is monotone and saturating (a chunk is counted once even if
almost all its voxels are phantom, so voxel share rises much faster than chunk
share). We fit p_voxel = 1 - (1 - p_chunk)^k with k>1 by least squares on the
anchors — one-parameter, monotone, exact at 0 and 1 — and report per-anchor
residuals as the honest uncertainty band.
"""
import json
import sys

import numpy as np

# chunk-level shares (Schurkai audit, 2026-07-14)
CHUNK = {
 "PHercMANBp":0.485, "PHerc0500P2":0.398, "PHerc0343P":0.318, "PHerc0009B":0.273,
 "PHercMAN5":0.270, "PHerc0332":0.259, "PHerc1299":0.254, "PHerc1451":0.222,
 "PHerc0826":0.204, "PHerc1545":0.194, "PHerc0483A":0.189, "PHerc0814":0.187,
 "PHerc0846B":0.186, "PHercMANB":0.185, "PHerc1218":0.179, "PHerc0257":0.179,
 "PHerc0211":0.178, "PHerc0175A":0.178, "PHerc0306B":0.177, "PHerc0358":0.176,
 "PHerc0125":0.169, "PHerc1447":0.169, "PHerc0490A":0.168, "PHerc0343":0.166,
 "PHerc0846A":0.158, "PHerc0191":0.156, "PHerc0800":0.155, "PHerc0175B":0.155,
 "PHercParis4":0.152, "PHerc0813":0.152, "PHerc0841":0.149, "PHerc0490B":0.149,
 "PHerc0483B":0.144, "PHerc0139":0.136, "PHerc0268":0.130, "PHerc1203":0.092,
}


def main(anchor_files):
    anchors = {}
    for f in anchor_files:
        d = json.load(open(f))
        name = f.split("survey_")[-1].replace(".json", "").split("_")[0]
        anchors[name] = d["sampled_phantom_frac"]
    print("anchors (voxel-level, measured):", {k: round(v, 4) for k, v in anchors.items()})

    xs = np.array([CHUNK[n] for n in anchors if n in CHUNK])
    ys = np.array([anchors[n] for n in anchors if n in CHUNK])
    ks = np.linspace(1.0, 25.0, 4000)
    best_k, best_sse = 1.0, 1e9
    for k in ks:
        pred = 1 - (1 - xs) ** k
        sse = float(((pred - ys) ** 2).sum())
        if sse < best_sse:
            best_sse, best_k = sse, k
    pred_anchor = 1 - (1 - xs) ** best_k
    resid = ys - pred_anchor
    print(f"fitted k = {best_k:.2f}; anchor residuals: "
          f"{[round(float(r), 3) for r in resid]} (max |r| = {np.abs(resid).max():.3f})")

    band = float(np.abs(resid).max())
    rows = []
    for n, c in sorted(CHUNK.items(), key=lambda kv: -kv[1]):
        v = 1 - (1 - c) ** best_k
        mark = " (measured)" if n in anchors else ""
        rows.append((n, c, anchors.get(n, v), band if n not in anchors else 0.0, mark))
    print(f"\n{'sample':14}{'chunk%':>8}{'voxel% (est)':>14}  ±band")
    for n, c, v, b, mark in rows:
        print(f"{n:14}{c*100:>7.1f}%{v*100:>13.1f}%  ±{b*100:.0f}%{mark}")

    out = {"model": "p_voxel = 1 - (1-p_chunk)^k", "k": round(best_k, 3),
           "anchors": {k: round(v, 4) for k, v in anchors.items()},
           "band_abs": round(band, 4),
           "estimates": {n: round((anchors.get(n, 1 - (1 - c) ** best_k)), 4)
                         for n, c in CHUNK.items()}}
    json.dump(out, open("calibration_m7_batch.json", "w"), indent=1)
    print("\nwrote calibration_m7_batch.json")


if __name__ == "__main__":
    main(sys.argv[1:])
