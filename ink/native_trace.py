"""Native Trace: grow verified surfaces directly in the scan where a scroll's ink is visible.

Several unread scrolls publish m7 surface *predictions* for their fine (77-78 keV) rescans while
carrying no usable segmentation there. This tool closes that gap end to end, and every stage is
gated by a measurement rather than an eyeball:

  1. seed generation  — sample the prediction volume, stratified over depth and radius from the
                        scroll axis, and keep only seeds whose CT value confirms real material.
                        This matters: 43.3% of predicted positives are phantoms over empty CT, and
                        the densest prediction block in PHerc0009B yields a seed sitting on CT = 0.
  2. tracing          — villa's own `vc_grow_seg_from_seed`, run on the published predictions with
                        their matching normal grids. A trace takes about 90 seconds on a laptop.
  3. seat gating      — `seat_mesh.seating_score` at the prediction->CT scale (x4). Correctly
                        seated surfaces score >= 15; a surface cutting across the windings scores
                        ~8; the gate keeps survivors and logs everything else.

The output is a ledger (JSON) plus the kept tifxyz surfaces — standard formats in, standard
formats out. Survivor rate observed on PHerc0009B: roughly one seed in six; at 90 s per attempt,
coverage is a function of patience, not luck.

Usage (PHerc0009B, all paths overridable):

    python ink/native_trace.py \
        --pred p9b_pred.zarr \
        --ct-l2 https://.../volumes/<fine>.zarr/2/ \
        --tracer /path/to/vc_grow_seg_from_seed \
        --params trace_params.json \
        --out traces/ --ledger ledger.json --seeds 30
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_surface import ChunkedVolume  # noqa: E402
from seat_mesh import sample_points, seating_score  # noqa: E402

SEAT_KEEP = 12.0        # keep for refinement; >= 15 is cleanly seated
CT_MATERIAL = 100.0     # median CT a seed neighbourhood must exceed


def generate_seeds(pred_path, ct_l2_url, count, seed=23):
    import zarr
    pred = zarr.open(os.path.join(pred_path, "0"), mode="r")
    ct = ChunkedVolume(ct_l2_url, threads=8)
    rng = np.random.default_rng(seed)
    depth_chunks = pred.shape[0] // 192
    centre_y, centre_x = pred.shape[1] / 2.0, pred.shape[2] / 2.0
    max_radius = min(centre_y, centre_x) * 0.75
    bands = [(zb, (lo, hi))
             for zb in range(max(2, depth_chunks // 4), depth_chunks - 2, 3)
             for (lo, hi) in ((0.2 * max_radius, 0.5 * max_radius),
                              (0.5 * max_radius, 0.75 * max_radius),
                              (0.75 * max_radius, max_radius))]
    rng.shuffle(bands)
    seeds = []
    for zb, (rlo, rhi) in bands:
        if len(seeds) >= count:
            break
        for _ in range(10):
            cz = int(zb + rng.integers(0, 3))
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(rlo, rhi)
            gy = int(centre_y + radius * np.sin(angle))
            gx = int(centre_x + radius * np.cos(angle))
            by, bx = gy // 192, gx // 192
            if not (1 <= by < pred.shape[1] // 192 - 1 and 1 <= bx < pred.shape[2] // 192 - 1):
                continue
            try:
                block = np.asarray(pred[cz * 192:(cz + 1) * 192,
                                        by * 192:(by + 1) * 192,
                                        bx * 192:(bx + 1) * 192])
            except Exception:
                continue
            fill = (block > 128).mean()
            if not (0.03 < fill < 0.25):        # sheets are thin; dense blocks are phantom halo
                continue
            hits = np.argwhere(block > 200)
            if len(hits) < 150:
                continue
            pick = hits[rng.choice(len(hits), size=12, replace=False)]
            pz = cz * 192 + pick[:, 0]
            py = by * 192 + pick[:, 1]
            px = bx * 192 + pick[:, 2]
            values = ct.at(pz, py, px)
            if np.median(values) > CT_MATERIAL:
                best = int(np.argmax(values))
                seeds.append({"x": int(px[best]), "y": int(py[best]), "z": int(pz[best]),
                              "ct": float(values[best])})
                break
    return seeds


def seat(surface_dir, ct_l2_url, scale=4.0):
    volume = ChunkedVolume(ct_l2_url, threads=10)
    points, normals = sample_points(surface_dir, 600)
    score, mean, coverage = seating_score(points, normals, volume, scale, level=2)
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--ct-l2", required=True)
    ap.add_argument("--tracer", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", default="traces")
    ap.add_argument("--ledger", default="ledger.json")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--threads", default="3")
    args = ap.parse_args()

    ledger = json.load(open(args.ledger)) if os.path.exists(args.ledger) else []
    done = {(r["x"], r["y"], r["z"]) for r in ledger}
    seeds = generate_seeds(args.pred, args.ct_l2, args.seeds)
    print(f"сидов к прогону: {len(seeds)}", flush=True)
    for i, s in enumerate(seeds):
        if (s["x"], s["y"], s["z"]) in done:
            continue
        out_dir = os.path.join(args.out, f"{len(ledger):03d}")
        os.makedirs(out_dir, exist_ok=True)
        started = time.time()
        env = dict(os.environ, OMP_NUM_THREADS=args.threads)
        subprocess.run(["nice", "-n", "20", args.tracer, "-v", args.pred, "-t", out_dir,
                        "-p", args.params, "-s", str(s["x"]), str(s["y"]), str(s["z"])],
                       env=env, stdout=open(os.path.join(out_dir, "log.txt"), "w"),
                       stderr=subprocess.STDOUT)
        row = dict(s)
        row["minutes"] = round((time.time() - started) / 60, 1)
        surfaces = sorted(glob.glob(os.path.join(out_dir, "auto_grown_*")))
        if surfaces:
            meta = json.load(open(os.path.join(surfaces[-1], "meta.json")))
            row["dir"] = surfaces[-1]
            row["area_cm2"] = round(meta.get("area_cm2", 0.0), 2)
            try:
                row["seating"] = round(seat(surfaces[-1], args.ct_l2), 2)
            except Exception as exc:
                row["seating"] = -1.0
                row["seat_error"] = str(exc)[:120]
        else:
            row["area_cm2"], row["seating"] = 0.0, -1.0
        # притяжка к листу как «шаг с проверкой»: до двух проходов, остаётся лучший
        if row.get("seating", -1) >= SEAT_KEEP and row.get("dir"):
            best_dir, best_score = row["dir"], row["seating"]
            src = row["dir"]
            for p in (1, 2):
                dst = os.path.join(out_dir, f"snap{p}")
                r = subprocess.run(["nice", "-n", "20", sys.executable,
                                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "snap_refine.py"),
                                    src, dst], capture_output=True, text=True,
                                   env=dict(os.environ, RAD="8"))
                score = None
                for line in r.stdout.splitlines():
                    if "посадка" in line:
                        try: score = float(line.split(":")[1])
                        except Exception: pass
                if score is None:
                    break
                if score > best_score:
                    best_dir, best_score = dst, score
                src = dst
            row["refined_dir"], row["refined_seating"] = best_dir, round(best_score, 2)
        ledger.append(row)
        json.dump(ledger, open(args.ledger, "w"), indent=1)
        kept = [r for r in ledger if max(r.get("refined_seating", 0), r.get("seating", -1)) >= SEAT_KEEP]
        print(f"  сид {i}: {row['area_cm2']} см², посадка {row['seating']} | "
              f"принято {len(kept)}, площадь {sum(r['area_cm2'] for r in kept):.1f} см²", flush=True)
    kept = [r for r in ledger if r.get("seating", -1) >= SEAT_KEEP]
    clean = [r for r in ledger if r.get("seating", -1) >= 15.0]
    print(f"ИТОГ: попыток {len(ledger)}, принято {len(kept)} ({sum(r['area_cm2'] for r in kept):.1f} см²), "
          f"из них чисто севших {len(clean)}", flush=True)


if __name__ == "__main__":
    main()
