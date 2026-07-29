#!/usr/bin/env python3
"""Chunk-adjacency test of the off-by-one blend hypothesis (villa#1114).

Hypothesis (bruniss/jrudolph, 2026-07-29): phantoms are a blending artifact —
the blend window extends one chunk too far in every direction, pulling in
empty chunks whose tiny blended values survive the softmax.

Test, with zero voxel downloads: zarr stores omit all-zero chunks, so an S3
key listing gives the exact set of data-bearing chunks for both the
prediction volume and the masked CT on the same grid. Classify every
prediction chunk by Chebyshev distance to the nearest CT-bearing chunk:

  d=0  supported (CT data inside the same chunk)
  d=1  halo ring — exactly what off-by-one blending predicts
  d>=2 beyond any blend window — NOT explained by the hypothesis

Usage: python halo_analysis.py PHerc0332 [PHerc1203 ...]   (or no args = all 36)
Writes halo_<sample>.json next to this script.
"""
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from full_batch import S3, SAMPLES, discover

BUCKET = S3


def list_keys(prefix):
    """All keys under prefix via paginated list-objects-v2."""
    keys, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={prefix}&max-keys=1000"
        if token:
            token_enc = urllib.parse.quote(token, safe="")
            url += f"&continuation-token={token_enc}"
        with urllib.request.urlopen(url, timeout=60) as r:
            text = r.read().decode()
        keys.extend(re.findall(r"<Key>([^<]+)</Key>", text))
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", text)
        if not m:
            return keys
        token = m.group(1)


def chunk_coords(keys, level_prefix):
    """Parse z/y/x chunk indices from zarr nested-store keys."""
    out = set()
    pat = re.compile(re.escape(level_prefix) + r"(\d+)/(\d+)/(\d+)$")
    for k in keys:
        m = pat.match(k)
        if m:
            out.add((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return out


def meta(url_zarr, level):
    with urllib.request.urlopen(f"{url_zarr}/{level}/.zarray", timeout=30) as r:
        d = json.load(r)
    return d["shape"], d["chunks"]


def analyze(sample):
    disc = discover(sample, SAMPLES[sample])
    if not disc:
        return None
    preds_url, ct_url, ct_level = disc
    pshape, pchunks = meta(preds_url, "0")
    cshape, cchunks = meta(ct_url, ct_level)
    if pshape != cshape:
        return {"sample": sample, "error": f"voxel grid mismatch {pshape} vs {cshape}"}
    pcz = pchunks[0]

    pgrid_total = 1
    cgrid = []
    for s, c in zip(pshape, pchunks):
        pgrid_total *= (s + c - 1) // c
    for s, c in zip(cshape, cchunks):
        cgrid.append((s + c - 1) // c)
    cgrid_total = cgrid[0] * cgrid[1] * cgrid[2]

    ppfx = preds_url.replace(BUCKET + "/", "") + "/0/"
    cpfx = ct_url.replace(BUCKET + "/", "") + f"/{ct_level}/"
    with ThreadPoolExecutor(2) as ex:
        fp = ex.submit(list_keys, ppfx)
        fc = ex.submit(list_keys, cpfx)
        P = chunk_coords(fp.result(), ppfx)
        C = chunk_coords(fc.result(), cpfx)

    # sanity: if either store materializes ~every chunk, existence != occupancy
    dense_store = len(C) > 0.98 * cgrid_total or len(P) > 0.98 * pgrid_total

    # voxel-space box distance: preds chunk box vs nearest CT chunk box.
    # buckets (Chebyshev, voxels): 0 = overlap (supported at chunk level),
    # (0, pcz] = one blend-chunk halo, (pcz, 2*pcz] , > 2*pcz.
    cz, cy, cx = cchunks

    def box_bucket(p):
        z0, y0, x0 = p[0] * pchunks[0], p[1] * pchunks[1], p[2] * pchunks[2]
        z1 = min(z0 + pchunks[0], pshape[0])
        y1 = min(y0 + pchunks[1], pshape[1])
        x1 = min(x0 + pchunks[2], pshape[2])
        margin = 2 * pcz
        best = None
        for zc in range(max(0, (z0 - margin) // cz), min(cgrid[0], (z1 + margin) // cz + 1)):
            gz = max(0, zc * cz - z1 + 1, z0 - (zc * cz + cz - 1))
            if gz > 2 * pcz:
                continue
            for yc in range(max(0, (y0 - margin) // cy), min(cgrid[1], (y1 + margin) // cy + 1)):
                gy = max(0, yc * cy - y1 + 1, y0 - (yc * cy + cy - 1))
                if max(gz, gy) > 2 * pcz:
                    continue
                for xc in range(max(0, (x0 - margin) // cx), min(cgrid[2], (x1 + margin) // cx + 1)):
                    if (zc, yc, xc) not in C:
                        continue
                    gx = max(0, xc * cx - x1 + 1, x0 - (xc * cx + cx - 1))
                    d = max(gz, gy, gx)
                    if best is None or d < best:
                        best = d
                        if best == 0:
                            return 0
        if best is None:
            return 3
        if best <= pcz:
            return 1
        return 2

    buckets = {0: 0, 1: 0, 2: 0, 3: 0}
    for p in P:
        buckets[box_bucket(p)] += 1

    n = len(P)
    rep = {
        "sample": sample,
        "preds_chunk_shape": pchunks, "ct_chunk_shape": cchunks,
        "preds_chunks": n, "ct_chunks": len(C),
        "dense_store_warning": dense_store,
        "supported_overlap": buckets[0],
        "halo_within_1chunk": buckets[1],
        "within_2chunks": buckets[2],
        "beyond_2chunks": buckets[3],
        "frac_supported": buckets[0] / n if n else 0,
        "frac_halo_1chunk": buckets[1] / n if n else 0,
        "frac_beyond_blend": (buckets[2] + buckets[3]) / n if n else 0,
    }
    return rep


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    names = sys.argv[1:] or list(SAMPLES)
    for name in names:
        out = os.path.join(here, f"halo_{name}.json")
        if os.path.exists(out):
            print(f"skip {name}", flush=True)
            continue
        try:
            rep = analyze(name)
        except Exception as e:
            print(f"{name} FAIL {type(e).__name__}: {str(e)[:100]}", flush=True)
            continue
        if rep is None:
            print(f"{name} artifacts not found", flush=True)
            continue
        json.dump(rep, open(out, "w"), indent=1)
        if "error" in rep:
            print(f"{name}: {rep['error']}", flush=True)
        else:
            print(f"{name}: preds_chunks={rep['preds_chunks']:,} supported={rep['frac_supported']:.3f} "
                  f"halo_1ch={rep['frac_halo_1chunk']:.3f} beyond={rep['frac_beyond_blend']:.4f}"
                  f"{' DENSE-STORE!' if rep['dense_store_warning'] else ''}", flush=True)


if __name__ == "__main__":
    main()
