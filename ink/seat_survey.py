"""For every scroll, work out which of its volumes each published surface actually sits in.

Meshes carry no record of the frame they were built in, so a surface published beside a scroll
is not necessarily renderable in that scroll's newest scan — and a mis-seated render looks
convincing. This walks the corpus, tries each volume at each plausible binning, and reports the
seating score from seat_mesh (calibrated: known-good meshes score 19-35, a surface cutting
across the windings scores 8).
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
import urllib.request

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_surface import ChunkedVolume          # noqa: E402
from seat_mesh import sample_points, seating_score  # noqa: E402

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
SCALES = (1.0, 2.0, 4.0, 8.0, 0.5)
GOOD = 15.0


def listing(prefix, delim="/"):
    query = {"list-type": "2", "prefix": prefix, "max-keys": "1000", "delimiter": delim}
    text = urllib.request.urlopen(BUCKET + "?" + urllib.parse.urlencode(query), timeout=120).read().decode()
    return [p for p in re.findall(r"<(?:Key|Prefix)>([^<]*)</(?:Key|Prefix)>", text) if p != prefix]


def meshes_of(scroll, cache_root="meshcache", max_meshes=6):
    """Every distinct mesh directory of the first segment that has any, downloaded once."""
    for segment in listing(f"{scroll}/segments/"):
        seg = segment.rstrip("/").split("/")[-1]
        keys = [k for k in listing(f"{scroll}/segments/{seg}/", "") if k.endswith("x.tif")]
        if not keys:
            continue
        found = []
        for key in keys[:max_meshes]:
            base = key[:-5]
            label = base.rstrip("/").split("/")[-1] or "mesh"
            out = os.path.join(cache_root, f"{scroll}_{seg}_{label}")
            if not os.path.exists(f"{out}/x.tif"):
                os.makedirs(out, exist_ok=True)
                try:
                    for a in "xyz":
                        urllib.request.urlretrieve(BUCKET + base + f"{a}.tif", f"{out}/{a}.tif")
                except Exception:
                    continue
            try:
                if (tifffile.imread(f"{out}/x.tif") > 0).mean() > 0.05:
                    found.append((label, out))
            except Exception:
                continue
        if found:
            return seg, found
    return None, []


def main():
    out_path = os.environ.get("OUT", "ink/seating.json")
    rows = json.load(open(out_path)) if os.path.exists(out_path) else []
    seen = {(r["scroll"], r.get("mesh", ""), r["volume"]) for r in rows}
    scrolls = [p.rstrip("/").split("/")[-1] for p in listing("", "/") if p.startswith("PHerc")]
    for scroll in scrolls:
        segment, meshes = meshes_of(scroll)
        if not meshes:
            print(f"{scroll}: пригодной сетки нет", flush=True)
            continue
        volumes = [p.rstrip("/").split("/")[-1] for p in listing(f"{scroll}/volumes/")
                   if p.endswith(".zarr/")]
        print(f"{scroll} / {segment[:30]}: сеток {len(meshes)}, томов {len(volumes)}", flush=True)
        for label, mesh_dir in meshes:
          try:
            points, normals = sample_points(mesh_dir, 600)
          except Exception as exc:
            print(f"  сетка {label[:34]}: не читается ({exc})", flush=True)
            continue
          print(f"  сетка {label[:44]}", flush=True)
          for name in volumes:
            if (scroll, label, name) in seen:
                continue
            url = f"{BUCKET}{scroll}/volumes/{name}/2/"
            try:
                volume = ChunkedVolume(url, threads=10)
            except Exception:
                print(f"    {name[:44]}: уровня 2 нет", flush=True)
                continue
            best = None
            for scale in SCALES:
                score, mean, coverage = seating_score(points, normals, volume, scale, level=2)
                if best is None or score > best[0]:
                    best = (score, scale, mean, coverage)
            score, scale, mean, coverage = best
            verdict = "садится" if score >= GOOD else ("частично" if score > 3 else "нет")
            rows.append({"scroll": scroll, "segment": segment, "mesh": label, "volume": name,
                         "best_scale": scale, "score": round(score, 2),
                         "surface_mean": round(mean, 1), "coverage": round(coverage, 3),
                         "verdict": verdict})
            json.dump(rows, open(out_path, "w"), indent=1)
            print(f"    {name[:44]:<44} масштаб {scale:>4} показатель {score:7.2f}  {verdict}", flush=True)
    print("ОБЗОР ГОТОВ", flush=True)


if __name__ == "__main__":
    main()
