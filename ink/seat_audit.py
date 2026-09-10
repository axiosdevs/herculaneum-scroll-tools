"""Corpus-wide audit: does each published surface actually sit on a sheet?

`seat_mesh` answers that question for one surface. This runs it over every segment the corpus
publishes — 327 of them across 16 scrolls — against the volume each segment names, and writes a
ledger. The point is not any single verdict but the distribution: a surface that renders
convincing papyrus while cutting across the windings is invisible to inspection and fatal to
anything built on it, and nobody has measured how often that happens.

Calibration (from seat_mesh): correctly seated meshes score 19-35, a surface known to cut across
the windings scores 8.5, threshold 15. Scores are reported at the binning that fits best, so a
mesh published in a coarser frame than its volume is detected rather than silently mis-scored.

Resumable: re-running skips segments already in the ledger.

    python ink/seat_audit.py --ledger seat_audit.json --scrolls PHerc0139,PHerc1667
    python ink/seat_audit.py --ledger seat_audit.json          # whole corpus
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request

import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_surface import ChunkedVolume  # noqa: E402
from seat_mesh import sample_points, seating_score  # noqa: E402

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
SCALES = (1.0, 2.0, 4.0, 0.5)
SEATED = 15.0
CUTTING = 8.5


def listing(prefix, delim="/"):
    query = {"list-type": "2", "prefix": prefix, "max-keys": "1000", "delimiter": delim}
    text = urllib.request.urlopen(BUCKET + "?" + urllib.parse.urlencode(query), timeout=120).read().decode()
    return [p for p in re.findall(r"<(?:Key|Prefix)>([^<]*)</(?:Key|Prefix)>", text) if p != prefix]


def volume_for(scroll, mesh_name):
    """The volume a mesh names in its own filename, else the finest volume of the scroll."""
    volumes = [p.rstrip("/").split("/")[-1] for p in listing(f"{scroll}/volumes/") if p.endswith(".zarr/")]
    stamp = re.search(r"on-(\d+)", mesh_name)
    if stamp:
        for v in volumes:
            if stamp.group(1) in v:
                return v
    def micron(name):
        m = re.search(r"(\d+\.?\d*)um", name)
        return float(m.group(1)) if m else 99.0
    return min(volumes, key=micron) if volumes else None


def fetch_mesh(scroll, segment, cache="auditcache"):
    keys = [k for k in listing(f"{scroll}/segments/{segment}/", "") if k.endswith("x.tif")]
    if not keys:
        return None, None
    base = keys[0][:-5]
    label = base.rstrip("/").split("/")[-1] or "mesh"
    out = os.path.join(cache, f"{scroll}_{segment}")
    if not os.path.exists(os.path.join(out, "x.tif")):
        os.makedirs(out, exist_ok=True)
        try:
            for a in "xyz":
                urllib.request.urlretrieve(BUCKET + base + f"{a}.tif", os.path.join(out, f"{a}.tif"))
        except Exception:
            return None, None
    try:
        if (tifffile.imread(os.path.join(out, "x.tif")) > 0).mean() < 0.02:
            return None, None
    except Exception:
        return None, None
    return out, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="seat_audit.json")
    ap.add_argument("--scrolls", default="")
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--keep-cache", action="store_true")
    args = ap.parse_args()

    ledger = json.load(open(args.ledger)) if os.path.exists(args.ledger) else []
    seen = {(r["scroll"], r["segment"]) for r in ledger}
    scrolls = ([s.strip() for s in args.scrolls.split(",") if s.strip()]
               or [p.rstrip("/").split("/")[-1] for p in listing("", "/") if p.startswith("PHerc")])

    for scroll in scrolls:
        segments = [p.rstrip("/").split("/")[-1] for p in listing(f"{scroll}/segments/")]
        if not segments:
            continue
        print(f"{scroll}: сегментов {len(segments)}", flush=True)
        for segment in segments:
            if (scroll, segment) in seen:
                continue
            mesh_dir, label = fetch_mesh(scroll, segment)
            row = {"scroll": scroll, "segment": segment}
            if not mesh_dir:
                row["verdict"] = "нет сетки"
                ledger.append(row)
                json.dump(ledger, open(args.ledger, "w"), indent=1)
                continue
            volume = volume_for(scroll, label)
            if not volume:
                row["verdict"] = "нет тома"
                ledger.append(row)
                json.dump(ledger, open(args.ledger, "w"), indent=1)
                continue
            url = f"{BUCKET}{scroll}/volumes/{volume}/{args.level}/"
            try:
                vol = ChunkedVolume(url, threads=10)
                points, normals = sample_points(mesh_dir, 500)
                best = (-1.0, None, 0.0)
                for scale in SCALES:
                    score, mean, coverage = seating_score(points, normals, vol, scale, level=args.level)
                    if score > best[0]:
                        best = (score, scale, coverage)
                row.update({"mesh": label, "volume": volume,
                            "score": round(best[0], 2), "scale": best[1],
                            "coverage": round(best[2], 3),
                            "verdict": "сидит" if best[0] >= SEATED
                                       else ("частично" if best[0] > CUTTING else "не сидит")})
            except Exception as exc:
                row["verdict"] = "ошибка"
                row["error"] = str(exc)[:120]
            ledger.append(row)
            json.dump(ledger, open(args.ledger, "w"), indent=1)
            if not args.keep_cache and mesh_dir and os.path.isdir(mesh_dir):
                for f in ("x.tif", "y.tif", "z.tif"):
                    try: os.remove(os.path.join(mesh_dir, f))
                    except OSError: pass
                try: os.rmdir(mesh_dir)
                except OSError: pass
            print(f"  {segment[:36]:<36} {row.get('score', '—'):>6} {row['verdict']}", flush=True)

    scored = [r for r in ledger if isinstance(r.get("score"), (int, float))]
    seated = [r for r in scored if r["score"] >= SEATED]
    cutting = [r for r in scored if r["score"] <= CUTTING]
    print(f"\nИТОГ: измерено {len(scored)} поверхностей | сидят {len(seated)} "
          f"({100*len(seated)/max(len(scored),1):.0f}%) | не сидят {len(cutting)} "
          f"({100*len(cutting)/max(len(scored),1):.0f}%)", flush=True)


if __name__ == "__main__":
    main()
