"""Measure where ink is detectable across the whole published corpus.

For every scroll that carries a surface volume, sample a few windows, run the published 2 um
ink checkpoint over them, and record what comes back. The point is not one scroll but the
table: which scans support ink recovery at all, and which do not — the answer decides where
scanning and segmentation effort is worth spending.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canonical_ink import predict          # noqa: E402
from text_score import text_score          # noqa: E402

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
WINDOW = int(os.environ.get("WINDOW", "512"))
PER_SCROLL = int(os.environ.get("PER_SCROLL", "3"))


def listing(prefix, delim="/"):
    query = {"list-type": "2", "prefix": prefix, "max-keys": "1000", "delimiter": delim}
    text = urllib.request.urlopen(BUCKET + "?" + urllib.parse.urlencode(query), timeout=120).read().decode()
    return [p for p in re.findall(r"<(?:Key|Prefix)>([^<]*)</(?:Key|Prefix)>", text) if p != prefix]


def micron_of(name):
    m = re.search(r"(\d+\.?\d*)um", name)
    return float(m.group(1)) if m else None


def kev_of(name):
    m = re.search(r"(\d+)keV", name)
    return int(m.group(1)) if m else None


def window_from(url, y0, x0, size):
    import numcodecs
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=24, max_retries=3))
    meta = session.get(url + ".zarray", timeout=60).json()
    shape = np.array(meta["shape"]); chunks = np.array(meta["chunks"])
    sep = meta.get("dimension_separator", "/")
    codec = numcodecs.get_codec(meta["compressor"]) if meta["compressor"] else None
    out = np.zeros((shape[0], size, size), np.uint8)
    keys = [(0, b, c)
            for b in range(y0 // chunks[1], (y0 + size - 1) // chunks[1] + 1)
            for c in range(x0 // chunks[2], (x0 + size - 1) // chunks[2] + 1)]

    def get(key):
        for _ in range(3):
            try:
                r = session.get(url + sep.join(map(str, key)), timeout=180)
                if r.status_code != 200:
                    return
                raw = codec.decode(r.content) if codec else r.content
                buf = np.frombuffer(raw, np.uint8)
                if buf.size != int(np.prod(chunks)):
                    return
                block = buf.reshape(tuple(chunks))
                _, yy, xx = np.array(key) * chunks
                ys, ye = max(y0, yy), min(y0 + size, yy + chunks[1])
                xs, xe = max(x0, xx), min(x0 + size, xx + chunks[2])
                if ye > ys and xe > xs:
                    out[:, ys - y0:ye - y0, xs - x0:xe - x0] = block[:, ys - yy:ye - yy, xs - xx:xe - xx]
                return
            except Exception:
                continue

    with ThreadPoolExecutor(12) as pool:
        list(pool.map(get, keys))
    return out, shape


def central_62(stack):
    depth = stack.shape[0]
    if depth >= 62:
        start = (depth - 62) // 2
        return stack[start:start + 62]
    pad = np.zeros((62, stack.shape[1], stack.shape[2]), np.uint8)
    pad[(62 - depth) // 2:(62 - depth) // 2 + depth] = stack
    return pad


def main():
    from model_resnet3d_3d_decoder import load_model
    checkpoint = os.environ.get("CKPT", "r152.ckpt")
    out_path = os.environ.get("OUT", "ink/atlas.json")
    rows = json.load(open(out_path)) if os.path.exists(out_path) else []
    done = {(r["scroll"], r["segment"], r["window"]) for r in rows}
    scrolls = [p.rstrip("/").split("/")[-1] for p in listing("", "/") if p.startswith("PHerc")]
    print(f"объектов в корпусе: {len(scrolls)}", flush=True)
    for scroll in scrolls:
        segments = [p.rstrip("/").split("/")[-1] for p in listing(f"{scroll}/segments/")]
        picked = None
        for segment in segments:
            volumes = [p for p in listing(f"{scroll}/segments/{segment}/surface-volumes/")
                       if p.endswith(".zarr/")]
            if volumes:
                volumes.sort(key=lambda p: micron_of(p) or 99)
                picked = (segment, volumes[0])
                break
        if not picked:
            print(f"{scroll}: поверхностных объёмов нет", flush=True)
            continue
        segment, volume = picked
        name = volume.rstrip("/").split("/")[-1]
        url = BUCKET + volume + "0/"
        try:
            probe, shape = window_from(url, 0, 0, 64)
        except Exception as exc:
            print(f"{scroll}: объём не читается ({exc})", flush=True)
            continue
        print(f"{scroll} / {segment[:28]} / {name[:40]} {tuple(shape)}", flush=True)
        rng = np.random.default_rng(0)
        tried = 0
        for _ in range(PER_SCROLL * 6):
            if tried >= PER_SCROLL:
                break
            y0 = int(rng.integers(0, max(1, shape[1] - WINDOW)))
            x0 = int(rng.integers(0, max(1, shape[2] - WINDOW)))
            key = (scroll, segment, f"{y0}_{x0}")
            if key in done:
                continue
            stack, _ = window_from(url, y0, x0, WINDOW)
            if (stack > 0).mean() < 0.5:
                continue
            tried += 1
            probs = predict(central_62(stack), checkpoint, load_model, reverse=False)
            um = micron_of(name) or 2.4
            # признак текста требует не меньше трёх строк в кадре: окно от ~4 мм
            span_mm = WINDOW * um / 1000.0
            if span_mm >= 4.0:
                score, period = text_score(probs, micron_per_pixel=um)
            else:
                score, period = float("nan"), float("nan")
            row = {
                "scroll": scroll, "segment": segment, "window": f"{y0}_{x0}",
                "volume": name, "micron": micron_of(name), "kev": kev_of(name),
                "layers": int(stack.shape[0]),
                "max_probability": round(float(probs.max()), 4),
                "ink_fraction": round(float((probs > 0.5).mean()), 5),
                "window_mm": round(float(span_mm), 2),
                "text_score": None if score != score else round(float(score), 4),
                "line_period_mm": None if period != period else round(float(period), 3),
            }
            rows.append(row)
            json.dump(rows, open(out_path, "w"), indent=1)
            ts = "—" if row["text_score"] is None else f"{row['text_score']:.3f}"
            print(f"    окно {y0},{x0} ({row['window_mm']} мм): макс {row['max_probability']:.2f} "
                  f"чернил {row['ink_fraction']*100:5.2f}% текст {ts}", flush=True)
    print("АТЛАС ГОТОВ", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.environ.get("VILLA_INFERENCE",
                                      "/Users/pc/defi/vesuvius/villa/ink-detection/optimized_inference"))
    main()
