"""Check the reproduction claim at scale, not on one lucky window.

For a scroll that publishes both a surface volume and the team's own ink map, sample many
windows, run the checkpoint over each, and correlate against the published map. One agreement
number proves nothing; a distribution does.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canonical_ink import predict  # noqa: E402

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
WINDOW = 512
LAYERS = (24, 86)


def listing(prefix, delim="/"):
    query = {"list-type": "2", "prefix": prefix, "max-keys": "1000", "delimiter": delim}
    text = urllib.request.urlopen(BUCKET + "?" + urllib.parse.urlencode(query), timeout=120).read().decode()
    return [p for p in re.findall(r"<(?:Key|Prefix)>([^<]*)</(?:Key|Prefix)>", text) if p != prefix]


def surface_window(url, y0, x0, size):
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
    return out


def main():
    from model_resnet3d_3d_decoder import load_model
    scroll = sys.argv[1]
    per = int(os.environ.get("PER", "6"))
    checkpoint = os.environ.get("CKPT", "r152.ckpt")
    out_path = os.environ.get("OUT", f"ink/validate_{scroll}.json")
    rows = json.load(open(out_path)) if os.path.exists(out_path) else []
    seen = {(r["segment"], r["window"]) for r in rows}
    for segment in [p.rstrip("/").split("/")[-1] for p in listing(f"{scroll}/segments/")]:
        inks = [k for k in listing(f"{scroll}/segments/{segment}/ink-detection/", "")
                if k.endswith(".tif")]
        volumes = [p for p in listing(f"{scroll}/segments/{segment}/surface-volumes/")
                   if p.endswith(".zarr/")]
        pairs = []
        for i in inks:
            m = re.search(r"volume-(\d+)", i)
            if m:
                pairs += [(i, v) for v in volumes if m.group(1) in v]
        if not pairs:
            continue
        ink_key, volume = pairs[0]
        url = BUCKET + volume + "0/"
        try:
            published = tifffile.imread(io.BytesIO(requests.get(BUCKET + ink_key, timeout=900).content))
        except Exception as exc:
            print(f"{segment[:34]}: карта не читается ({exc})", flush=True)
            continue
        print(f"{segment[:34]}: карта {published.shape}", flush=True)
        rng = np.random.default_rng(0)
        picked = 0
        for _ in range(per * 10):
            if picked >= per:
                break
            y0 = int(rng.integers(0, max(1, published.shape[0] - WINDOW)))
            x0 = int(rng.integers(0, max(1, published.shape[1] - WINDOW)))
            if (segment, f"{y0}_{x0}") in seen:
                continue
            share = float((published[y0:y0 + WINDOW, x0:x0 + WINDOW] > 200).mean())
            if not (0.05 < share < 0.60):        # насыщенные и пустые окна ничего не проверяют
                continue
            stack = surface_window(url, y0, x0, WINDOW)
            if (stack > 0).mean() < 0.5 or stack.shape[0] < 62:
                continue
            picked += 1
            lo = (stack.shape[0] - 62) // 2 if stack.shape[0] != 109 else LAYERS[0]
            ours = predict(stack[lo:lo + 62], checkpoint, load_model, reverse=False)
            theirs = published[y0:y0 + WINDOW, x0:x0 + WINDOW].astype(np.float32) / 255.0
            r = float(np.corrcoef(ours.ravel(), theirs.ravel())[0, 1])
            rows.append({"scroll": scroll, "segment": segment, "window": f"{y0}_{x0}",
                         "ink_share": round(share, 4), "r": round(r, 4)})
            json.dump(rows, open(out_path, "w"), indent=1)
            print(f"    окно {y0},{x0}: чернил {share*100:4.1f}%  r = {r:+.3f}", flush=True)
    vals = np.array([row["r"] for row in rows])
    if len(vals):
        print(f"\nИТОГ: окон {len(vals)}, медиана r = {np.median(vals):+.3f}, "
              f"минимум {vals.min():+.3f}, доля r>0.7: {100*(vals > 0.7).mean():.0f}%", flush=True)


if __name__ == "__main__":
    from inference_env import ensure
    ensure()
    main()
