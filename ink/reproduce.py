"""One command that checks the whole claim: render a window of a scroll the team has already
published an ink map for, run the published checkpoint over it, and print the correlation.

    python ink/reproduce.py            # PHerc0139, 78 keV, downloads what it needs

Nothing is kept locally except the checkpoint. Expect r ~= 0.96 against the team's own
surface volume, which is what "the pipeline is right" means here.
"""
from __future__ import annotations

import io
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canonical_ink import predict  # noqa: E402

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
SCROLL = "PHerc0139"
SEGMENT = "20250108000000-w025_2025010863"
SURFACE = f"{BUCKET}{SCROLL}/segments/{SEGMENT}/surface-volumes/2.399um-0.22m-78keV-volume-20260102150214.zarr/0/"
INK_MAP = (f"{BUCKET}{SCROLL}/segments/{SEGMENT}/ink-detection/"
           f"{SCROLL}-20250108000000-2.399um-0.22m-78keV-volume-20260102150214-20260417190342-"
           f"new_canon_autoresearch_recipe-tile256-stride128.tif")
CHECKPOINT_URL = ("https://huggingface.co/scrollprize/ink_canonical_2um/resolve/main/"
                  "r152_3ddec_v2_l5_epoch13.ckpt")
LAYERS = (24, 86)          # откалибровано: центральные 62 слоя из 109
WINDOW = 512


def fetch_checkpoint(path="r152.ckpt"):
    if not os.path.exists(path):
        print("качаю чекпойнт (1.4 ГБ, один раз)...", flush=True)
        urllib.request.urlretrieve(CHECKPOINT_URL, path)
    return path


def fetch_surface_window(y0, x0, size):
    import numcodecs
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=24, max_retries=3))
    meta = session.get(SURFACE + ".zarray", timeout=60).json()
    shape = np.array(meta["shape"])
    chunks = np.array(meta["chunks"])
    sep = meta.get("dimension_separator", "/")
    codec = numcodecs.get_codec(meta["compressor"]) if meta["compressor"] else None
    out = np.zeros((shape[0], size, size), np.uint8)
    keys = [(0, b, c)
            for b in range(y0 // chunks[1], (y0 + size - 1) // chunks[1] + 1)
            for c in range(x0 // chunks[2], (x0 + size - 1) // chunks[2] + 1)]

    def get(key):
        r = session.get(SURFACE + sep.join(map(str, key)), timeout=240)
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

    with ThreadPoolExecutor(12) as pool:
        list(pool.map(get, keys))
    return out


def main():
    from model_resnet3d_3d_decoder import load_model
    checkpoint = fetch_checkpoint()
    print("скачиваю опубликованную карту чернил...", flush=True)
    published = tifffile.imread(io.BytesIO(requests.get(INK_MAP, timeout=900).content))
    # окно с умеренной долей чернил: у насыщенных корреляция ни о чём не говорит
    best = None
    step = 2048
    for y in range(0, published.shape[0] - WINDOW, step):
        for x in range(0, published.shape[1] - WINDOW, step):
            share = (published[y:y + WINDOW, x:x + WINDOW] > 200).mean()
            if 0.10 < share < 0.35 and (best is None or share > best[0]):
                best = (share, y, x)
    if best is None:
        print("не нашёл окна с умеренной долей чернил"); return 1
    share, y0, x0 = best
    print(f"окно y={y0} x={x0}, чернил по их карте {share*100:.1f}%", flush=True)
    stack = fetch_surface_window(y0, x0, WINDOW)
    print(f"поверхностный объём {stack.shape}, беру слои {LAYERS[0]}-{LAYERS[1]}", flush=True)
    ours = predict(stack[LAYERS[0]:LAYERS[1]], checkpoint, load_model, reverse=False)
    theirs = published[y0:y0 + WINDOW, x0:x0 + WINDOW].astype(np.float32) / 255.0
    r = float(np.corrcoef(ours.ravel(), theirs.ravel())[0, 1])
    print(f"\nкорреляция с опубликованной картой: r = {r:+.3f}")
    print("ожидается около +0.9; ниже 0.5 означает, что одно из четырёх соглашений нарушено")
    return 0 if r > 0.5 else 1


if __name__ == "__main__":
    sys.path.insert(0, os.environ.get("VILLA_INFERENCE",
                                      "/Users/pc/defi/vesuvius/villa/ink-detection/optimized_inference"))
    sys.exit(main())
