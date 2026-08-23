"""Export ink probability maps as 8-bit PNG plus a manifest, so results can be checked."""
from __future__ import annotations

import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_score import text_score  # noqa: E402


def export(pattern, out_dir, scroll, downsample=2):
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for path in sorted(glob.glob(pattern)):
        prob = np.load(path)
        name = os.path.basename(path)[:-4]
        score, period = text_score(prob)
        image = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
        if downsample > 1:
            image = cv2.resize(image, (image.shape[1] // downsample, image.shape[0] // downsample),
                               interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{out_dir}/{name}.png", image)
        manifest.append({
            "scroll": scroll,
            "window": name,
            "shape": list(prob.shape),
            "micron_per_pixel": 2.399,
            "ink_fraction_over_0.5": round(float((prob > 0.5).mean()), 5),
            "max_probability": round(float(prob.max()), 4),
            "text_score": round(float(score), 4),
            "line_period_mm": round(float(period), 3),
            "png_downsample": downsample,
        })
    with open(f"{out_dir}/manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


if __name__ == "__main__":
    rows = export(sys.argv[1], sys.argv[2], sys.argv[3],
                  int(sys.argv[4]) if len(sys.argv) > 4 else 2)
    print(f"выгружено карт: {len(rows)}")
    for r in sorted(rows, key=lambda x: -x["text_score"])[:5]:
        print(f"  {r['window'][:44]:<44} чернил {r['ink_fraction_over_0.5']*100:5.2f}% "
              f"признак текста {r['text_score']:.3f}")
