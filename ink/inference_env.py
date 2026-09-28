"""Make the villa model loader importable without a local villa checkout.

The checkpoint loader lives in ScrollPrize/villa (ink-detection/optimized_inference). A judge
running `python ink/reproduce.py` from a fresh clone should not need to clone villa first, so:
use VILLA_INFERENCE if set (point it at a villa checkout's ink-detection/optimized_inference),
else fetch the loader files from villa's main branch into a cache directory once.
"""
from __future__ import annotations

import os
import sys
import urllib.request

RAW = "https://raw.githubusercontent.com/ScrollPrize/villa/main/ink-detection/optimized_inference/"
FILES = ("model_resnet3d_3d_decoder.py", "models/__init__.py",
         "models/resnetall.py", "models/non_local_helper.py")


def ensure() -> str:
    override = os.environ.get("VILLA_INFERENCE")
    if override and os.path.exists(os.path.join(override, "model_resnet3d_3d_decoder.py")):
        sys.path.insert(0, override)
        return override
    cache = os.path.join(os.path.expanduser("~"), ".cache", "scroll-tools-inference")
    os.makedirs(os.path.join(cache, "models"), exist_ok=True)
    for name in FILES:
        path = os.path.join(cache, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(RAW + name, path)
    sys.path.insert(0, cache)
    return cache
