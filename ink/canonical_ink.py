"""Run a published Vesuvius ink checkpoint over a rendered layer stack.

The convention the checkpoints expect: 62 layers, uint8 clipped to [0, 200] then scaled
to [0, 1], 256-pixel tiles at stride 128, logits at quarter resolution. Depth in the
published surface volumes runs opposite to the geometric normal, so a stack rendered
along the normal must be reversed.
"""
from __future__ import annotations

import os

import numpy as np
import torch

CLIP = 200
_MODELS: dict = {}


def device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load(checkpoint: str, dev: torch.device, frames: int, loader):
    key = (checkpoint, str(dev), frames)
    if key not in _MODELS:
        net = loader(checkpoint, dev, num_frames=frames)
        net.eval()
        _MODELS[key] = net
    return _MODELS[key]


def predict(stack, checkpoint, loader, dev=None, reverse=True, tile=256, stride=128,
            batch=None, denom=CLIP):
    """stack: (layers, H, W) uint8 -> (H, W) float32 ink probability."""
    dev = device(dev)
    batch = batch or int(os.environ.get("BATCH", "8" if dev.type == "cuda" else "1"))
    net = load(checkpoint, dev, stack.shape[0], loader)
    _, height, width = stack.shape
    total = np.zeros((height, width), np.float32)
    weight = np.zeros((height, width), np.float32)
    window = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32) + 1e-3
    tiles = [(y, x)
             for y in range(0, max(height - tile, 0) + 1, stride)
             for x in range(0, max(width - tile, 0) + 1, stride)
             if stack[:, y:y + tile, x:x + tile].shape[1:] == (tile, tile)
             and (stack[:, y:y + tile, x:x + tile] != 0).any(0).mean() >= 0.05]
    autocast = torch.autocast(dev.type, dtype=torch.float16, enabled=(dev.type == "cuda"))
    with torch.no_grad(), autocast:
        for start in range(0, len(tiles), batch):
            group = tiles[start:start + batch]
            block = np.empty((len(group), stack.shape[0], tile, tile), np.float32)
            for k, (y, x) in enumerate(group):
                patch = np.clip(stack[:, y:y + tile, x:x + tile], 0, CLIP).astype(np.float32) / denom
                block[k] = patch[::-1] if reverse else patch
            out = net.forward(torch.from_numpy(block)[:, None].to(dev))
            if isinstance(out, (list, tuple)):
                out = out[0]
            out = getattr(out, "logits", out)
            probs = torch.nn.functional.interpolate(
                torch.sigmoid(out if out.ndim == 4 else out[:, None]),
                size=(tile, tile), mode="bilinear", align_corners=False)[:, 0]
            probs = probs.float().cpu().numpy()
            for k, (y, x) in enumerate(group):
                total[y:y + tile, x:x + tile] += probs[k] * window
                weight[y:y + tile, x:x + tile] += window
    return np.divide(total, weight, out=np.zeros_like(total), where=weight > 0)
