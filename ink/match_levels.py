"""Put a rendered stack on the intensity scale the ink checkpoint responds to.

This rescale works, and the reason it works is not known. Both halves are stated because
the first is useful and the second is what stops the next person repeating my week.

Measured on one PHerc1451 surface, planting synthetic ink of known contrast and asking
whether the model recovers it (ink/detectability.py):

    as rendered   plant 8 -> +0.006 lift, r 0.15   sensitivity: none
    level-matched plant 8 -> +0.069 lift, r 0.53   sensitivity: 8

An elevenfold difference from one rescale. Before it, thirty-five canvases on that scroll
read as blank, and every one was the model saying nothing rather than the papyrus being
empty. A survey run without a sensitivity check of some kind is not evidence of absence.

**The explanation I first gave here was wrong.** I argued that `ink_canonical_2um`
normalises as clip(0, 200) / 200 -- a fixed divisor assuming the brightness of its training
scans -- so handing it a median of 128 where it learned 42 puts it off its domain. Two
measurements refute that:

    team stack brightened to median 128        r 0.994 with their published ink map
    team stack contrast scaled 0.25x - 2.0x    r 0.918 - 0.996

The model tolerates three times the brightness and four times the contrast range without
losing its letters. Whatever this remap is fixing on our renders, it is not that, and the
difference between our stacks and theirs is still open.

The mapping is piecewise-linear through the 5th, 50th and 95th percentiles of a reference
scan.

    from match_levels import match_levels
    stack = match_levels(stack)          # then infer as usual
"""
from __future__ import annotations

import numpy as np

# 5th, 50th and 95th percentiles of the team's published PHerc0139 surface volume, the scan
# this checkpoint demonstrably reads (our renderer reproduces their ink map there at 0.963,
# and their own stack through this model reaches 0.996).
REFERENCE = (29.0, 42.0, 128.0)


def levels(stack):
    """The three percentiles this module works in."""
    return tuple(float(v) for v in np.percentile(np.asarray(stack), [5, 50, 95]))


def match_levels(stack, reference=REFERENCE):
    """Rescale so the stack's 5/50/95 percentiles sit on the reference's."""
    a = np.asarray(stack, np.float32)
    p5, p50, p95 = np.percentile(a, [5, 50, 95])
    r5, r50, r95 = reference
    out = np.empty_like(a)
    low = a <= p50
    out[low] = r5 + (a[low] - p5) * (r50 - r5) / max(p50 - p5, 1e-6)
    out[~low] = r50 + (a[~low] - p50) * (r95 - r50) / max(p95 - p50, 1e-6)
    return np.clip(out, 0, 255).astype(np.asarray(stack).dtype)
