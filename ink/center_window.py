"""Is the depth window on the sheet, or in the gap between windings?

An automatically grown surface is not guaranteed to sit on papyrus. It can run parallel to
a sheet and a little beside it, and a tracer has no reason to notice: its own objective is
satisfied by following the winding, not by landing on the material. The render that follows
is then a picture of the gap. It looks like papyrus, it has texture and contrast, and an ink
model over it reports nothing -- correctly, because there is no ink in air.

Measured on PHerc1451: a surface whose 62-layer window was centred at the mesh found the
sheet peak at **-149 um**, twice the window's own half-width of 74 um. Every one of that
scroll's canvases was a picture of the gap, and the survey's clean negative meant nothing.
Nothing in the render, the seating score or the ink map said so.

The test is one cheap render and one array. Sample the same surface over a window several
times wider than the one you intend to use -- +-360 um reaches past a winding either side --
and take the mean brightness per layer. Papyrus is denser than the gap, so the sheet is a
peak. Where that peak sits relative to the centre is the offset the render needs.

    from center_window import depth_profile, sheet_offset, window_verdict
    prof = depth_profile(deep_stack)                 # +-150 layers, coarse in plane is fine
    zoff, contrast = sheet_offset(prof)              # layers to shift the window by
    ...
    window_verdict(depth_profile(stack))             # on an ordinary 62-layer stack

`window_verdict` is the retrospective half: run it on a render you already have and it says
whether that render ever saw a sheet, without another pass over the volume. A window whose
brightest layer is at its own edge is not centred on anything -- the sheet is outside it.
"""
from __future__ import annotations

import numpy as np

FLAT_CONTRAST = 0.10    # below this the window holds no sheet/gap structure at all
EDGE_LAYERS = 4         # a peak this close to either end means the sheet is outside


def depth_profile(stack, sub=8):
    """Mean brightness per layer, subsampled in plane.

    In-plane detail is irrelevant here and subsampling by 8 makes this cheap enough to run
    before every render.
    """
    a = np.asarray(stack)
    return np.array([a[i, ::sub, ::sub].astype(np.float32).mean() for i in range(a.shape[0])])


def _smooth(profile, k=5):
    """Boxcar, padded by repeating the end values.

    Zero-padding would be wrong here in a way that matters: it drops the first and last
    samples toward zero and manufactures a dark edge, which is exactly the feature the
    'edge' verdict below reads. A profile of pure noise scored a false sheet offset until
    this padding was fixed.
    """
    if k <= 1 or profile.size < k:
        return profile
    pad = k // 2
    padded = np.concatenate([np.full(pad, profile[0]), profile, np.full(pad, profile[-1])])
    kern = np.ones(k, np.float32) / k
    return np.convolve(padded, kern, mode="valid")


def sheet_offset(profile, max_shift=None):
    """Layers from the window's centre to the nearest sheet, and the sheet's contrast.

    Returns (offset, contrast). The offset is signed in layers -- add it to the render's
    depth offset to centre the next window on the sheet. Contrast is (peak - floor) / peak
    over the searched span; below FLAT_CONTRAST there is no sheet to centre on and the
    offset returned is 0.

    The *nearest* peak, not the brightest: at a winding period of about 700 um a wide
    enough probe sees more than one sheet, and the neighbouring winding is the wrong
    answer even when it happens to be denser.
    """
    p = _smooth(np.asarray(profile, np.float32))
    n = p.size
    mid = (n - 1) / 2.0
    if max_shift is not None:
        lo, hi = int(max(0, mid - max_shift)), int(min(n, mid + max_shift) + 1)
        p_search = p[lo:hi]
        base = lo
    else:
        p_search, base = p, 0
    floor, peak = float(p_search.min()), float(p_search.max())
    contrast = (peak - floor) / max(peak, 1e-6)
    if contrast < FLAT_CONTRAST:
        return 0, contrast
    # local maxima, then the one closest to the centre
    interior = np.arange(1, p_search.size - 1)
    rising = p_search[interior] >= p_search[interior - 1]
    falling = p_search[interior] >= p_search[interior + 1]
    peaks = interior[rising & falling]
    # only peaks that are actually sheets, not ripples on the gap
    peaks = peaks[p_search[peaks] >= floor + 0.5 * (peak - floor)]
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(p_search))])
    idx = base + int(peaks[np.argmin(np.abs(base + peaks - mid))])
    return int(round(idx - mid)), contrast


def window_verdict(profile):
    """What a render already in hand actually saw: 'centred', 'edge' or 'flat'.

    'flat'    -- the window never left material or never entered it; no sheet in range.
    'edge'    -- the brightest layer is at the window's own boundary, so the sheet is
                 outside it and the render is of the approach to a sheet, not of one.
    'centred' -- there is a sheet and the window is on it.
    """
    p = _smooth(np.asarray(profile, np.float32))
    n = p.size
    floor, peak = float(p.min()), float(p.max())
    contrast = (peak - floor) / max(peak, 1e-6)
    if contrast < FLAT_CONTRAST:
        return "flat", 0, contrast
    idx = int(np.argmax(p))
    off = int(round(idx - (n - 1) / 2.0))
    if idx < EDGE_LAYERS or idx > n - 1 - EDGE_LAYERS:
        return "edge", off, contrast
    return "centred", off, contrast
