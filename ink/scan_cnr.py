"""How many grey levels is ink worth on this scan, against this scan's own noise?

Before committing a month to a scroll it is worth knowing whether its scan can carry ink at
all. Brightness cannot tell you and contrast in the large cannot either: what decides it is
how far ink moves a voxel compared with how far noise moves one, at the width of a stroke.

Measured on PHerc0139, one of the scrolls the published `ink_canonical_2um` checkpoint
reads confidently, against that scroll's own published ink map:

    ink is brighter than bare papyrus by   13.3 grey levels
    scatter at a 0.35 mm stroke scale      24.6 grey levels
    single-layer ink CNR                    0.54

That is the whole margin the field works with. One layer is well under a coin toss, and the
reason the recipe reads 62 of them is that averaging buys the factor of about eight that
makes it legible -- roughly 4.3 by the time the model sees anything. Anything that costs a
factor of two here costs more than it looks.

Two measurements, because most scrolls have no published ink map:

    ink_cnr(stack, ink_map)   direct, where a map exists
    sheet_cnr(stack)          proxy, from the sheet-to-gap contrast, needs nothing

On three scrolls, every window centred on its sheet:

    PHerc0009B 77 keV   ink/noise 0.69   sheet/noise 2.33   letters published
    PHerc0139  78 keV   ink/noise 0.54   sheet/noise 1.15   letters published
    PHerc1451  78 keV        --          sheet/noise 0.52   no ink output at all

Both scans with published letters sit between 1.15 and 2.33 on the proxy. PHerc1451's 78 keV
scan -- surface predictions published, no ink output, the scroll this was needed for -- reads
**0.52, below the readable pair by a factor of 2.2**. A planted-ink probe on the same renders
came in at about half the PHerc0139 reference's sensitivity independently, and the factor
survives the choice of filter scale: at 50 um instead of a stroke's width it is 2.83 against
1.43 for those two, the same two to one.

Three points order correctly and that is what the proxy is offered for. They are not enough
to convert a proxy reading into an ink CNR, and this module does not try; it reports the
measurement and the reference scans beside it.

So a null on PHerc1451 is a property of its scan rather than a statement about its papyrus --
a different and more useful sentence than "no text found", and one line of measurement to
obtain before committing a month to a scroll.

    python ink/scan_cnr.py <stack.npy> [ink_map.tif]
"""
from __future__ import annotations

import sys

import numpy as np

STROKE_MM = 0.35        # stroke width on these scrolls
REFERENCE = {           # measured on centred windows, for comparison against a scan in hand
    "PHerc0009B 77 keV (буквы опубликованы)": {"ink_cnr": 0.687, "sheet_cnr": 2.333},
    "PHerc0139 78 keV (буквы опубликованы)": {"ink_cnr": 0.541, "sheet_cnr": 1.153},
    "PHerc1451 78 keV (не прочитан)": {"ink_cnr": None, "sheet_cnr": 0.516},
}


def _blur(a, k):
    try:
        import cv2
        return cv2.GaussianBlur(a, (k, k), 0)
    except Exception:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(a, k / 6.0)


def stroke_noise(layer, micron_per_pixel=2.399):
    """Scatter left after removing everything coarser than a stroke.

    Ink is a stroke-width feature, so structure broader than that -- the sheet, the fibres,
    the illumination of the render -- is not what competes with it. Subtracting a blur at
    that scale leaves what does.
    """
    a = np.asarray(layer, np.float32)
    k = int(round(STROKE_MM * 1000.0 / micron_per_pixel))
    k = max(3, k | 1)
    resid = a - _blur(a, k)
    # A blur has to invent something beyond the edge, and whatever it invents leaves a
    # residual there that is not noise. On a plain gradient that border alone reported a
    # scatter of 0.81 grey levels out of nothing at all.
    m = k // 2
    if a.shape[0] > 2 * m + 8 and a.shape[1] > 2 * m + 8:
        resid, a = resid[m:-m, m:-m], a[m:-m, m:-m]
    live = a > 0
    if live.sum() < 100:
        return float("nan")
    return float(resid[live].std())


def sheet_cnr(stack, micron_per_pixel=2.399, sub=8):
    """Sheet-to-gap contrast over stroke-scale noise. Needs no ink map.

    A proxy, and named as one: it measures the dose the scan delivered to this material,
    which is what sets the ink margin, without claiming to measure ink. Use it to compare
    scans, not to predict a number of letters.

    It is only comparable between windows the sheet is centred in. A window running off the
    sheet has a ramp across it rather than a peak, and the ramp is larger than the sheet's
    own contrast: a PHerc0009B window whose sheet sat at layer 5 of 62 read 3.03, and the
    same data centred reads something else entirely. `window` carries that verdict, and a
    reading taken on anything but 'centred' should not be compared with one that is.
    """
    a = np.asarray(stack)
    prof = np.array([a[i, ::sub, ::sub].astype(np.float32).mean() for i in range(a.shape[0])])
    contrast = float(prof.max() - prof.min())
    layer = np.asarray(a[int(np.argmax(prof))])
    noise = stroke_noise(layer, micron_per_pixel)
    try:
        from center_window import window_verdict
        verdict = window_verdict(prof)[0]
    except Exception:
        verdict = None
    return {"contrast": round(contrast, 2), "noise": round(noise, 2),
            "sheet_cnr": round(contrast / max(noise, 1e-6), 3),
            "sheet_layer": int(np.argmax(prof)), "window": verdict}


def ink_cnr(stack, ink_map, micron_per_pixel=2.399, ink_percentile=90):
    """Ink-to-papyrus separation over stroke-scale noise, from a published ink map.

    Reported at the layer where it is largest rather than at the sheet's centre: the two are
    not the same layer, and on PHerc0139 they sit four apart -- ink is on the face.
    """
    a = np.asarray(stack)
    m = np.asarray(ink_map)
    while m.ndim > 2:
        m = m[0]
    inked = m[m > 0]
    if inked.size == 0:
        return None
    thr = max(float(np.percentile(inked, ink_percentile)), 1.0)
    mask = m >= thr
    if mask.sum() < 1000 or (~mask).sum() < 1000:
        return None
    best = None
    for li in range(a.shape[0]):
        layer = np.asarray(a[li]).astype(np.float32)
        live = layer > 0
        ink_px, bare_px = layer[mask & live], layer[(~mask) & live]
        if ink_px.size < 500 or bare_px.size < 500:
            continue
        contrast = float(ink_px.mean() - bare_px.mean())
        noise = stroke_noise(layer, micron_per_pixel)
        if not np.isfinite(noise) or noise <= 0:
            continue
        cnr = abs(contrast) / noise
        if best is None or cnr > best["ink_cnr"]:
            best = {"layer": li, "contrast": round(contrast, 2),
                    "noise": round(noise, 2), "ink_cnr": round(cnr, 3)}
    return best


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip().splitlines()[-1])
        return 1
    stack = np.load(sys.argv[1], mmap_mode="r")
    s = sheet_cnr(stack)
    print(f"лист/шум {s['sheet_cnr']:.3f}  (перепад {s['contrast']}, шум {s['noise']}, "
          f"слой {s['sheet_layer']})")
    if len(sys.argv) > 2:
        import tifffile
        r = ink_cnr(stack, tifffile.imread(sys.argv[2]))
        if r:
            print(f"чернила/шум {r['ink_cnr']:.3f}  (разделение {r['contrast']:+.2f} "
                  f"уровней на слое {r['layer']})")
    print("\nдля сравнения:")
    for name, ref in REFERENCE.items():
        ink = f"{ref['ink_cnr']:.3f}" if ref["ink_cnr"] else "нет карты"
        print(f"  {name:32s} лист/шум {ref['sheet_cnr']:.2f}  чернила/шум {ink}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
