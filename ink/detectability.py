"""Would this window have shown ink, if there were any? Plant some and see.

Every null result in this field is ambiguous. "The detector found nothing here" can mean
the papyrus is blank, or it can mean the surface is off the sheet, the depth window is
misplaced, the scan is too coarse, or the model is out of its domain -- and nothing in the
output distinguishes them. I have twice taken the second for the first.

This settles it per window. Take the rendered stack, add a synthetic ink layer of known
strength where ink would physically sit -- on the sheet's face, a couple of voxels thick,
following the sheet rather than a flat plane -- and run the same model again. If the
planted ink comes back, the window is readable and its emptiness is a real observation. If
it does not, the window is blind and its emptiness says nothing at all.

The output is a sensitivity threshold in the units the physics is in: the smallest planted
contrast, in CT intensity, that the model recovers above a chosen margin. Calibrated on
PHerc0139, where blank regions are known from the team's published map, it says what this
pipeline can and cannot see; carried to an unread scroll, it turns "no text found" into
"no text found, and here is the strength of text that would have been found".

    from detectability import plant_ink, probe
    result = probe(stack, predict_fn, amplitudes=(4, 8, 16, 32))

`predict_fn(stack) -> probability map` is whatever inference path is already in use, so the
probe measures the pipeline as it actually runs rather than an idealised copy of it.

**A sensitivity gained by preprocessing is not a sensitivity gained.** The probe answers for
the pipeline it is handed, and a pipeline can be made more sensitive to planted ink while
becoming worse at real letters. Measured: smoothing our PHerc1451 stack with a Gaussian far
narrower than a stroke -- sigma 19 um against a 350 um stroke -- took its planted-ink
threshold from nothing at all to 16, better than the unsmoothed reference. The same smoothing
applied to the team's PHerc0139 surface volume took agreement with their published ink map
from **0.858 down to 0.448**, and the ink fraction from 2.0% to 6.6%. The model reads
something finer than stroke shape, and the probe cannot see that being destroyed.

So any change that moves this number should be checked against a reference where the truth is
published, and `reproduce.py` exists for exactly that. The probe tells you a window is blind;
it does not tell you a pipeline is good.

**What it is good for is settling a choice no other signal can.** The layer order along the
normal -- which way the model reads the sheet -- is a free parameter every survey has to fix,
and `orientation` fixes it without labels: plant ink on each face, read both ways, and the one
combination that answers positively is the order and the face. On PHerc0139 as published it
says forward, which the team's ink map confirms; on the same stack flipped in depth it says
reverse, which it has to. Neither the amount of ink reported nor a periodicity score over it
can do this -- the wrong order reported 20 times less ink than the right one on PHerc0139 and
8 times more on a PHerc1451 surface.

A retraction belongs here. An earlier version of this docstring said the wrong order always
reports more ink, from that one PHerc1451 surface. The flipped reference refutes it, and the
same test showed why the claim could not have been checked with this module as it then was:
it planted ink on one face only, and so reported a readable stack whose ink is on the other
face as blind.
"""
from __future__ import annotations

import numpy as np

SHEET_BAND = 6          # voxels either side of the sheet centre that count as its face
STROKE_MM = 0.35        # typical Herculaneum stroke width
LINE_MM = 2.0           # typical line spacing


def sheet_depth(stack, search=12):
    """Per-pixel depth of the sheet centre, from the stack's own brightness.

    Taken over a band around the middle: the neighbouring windings are only a few hundred
    microns away and a wide band would find one of those instead.
    """
    C = stack.shape[0]
    mid = (C - 1) / 2.0
    lo, hi = int(max(0, mid - search)), int(min(C, mid + search) + 1)
    band = stack[lo:hi].astype(np.float32)
    depth = np.arange(lo, hi, dtype=np.float32)[:, None, None]
    base = np.percentile(band, 20, axis=0, keepdims=True)
    w = np.clip(band - base, 0, None)
    tot = w.sum(0)
    return np.where(tot > 1e-6, (w * depth).sum(0) / np.maximum(tot, 1e-6), mid)


def script_mask(shape, micron_per_pixel=2.401, seed=0):
    """A plausible page of writing: strokes of the right width, arranged in lines."""
    rng = np.random.default_rng(seed)
    h, w = shape
    stroke = max(2, int(round(STROKE_MM * 1000 / micron_per_pixel)))
    line = max(stroke * 3, int(round(LINE_MM * 1000 / micron_per_pixel)))
    mask = np.zeros(shape, np.float32)
    if h < line * 2 or w < stroke * 8:
        raise ValueError(
            f"окно {h}x{w} px мало для пробы: при шаге строк {LINE_MM} мм "
            f"({line} px) в него не помещается ни одной строки")
    for y in range(line, h - line, line):
        x = rng.integers(0, line)
        while x < w - stroke * 4:
            glyph_w = rng.integers(stroke * 2, stroke * 4)
            glyph_h = rng.integers(stroke * 2, stroke * 3)
            y0 = y + int(rng.integers(-stroke, stroke))
            if 0 <= y0 < h - glyph_h:
                # a hollow-ish mark: two verticals and a bar, closer to a letter than a blob
                mask[y0:y0 + glyph_h, x:x + stroke] = 1.0
                mask[y0:y0 + glyph_h, x + glyph_w - stroke:x + glyph_w] = 1.0
                mid = y0 + glyph_h // 2
                mask[mid:mid + stroke, x:x + glyph_w] = 1.0
            x += glyph_w + int(rng.integers(stroke, stroke * 3))
    return mask


def plant_ink(stack, amplitude, micron_per_pixel=2.401, seed=0, band=SHEET_BAND, face="near"):
    """Add a synthetic ink layer of the given contrast on one face of the sheet.

    Ink lies on the surface, not in the middle of the sheet, so the layer follows the measured
    sheet depth rather than a flat plane -- a flat layer would be a test of the renderer's
    geometry, not of the detector. `face` is which side of the sheet's centre it goes on:
    'near' toward layer 0, 'far' toward the last layer.

    Which face is right is a property of the segment, not of this function, and the first
    version of this module hard-coded 'near'. Measured on PHerc0139's published surface volume
    flipped in depth -- a stack that reads its letters perfectly well in reverse -- that made
    the probe report no threshold in either order: its ink was being planted on the face the
    model does not read ink from. `orientation` below plants on both faces.
    """
    if face not in ("near", "far"):
        raise ValueError(f"face must be 'near' or 'far', not {face!r}")
    out = stack.astype(np.float32).copy()
    C, H, W = out.shape
    centre = sheet_depth(stack)
    mask = script_mask((H, W), micron_per_pixel, seed)
    face = centre - band / 2.0 if face == "near" else centre + band / 2.0
    zz = np.arange(C, dtype=np.float32)[:, None, None]
    profile = np.exp(-0.5 * ((zz - face[None]) / (band / 2.0)) ** 2)
    out += amplitude * profile * mask[None]
    return np.clip(out, 0, 255).astype(stack.dtype), mask


def probe(stack, predict_fn, amplitudes=(4, 8, 16, 32), micron_per_pixel=2.401,
          seed=0, threshold=0.5, margin=0.05, face="near"):
    """Smallest planted contrast this pipeline recovers on this window.

    Returns a dict with one row per amplitude -- the correlation between the recovered map
    and the planted mask, and the rise in ink fraction inside the strokes over the same
    pixels before planting --
    plus `sensitivity`, the lowest amplitude whose *ink lift* clears `margin`.

    The lift is the decision, not the correlation. Calibrated on PHerc0139, a planted
    contrast of 2 out of 255 already moves the raw probabilities enough for r to read 0.44
    while not one pixel crosses the 0.5 threshold -- a response too faint to be a detection.
    The lift asks the question that matters: did the planted strokes actually come out as
    ink? None means the window is blind, and whatever it reported about real ink carries no
    information.
    """
    base = predict_fn(stack)
    rows = []
    sensitivity = None
    for amp in amplitudes:
        planted, mask = plant_ink(stack, amp, micron_per_pixel, seed, face=face)
        got = predict_fn(planted)
        delta = got.astype(np.float32) - base.astype(np.float32)
        m = mask.astype(bool)
        if m.sum() < 16 or (~m).sum() < 16:
            continue
        # a detector that did not move at all has no correlation to report, and
        # corrcoef would hand back a NaN that reads as a missing measurement
        r = 0.0 if delta.std() < 1e-12 else float(
            np.corrcoef(delta.ravel(), mask.ravel())[0, 1])
        # The lift has to be the change planting caused, not a comparison of one map with
        # itself: inside-versus-outside on a single map measures whatever was already there.
        # Measured on PHerc1451 that confound gave the same lift at every amplitude, and a
        # negative one where the baseline happened to be inkier outside the strokes.
        lift = float((got[m] > threshold).mean() - (base[m] > threshold).mean())
        rows.append({"amplitude": float(amp), "r_with_planted": round(r, 3),
                     "ink_lift": round(lift, 3)})
        if sensitivity is None and lift >= margin:
            sensitivity = float(amp)
    return {"sensitivity": sensitivity, "baseline_ink_pct": round(float((base > threshold).mean()) * 100, 2),
            "rows": rows}


def orientation(stack, predict_for, amplitude=32, micron_per_pixel=2.401, seed=0,
                threshold=0.5, margin=0.02, gap=0.01):
    """Which way to read this stack, and on which face its ink should be -- without labels.

    `predict_for(reverse)` returns the pipeline's predict function for one layer order. Ink is
    planted on each face in turn and the stack read both ways, four runs in all. On a stack the
    model can read, one combination answers with a positive lift; the wrong order answers with
    nothing at all, on either face; and the right order with ink on the wrong face answers with
    a *negative* lift, because a bright layer where the model expects bare papyrus reads as the
    absence of ink.

    Measured at amplitude 32 on PHerc0139's published surface volume, as published and flipped
    in depth -- so the right answer is known both ways from the team's own ink map -- and on
    two PHerc1451 surfaces that have no labels at all:

                          fwd near   fwd far   rev near   rev far    verdict
        PHerc0139          +0.056    -0.233     0.000      0.000     forward  (letters read so)
        same, flipped       0.000     0.000    -0.091     +0.025     reverse  (must be, by construction)
        PHerc1451 r5/014   +0.231    +0.048     0.000     +0.001     forward
        PHerc1451 r6/056   +0.183    +0.006    -0.161     -0.066     forward

    The flipped row is the one that matters: the verdict follows the data and not the side the
    ink is planted on, which is what a one-face probe could not show.

    Two things this is *not* measured by. The amount of ink the model reports says nothing about
    order: reading the wrong way reported 20 times less ink than the right way on PHerc0139, and
    8 times more on r6/056. And a probe planting on one face only -- the first version of this
    module -- answers 'blind' on a perfectly readable stack whose ink is on the other face.

    Returns every combination's lift and baseline ink, and `verdict`: the winning (order, face)
    if its lift clears `margin` and beats the runner-up by `gap`, else None.
    """
    combos = {}
    for reverse in (False, True):
        fn = predict_for(reverse)
        for face in ("near", "far"):
            res = probe(stack, fn, amplitudes=(amplitude,), micron_per_pixel=micron_per_pixel,
                        seed=seed, threshold=threshold, margin=margin, face=face)
            lift = res["rows"][0]["ink_lift"] if res["rows"] else 0.0
            combos[("reverse" if reverse else "forward", face)] = {
                "lift": lift, "baseline_ink_pct": res["baseline_ink_pct"]}
    ranked = sorted(combos.items(), key=lambda kv: kv[1]["lift"], reverse=True)
    best, runner = ranked[0], ranked[1]
    clear = best[1]["lift"] >= margin and best[1]["lift"] - runner[1]["lift"] >= gap
    verdict = best[0] if clear else None
    return {"verdict": verdict, "amplitude": float(amplitude),
            "combinations": {f"{o} {f}": v for (o, f), v in combos.items()}}
