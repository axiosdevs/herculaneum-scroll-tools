"""Score an ink map by how much it looks like writing rather than material.

Ink fraction is a poor guide: the highest-coverage windows are broad material responses
and the sparsest are isolated specks. Script instead shows up as periodic rows. The score
is the share of profile energy concentrated at one period in the 1.0-3.5 mm band, gated on
at least three resolved rows. Calibrated against windows of PHerc0139 where the answer is
known: AUC 0.885, with 92% of blank windows scoring exactly zero.
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_ROWS = 3
PERIOD_MM = (1.0, 3.5)


def text_score(prob_map, micron_per_pixel=2.399, threshold=0.5):
    """Return (score, period_mm). Score 0 means "no rows resolved"."""
    mask = (prob_map > threshold).astype(np.float32)
    coverage = mask.mean()
    if coverage < 0.01 or coverage > 0.85:
        return 0.0, 0.0
    best = (0.0, 0.0)
    for turn in range(4):
        view = np.rot90(mask, turn) if turn < 2 else np.rot90(mask.T, turn - 2)
        profile = view.mean(1)
        profile = profile - cv2.GaussianBlur(profile.reshape(-1, 1), (1, 301), 0).ravel()
        if profile.std() < 1e-5:
            continue
        windowed = (profile - profile.mean()) * np.hanning(len(profile))
        power = np.abs(np.fft.rfft(windowed)) ** 2
        freq = np.fft.rfftfreq(len(windowed), d=micron_per_pixel / 1000)
        band = (freq > 1 / PERIOD_MM[1]) & (freq < 1 / PERIOD_MM[0])
        if not band.any():
            continue
        peak = int(np.argmax(np.where(band, power, 0)))
        half = max(2, len(power) // 200)
        purity = float(power[max(1, peak - half):peak + half + 1].sum() / (power[1:].sum() + 1e-9))
        period = 1 / freq[peak] if freq[peak] > 0 else 0.0
        smooth = cv2.GaussianBlur(view.mean(1).reshape(-1, 1), (1, 31), 0).ravel()
        smooth = smooth - cv2.GaussianBlur(smooth.reshape(-1, 1), (1, 301), 0).ravel()
        gap = max(3, int(period * 1000 / micron_per_pixel * 0.4))
        rows = [i for i in range(gap, len(smooth) - gap)
                if smooth[i] == smooth[max(0, i - gap):i + gap + 1].max()
                and smooth[i] > 0.5 * smooth.std()]
        if len(rows) < MIN_ROWS:
            continue
        if purity > best[0]:
            best = (purity, period)
    return best
