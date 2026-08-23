"""Recover the map between a coarse scan's coordinate frame and a finer rescan.

Several scrolls carry segmentation only in a coarse 8.6-9.4 um / 113-116 keV frame while
the 77-78 keV rescan that shows ink has no surfaces at all. The two scans of one scroll
differ by a translation with a small linear tilt and no rotation, which is recoverable
from the volumes alone: match pyramid levels to a common voxel size, cross-correlate
occupancy profiles along z, then cross-correlate individual slices in plane.

On PHerc0009B the slices match at r = 0.86 and the resulting map renders clean papyrus in
a volume that had no surfaces.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests


def fetch_volume(url, threads=12):
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=threads * 2, max_retries=3))
    meta = session.get(url.rstrip("/") + "/.zarray", timeout=60).json()
    shape = np.array(meta["shape"])
    chunks = np.array(meta["chunks"])
    sep = meta.get("dimension_separator", ".")
    codec = None
    if meta["compressor"]:
        import numcodecs
        codec = numcodecs.get_codec(meta["compressor"])
    out = np.zeros(tuple(shape), np.uint8)
    keys = [(a, b, c)
            for a in range(int(np.ceil(shape[0] / chunks[0])))
            for b in range(int(np.ceil(shape[1] / chunks[1])))
            for c in range(int(np.ceil(shape[2] / chunks[2])))]

    def get(key):
        for _ in range(3):
            try:
                r = session.get(url.rstrip("/") + "/" + sep.join(map(str, key)), timeout=120)
                if r.status_code != 200:
                    return
                raw = codec.decode(r.content) if codec else r.content
                buf = np.frombuffer(raw, np.uint8)
                if buf.size != int(np.prod(chunks)):
                    return
                lo = np.array(key) * chunks
                hi = np.minimum(lo + chunks, shape)
                out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = buf.reshape(tuple(chunks))[
                    :hi[0] - lo[0], :hi[1] - lo[1], :hi[2] - lo[2]]
                return
            except Exception:
                continue

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(get, keys))
    return out


def z_shift(coarse, fine):
    """Shift in fine-voxel units, from how much of each slice is occupied."""
    pc = np.array([(coarse[z] > 0).mean() for z in range(coarse.shape[0])], np.float32)
    pf = np.array([(fine[z] > 0).mean() for z in range(fine.shape[0])], np.float32)
    n = len(pc) + len(pf)
    corr = np.fft.irfft(np.fft.rfft(pc - pc.mean(), n) * np.conj(np.fft.rfft(pf - pf.mean(), n)), n)
    lag = int(np.argmax(corr))
    return lag - n if lag > n // 2 else lag


def plane_shift(a, b):
    """(dy, dx, r) aligning image a onto image b."""
    a = a.astype(np.float32) - a.mean()
    b = b.astype(np.float32) - b.mean()
    corr = np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=a.shape)
    peak = np.unravel_index(corr.argmax(), corr.shape)
    dy = peak[0] - a.shape[0] if peak[0] > a.shape[0] // 2 else peak[0]
    dx = peak[1] - a.shape[1] if peak[1] > a.shape[1] // 2 else peak[1]
    shifted = np.roll(np.roll(a, -dy, 0), -dx, 1)
    return dy, dx, float(np.corrcoef(shifted.ravel(), b.ravel())[0, 1])


def rescale(volume, factor):
    out = np.stack([cv2.resize(volume[z], (int(volume.shape[2] * factor), int(volume.shape[1] * factor)),
                               interpolation=cv2.INTER_AREA) for z in range(volume.shape[0])])
    return np.stack([cv2.resize(out[:, :, x], (int(volume.shape[0] * factor), out.shape[1]),
                                interpolation=cv2.INTER_AREA) for x in range(out.shape[2])], axis=2)


def estimate(coarse_url, fine_url, coarse_um, fine_um, samples=6):
    coarse = fetch_volume(coarse_url).astype(np.float32)
    fine = fetch_volume(fine_url).astype(np.float32)
    scaled = rescale(coarse, coarse_um / fine_um)
    lag = z_shift(scaled, fine)
    rows = []
    step = max(1, fine.shape[0] // samples)
    for zf in range(fine.shape[0] // 6, fine.shape[0] - fine.shape[0] // 6, step):
        zc = zf + lag
        if not (0 <= zc < scaled.shape[0]):
            continue
        h = min(scaled.shape[1], fine.shape[1])
        w = min(scaled.shape[2], fine.shape[2])
        a, b = scaled[zc, :h, :w], fine[zf, :h, :w]
        if a.std() < 1 or b.std() < 1:
            continue
        dy, dx, r = plane_shift(a, b)
        rows.append((zf, dy, dx, r))
    return lag, rows


if __name__ == "__main__":
    lag, rows = estimate(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
    print(f"сдвиг по z (в шаге тонкого скана): {lag}")
    for zf, dy, dx, r in rows:
        print(f"  z={zf}: dy={dy:+d} dx={dx:+d} r={r:+.3f}")
