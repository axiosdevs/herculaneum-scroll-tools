"""Tests for the four conventions the ink checkpoints depend on."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from render_surface import grid_normals, upsample_grid          # noqa: E402
from register_scans import plane_shift, z_shift                 # noqa: E402
from text_score import text_score                               # noqa: E402


def test_normal_of_a_plane_points_along_z():
    x = np.tile(np.arange(10, dtype=np.float32), (10, 1))
    y = np.tile(np.arange(10, dtype=np.float32)[:, None], (1, 10))
    z = np.zeros((10, 10), np.float32)
    nx, ny, nz = grid_normals(x, y, z)
    assert abs(nx[5, 5]) < 1e-5 and abs(ny[5, 5]) < 1e-5
    assert abs(abs(nz[5, 5]) - 1.0) < 1e-5


def test_normal_is_perpendicular_to_a_tilted_plane():
    u = np.arange(12, dtype=np.float32)
    x = np.tile(u, (12, 1))
    y = np.tile(u[:, None], (1, 12))
    z = 0.5 * x + 0.25 * y
    nx, ny, nz = grid_normals(x, y, z)
    n = np.array([nx[6, 6], ny[6, 6], nz[6, 6]])
    for tangent in (np.array([1, 0, 0.5]), np.array([0, 1, 0.25])):
        assert abs(float(n @ tangent)) < 1e-4


def test_upsample_uses_the_cell_corner_convention():
    row = np.tile(np.arange(4, dtype=np.float32), (4, 1))
    out = upsample_grid(row, 2, 8, 8)
    # pixel p must read grid coordinate p / 2, so the first samples are 0, 0.5, 1.0 ...
    assert np.allclose(out[0, :3], [0.0, 0.5, 1.0], atol=1e-5)


def test_plane_shift_recovers_a_known_translation():
    rng = np.random.default_rng(0)
    a = rng.random((64, 64)).astype(np.float32)
    b = np.roll(np.roll(a, 5, 0), -3, 1)
    dy, dx, r = plane_shift(a, b)
    assert (dy, dx) == (-5, 3)
    assert r > 0.99


def test_z_shift_recovers_a_known_offset():
    rng = np.random.default_rng(1)
    base = np.zeros((40, 8, 8), np.uint8)
    for z in range(10, 30):
        base[z, :z % 8 + 1] = 1
    coarse = base
    fine = np.zeros_like(base)
    fine[:-6] = base[6:]
    assert abs(z_shift(coarse, fine) - 6) <= 1


def test_text_score_separates_rows_from_noise():
    rows = np.zeros((600, 600), np.float32)
    for y in range(50, 550, 60):          # шаг 60 px x 24 um = 1.44 mm, внутри полосы 1.0-3.5
        for x in range(6, 594, 18):
            rows[y:y + 14, x:x + 9] = 1.0
    rng = np.random.default_rng(2)
    noise = (rng.random((600, 600)) < 0.05).astype(np.float32)
    row_score, period = text_score(rows, micron_per_pixel=24.0)
    noise_score, _ = text_score(noise, micron_per_pixel=24.0)
    assert row_score > noise_score
    assert 1.0 <= period <= 3.5


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                passed += 1
                print(f"ok   {name}")
            except AssertionError as exc:
                failed += 1
                print(f"FAIL {name}: {exc}")
    print(f"\n{passed} прошло, {failed} провалено")
    sys.exit(1 if failed else 0)
