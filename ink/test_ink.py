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


def test_normals_are_defined_at_the_grid_border():
    """A zero gradient at the border leaves the outermost cell unrendered — a visible seam
    once windows are stitched together."""
    u = np.arange(8, dtype=np.float32)
    x = np.tile(u, (8, 1))
    y = np.tile(u[:, None], (1, 8))
    z = np.zeros((8, 8), np.float32)
    nx, ny, nz = grid_normals(x, y, z)
    for r, c in ((0, 0), (0, 7), (7, 0), (7, 7)):
        assert abs(abs(nz[r, c]) - 1.0) < 1e-5, (r, c, nz[r, c])


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


def test_sheet_offset_recovers_a_known_displacement():
    """A bright sheet displaced by a smooth field is found where it was put."""
    from flatten_stack import sheet_offset, gather
    depth, size = 120, 256
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    truth = 6.0 * np.sin(2 * np.pi * yy / size) * np.cos(2 * np.pi * xx / size)
    centre = (depth - 1) / 2.0 + truth
    z = np.arange(depth, dtype=np.float32)[:, None, None]
    stack = (180.0 * np.exp(-0.5 * ((z - centre[None]) / 2.5) ** 2) + 30.0).astype(np.uint8)
    found = sheet_offset(stack, max_shift=12, smooth_px=4, search=12)
    assert np.abs(found - truth).mean() < 1.0, np.abs(found - truth).mean()


def test_flattening_cannot_manufacture_contrast():
    """On a stack whose sheet is destroyed, re-centring must invent nothing.

    Guards the obvious failure mode: picking each pixel's brightest depth raises any
    contrast statistic by construction. Smoothing the shift field is what stops it.
    """
    from flatten_stack import sheet_offset, gather
    from seat_mesh import sheet_contrast
    rng = np.random.default_rng(11)
    depth, size = 120, 256
    noise = rng.integers(0, 255, (depth, size, size)).astype(np.uint8)
    flat = gather(noise, sheet_offset(noise, max_shift=12, smooth_px=4, search=12), 62)
    assert abs(sheet_contrast(flat)) < 1.0, sheet_contrast(flat)


def test_voxel_size_recovered_from_mesh_metadata():
    """area_cm2 / area_vx2 gives the frame a mesh was laid out in, to three digits."""
    import json, tempfile
    from resolve_frame import voxel_size_from_meta
    d = tempfile.mkdtemp()
    # the published PHerc0009B segment: 26.24 cm2 over 29,937,600 voxel^2
    json.dump({"area_cm2": 26.239425659179688, "area_vx2": 29937600.0},
              open(os.path.join(d, "meta.json"), "w"))
    um = voxel_size_from_meta(d)
    assert abs(um - 9.363) < 0.01, um


def test_residual_shift_recovers_a_known_offset():
    """The leftover a chunk test cannot see is found by one cross-correlation."""
    from resolve_frame import residual_shift
    rng = np.random.default_rng(3)
    a = rng.random((256, 256)).astype(np.float32)
    b = np.roll(np.roll(a, 17, axis=0), -23, axis=1)
    r, dy, dx = residual_shift(b, a)
    assert (dy, dx) == (17, -23), (dy, dx)
    assert r > 0.9, r


# -- detectability: the probe has to be able to say "blind", and has to not be fooled by a
# -- detector whose output already resembles the planted mask.

TEST_UM = 40.0   # coarse, so a 200 px test window is 8 mm across and holds several lines


def _sheet_stack(C=62, H=200, W=200, centre=31.0):
    """A stack with a bright sheet at a known depth and nothing written on it."""
    z = np.arange(C, dtype=np.float32)[:, None, None]
    return (40 + 120 * np.exp(-0.5 * ((z - centre) / 4.0) ** 2)
            + np.zeros((C, H, W), np.float32)).astype(np.uint8)


def test_planted_ink_lands_on_the_sheet_not_in_the_middle_of_the_stack():
    from detectability import plant_ink
    stack = _sheet_stack(centre=40.0)
    planted, mask = plant_ink(stack, 60, micron_per_pixel=TEST_UM)
    delta = planted.astype(np.float32) - stack.astype(np.float32)
    inked = delta[:, mask.astype(bool)].mean(axis=1)
    peak = float(np.argmax(inked))
    # on the near face of the sheet, not at its bright centre and not at the stack's middle
    assert 33 <= peak <= 40, peak
    assert peak < 40.0, peak


def test_probe_reports_blind_when_the_model_ignores_the_stack():
    from detectability import probe
    stack = _sheet_stack()
    rng = np.random.default_rng(3)
    fixed = rng.random((200, 200)).astype(np.float32) * 0.2

    def deaf(_stack):
        return fixed

    out = probe(stack, deaf, amplitudes=(8, 32, 64), micron_per_pixel=TEST_UM)
    assert out["sensitivity"] is None, out
    assert all(abs(r["ink_lift"]) < 1e-9 for r in out["rows"]), out["rows"]


def test_probe_finds_the_threshold_when_the_model_responds():
    from detectability import probe
    stack = _sheet_stack()

    def sees(s):
        # responds to whatever was added on the sheet's near face
        return np.clip((s[30:38].astype(np.float32).max(0) - 150.0) / 40.0, 0, 1)

    out = probe(stack, sees, amplitudes=(2, 64), micron_per_pixel=TEST_UM)
    assert out["sensitivity"] == 64.0, out


def test_probe_lift_is_before_versus_after_not_inside_versus_outside():
    """The bug this metric had: a detector already brighter where the mask is scores a lift
    without responding to the plant at all. Measured on PHerc1451 that confound gave the
    same lift at every amplitude."""
    from detectability import probe, script_mask
    stack = _sheet_stack()
    biased = script_mask((200, 200), TEST_UM).astype(np.float32) * 0.9

    def biased_but_deaf(_stack):
        return biased

    out = probe(stack, biased_but_deaf, amplitudes=(8, 64), micron_per_pixel=TEST_UM)
    assert out["sensitivity"] is None, out
    assert all(abs(r["ink_lift"]) < 1e-9 for r in out["rows"]), out["rows"]


def test_probe_refuses_a_window_too_narrow_to_hold_a_line_of_text():
    from detectability import script_mask
    try:
        script_mask((40, 40), TEST_UM)
    except ValueError:
        return
    raise AssertionError("узкое окно должно быть отклонено, а не размечено пустой маской")


# -- center_window: the failure that made a whole scroll's survey meaningless

def _profile_with_sheet_at(n, peak_index, floor=30.0, peak=90.0, width=12.0):
    z = np.arange(n, dtype=np.float32)
    return floor + (peak - floor) * np.exp(-0.5 * ((z - peak_index) / width) ** 2)


def test_sheet_offset_recovers_a_known_displacement():
    from center_window import sheet_offset
    for true_off in (-62, -20, 0, 35):
        prof = _profile_with_sheet_at(301, 150 + true_off)
        off, contrast = sheet_offset(prof)
        assert abs(off - true_off) <= 2, (true_off, off)
        assert contrast > 0.5, contrast


def test_sheet_offset_picks_the_nearer_winding_not_the_brighter_one():
    """At a ~700 um winding period a wide probe sees more than one sheet. The neighbour is
    the wrong answer even when it is denser."""
    from center_window import sheet_offset
    near = _profile_with_sheet_at(601, 300 + 40, floor=30.0, peak=80.0)
    far = _profile_with_sheet_at(601, 300 - 290, floor=0.0, peak=60.0)
    off, _ = sheet_offset(near + far)
    assert abs(off - 40) <= 3, off


def test_sheet_offset_declines_to_guess_on_a_flat_window():
    from center_window import sheet_offset
    rng = np.random.default_rng(5)
    prof = 100.0 + rng.normal(0, 0.4, 200).astype(np.float32)
    off, contrast = sheet_offset(prof)
    assert off == 0, off
    assert contrast < 0.10, contrast


def test_window_verdict_calls_a_gap_render_by_its_name():
    """The PHerc1451 case, to its measured numbers: the sheet peak sat 62 layers outside a
    62-layer window, and what the window held was the sheet's flank -- brightest at its own
    edge, 58 falling to 47 across the window."""
    from center_window import window_verdict
    deep = _profile_with_sheet_at(301, 150 - 62, width=25.0)
    window = deep[150 - 31:150 + 31]
    verdict, off, contrast = window_verdict(window)
    assert verdict == "edge", (verdict, off, contrast)
    assert off < 0, off
    assert contrast > 0.15, contrast


def test_window_verdict_calls_a_window_with_no_sheet_in_it_flat():
    """Far enough into the gap there is not even a flank -- and 'flat' is the honest word
    for a render that never met material."""
    from center_window import window_verdict
    deep = _profile_with_sheet_at(301, 150 - 62, width=12.0)
    verdict, _, contrast = window_verdict(deep[150 - 31:150 + 31])
    assert verdict == "flat", (verdict, contrast)


def test_window_verdict_accepts_a_window_on_the_sheet():
    from center_window import window_verdict
    deep = _profile_with_sheet_at(301, 150)
    verdict, off, _ = window_verdict(deep[150 - 31:150 + 31])
    assert verdict == "centred", verdict
    assert abs(off) <= 2, off


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
