"""Tests for surface_support: a surface half over material, half over nothing."""
import numpy as np, pytest
zarr = pytest.importorskip("zarr")
from surface_support import read_surface, write_surface, support_map, report

def _ct(tmp_path):
    path = tmp_path / "ct.zarr"
    arr = zarr.open_array(str(path), mode="w", shape=(32, 32, 32), chunks=(16, 16, 16),
                          dtype="u1", zarr_format=2)
    arr[:, :16, :] = 200            # material only in the first half along y
    return arr, str(path)

def _surface(tmp_path, rows=8, cols=8):
    """Half the quads sit at y=8 (supported), half at y=24 (empty)."""
    pts = np.zeros((rows, cols, 3), np.float32)
    pts[..., 0] = 10                                  # z
    pts[..., 1] = np.where(np.arange(cols) < cols // 2, 8, 24)[None, :]
    pts[..., 2] = np.arange(rows)[:, None] + 1        # x
    valid = np.ones((rows, cols), bool)
    write_surface(tmp_path / "seg", pts, valid)
    return pts, valid

def test_roundtrip_reads_back_what_was_written(tmp_path):
    pts, valid = _surface(tmp_path)
    back, back_valid = read_surface(tmp_path / "seg")
    np.testing.assert_allclose(back[valid], pts[valid], atol=1e-4)
    assert back_valid.sum() == valid.sum()

def test_half_the_surface_is_unsupported(tmp_path):
    pts, valid = _surface(tmp_path)
    ct, _ = _ct(tmp_path)
    supported = support_map(pts, valid, ct)
    result = report(pts, valid, supported)
    assert result["quads"] == 64
    assert result["frac_supported"] == pytest.approx(0.5)
    assert supported[:, :4].all() and not supported[:, 4:].any()

def test_dilation_reaches_across_a_gap(tmp_path):
    """A quad two voxels outside the material counts once dilation covers it."""
    pts = np.zeros((1, 2, 3), np.float32)
    pts[0, 0] = (10, 15, 5)                          # inside
    pts[0, 1] = (10, 17, 5)                          # two voxels past the edge
    valid = np.ones((1, 2), bool)
    ct, _ = _ct(tmp_path)
    assert support_map(pts, valid, ct).tolist() == [[True, False]]
    assert support_map(pts, valid, ct, dilation=2).tolist() == [[True, True]]

def test_points_outside_the_volume_are_unsupported(tmp_path):
    pts = np.zeros((1, 2, 3), np.float32)
    pts[0, 0] = (10, 8, 5)
    pts[0, 1] = (999, 8, 5)                          # past the end of the volume
    valid = np.ones((1, 2), bool)
    ct, _ = _ct(tmp_path)
    assert support_map(pts, valid, ct).tolist() == [[True, False]]

def test_trim_keeps_only_supported_quads(tmp_path):
    pts, valid = _surface(tmp_path)
    ct, _ = _ct(tmp_path)
    supported = support_map(pts, valid, ct)
    write_surface(tmp_path / "trimmed", pts, valid & supported)
    _, kept = read_surface(tmp_path / "trimmed")
    assert kept.sum() == 32
    assert kept[:, :4].all() and not kept[:, 4:].any()

def test_area_uses_the_voxel_size(tmp_path):
    pts, valid = _surface(tmp_path)
    ct, _ = _ct(tmp_path)
    result = report(pts, valid, support_map(pts, valid, ct), voxelsize_um=8.64)
    assert result["area_cm2"] == pytest.approx(64 * (8.64e-4) ** 2, rel=1e-3)
    assert result["supported_area_cm2"] == pytest.approx(result["area_cm2"] / 2, rel=1e-3)
