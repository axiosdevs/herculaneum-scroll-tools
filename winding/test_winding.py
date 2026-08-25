"""Seeded-error test for the winding verifier.

125/125 CONSISTENT on the released PHercParis4 annotations shows only that clean data raises
no alarms. Detection power needs the other direction: plant a known error and require the
verdict to flip. The seeded error here is the canonical annotation mistake — one point taken
from the neighbouring wrap, i.e. displaced radially by exactly one winding gap.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from verify import analyze


def synthetic(seed=0, n=60, r0=300.0, gap=14.0, noise=0.6):
    rng = np.random.default_rng(seed)
    ang = np.linspace(0.0, 6.0 * np.pi, n)                # три оборота
    r = r0 + gap * ang / (2.0 * np.pi) + rng.normal(0, noise, n)
    z = np.linspace(1000.0, 1400.0, n)
    pts = np.c_[r * np.cos(ang), r * np.sin(ang), z]
    umb = {"control_points": [{"z": 900.0, "x": 0.0, "y": 0.0},
                              {"z": 1500.0, "x": 0.0, "y": 0.0}]}
    return pts, umb, gap


def test_clean_spiral_is_consistent():
    pts, umb, _ = synthetic()
    assert analyze("clean", pts, umb)["verdict"] == "CONSISTENT"


def test_one_point_from_the_next_wrap_is_flagged():
    pts, umb, gap = synthetic()
    bad = pts.copy()
    k = 30
    scale = 1.0 + gap / np.hypot(bad[k, 0], bad[k, 1])    # радиальный сдвиг на один виток
    bad[k, 0] *= scale
    bad[k, 1] *= scale
    assert analyze("seeded", bad, umb)["verdict"] == "SUSPECT"


def test_flag_survives_any_seeded_position():
    for k in (5, 20, 45, 58):
        pts, umb, gap = synthetic(seed=k)
        scale = 1.0 + gap / np.hypot(pts[k, 0], pts[k, 1])
        pts[k, 0] *= scale
        pts[k, 1] *= scale
        assert analyze("seeded", pts, umb)["verdict"] == "SUSPECT", k


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); passed += 1; print("ok  ", name)
            except AssertionError as exc:
                failed += 1; print("FAIL", name, exc)
    print(f"{passed} прошло, {failed} провалено")
    sys.exit(1 if failed else 0)
