# Herculaneum Scroll Tools: Dual-Energy Ink Contrast & Cross-Scan Registration

Two open-source utilities for the Vesuvius Challenge, built to attack problems the
current pipeline does not address directly:

1. **Dual-energy co-rendering** — combine the two X-ray energies a scroll was scanned
   at into a single "high-Z contrast" map, surfacing metal-bearing material (a candidate
   signal for metallic inks and mineral inclusions) directly from physics, with no ML.
2. **Cross-scan registration** — align an *old* scan's coordinate frame (and every
   segmentation / label built on it) to a *newer, higher-resolution* scan of the same
   scroll, so years of prior segmentation work transfers onto the new data instead of
   being redone.

Both stream data directly from the public `vesuvius-challenge-open-data` S3 bucket and
`dl.ash2txt.org` — no local copy of a full scroll is needed. Everything runs on a laptop.

MIT-licensed. Standard formats in (OME-Zarr, tifxyz, `.volpkg` affines), standard formats
out (PNG maps, NumPy arrays).

---

## 1. Dual-energy co-rendering (`dual_energy/`)

Several Herculaneum scrolls were scanned at **two X-ray energies** (e.g. PHerc0332 /
Scroll 3 at 53 keV and 70 keV). X-ray attenuation is energy-dependent, and that
dependence is much stronger for high-atomic-number (high-Z) elements than for the
carbon/organic matrix of papyrus. Taking the **ratio of the two energies** therefore
isolates dense, metal-bearing material — exactly the kind of trace-metal signature some
ancient inks carry — using nothing but physics.

**What the tool does**

- Reads the surface of a traced segment from its PPM map, samples both energy volumes
  along the surface normal (using the official `.volpkg` affine transforms to align the
  two energies), and builds a per-pixel **53/70 keV ratio map** across the whole segment.
- The legacy energy volumes are stored as *uncompressed single-strip TIFFs*, so the tool
  fetches only the rows it needs via HTTP range requests — a full segment scan streams in
  minutes without downloading the multi-hundred-GB volume.
- Resumable: every patch is cached as an `.npz`, so an interrupted scan continues where it
  stopped.

**Validation.** On Scroll 3, dense inclusions show a 53/70 ratio of ~1.13 versus ~0.97 for
background papyrus — a clear, reproducible high-Z separation (the physics works). See
`examples/dual_energy_metal_map.png` for the assembled 33 cm² map.

**Honest scope.** This surfaces *metal*, which is a *candidate* ink signal, not a proof of
text. On Scroll 3 the high-Z clusters did not form letters (its ink does not appear to be
metallic), but the tool is a general, physics-grounded contrast channel for any
dual-energy scroll, and a natural second input channel for ink-detection models.

**Run it**

```bash
pip install -r requirements.txt
# single 5x5 mm patch (quick sanity check, prints ratio stats + saves ratio/max maps):
python dual_energy/sample_patch.py --ppm <segment>.ppm --size 600 --step 2 --out_prefix demo
# full segment scan (resumable), then assemble the map:
python dual_energy/scan_segment.py        # edit PPM / volume UUIDs at the top
python dual_energy/assemble_map.py        # -> metal_map_overlay.png + density map
```

Energy volumes, affines and PPM paths are the standard `.volpkg` layout; the header of
`sample_patch.py` documents the exact Scroll-3 UUIDs used as the reference example.

---

## 2. Cross-scan registration (`registration/`)

Scrolls get re-scanned at ever higher resolution (Scroll 3: 7.91 µm in 2023 → 2.4 µm in
2025). Between scans the physical scroll is re-mounted, so the coordinate frames do **not**
match — and every segmentation, PPM and ink label built on the old scan is stranded on the
old data. This tool recovers the transform between the two frames so that prior work
transfers forward.

**Method (coarse → fine)**

1. **Coarse global alignment** at a shared downsampled resolution: recover the rigid
   relationship (in the reference example: horizontal flip + ~300.5° rotation, with a
   z-axis inversion `z_new = Z0 − z_old` and a z-linear in-plane drift). Parameters live in
   `registration/coarse_transform.json`; `map_to_new.py` exposes
   `p791L0_to_p24L0(pts)` mapping old-frame voxel coordinates to new-frame voxel
   coordinates.
2. **Fine snap to the surface.** The coarse map leaves a ±230 µm residual — larger than a
   sheet gap. `render_new_scan.py` removes it by snapping each mapped surface point, along
   its normal, to the nearest sheet in the organizers' published surface-prediction volume.
   Result: 100% of rays hit a sheet, median absolute residual ~29 µm, and a smooth offset
   field.

The payoff: you can **re-render any legacy segment window straight out of the new
high-resolution scan** (`render_new_scan.py`), inheriting the old segmentation but gaining
the new scan's detail. `examples/registration_overlay.png` shows old-vs-mapped cross
sections agreeing; `examples/rerender_from_new_scan.png` shows a legacy Scroll-3 segment
re-rendered at 4.8 µm from the 2025 scan (papyrus fibres, cracks and inclusions resolve
cleanly).

`render_tifxyz.py` is a small standalone helper: render flattened surface layers from any
`tifxyz` mesh + a scroll volume zarr, in the winners' chunk layout, ready for an
ink-detection model.

**Run it**

```bash
python registration/map_to_new.py         # sanity: prints a mapped test coordinate
python registration/render_new_scan.py --u0 <col> --v0 <row> --w 1500 --h 1245 --out demo
```

---

## Why these help the challenge

- **Registration** is directly reusable: it turns "we rescanned it, now re-segment
  everything" into "we rescanned it, the old segments still apply." Any scroll with an
  old + new scan pair benefits.
- **Dual-energy** adds an independent, ML-free physical contrast channel — useful both as a
  standalone metal map and as an extra input band for ink models on multi-energy scrolls.
- Both are self-contained, laptop-runnable, stream from the public buckets, and emit
  standard formats for easy integration.

Feedback and PRs welcome. Released under MIT so any of it can be folded into VC3D or the
community tooling.
