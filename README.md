# Herculaneum Scroll Tools

Four open-source utilities for the Vesuvius Challenge, built to attack problems the
current pipeline does not address directly:

1. **Dual-energy co-rendering** — combine the two X-ray energies a scroll was scanned
   at into a single "high-Z contrast" map, surfacing metal-bearing material (a candidate
   signal for metallic inks and mineral inclusions) directly from physics, with no ML.
2. **Cross-scan registration** — align an *old* scan's coordinate frame (and every
   segmentation / label built on it) to a *newer, higher-resolution* scan of the same
   scroll, so years of prior segmentation work transfers onto the new data instead of
   being redone.
3. **CT-consistency QA** ([villa#1114](https://github.com/ScrollPrize/villa/issues/1114)) —
   measure and clean *phantom* voxels in published surface predictions; includes exact
   voxel-level phantom fractions for **all 13 grand-prize-eligible scrolls** (below).
4. **Winding-constraint annotator + verifier** — annotate winding constraints on
   flattened renders and export native spiral-input files; validated 125/125 on the
   released PHercParis4 annotations.

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

---

## 3. CT-consistency QA for surface predictions (`ct_support/`)

Addresses [ScrollPrize/villa#1114](https://github.com/ScrollPrize/villa/issues/1114):
published surface-prediction volumes can contain large *phantom* regions — positive
voxels sitting where the masked CT reads exactly 0 (outside the scroll). On the
PHerc0332 m7 predictions ~70% of positive voxels are phantoms, and seed-growers that
don't consult the CT can ride the phantom shell.

`ct_support.py` streams the prediction zarr and the masked CT zarr together
(no local copies) and provides four CPU-only, resumable modes:

- **`survey`** — per-plane phantom/support statistics (spot planes or a z-stride
  across the whole scroll), JSON + CSV report;
- **`slabs`** — chunk-aligned slab survey: reads whole z-chunk slabs in bounded-memory
  Y-stripes, so **100% of transferred bytes are used** (plane mode pays ~chunk-depth×
  amplification on remote zarrs) and every plane inside a slab is measured exactly;
- **`chunks`** — per-cube support map for cube-level filtering (on PHerc0332 the
  distribution is strongly bimodal — in our 128³ test window 96.5% of cubes are
  cleanly keep/drop at a 0.5 threshold);
- **`clean`** — write a cleaned (`preds AND ct>0`) copy of a z-window to a local zarr.

**Independent reproduction of villa#1114** (this tool, 2026-07-23, live S3 objects):
planes z∈{2000, 4224, 6000} → phantom fractions **0.6810 / 0.6717 / 0.6459**, positive
counts 2,259,758 / 2,379,932 / 2,110,813 — matching the issue's reported numbers
exactly. Full strided survey report: `ct_support/survey_pherc0332.json`.

### Measured voxel-level phantom fractions — all 13 grand-prize scrolls

Exact voxel-level measurements of the published m7 surface predictions
(run `20260413222639`), preds level 0 vs. the masked CT level on the same grid,
threshold 127. Every 12th z-chunk slab, all planes inside each slab measured
exactly (`full_batch.py` is the driver; per-plane data in each
`ct_support/survey_PHerc*.json`):

| scroll (GP-eligible) | positive voxels sampled | phantom voxels | phantom % | planes measured |
|---|---|---|---|---|
| PHerc1218 | 1,199,708,221 | 602,514,869 | **50.2%** | 224 |
| PHerc0358 | 1,021,436,239 | 503,603,333 | **49.3%** | 148 |
| PHerc0257 | 9,596,604,615 | 4,643,455,605 | **48.4%** | 1728 |
| PHerc1545 | 8,511,544,764 | 4,116,708,043 | **48.4%** | 1920 |
| PHerc0826 | 9,617,341,090 | 4,651,998,256 | **48.4%** | 1536 |
| PHerc0125 | 1,177,475,648 | 559,842,643 | **47.5%** | 182 |
| PHerc1447 | 14,344,637,261 | 6,290,066,086 | **43.8%** | 2112 |
| PHerc0813 | 9,077,181,876 | 3,936,984,632 | **43.4%** | 1536 |
| PHerc0211 | 8,665,801,539 | 3,756,497,458 | **43.3%** | 1728 |
| PHerc0800 | 27,053,626,452 | 11,281,693,092 | **41.7%** | 2112 |
| PHerc0191 | 12,232,136,550 | 4,874,448,732 | **39.8%** | 1728 |
| PHerc0268 | 23,824,135,553 | 7,919,731,041 | **33.2%** | 1344 |
| PHerc1203 | 136,835,937 | 41,379,993 | **30.2%** | 38 |
| **total** | **126,458,465,745** | **53,178,923,783** | **42.1%** | **16,336** |

Additional measured anchors: PHerc0139 37.1%, PHerc0332 69.4%, PHercMANBp 95.5%.

In other words: across 16,336 exactly-measured planes of the 13 grand-prize scrolls,
**126.5 billion prediction-positive voxels were checked and 53.2 billion of them
(42.1%) sit outside the scroll** (masked CT reads exactly 0). Roughly every second
"papyrus surface" voxel handed to teams for these scrolls is phantom. Any
grower/trainer consuming these predictions without a CT filter inherits that
contamination; `clean` mode removes it in one pass.

**Chunk→voxel calibration** (`ct_support/calibrate.py`): fitting
`p_voxel = 1 − (1 − p_chunk)^k` over all 16 measured anchors against the chunk-level
audit gives **k = 3.30** (max residual ±6.6 pts); estimates for the remaining
m7-batch samples are in `ct_support/calibration_m7_batch.json`.

```bash
PRED=https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0332/representations/predictions/surfaces/20251211183505-surface-20260413222639-surface-m7-L2-th0.2.zarr
CT=https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr

python ct_support/ct_support.py survey --preds $PRED --ct $CT --planes 2000,4224,6000
python ct_support/ct_support.py slabs  --preds $PRED --ct $CT --slab-stride 12 --out report.json
python ct_support/ct_support.py chunks --preds $PRED --ct $CT --z0 4224 --z1 4352 --out cubes.npz
python ct_support/ct_support.py clean  --preds $PRED --ct $CT --z0 4200 --z1 4400 --out cleaned.zarr

# measure every sample in the m7 batch (auto-discovers preds+CT on S3, resumable):
python ct_support/full_batch.py
```

---

## 4. Winding-constraint annotator + verifier (`winding/`)

Winding constraints are the organizers' stated #1 lever for unrolling scrolls at
scale ("we believe that the fastest way to unroll scrolls at scale is to develop
methods for creating winding constraints…", and explicitly: "There is no required
annotation tool or generation method" — scrollprize.org/open_problems/winding_annotations).

This module is a lightweight path: annotate on the *flattened* segment render,
export **native spiral-input files**.

- `annotator.py` — click paths on a flattened render; each collection is a
  same-winding (or relative-winding, with `wind_a`) constraint. Points are mapped
  through the segment's tifxyz to full-resolution volume voxels and written in the
  exact `same_windings.json` / `relative_windings.json` schema used by the
  spiral-input dataset (VC3D point collections) — verified against the released
  PHercParis4 files field-for-field. `--selftest` runs headless and validates the
  round-trip on a real tifxyz.
- `tifxyz_map.py` — flattened (u,v) → (x,y,z) bilinear lookup with missing-cell
  handling and remote fetch helper.
- `verify.py` — per-collection geometry report + a CT cross-section overlay with
  the umbilicus marked (see `examples/winding_verify_overlay.png`, drawn from the
  released PHercParis4 annotations over the Scroll 1 volume). Flags gross errors
  (>12σ trend outliers); honest scope note in the docstring: subtle single-gap
  hops need the spiral fit itself — the overlay is the fast human check.

Validation: on the released human-verified PHercParis4 `same_windings.json`
(125 collections, ~6k points) the verifier reports **125/125 CONSISTENT**, and
the annotator's exported files match the native schema exactly.

```bash
# annotate (GUI): draw on a flattened render, export native constraint JSONs
python winding/annotator.py --image flat.png --tifxyz /path/to/tifxyz --out out/

# verify + overlay
python winding/verify.py --constraints out/same_windings.json \
  --umbilicus umbilicus.json \
  --ct https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr \
  --ct-level 3 --ct-scale 8 --overlay overlay.png
```
