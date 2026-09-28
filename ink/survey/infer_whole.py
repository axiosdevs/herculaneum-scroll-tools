"""One GPU process behind the render queue: a whole surface per job, scored as one canvas.

Depth order is a property of the surface, so it is settled once on the first job and costs
one inference per surface after that. The canvas is 75 cells wide -- about 18 mm at 2.4 um
-- which puts twelve FFT bins inside the 1.0-3.5 mm line band, where a single 4.8 mm window
puts three and pins every period to 1.20 mm whatever the map holds.
"""
import glob, json, os, sys, time
import numpy as np

sys.path.insert(0, "/workspace/ink")
sys.path.insert(0, "/workspace/optimized_inference")
sys.path.insert(0, "/workspace")
from text_score import text_score
from scan_cnr import sheet_cnr
from center_window import depth_profile, window_verdict
from run_r152 import predict

REF_LEVELS = (29.0, 42.0, 128.0)   # 5/50/95 percentiles of PHerc0139's published surface volume


def match_levels(stack, ref=REF_LEVELS, out=None):
    """Put this stack on the intensity scale the published checkpoint responds to.

    Empirical, and stated as such. Remapping our PHerc1451 render onto PHerc0139's
    percentiles raises the response to planted ink about elevenfold (+0.006 -> +0.069).
    The mechanism is NOT the one first proposed here: that the fixed clip(0,200)/200
    divisor put the model off its trained brightness. Measurement refutes it -- brightening
    the team's own stack to median 128 left agreement at r=0.994, and scaling its contrast
    over 0.25x-2.0x held r=0.918-0.996. The model tolerates both. The rescale therefore
    helps for a reason not yet identified, and the claim made is the measured effect, not
    an explanation of it.

    Done in slabs along depth: a whole-surface stack is ~2 GB as uint8 and a float32 copy
    of it is ~9 GB, which is what was getting this worker OOM-killed mid-survey.
    """
    a = np.asarray(stack)
    p5, p50, p95 = np.percentile(a[::4, ::4, ::4].astype(np.float32), [5, 50, 95])
    r5, r50, r95 = ref
    if out is None:
        out = np.empty(a.shape, np.uint8)
    lo_slope = (r50 - r5) / max(p50 - p5, 1e-6)
    hi_slope = (r95 - r50) / max(p95 - p50, 1e-6)
    for i in range(0, a.shape[0], 4):
        b = a[i:i + 4].astype(np.float32)
        lo = b <= p50
        b[lo] = r5 + (b[lo] - p5) * lo_slope
        b[~lo] = r50 + (b[~lo] - p50) * hi_slope
        out[i:i + 4] = np.clip(b, 0, 255).astype(np.uint8)
        del b
    return out

BAND_PX = 1024          # rows of canvas handed to the model at once
BAND_OVERLAP = 256      # twice the 128 stride, so every output row has full tile context


def predict_banded(stack, ckpt, reverse, band=BAND_PX, overlap=BAND_OVERLAP):
    """The published recipe, run over a whole surface without holding it in memory.

    A whole-surface canvas is 6000 px square by 62 layers. Handing that to the model at
    once costs tens of gigabytes and is what was killing this worker on a box shared with
    the renderers. The window the model actually sees is 256 px wide with a 128 stride, so
    a row's output depends on nothing further than 128 px away: cutting the canvas into
    bands with 256 px of overlap and keeping each band's interior is arithmetically the
    same result, at memory set by the band rather than by the surface.
    """
    H = stack.shape[1]
    if H <= band + overlap:
        return predict(match_levels(stack), ckpt, reverse=reverse)
    out = None
    y = 0
    while y < H:
        lo = max(0, y - overlap)
        hi = min(H, y + band + overlap)
        piece = predict(match_levels(stack[:, lo:hi]), ckpt, reverse=reverse)
        if out is None:
            out = np.empty((H, piece.shape[1]), piece.dtype)
        keep_lo, keep_hi = y, min(H, y + band)
        out[keep_lo:keep_hi] = piece[keep_lo - lo:keep_hi - lo]
        del piece
        y += band
    return out

HERE, QUEUE = "/workspace", "/workspace/queue"
MAPS, CANV = "/workspace/scan_maps", "/workspace/canvases"
LEDGER, CKPT = "/workspace/ink_scan.json", "/workspace/r152.ckpt"
VOXEL_UM = 2.399
QSUB = 8        # in-plane subsample for the per-canvas quality read

os.makedirs(MAPS, exist_ok=True)
os.makedirs(CANV, exist_ok=True)
ledger = json.load(open(LEDGER)) if os.path.exists(LEDGER) else []
done = {r["dir"] for r in ledger if r.get("whole")}
# The depth order is not a free choice and it must not be settled by a text score. Measured
# on the flattest PHerc1451 surface: forward recovers planted ink at an amplitude of 32, the
# same threshold the team's own readable surface volume gives, and reverse recovers none at
# all -- negative lifts at every amplitude -- while reporting 26.6% "ink" against forward's
# 3.2%. A text score cannot tell those apart, and the first survey locked onto reverse on its
# first canvas and read the wrong face of every sheet after it.
polarity = os.environ.get("POLARITY", "fwd")
idle = 0
print("инференс запущен", flush=True)
while True:
    jobs = sorted(glob.glob(os.path.join(QUEUE, "*_whole.npy")))
    if not jobs:
        idle += 1
        if idle > 360:
            print("очередь пуста долго — выходим", flush=True)
            break
        time.sleep(15)
        continue
    idle = 0
    for job in jobs:
        name = os.path.basename(job)[:-4]
        meta_p = os.path.join(QUEUE, name + ".json")
        if not os.path.exists(meta_p):
            continue
        size = os.path.getsize(job)
        time.sleep(1.5)
        if os.path.getsize(job) != size:
            continue
        meta = json.load(open(meta_p))
        if meta["dir"] in done:
            os.remove(job); os.remove(meta_p); continue
        free_gb = int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) / 1048576
        waited = 0
        while free_gb < 40 and waited < 900:
            time.sleep(20)
            waited += 20
            free_gb = int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) / 1048576
        print(f"  {name}: старт, свободно {free_gb:.0f} ГБ", flush=True)
        stack = np.load(job, mmap_mode="r")
        orders = (("rev", True), ("fwd", False)) if polarity is None else ((polarity, polarity == "rev"),)
        best = None
        for tag, rev in orders:
            try:
                p = predict_banded(stack, CKPT, rev)
            except Exception as exc:
                print(f"  {name}: инференс не удался — {exc}", flush=True)
                continue
            ts = text_score(p)
            cand = {"order": tag, "text": round(float(ts[0]), 3), "period_mm": round(float(ts[1]), 2),
                    "ink_pct": round(float((p > 0.5).mean()) * 100, 2),
                    "max": round(float(p.max()), 3), "median": round(float(np.median(p)), 3),
                    "map": os.path.relpath(os.path.join(CANV, name + f"_{tag}.npy"), HERE)}
            np.save(os.path.join(CANV, name + f"_{tag}.npy"), p)
            if best is None or (cand["text"], cand["max"]) > (best["text"], best["max"]):
                best = cand
                mm = p.shape[0] * VOXEL_UM / 1000
        try:
            stack_for_quality = np.array(stack[:, ::QSUB, ::QSUB], copy=True)
        except Exception:
            stack_for_quality = None
        del stack
        os.remove(job); os.remove(meta_p)
        open(os.path.join(QUEUE, name + ".done"), "w").write("ok")
        if best is None:
            continue
        polarity = polarity or best["order"]
        # Every canvas carries what its own null is worth: a window the sheet is not in,
        # or a scan without the margin to hold ink, cannot report an absence of writing.
        try:
            # subsampled 8x in plane, so the stroke-scale filter has to be told the
            # pixel size it is now looking at or it measures noise at 2.8 mm instead
            q = sheet_cnr(stack_for_quality, micron_per_pixel=VOXEL_UM * QSUB, sub=1)
        except Exception:
            q = {}
        row = dict(meta); row.update(best); row.update(
            {"sheet_cnr": q.get("sheet_cnr"), "window": q.get("window"),
             "noise": q.get("noise")})
        ledger.append(row)
        done.add(meta["dir"])
        json.dump(ledger, open(LEDGER, "w"), indent=1)
        print(f"ПОЛОТНО {meta['dir']}: {mm:.1f} мм, чернил {row['ink_pct']:.2f}%, "
              f"текст {row['text']:.3f} период {row['period_mm']:.2f} мм", flush=True)
