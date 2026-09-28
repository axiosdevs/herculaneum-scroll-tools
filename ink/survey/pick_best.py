"""Choose which surfaces are worth a full canvas.

Rendering a 14.4 mm canvas costs about ninety minutes. The seating scan costs a minute and a
half and says whether the surface lies flat enough on its sheet for a stroke to survive:
rendering the team's own PHerc0139 mesh with this renderer scatters 26.4 at stroke scale, and
our surfaces range from 7 to 46. Only the flat, centred ones get the ninety minutes.
"""
import glob, json, os
import numpy as np

rows = []
for f in glob.glob(f"/workspace/{os.environ.get('SCAN', 'seating_scan')}_*.json"):
    try:
        rows += json.load(open(f))
    except Exception:
        pass
ok = [r for r in rows if r.get("noise") and np.isfinite(r["noise"])]
print(f"отранжировано {len(ok)} поверхностей", flush=True)

# the reference is this renderer on the team's seated PHerc0139 mesh; another scan can be
# noisier or quieter as a whole, so REF and TOL can be set for it
REF = float(os.environ.get("REF", "26.4"))
TOL = float(os.environ.get("TOL", "1.15"))
picked = [r for r in ok if r["window"] == "centred" and r["noise"] <= REF * TOL]
picked.sort(key=lambda r: r["noise"])
print(f"окно centred и шум <= {REF*TOL:.1f}: {len(picked)}", flush=True)
area = sum(r.get("area_cm2") or 0 for r in picked)
print(f"суммарная площадь отобранного: {area:.1f} см²", flush=True)

out = [{"dir": r["dir"], **({"y0": r["y0"], "x0": r["x0"]} if r.get("y0") is not None else {}),
        "seating": r.get("seating") or 0.0,
        "area_cm2": r.get("area_cm2") or 0.0, "noise": r["noise"],
        "sheet_cnr": r["sheet_cnr"], "window": r["window"]} for r in picked]
json.dump(out, open(os.environ.get("PICKS_OUT", "/workspace/picks_flat.json"), "w"),
          ensure_ascii=False, indent=1)
for r in out[:15]:
    print(f"  {r['dir'].split('/workspace/')[-1][:34]:36s} шум {r['noise']:6.2f} "
          f"лист/шум {r['sheet_cnr']:5.2f}")
