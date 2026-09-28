"""One sheet of every finished canvas, so a person can scan them all at once."""
import glob, os
import numpy as np, cv2

CANV = "/workspace/canvases"
CELL = 460
files = sorted(glob.glob(f"{CANV}/*_rev.npy")) + sorted(glob.glob(f"{CANV}/*_fwd.npy"))
seen, use = set(), []
for f in files:
    k = os.path.basename(f).rsplit("_", 1)[0]
    if k in seen:
        continue
    seen.add(k)
    use.append(f)
use = use[-24:]
cols = 6
rows = (len(use) + cols - 1) // cols
sheet = np.zeros((rows * (CELL + 22), cols * CELL), np.uint8)
for i, f in enumerate(use):
    a = np.clip(np.load(f).astype(np.float32), 0, 1)
    lo, hi = np.percentile(a, [40, 99.5])
    st = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.resize((st * 255).astype(np.uint8), (CELL, CELL), interpolation=cv2.INTER_AREA)
    r, c = divmod(i, cols)
    y0 = r * (CELL + 22)
    sheet[y0 + 22:y0 + 22 + CELL, c * CELL:(c + 1) * CELL] = img
    name = os.path.basename(f).replace("auto_grown_", "").replace("_whole", "")[:26]
    cv2.putText(sheet, name, (c * CELL + 4, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, 255, 1)
cv2.imwrite("/workspace/contact.png", sheet, [cv2.IMWRITE_PNG_COMPRESSION, 6])
print(f"лист из {len(use)} полотен: {sheet.shape}")
