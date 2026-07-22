import argparse, json, os, sys, time
from collections import defaultdict

import numpy as np
import requests

BASE = "https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/volumes"
VOL53 = "20231027191953"   # 53 keV, 3.24 um
VOL70 = "20231201141544"   # 70 keV, 3.24 um
W = H = 9414
NSLICES = {VOL53: 22941, VOL70: 22932}
DATA_OFF = 368  # uncompressed single-strip TIFF, uint16 LE


def load_affine(path):
    with open(path) as f:
        t = json.load(f)
    M = np.array(t["params"], dtype=np.float64)
    if M.shape == (3, 4):
        M = np.vstack([M, [0, 0, 0, 1]])
    return M


def read_ppm_patch(ppm_path, v0, u0, size, step):
    with open(ppm_path, "rb") as f:
        head = f.read(400)
        hdr_end = head.index(b"<>\n") + 3
        meta = dict(l.split(": ") for l in head[:head.index(b"<>")].decode().strip().split("\n"))
    w, h = int(meta["width"]), int(meta["height"])
    mm = np.memmap(ppm_path, dtype="<f8", mode="r", offset=hdr_end, shape=(h, w, 6))
    patch = np.array(mm[v0:v0 + size:step, u0:u0 + size:step])
    return patch[..., :3], patch[..., 3:], (w, h)


class SliceReader:
    """ROI reads from remote uncompressed single-strip TIFF slices via HTTP ranges."""

    def __init__(self, vol):
        self.vol = vol
        self.sess = requests.Session()

    def fetch_rows(self, z, y_min, y_max):
        url = f"{BASE}/{self.vol}/{z:05d}.tif"
        start = DATA_OFF + y_min * W * 2
        end = DATA_OFF + (y_max + 1) * W * 2 - 1
        for attempt in range(7):
            try:
                r = self.sess.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                if r.status_code in (200, 206):
                    buf = np.frombuffer(r.content, dtype="<u2")
                    return buf.reshape(y_max - y_min + 1, W)
            except Exception:
                self.sess = requests.Session()
            time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"fetch failed {url}")


def sample_volume(vol, pts):
    """pts: (N,3) float voxel coords (x,y,z). Returns nearest-neighbor uint16 values."""
    xi = np.clip(np.rint(pts[:, 0]).astype(np.int64), 0, W - 1)
    yi = np.clip(np.rint(pts[:, 1]).astype(np.int64), 0, H - 1)
    zi = np.clip(np.rint(pts[:, 2]).astype(np.int64), 0, NSLICES[vol] - 1)
    out = np.zeros(len(pts), dtype=np.uint16)
    reader = SliceReader(vol)
    by_z = defaultdict(list)
    for i, z in enumerate(zi):
        by_z[int(z)].append(i)
    print(f"[{vol}] slices needed: {len(by_z)}", flush=True)
    t0 = time.time()
    for n, (z, idxs) in enumerate(sorted(by_z.items())):
        idxs = np.array(idxs)
        ys = yi[idxs]
        # fetch only needed row-runs, coalescing gaps < 64 rows
        uy = np.unique(ys)
        splits = np.where(np.diff(uy) > 64)[0] + 1
        for run in np.split(uy, splits):
            cy, cy2 = int(run[0]), int(run[-1])
            sel = idxs[(ys >= cy) & (ys <= cy2)]
            rows = SliceReader.fetch_rows(reader, z, cy, cy2)
            out[sel] = rows[yi[sel] - cy, xi[sel]]
        if n % 20 == 0:
            print(f"[{vol}] {n}/{len(by_z)} slices, {time.time()-t0:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppm", required=True)
    ap.add_argument("--u0", type=int, default=-1)
    ap.add_argument("--v0", type=int, default=-1)
    ap.add_argument("--size", type=int, default=800)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--depth", type=int, default=10, help="+- voxels along normal at 3.24um")
    ap.add_argument("--zstep", type=int, default=2)
    ap.add_argument("--out_prefix", default="patch")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    A = load_affine(os.path.join(here, "20231027191953-to-20231117143551.json"))  # 3.24(53) -> 7.91
    B = load_affine(os.path.join(here, "20231027191953-to-20231201141544.json"))  # 53 -> 70 (3.24)
    Ainv = np.linalg.inv(A)

    xyz, nrm, (w, h) = read_ppm_patch(
        args.ppm,
        args.v0 if args.v0 >= 0 else 0,
        args.u0 if args.u0 >= 0 else 0,
        args.size, args.step)
    if args.u0 < 0:  # default: center patch
        xyz, nrm, _ = read_ppm_patch(args.ppm, (2491 - args.size) // 2, (25706 - args.size) // 2, args.size, args.step)
    ph, pw = xyz.shape[:2]
    valid = np.linalg.norm(xyz.reshape(-1, 3), axis=1) > 0
    print(f"patch {ph}x{pw}, valid {valid.mean()*100:.0f}%", flush=True)

    # to 3.24um 53keV frame
    P = xyz.reshape(-1, 3)
    P53 = (Ainv @ np.c_[P, np.ones(len(P))].T).T[:, :3]
    Nv = nrm.reshape(-1, 3)
    R = Ainv[:3, :3] / np.linalg.norm(Ainv[:3, :3], axis=0).mean()
    N53 = (R @ Nv.T).T
    N53 /= np.clip(np.linalg.norm(N53, axis=1, keepdims=True), 1e-9, None)

    offsets = np.arange(-args.depth, args.depth + 1, args.zstep, dtype=np.float64)
    print(f"sampling {len(offsets)} depth offsets x {valid.sum()} px x 2 energies", flush=True)

    stacks = {}
    all_valid = np.tile(valid, len(offsets))
    for vol, transform in ((VOL53, None), (VOL70, B)):
        pts_all = np.concatenate([P53 + N53 * t for t in offsets])
        if transform is not None:
            pts_all = (transform @ np.c_[pts_all, np.ones(len(pts_all))].T).T[:, :3]
        vals = np.zeros(len(pts_all), dtype=np.uint16)
        vals[all_valid] = sample_volume(vol, pts_all[all_valid])
        stacks[vol] = vals.reshape(len(offsets), ph, pw)
        np.save(f"{args.out_prefix}_{vol}.npy", stacks[vol])
        print(f"[{vol}] done", flush=True)

    import cv2
    for name, arr in (("53", stacks[VOL53]), ("70", stacks[VOL70])):
        m = arr.max(axis=0).astype(np.float32)
        m = (m - m.min()) / max(1, np.ptp(m)) * 255
        cv2.imwrite(f"{args.out_prefix}_max{name}.png", m.astype(np.uint8))
    a = stacks[VOL53].mean(axis=0).astype(np.float32)
    b = stacks[VOL70].mean(axis=0).astype(np.float32)
    ratio = (a + 1) / (b + 1)
    lo, hi = np.percentile(ratio[ratio > 0], [2, 98])
    rn = np.clip((ratio - lo) / max(1e-6, hi - lo), 0, 1) * 255
    cv2.imwrite(f"{args.out_prefix}_ratio.png", rn.astype(np.uint8))
    print("saved ratio + max maps", flush=True)


if __name__ == "__main__":
    main()
