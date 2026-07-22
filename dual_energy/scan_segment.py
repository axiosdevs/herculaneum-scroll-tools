import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample_patch import load_affine, read_ppm_patch, sample_volume, VOL53, VOL70

PPM = "20240618142020.ppm"
PATCH = 600      # ppm px (7.91um) per patch side
STEP = 4         # sampling stride inside patch
OFFSETS = np.array([-4.0, 0.0, 4.0])  # depth offsets, voxels at 3.24um
OUT = "scan"

here = os.path.dirname(os.path.abspath(__file__))
A = load_affine(os.path.join(here, "20231027191953-to-20231117143551.json"))
B = load_affine(os.path.join(here, "20231027191953-to-20231201141544.json"))
Ainv = np.linalg.inv(A)
Rrot = Ainv[:3, :3] / np.linalg.norm(Ainv[:3, :3], axis=0).mean()


def process_patch(args):
    try:
        return _process_patch(args)
    except Exception as e:
        return f"patch_v{args[0]}_u{args[1]}", f"FAILED {type(e).__name__}"


def _process_patch(args):
    v0, u0 = args
    tag = f"{OUT}/patch_v{v0}_u{u0}.npz"
    if os.path.exists(tag):
        return tag, "cached"
    xyz, nrm, _ = read_ppm_patch(os.path.join(here, PPM), v0, u0, PATCH, STEP)
    ph, pw = xyz.shape[:2]
    P = xyz.reshape(-1, 3)
    valid = np.linalg.norm(P, axis=1) > 0
    if valid.mean() < 0.3:
        np.savez_compressed(tag, empty=True)
        return tag, "empty"
    P53 = (Ainv @ np.c_[P, np.ones(len(P))].T).T[:, :3]
    N = nrm.reshape(-1, 3)
    N53 = (Rrot @ N.T).T
    N53 /= np.clip(np.linalg.norm(N53, axis=1, keepdims=True), 1e-9, None)

    all_valid = np.tile(valid, len(OFFSETS))
    out = {}
    for vol, transform in ((VOL53, None), (VOL70, B)):
        pts = np.concatenate([P53 + N53 * t for t in OFFSETS])
        if transform is not None:
            pts = (transform @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
        vals = np.zeros(len(pts), dtype=np.uint16)
        vals[all_valid] = sample_volume(vol, pts[all_valid])
        out[vol] = vals.reshape(len(OFFSETS), ph, pw)
    np.savez_compressed(tag, v53=out[VOL53], v70=out[VOL70], valid=valid.reshape(ph, pw))
    return tag, "done"


def main():
    os.makedirs(OUT, exist_ok=True)
    W, H = 25706, 2491
    jobs = [(v0, u0) for v0 in range(0, H - PATCH + 1, PATCH) for u0 in range(0, W - PATCH + 1, PATCH)]
    print(f"{len(jobs)} patches", flush=True)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=16) as ex:
        for tag, status in ex.map(process_patch, jobs):
            done += 1
            print(f"[{done}/{len(jobs)}] {status} {tag} {time.time()-t0:.0f}s", flush=True)
    print("SCAN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
