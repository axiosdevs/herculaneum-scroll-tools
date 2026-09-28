"""Audit `detectability.orientation` against published ink maps whose layer order is known.

The orientation check -- plant ink on each face, read the stack both ways, take the one
combination that answers -- was built and validated on a single PHerc0139 window, where it is
right as published and right again with the stack flipped in depth. This asks whether that
generalises. For each published segment it picks the window with the most map coverage, reads
the centre 62 layers of the segment's own surface volume both ways with the published
checkpoint, and runs the check. The team's order is taken from the map itself: whichever read
correlates with the published map.

Result on the 44 segments audited, all 18 of PHerc0009B and 26 of PHerc0139
(`orientation_audit.jsonl`):

    the team's order                forward on all 44 (forward read vs published map r >= 0.56,
                                    reverse read r <= 0.50)
    the check says forward          12
    the check says reverse          16
    the check gives no verdict      16

No better than a coin. It is not a way to choose a layer order, and nothing in this repository
relies on it for one any more.

    SHARD=0 NSHARD=4 CKPT=r152.ckpt python ink/audit/orientation_audit.py

Needs a CUDA card and the published checkpoint; about ninety seconds a segment on a 4090.
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np, requests, numcodecs, fsspec, tifffile, zarr
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import inference_env                                              # noqa: E402
sys.path.insert(0, inference_env.ensure())
from canonical_ink import predict                                 # noqa: E402
from model_resnet3d_3d_decoder import load_model                  # noqa: E402
from detectability import plant_ink                               # noqa: E402
from text_score import text_score                                 # noqa: E402

B = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
CKPT = os.environ.get("CKPT", "r152.ckpt")
W, NL, AMP, TH = 1920, 62, 32, 0.5
SHARD, NSHARD = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARD", "1"))
SEGMENTS = os.environ.get("SEGMENTS", os.path.join(HERE, "segments.json"))
OUT = os.environ.get("OUT", f"orientation_audit_{SHARD}.jsonl")
sess = requests.Session()
sess.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=48, max_retries=2))


def map_zarr(url):
    f = fsspec.open(url, "rb", block_size=2**20).open()
    tf = tifffile.TiffFile(f)
    return zarr.open(tf.pages[0].aszarr(), mode="r"), tf


def read_volume(base, z0, y0, x0):
    meta = sess.get(base + ".zarray", timeout=60).json()
    ch = np.array(meta["chunks"]); sep = meta.get("dimension_separator", "/")
    codec = numcodecs.get_codec(meta["compressor"]) if meta["compressor"] else None
    out = np.zeros((NL, W, W), np.uint8)
    keys = sorted({(z // ch[0], y // ch[1], x // ch[2])
                   for z in list(range(z0, z0 + NL, ch[0])) + [z0 + NL - 1]
                   for y in range(y0, y0 + W, ch[1]) for x in range(x0, x0 + W, ch[2])})

    def get(k):
        for _ in range(4):
            try:
                r = sess.get(base + sep.join(map(str, k)), timeout=45)
                if r.status_code == 404:
                    return k, None
                if r.status_code == 200:
                    raw = codec.decode(r.content) if codec else r.content
                    b = np.frombuffer(raw, np.uint8)
                    return k, (b.reshape(tuple(ch)) if b.size == int(np.prod(ch)) else None)
            except Exception:
                time.sleep(2)
        return k, None

    with ThreadPoolExecutor(32) as ex:
        for k, blk in ex.map(get, keys):
            if blk is None:
                continue
            o = np.array(k) * ch
            zs, ze = max(z0, o[0]), min(z0 + NL, o[0] + ch[0])
            ys, ye = max(y0, o[1]), min(y0 + W, o[1] + ch[1])
            xs, xe = max(x0, o[2]), min(x0 + W, o[2] + ch[2])
            if ze > zs and ye > ys and xe > xs:
                out[zs - z0:ze - z0, ys - y0:ye - y0, xs - x0:xe - x0] = \
                    blk[zs - o[0]:ze - o[0], ys - o[1]:ye - o[1], xs - o[2]:xe - o[2]]
    return out


def corr(a, b):
    a = a.ravel().astype(np.float32); b = b.ravel().astype(np.float32)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def audit(r):
    rec = {"segment": r["segment"], "ink_map": r["ink_maps"][0]}
    mz, tf = map_zarr(B + r["ink_maps"][0])
    s = 32
    coarse = np.asarray(mz[::s, ::s])
    cov = (coarse > 0).astype(np.float32); ink = (coarse > 127).astype(np.float32)
    k = W // s
    if coarse.shape[0] <= k or coarse.shape[1] <= k:
        raise ValueError("map smaller than the window")
    cs = sliding_window_view(cov, (k, k))[::4, ::4].mean(axis=(-1, -2))
    iy, ix = np.unravel_index(int(np.argmax(cs)), cs.shape)
    y0, x0 = iy * 4 * s, ix * 4 * s
    rec.update({"y0": int(y0), "x0": int(x0), "coverage": round(float(cs.max()), 3),
                "map_ink_pct_segment": round(float(ink[cov > 0].mean() * 100) if cov.sum() else 0.0, 2)})
    pub = np.asarray(mz[y0:y0 + W, x0:x0 + W]).astype(np.float32) / 255.0
    tf.close()
    base = B + r["volume"] + "0/"
    L = sess.get(base + ".zarray", timeout=60).json()["shape"][0]
    z0 = max(0, (L - NL) // 2)
    stack = read_volume(base, z0, y0, x0)
    rec.update({"layers": int(L), "z0": int(z0)})

    def pf(st, rev):
        return predict(st, CKPT, load_model, dev="cuda", reverse=rev)

    bf, br = pf(stack, False), pf(stack, True)
    lifts = {}
    for rev, b0 in ((False, bf), (True, br)):
        for face in ("near", "far"):
            planted, mask = plant_ink(stack, AMP, 2.4, face=face)
            got = pf(planted, rev)
            m = mask.astype(bool)
            lifts[("reverse" if rev else "forward") + " " + face] = round(
                float((got[m] > TH).mean() - (b0[m] > TH).mean()), 3)
    ranked = sorted(lifts.items(), key=lambda kv: kv[1], reverse=True)
    clear = ranked[0][1] >= 0.02 and ranked[0][1] - ranked[1][1] >= 0.01
    verdict = ranked[0][0] if clear else None
    cf, cr = corr(bf, pub), corr(br, pub)
    team = "forward" if cf >= cr else "reverse"
    rec.update({"lifts": lifts, "verdict": verdict, "corr_pub_fwd": round(cf, 3),
                "corr_pub_rev": round(cr, 3), "team_order": team,
                "pub_ink_pct": round(float((pub > TH).mean() * 100), 2),
                "fwd_ink_pct": round(float((bf > TH).mean() * 100), 2),
                "rev_ink_pct": round(float((br > TH).mean() * 100), 2),
                "text_pub": round(float(text_score(pub)[0]), 3),
                "text_fwd": round(float(text_score(bf)[0]), 3),
                "text_rev": round(float(text_score(br)[0]), 3)})
    rec["disagree"] = verdict is not None and verdict.split()[0] != team
    return rec


def main():
    rows = json.load(open(SEGMENTS))[SHARD::NSHARD]
    done = {json.loads(l)["segment"] for l in open(OUT)} if os.path.exists(OUT) else set()
    for r in rows:
        if r["segment"] in done:
            continue
        t0 = time.time()
        try:
            rec = audit(r)
        except Exception as exc:
            rec = {"segment": r["segment"], "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
        rec["seconds"] = round(time.time() - t0)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{r['segment'][:44]:46s} check {str(rec.get('verdict')):14s} "
              f"team {str(rec.get('team_order')):8s} {'DISAGREE' if rec.get('disagree') else ''}"
              f" {rec.get('error', '')} ({rec['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
