#!/usr/bin/env python3
"""Full-batch voxel-level phantom surveys for every sample in the 2026-04-13
m7 surface-prediction batch (villa#1114). Auto-discovers each sample's
prediction zarr + matching masked-CT zarr/level from S3, then runs the
retry-hardened strided survey. Fully resumable: finished samples are skipped.
"""
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ct_support as cs

S3 = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
RUN_ID = "20260413222639"

# sample -> volume id (from Schurkai audit table, villa#1114)
SAMPLES = {
 "PHercMANBp":"20251216152116", "PHerc0500P2":"20250526151718",
 "PHerc0343P":"20260304131111", "PHerc0009B":"20260319104112",
 "PHercMAN5":"20260311104824", "PHerc0332":"20251211183505",
 "PHerc1299":"20260309130042", "PHerc1451":"20260319101107",
 "PHerc0826":"20250821151701", "PHerc1545":"20250821151648",
 "PHerc0483A":"20250521140913", "PHerc0814":"20260309142202",
 "PHerc0846B":"20250804142305", "PHercMANB":"20260323091048",
 "PHerc1218":"20250521120456", "PHerc0257":"20250821151750",
 "PHerc0211":"20250821151803", "PHerc0175A":"20250521115057",
 "PHerc0306B":"20250521133212", "PHerc0358":"20250821151737",
 "PHerc0125":"20250821151825", "PHerc1447":"20250521151220",
 "PHerc0490A":"20250521151210", "PHerc0343":"20250521140437",
 "PHerc0846A":"20260319102732", "PHerc0191":"20250821151635",
 "PHerc0800":"20250521135224", "PHerc0175B":"20250521125822",
 "PHercParis4":"20260411134726", "PHerc0813":"20250821151723",
 "PHerc0841":"20260319124803", "PHerc0490B":"20250521151215",
 "PHerc0483B":"20251124083638", "PHerc0139":"20250728140407",
 "PHerc0268":"20251110183117", "PHerc1203":"20260319130212",
}


def list_prefixes(prefix):
    url = f"{S3}/?list-type=2&prefix={prefix}&delimiter=/"
    with urllib.request.urlopen(url, timeout=30) as r:
        text = r.read().decode()
    return re.findall(r"<Prefix>([^<]+)</Prefix>", text)


def discover(sample, volid):
    preds = None
    for p in list_prefixes(f"{sample}/representations/predictions/surfaces/"):
        if volid in p and RUN_ID in p and p.endswith(".zarr/"):
            preds = f"{S3}/{p.rstrip('/')}"
            break
    ct = None
    for p in list_prefixes(f"{sample}/volumes/"):
        if volid in p and "masked" in p and p.endswith(".zarr/"):
            ct = f"{S3}/{p.rstrip('/')}"
            break
    if not preds or not ct:
        return None
    m = re.search(r"m7-L(\d)", preds)
    ct_level = m.group(1) if m else "2"
    return preds, ct, ct_level


class Args:  # namespace for cs.cmd_survey
    pass


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    GP = ["PHerc1203", "PHerc0268", "PHerc0139", "PHerc0813", "PHerc0125",
          "PHerc1447", "PHerc0800", "PHerc0358", "PHerc0211", "PHerc0257",
          "PHerc1218", "PHerc1545", "PHerc0826", "PHerc0191"]
    # grand-prize-eligible scrolls first (teams pick targets from these),
    # cheapest-first within the group; then the rest alphabetically.
    order = [s for s in GP if s in SAMPLES] + sorted(set(SAMPLES) - set(GP))
    if len(sys.argv) > 1:
        order = sys.argv[1:]
    done, failed = [], []
    for i, name in enumerate(order):
        out = os.path.join(here, f"survey_{name}.json")
        if os.path.exists(out):
            print(f"[{i+1}/36] skip {name} (done)", flush=True)
            done.append(name)
            continue
        try:
            disc = discover(name, SAMPLES[name])
        except Exception as e:
            print(f"[{i+1}/36] {name} DISCOVERY_FAIL {type(e).__name__}", flush=True)
            failed.append(name)
            continue
        if not disc:
            print(f"[{i+1}/36] {name} ARTIFACTS_NOT_FOUND", flush=True)
            failed.append(name)
            continue
        preds, ct, lvl = disc
        print(f"[{i+1}/36] SURVEY {name} (ct-level {lvl})", flush=True)
        a = Args()
        a.preds, a.ct = preds, ct
        a.preds_level, a.ct_level = "0", lvl
        a.thr, a.out = 127, out
        a.slab_stride = 12
        t0 = time.time()
        try:
            cs.cmd_slabs(a)
            done.append(name)
            print(f"    {name} OK in {(time.time()-t0)/60:.0f} min", flush=True)
        except SystemExit as e:
            print(f"    {name} GRID_FAIL: {e}", flush=True)
            failed.append(name)
        except Exception as e:
            print(f"    {name} FAIL: {type(e).__name__}: {str(e)[:120]}", flush=True)
            failed.append(name)
    print(f"FULL_BATCH_DONE ok={len(done)} failed={len(failed)} {failed}", flush=True)


if __name__ == "__main__":
    main()
