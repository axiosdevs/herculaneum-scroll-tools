"""Run the detectability probe: would this window have shown ink, if there were any?

`detectability.py` holds the measurement; this is the way to run it.

    python ink/probe.py --self-test      # seconds, no network: the probe on a known detector
    python ink/probe.py --reference      # the team's own PHerc0139 window, where letters read
    python ink/probe.py --stack mine.npy # a rendered stack of your own

The reference run is the one that makes the others mean something. It plants ink of known
contrast on the surface volume the published checkpoint reads letters from and reports the
faintest amplitude it recovers -- 32 of 255 when this was written. A pipeline that cannot
reach that on a stack known to be readable is not measuring the papyrus, and a null result
from it says nothing about what is written there.

Two things the reference run taught, both worth having before trusting your own number:

  * **Above about 32 the test inverts.** Planting 64 on the reference moved the recovered ink
    *down*, lift -0.163. Planting harder than the physics is a different experiment, not a
    stronger one, and a check run only at large amplitudes reports failure on a sound pipeline.
  * **The amount of ink reported says nothing about which way to read.** The wrong layer
    order reported 20 times less ink than the right one on PHerc0139 and 8 times more on a
    PHerc1451 surface. `--orientation` settles it instead: ink planted on each face, the stack
    read both ways, and the one combination that answers is the order and the face.

Needs torch and the published checkpoint for the reference and stack modes (downloaded once,
1.4 GB). The self-test needs neither.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detectability import probe, script_mask                       # noqa: E402

REFERENCE_SENSITIVITY = 32.0     # measured on PHerc0139's published surface volume
REFERENCE_WINDOW = (24, 86)      # the 62 layers of 109 that reproduce their letters, forward


def self_test():
    """The probe against two detectors whose answers are known in advance.

    No network, no checkpoint, a couple of seconds: enough to see that it fires on a detector
    that responds and says `blind` on one that does not -- including the case that fooled this
    module once, a detector whose output already looks like the planted mask without reacting
    to it at all.
    """
    um = 40.0        # coarse, so a small test window is still several line spacings across
    C, H, W = 62, 200, 200
    z = np.arange(C, dtype=np.float32)[:, None, None]
    stack = (40 + 120 * np.exp(-0.5 * ((z - 31) / 4.0) ** 2)
             + np.zeros((C, H, W), np.float32)).astype(np.uint8)

    def sees(s):
        return np.clip((s[26:34].astype(np.float32).max(0) - 150.0) / 40.0, 0, 1)

    rng = np.random.default_rng(3)
    deaf_out = rng.random((H, W)).astype(np.float32) * 0.2
    biased_out = script_mask((H, W), um).astype(np.float32) * 0.9

    cases = [("детектор, который реагирует", sees, "порог найден"),
             ("детектор, который молчит", lambda _s: deaf_out, "слепо"),
             ("детектор, чей выход похож на маску, но не реагирует", lambda _s: biased_out, "слепо")]
    ok = True
    for name, fn, want in cases:
        out = probe(stack, fn, amplitudes=(8, 32, 64), micron_per_pixel=um)
        got = "порог найден" if out["sensitivity"] is not None else "слепо"
        mark = "ок " if got == want else "ПЛОХО"
        ok &= got == want
        print(f"  {mark} {name}: {got} (порог {out['sensitivity']})")
    print("самопроверка пройдена" if ok else "САМОПРОВЕРКА ПРОВАЛЕНА")
    return 0 if ok else 1


def _predict_fn(ckpt, reverse, device):
    """The same inference path reproduce.py uses, so the probe measures what the survey runs."""
    import inference_env
    sys.path.insert(0, inference_env.ensure())
    from canonical_ink import predict
    from model_resnet3d_3d_decoder import load_model
    return lambda stack: predict(stack, ckpt, load_model, dev=device, reverse=reverse)


def run(stack, ckpt, amplitudes, reverse, device, micron_per_pixel):
    predict_fn = _predict_fn(ckpt, reverse, device)
    return probe(stack, predict_fn, amplitudes=amplitudes, micron_per_pixel=micron_per_pixel)


def report(result, label):
    s = result["sensitivity"]
    print(f"\n{label}")
    print(f"  фон: чернил {result['baseline_ink_pct']}%")
    for row in result["rows"]:
        print(f"  амплитуда {row['amplitude']:5.0f}   подъём {row['ink_lift']:+.3f}"
              f"   r с посаженной маской {row['r_with_planted']:+.3f}")
    if s is None:
        print("  ПОРОГ НЕ НАЙДЕН — окно слепое. Его пустота ничего не говорит о папирусе.")
    else:
        print(f"  порог {s:.0f} из 255 — текст такой силы это окно показало бы.")
        if s > REFERENCE_SENSITIVITY:
            print(f"  (эталон PHerc0139 — {REFERENCE_SENSITIVITY:.0f}; это окно менее чувствительно)")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true",
                    help="проверить сам инструмент, без сети и чекпойнта")
    ap.add_argument("--reference", action="store_true",
                    help="прогнать на томе PHerc0139, где модель читает буквы")
    ap.add_argument("--stack", help="свой стек: .npy формы (слои, H, W)")
    ap.add_argument("--orientation", action="store_true",
                    help="решить без разметки, в какую сторону читать стек и на какой грани чернила")
    ap.add_argument("--amplitudes", default="8,16,32,64",
                    help="силы посаженных чернил в уровнях серого")
    ap.add_argument("--reverse", action="store_true",
                    help="читать лист с другой стороны (сначала сравните обе)")
    ap.add_argument("--device", default="cuda" if os.environ.get("CUDA") else "cpu")
    ap.add_argument("--micron-per-pixel", type=float, default=2.399)
    ap.add_argument("--checkpoint", default="r152.ckpt")
    ap.add_argument("--json", help="куда положить результат")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not (args.reference or args.stack):
        ap.print_help()
        return 1

    amps = tuple(float(v) for v in args.amplitudes.split(","))
    if max(amps) > 64:
        print("осторожно: выше ~32 проба переворачивается — см. докстроку", file=sys.stderr)

    from reproduce import fetch_checkpoint, fetch_surface_window
    ckpt = fetch_checkpoint(args.checkpoint)
    out = {}

    if args.reference:
        lo, hi = REFERENCE_WINDOW
        print(f"качаю окно тома PHerc0139, слои {lo}-{hi}...", flush=True)
        full = fetch_surface_window(13200, 12280, 1920)
        res = run(full[lo:hi], ckpt, amps, args.reverse, args.device, args.micron_per_pixel)
        s = report(res, "эталон PHerc0139 (там, где модель читает буквы)")
        out["reference"] = res
        if s is None:
            print("\n  Проба не сработала на заведомо читаемом стеке — дело в конвейере,\n"
                  "  а не в папирусе. Сначала почините это.")

    if args.stack:
        stack = np.array(np.load(args.stack, mmap_mode="r"))
        print(f"\nваш стек {tuple(stack.shape)}", flush=True)
        if args.orientation:
            from detectability import orientation
            res = orientation(stack, lambda rev: _predict_fn(ckpt, rev, args.device),
                              amplitude=32, micron_per_pixel=args.micron_per_pixel)
            for combo, v in res["combinations"].items():
                print(f"  {combo:14s} подъём {v['lift']:+.3f}   фон {v['baseline_ink_pct']:.2f}%")
            if res["verdict"] is None:
                print("  ни одно сочетание не отвечает — стек слеп в обе стороны")
            else:
                order, face = res["verdict"]
                print(f"  читать: {'вперёд' if order == 'forward' else 'назад'}, "
                      f"чернила на {'ближней' if face == 'near' else 'дальней'} грани")
            out["orientation"] = res
        else:
            res = run(stack, ckpt, amps, args.reverse, args.device, args.micron_per_pixel)
            report(res, os.path.basename(args.stack))
            out["stack"] = res

    if args.json:
        json.dump(out, open(args.json, "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
