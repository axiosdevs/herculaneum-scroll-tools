#!/usr/bin/env python3
"""Winding-constraint annotator: draw paths on a flattened segment render, export
native spiral-fit constraint files (same_windings.json / relative_windings.json).

Workflow
--------
1. Open a flattened surface render of a segment (PNG/TIFF you already have, e.g.
   from vc_render or render_tifxyz.py) together with its tifxyz directory.
2. Click points along a papyrus feature (a text row, fiber, or column gap).
   Each *collection* of points is one constraint:
     - in `same` mode every point in the collection lies on the SAME wrap;
     - in `relative` mode each point carries a winding value `wind_a`
       (constant per collection unless you change it), expressing wraps
       relative to the collection's reference.
3. Export: points are mapped through the tifxyz to full-resolution volume
   voxels (x, y, z) and written in the exact JSON schema used by the
   spiral-input dataset (VC3D point collections), so they drop straight into
   the spiral-fit inputs.

Keys
----
  left-click  add point to current collection
  n           start a new collection
  m           toggle mode same <-> relative
  +/-         (relative mode) increment / decrement wind_a for new points
  u           undo last point
  w           write JSON files and print a summary
  q           quit

Non-interactive check: `--selftest` synthesizes a few collections and verifies
the exported files round-trip and map into the volume, printing a report
(useful on headless machines / CI).
"""
import argparse
import json
import os
import time

import numpy as np

from tifxyz_map import TifxyzMap, fetch_remote

PALETTE = [(0.00, 0.75, 0.63), (0.84, 0.39, 0.78), (0.58, 0.06, 0.08),
           (0.10, 0.45, 0.85), (0.95, 0.60, 0.10), (0.35, 0.70, 0.20)]


class Session:
    def __init__(self, tif_map):
        self.map = tif_map
        self.collections = []          # each: dict(name, mode, wind_a, pts=[(u,v,xyz)])
        self.mode = "same"
        self.wind_a = 0.0
        self.new_collection()

    def new_collection(self):
        idx = len(self.collections) + 1
        name = (f"same_wrap{idx}" if self.mode == "same" else f"wraps{idx}")
        self.collections.append({"name": name, "mode": self.mode,
                                 "wind_a": self.wind_a, "pts": []})

    def add(self, u, v):
        xyz = self.map.lookup_render_px(u, v)
        if xyz is None:
            return None
        self.collections[-1]["pts"].append((float(u), float(v), xyz))
        return xyz

    def undo(self):
        if self.collections[-1]["pts"]:
            self.collections[-1]["pts"].pop()

    def export(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        now = int(time.time() * 1000)
        outs = {"same": {"collections": {}}, "relative": {"collections": {}}}
        cid = {"same": 0, "relative": 0}
        for c in self.collections:
            if not c["pts"]:
                continue
            kind = "same" if c["mode"] == "same" else "relative"
            cid[kind] += 1
            k = str(cid[kind])
            col = {
                "color": list(PALETTE[(cid[kind] - 1) % len(PALETTE)]),
                "metadata": {"winding_is_absolute": False},
                "name": c["name"],
                "points": {},
            }
            if kind == "relative":
                col["autoFillConstant"] = 0.0
                col["autoFillMode"] = 1
            for i, (u, v, xyz) in enumerate(c["pts"]):
                col["points"][str(i)] = {
                    "creation_time": now + i,
                    "p": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                    "wind_a": (None if kind == "same" else float(c["wind_a"])),
                }
            outs[kind]["collections"][k] = col
        paths = {}
        for kind, fname in (("same", "same_windings.json"),
                            ("relative", "relative_windings.json")):
            if outs[kind]["collections"]:
                p = os.path.join(out_dir, fname)
                json.dump(outs[kind], open(p, "w"), indent=1)
                paths[kind] = p
        return paths


def run_gui(a):
    import matplotlib
    import matplotlib.pyplot as plt
    import cv2
    img = cv2.imread(a.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"cannot read {a.image}")
    tm = TifxyzMap(a.tifxyz)
    ses = Session(tm)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(img, cmap="gray")
    title = lambda: ax.set_title(
        f"mode={ses.mode} wind_a={ses.wind_a:+.1f} | collection "
        f"#{len(ses.collections)} ({len(ses.collections[-1]['pts'])} pts) — "
        "click=add n=new m=mode +/-=wind u=undo w=write q=quit")
    title()
    art = []

    def redraw():
        for x in art:
            x.remove()
        art.clear()
        for i, c in enumerate(ses.collections):
            if not c["pts"]:
                continue
            us = [p[0] for p in c["pts"]]; vs = [p[1] for p in c["pts"]]
            col = PALETTE[i % len(PALETTE)]
            art.append(ax.plot(us, vs, "-o", ms=4, color=col)[0])
        title(); fig.canvas.draw_idle()

    def on_click(e):
        if e.inaxes != ax or e.button != 1 or e.xdata is None:
            return
        if ses.add(e.xdata, e.ydata) is None:
            print("point outside valid surface — ignored")
        redraw()

    def on_key(e):
        if e.key == "n":
            ses.new_collection()
        elif e.key == "m":
            ses.mode = "relative" if ses.mode == "same" else "same"
            ses.new_collection()
        elif e.key == "+":
            ses.wind_a += 1; ses.collections[-1]["wind_a"] = ses.wind_a
        elif e.key == "-":
            ses.wind_a -= 1; ses.collections[-1]["wind_a"] = ses.wind_a
        elif e.key == "u":
            ses.undo()
        elif e.key == "w":
            paths = ses.export(a.out)
            print("wrote:", paths)
        elif e.key == "q":
            plt.close(fig)
        redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    paths = ses.export(a.out)
    print("final export:", paths)


def run_selftest(a):
    tm = TifxyzMap(a.tifxyz)
    ses = Session(tm)
    h, w = tm.gh / tm.scale, tm.gw / tm.scale   # render-pixel extents
    rng = np.random.default_rng(0)
    added = 0
    for ci in range(3):
        if ci:
            ses.new_collection()
        v = (0.2 + 0.3 * ci) * h
        for u in np.linspace(0.1 * w, 0.9 * w, 12):
            if ses.add(u, v + rng.normal(0, 2)) is not None:
                added += 1
    ses.mode = "relative"; ses.wind_a = 3.0; ses.new_collection()
    for u in np.linspace(0.2 * w, 0.8 * w, 8):
        ses.add(u, 0.85 * h)
    paths = ses.export(a.out)
    print(f"selftest: added {added} same pts + relative collection")
    for kind, p in paths.items():
        d = json.load(open(p))
        ncol = len(d["collections"])
        npts = sum(len(c["points"]) for c in d["collections"].values())
        one = next(iter(next(iter(d["collections"].values()))["points"].values()))
        print(f"  {kind}: {ncol} collections, {npts} points, sample p={one['p']}, "
              f"wind_a={one['wind_a']}")
    print("SELFTEST_OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="flattened surface render (PNG/TIFF)")
    ap.add_argument("--tifxyz", required=True, help="dir with x/y/z.tif + meta.json "
                    "(use winding/fetch to pull a remote one)")
    ap.add_argument("--out", default="constraints_out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        run_selftest(a)
    else:
        if not a.image:
            raise SystemExit("--image is required (or use --selftest)")
        run_gui(a)


if __name__ == "__main__":
    main()
