"""tifxyz_map: flattened (u,v) -> volume (x,y,z) lookup for tifxyz meshes.

A tifxyz mesh is three TIFFs (x.tif, y.tif, z.tif) holding, per flattened grid
cell, the full-resolution voxel coordinate of the papyrus surface, plus a
meta.json whose "scale" gives the flattened-grid -> render-pixel factor
(e.g. scale 0.05 means 1 grid cell = 20 render pixels).

Invalid/missing cells are encoded as 0 or -1 in all three channels; this module
treats any cell with a non-positive vector norm as missing and interpolates only
across valid neighbours.
"""
import json
import os

import numpy as np
import tifffile


class TifxyzMap:
    def __init__(self, tifxyz_dir):
        self.dir = tifxyz_dir
        self.X = tifffile.imread(os.path.join(tifxyz_dir, "x.tif")).astype(np.float64)
        self.Y = tifffile.imread(os.path.join(tifxyz_dir, "y.tif")).astype(np.float64)
        self.Z = tifffile.imread(os.path.join(tifxyz_dir, "z.tif")).astype(np.float64)
        meta_path = os.path.join(tifxyz_dir, "meta.json")
        self.meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        s = self.meta.get("scale", [1.0, 1.0])
        self.scale = float(s[0]) if isinstance(s, (list, tuple)) else float(s)
        self.valid = (np.abs(self.X) + np.abs(self.Y) + np.abs(self.Z)) > 0
        self.gh, self.gw = self.X.shape

    def render_to_grid(self, u_px, v_px):
        """Render-pixel coords -> fractional grid coords."""
        return u_px * self.scale, v_px * self.scale

    def lookup_grid(self, gu, gv):
        """Bilinear (x,y,z) at fractional grid coords; None if all corners missing."""
        u0, v0 = int(np.floor(gu)), int(np.floor(gv))
        if not (0 <= v0 < self.gh - 1 and 0 <= u0 < self.gw - 1):
            u0 = min(max(u0, 0), self.gw - 1)
            v0 = min(max(v0, 0), self.gh - 1)
            if not self.valid[v0, u0]:
                return None
            return np.array([self.X[v0, u0], self.Y[v0, u0], self.Z[v0, u0]])
        fu, fv = gu - u0, gv - v0
        w = np.array([(1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv])
        cs = [(v0, u0), (v0, u0 + 1), (v0 + 1, u0), (v0 + 1, u0 + 1)]
        vals, ws = [], []
        for wi, (vv, uu) in zip(w, cs):
            if self.valid[vv, uu]:
                vals.append([self.X[vv, uu], self.Y[vv, uu], self.Z[vv, uu]])
                ws.append(wi)
        if not ws:
            return None
        vals = np.array(vals); ws = np.array(ws); ws /= ws.sum()
        return (vals * ws[:, None]).sum(axis=0)

    def lookup_render_px(self, u_px, v_px):
        """Render-pixel coords (as drawn on a flattened image) -> (x,y,z) voxels."""
        gu, gv = self.render_to_grid(u_px, v_px)
        return self.lookup_grid(gu, gv)


def fetch_remote(seg_url, dest):
    """Download x/y/z.tif + meta.json of a remote tifxyz dir to dest/."""
    import urllib.request
    os.makedirs(dest, exist_ok=True)
    for f in ("x.tif", "y.tif", "z.tif", "meta.json"):
        p = os.path.join(dest, f)
        if not os.path.exists(p):
            urllib.request.urlretrieve(f"{seg_url}/{f}", p)
    return dest
