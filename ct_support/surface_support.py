"""Check a traced surface against the CT it is supposed to lie on.

`audit_ct_support` answers the question for a prediction volume; this answers it
one layer down, for the surface a tracer produced from one. The tracer follows
predicted sheets and has no support constraint, so a patch keeps growing into
regions the volume does not cover: measured on PHerc1218, 55% of a traced
surface stood over voxels where the masked CT reads exactly 0.

Given a tifxyz surface (the `x.tif` / `y.tif` / `z.tif` triple every tracer
writes) and the CT volume it should rest on, this reports what share of the
surface has material under it, writes a per-quad support map, and can emit a
trimmed copy keeping only the supported part.

    python surface_support.py report --surface path/to/segment --ct ct.zarr/0
    python surface_support.py trim   --surface path/to/segment --ct ct.zarr/0 \
        --out trimmed/ --dilation 1

Reads are chunk-batched, so a remote CT costs one request per touched chunk
rather than one per quad.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# tifxyz I/O — plain uncompressed float32 TIFFs, no third-party reader needed
# --------------------------------------------------------------------------- #

def read_tif(path: Path) -> np.ndarray:
    """Read a single-channel float32 TIFF (strip or tile layout, no compression)."""
    data = path.read_bytes()
    order = "<" if data[:2] == b"II" else ">"
    ifd = struct.unpack(order + "I", data[4:8])[0]
    count = struct.unpack(order + "H", data[ifd:ifd + 2])[0]

    tags: Dict[int, Tuple[int, int, int]] = {}
    for i in range(count):
        entry = ifd + 2 + i * 12
        tag, typ, n = struct.unpack(order + "HHI", data[entry:entry + 8])
        value = struct.unpack(order + "I", data[entry + 8:entry + 12])[0]
        if typ == 3 and n == 1:
            value = struct.unpack(order + "H", data[entry + 8:entry + 10])[0]
        tags[tag] = (typ, n, value)

    width, height = tags[256][2], tags[257][2]
    if tags.get(259, (0, 0, 1))[2] != 1:
        raise ValueError(f"{path}: compressed TIFFs are not supported")

    def offsets(tag: Tuple[int, int, int]) -> list:
        if tag[1] == 1:
            return [tag[2]]
        return list(struct.unpack(order + f"{tag[1]}I", data[tag[2]:tag[2] + 4 * tag[1]]))

    if 273 in tags:                                   # strip layout
        chunks = offsets(tags[273])
        sizes = offsets(tags[279])
        raw = b"".join(data[o:o + c] for o, c in zip(chunks, sizes))
        return np.frombuffer(raw, dtype=order + "f4").reshape(height, width)

    tile_w, tile_h = tags[322][2], tags[323][2]       # tile layout
    chunks, sizes = offsets(tags[324]), offsets(tags[325])
    across = (width + tile_w - 1) // tile_w
    out = np.zeros((height, width), np.float32)
    for index, (offset, size) in enumerate(zip(chunks, sizes)):
        ty, tx = divmod(index, across)
        tile = np.frombuffer(data[offset:offset + size], dtype=order + "f4")
        tile = tile.reshape(tile_h, tile_w)
        y0, x0 = ty * tile_h, tx * tile_w
        y1, x1 = min(y0 + tile_h, height), min(x0 + tile_w, width)
        out[y0:y1, x0:x1] = tile[:y1 - y0, :x1 - x0]
    return out


def read_surface(directory: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (points, valid) for a tifxyz surface: points are (H, W, 3) in z,y,x."""
    x = read_tif(directory / "x.tif")
    y = read_tif(directory / "y.tif")
    z = read_tif(directory / "z.tif")
    valid = (x > 0) & (y > 0) & (z > 0)
    return np.stack([z, y, x], axis=-1), valid


def write_surface(directory: Path, points: np.ndarray, valid: np.ndarray,
                  meta: Optional[dict] = None) -> None:
    """Write a tifxyz triple, blanking quads that are not valid."""
    directory.mkdir(parents=True, exist_ok=True)
    for axis, name in ((2, "x.tif"), (1, "y.tif"), (0, "z.tif")):
        plane = np.where(valid, points[..., axis], -1.0).astype("<f4")
        _write_tif(directory / name, plane)
    if meta is not None:
        (directory / "meta.json").write_text(json.dumps(meta, indent=1))


def _write_tif(path: Path, plane: np.ndarray) -> None:
    height, width = plane.shape
    body = plane.tobytes()
    header_size = 8
    n_entries = 9
    data_offset = header_size + 2 + 12 * n_entries + 4   # header + IFD + next-IFD pointer
    entries = [(256, 4, width), (257, 4, height), (258, 3, 32), (259, 3, 1),
               (262, 3, 1), (273, 4, data_offset),
               (277, 3, 1), (279, 4, len(body)), (339, 3, 3)]
    assert len(entries) == n_entries
    ifd = struct.pack("<H", len(entries))
    for tag, typ, value in entries:
        ifd += struct.pack("<HHII", tag, typ, 1, value)
    ifd += struct.pack("<I", 0)
    path.write_bytes(b"II\x2a\x00" + struct.pack("<I", header_size) + ifd + body)


# --------------------------------------------------------------------------- #
# support test
# --------------------------------------------------------------------------- #

def open_ct(path: str, anon: bool = False):
    """Open a CT volume: local zarr array, or one over http(s)/s3."""
    import zarr

    if path.startswith(("s3://", "gs://", "gcs://")):
        import fsspec
        from zarr.storage import FsspecStore
        protocol, rest = path.split("://", 1)
        options = {"anon": True} if anon and protocol in ("s3", "gs", "gcs") else {}
        fs = fsspec.filesystem(protocol, asynchronous=True, **options)
        return zarr.open_array(FsspecStore(fs, path=rest), mode="r")
    return zarr.open_array(path, mode="r")


def support_map(points: np.ndarray, valid: np.ndarray, ct,
                threshold: int = 0, dilation: int = 0) -> np.ndarray:
    """Per-quad support: True where the CT holds material under the surface point.

    Points are grouped by CT chunk so each chunk is fetched once, which keeps a
    remote volume to one request per touched chunk instead of one per quad.
    """
    supported = np.zeros(valid.shape, dtype=bool)
    if not valid.any():
        return supported

    shape = np.asarray(ct.shape)
    chunks = np.asarray(ct.chunks)
    rows, cols = np.nonzero(valid)
    voxels = np.rint(points[rows, cols]).astype(np.int64)

    inside = np.all((voxels >= 0) & (voxels < shape), axis=1)
    rows, cols, voxels = rows[inside], cols[inside], voxels[inside]
    if len(voxels) == 0:
        return supported

    keys = voxels // chunks
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    rows, cols, voxels, keys = rows[order], cols[order], voxels[order], keys[order]
    boundaries = np.flatnonzero(np.any(np.diff(keys, axis=0) != 0, axis=1)) + 1

    for start, stop in zip(np.r_[0, boundaries], np.r_[boundaries, len(keys)]):
        key = keys[start]
        lo = key * chunks
        hi = np.minimum(lo + chunks, shape)
        if dilation:
            lo = np.maximum(lo - dilation, 0)
            hi = np.minimum(hi + dilation, shape)
        block = np.asarray(ct[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]) > threshold
        if dilation:
            block = _dilate(block, dilation)
        local = voxels[start:stop] - lo
        supported[rows[start:stop], cols[start:stop]] = block[
            local[:, 0], local[:, 1], local[:, 2]]

    return supported


def _dilate(block: np.ndarray, radius: int) -> np.ndarray:
    """Chebyshev dilation by shifting; radius is small, so shifts beat a filter."""
    out = block.copy()
    for dz in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dz == dy == dx == 0:
                    continue
                out |= _shift(block, dz, dy, dx)
    return out


def _shift(block: np.ndarray, dz: int, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(block)
    src = tuple(slice(max(0, -d), block.shape[i] - max(0, d))
                for i, d in enumerate((dz, dy, dx)))
    dst = tuple(slice(max(0, d), block.shape[i] - max(0, -d))
                for i, d in enumerate((dz, dy, dx)))
    out[dst] = block[src]
    return out


def report(points: np.ndarray, valid: np.ndarray, supported: np.ndarray,
           voxelsize_um: Optional[float] = None) -> dict:
    total = int(valid.sum())
    kept = int((valid & supported).sum())
    out = {
        "quads": total,
        "supported": kept,
        "unsupported": total - kept,
        "frac_supported": (kept / total) if total else 0.0,
        "frac_unsupported": ((total - kept) / total) if total else 0.0,
    }
    if voxelsize_um:
        # one quad spans one grid step; area in cm2 at the surface's own sampling
        area = (voxelsize_um * 1e-4) ** 2
        out["area_cm2"] = total * area
        out["supported_area_cm2"] = kept * area
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _load(args):
    directory = Path(args.surface)
    points, valid = read_surface(directory)
    ct = open_ct(args.ct, anon=args.anon)
    supported = support_map(points, valid, ct, threshold=args.threshold,
                            dilation=args.dilation)
    meta_path = directory / "meta.json"
    voxelsize = None
    if meta_path.exists():
        try:
            voxelsize = json.loads(meta_path.read_text()).get("voxelsize")
        except Exception:
            voxelsize = None
    if args.voxelsize:
        voxelsize = args.voxelsize
    return directory, points, valid, supported, voxelsize


def cmd_report(args):
    _, points, valid, supported, voxelsize = _load(args)
    result = report(points, valid, supported, voxelsize)
    print(f"quads {result['quads']:,} | supported {result['frac_supported']:.4f} "
          f"| unsupported {result['frac_unsupported']:.4f} "
          f"({result['unsupported']:,} quads)")
    if "area_cm2" in result:
        print(f"area {result['area_cm2']:.4f} cm2, of which supported "
              f"{result['supported_area_cm2']:.4f} cm2")
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=1))
        print(f"wrote {args.output}")
    if args.map:
        _write_png(Path(args.map), valid, supported)
        print(f"wrote {args.map}")


def cmd_trim(args):
    _, points, valid, supported, voxelsize = _load(args)
    before = report(points, valid, supported, voxelsize)
    keep = valid & supported
    write_surface(Path(args.out), points, keep,
                  meta={"voxelsize": voxelsize} if voxelsize else None)
    print(f"kept {int(keep.sum()):,} of {before['quads']:,} quads "
          f"({before['frac_supported']:.4f}) -> {args.out}")
    if args.map:
        _write_png(Path(args.map), valid, supported)
        print(f"wrote {args.map}")


def _write_png(path: Path, valid: np.ndarray, supported: np.ndarray) -> None:
    """Grey where there is no surface, white supported, black unsupported."""
    image = np.full(valid.shape, 128, np.uint8)
    image[valid & supported] = 255
    image[valid & ~supported] = 0
    height, width = image.shape
    raw = b"".join(b"\x00" + image[i].tobytes() for i in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b""))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure and trim a traced surface against the CT under it.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--surface", required=True,
                       help="tifxyz directory holding x.tif / y.tif / z.tif")
        p.add_argument("--ct", required=True,
                       help="CT array (local path, or s3://... for a public bucket)")
        p.add_argument("--threshold", type=int, default=0,
                       help="CT value above which a voxel counts as material")
        p.add_argument("--dilation", type=int, default=0,
                       help="accept support within this many voxels (masks are imperfect)")
        p.add_argument("--voxelsize", type=float, default=None,
                       help="micrometres per voxel, for the area figures")
        p.add_argument("--anon", action="store_true", help="anonymous object storage access")
        p.add_argument("--map", default=None, help="write a per-quad support PNG here")

    rep = sub.add_parser("report", help="measure support, write nothing")
    common(rep)
    rep.add_argument("--output", default=None, help="write the JSON report here")
    rep.set_defaults(func=cmd_report)

    trim = sub.add_parser("trim", help="write a copy keeping only supported quads")
    common(trim)
    trim.add_argument("--out", required=True, help="output tifxyz directory")
    trim.set_defaults(func=cmd_trim)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
