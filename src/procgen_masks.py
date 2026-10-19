# SPDX-License-Identifier: MIT
"""Raster helpers for wear masks, numpy only (Blender's Python has neither scipy nor Pillow).

Near-Gaussian blur from box passes, smooth value noise, compact splats and downward runs
combined into potentials, coverage thresholds, and an RGB or grey PNG writer built on the
standard library.

Arrays are float32, rows first (row 0 at the top), values in 0..1. Positions are continuous
pixel coordinates: pixel (r, c) covers [r, r + 1) x [c, c + 1), so its centre is at +0.5.
Every random choice comes from a `random.Random` the caller seeds, so a mask is the same
for the same seed.
"""

from __future__ import annotations

import math
import random
import struct
import zlib
from pathlib import Path

import numpy as np


def box_blur(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    """Mean over a window of 2r + 1 pixels along `axis`, zero outside the array."""
    if r < 1:
        return a
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    c = np.zeros((n + 1, *a.shape[1:]), np.float64)
    np.cumsum(a, axis=0, dtype=np.float64, out=c[1:])
    idx = np.arange(n)
    hi = np.minimum(idx + r + 1, n)
    lo = np.maximum(idx - r, 0)
    out = ((c[hi] - c[lo]) / (2 * r + 1)).astype(np.float32)
    return np.moveaxis(out, 0, axis)


def box_radius(sigma: float) -> int:
    """Radius of three box passes with about this standard deviation: three passes of
    radius r have a variance of ((2r + 1)^2 - 1) / 4."""
    return max(0, round((math.sqrt(4 * sigma * sigma + 1) - 1) / 2))


def gauss_blur(a: np.ndarray, sx: float, sy: float | None = None) -> np.ndarray:
    """Near-Gaussian blur, sigma in pixels across columns (`sx`) and rows (`sy`)."""
    out = np.asarray(a, np.float32)
    for axis, s in ((0, sx if sy is None else sy), (1, sx)):
        r = box_radius(s)
        for _ in range(3):
            out = box_blur(out, r, axis)
    return out


def generator(rng: random.Random) -> np.random.Generator:
    """A numpy generator seeded from a stdlib one, for bulk random arrays."""
    return np.random.default_rng(rng.getrandbits(64))


def value_noise(
    shape: tuple[int, int],
    cell: float,
    rng: random.Random,
    octaves: int = 1,
    gain: float = 0.5,
    cell_y: float | None = None,
) -> np.ndarray:
    """Smooth value noise in 0..1: random values on a lattice every `cell` pixels across
    (`cell_y` down), smoothstep-interpolated, summed over octaves of doubling frequency."""
    h, w = shape
    gen = generator(rng)
    out = np.zeros(shape, np.float32)
    cx, cy = float(cell), float(cell if cell_y is None else cell_y)
    amp, total = 1.0, 0.0
    for _ in range(octaves):
        gw, gh = int(w / cx) + 3, int(h / cy) + 3
        lattice = gen.random((gh, gw), dtype=np.float32)
        xs = (np.arange(w) + 0.5) / cx
        ys = (np.arange(h) + 0.5) / cy
        ix, iy = xs.astype(int), ys.astype(int)
        tx, ty = xs - ix, ys - iy
        tx, ty = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)
        rows = lattice[:, ix] * (1 - tx) + lattice[:, ix + 1] * tx
        out += amp * (rows[iy] * (1 - ty)[:, None] + rows[iy + 1] * ty[:, None]).astype(
            np.float32
        )
        total += amp
        amp *= gain
        cx, cy = max(cx / 2, 1.0), max(cy / 2, 1.0)
    return out / total


def noise_1d(n: int, cell: float, rng: random.Random) -> np.ndarray:
    """Smooth 1-D value noise in 0..1 over n samples, a random value every `cell` samples."""
    k = int(n / cell) + 3
    knots = np.array([rng.random() for _ in range(k)], np.float32)
    t = (np.arange(n) + 0.5) / cell
    i = t.astype(int)
    f = t - i
    f = f * f * (3 - 2 * f)
    return knots[i] * (1 - f) + knots[i + 1] * f


def combine(window: np.ndarray, value: np.ndarray, mode: str) -> None:
    """Merge `value` into `window` in place: "max", "add" or "screen" (1 - (1-a)(1-b),
    which saturates at 1 so overlapping stains darken without overflowing)."""
    if mode == "max":
        np.maximum(window, value, out=window)
    elif mode == "add":
        window += value
    elif mode == "screen":
        window += value - window * value
    else:
        raise ValueError(mode)


def _window(dst: np.ndarray, r0: float, r1: float, c0: float, c1: float):
    h, w = dst.shape
    a, b = max(math.floor(r0), 0), min(math.ceil(r1) + 1, h)
    c, d = max(math.floor(c0), 0), min(math.ceil(c1) + 1, w)
    return a, b, c, d


def splat(
    dst: np.ndarray,
    row: float,
    col: float,
    ry: float,
    rx: float,
    amp: float = 1.0,
    mode: str = "max",
) -> None:
    """A compact bump (1 - d^2)^2 centred on (row, col) with radii ry, rx in pixels."""
    ry, rx = max(ry, 0.5), max(rx, 0.5)
    a, b, c, d = _window(dst, row - ry, row + ry, col - rx, col + rx)
    if a >= b or c >= d:
        return
    yy = (np.arange(a, b) + 0.5 - row) / ry
    xx = (np.arange(c, d) + 0.5 - col) / rx
    d2 = yy[:, None] ** 2 + xx[None, :] ** 2
    combine(
        dst[a:b, c:d], (amp * np.clip(1 - d2, 0, None) ** 2).astype(np.float32), mode
    )


def rect(
    dst: np.ndarray,
    r0: float,
    r1: float,
    c0: float,
    c1: float,
    amp: float = 1.0,
    soft: float = 1.0,
    mode: str = "max",
) -> None:
    """A filled rectangle of pixel rows r0..r1 and columns c0..c1 with edges ramped over
    `soft` pixels (antialiased rather than blurred)."""
    a, b, c, d = _window(dst, r0 - soft, r1 + soft, c0 - soft, c1 + soft)
    if a >= b or c >= d:
        return
    yy = np.arange(a, b) + 0.5
    xx = np.arange(c, d) + 0.5
    fy = np.clip(np.minimum(yy - r0, r1 - yy) / soft + 0.5, 0, 1)
    fx = np.clip(np.minimum(xx - c0, c1 - xx) / soft + 0.5, 0, 1)
    combine(dst[a:b, c:d], (amp * fy[:, None] * fx[None, :]).astype(np.float32), mode)


def run(
    dst: np.ndarray,
    row: float,
    col: float,
    length: float,
    width: tuple[float, float],
    amp: tuple[float, float],
    rng: random.Random,
    wobble: float = 0.0,
    beads: float = 0.0,
    mode: str = "screen",
    step: float = 40.0,
    neck: float = 1.0,
) -> None:
    """A streak running down from (row, col): `length` pixels long, its full width tapering
    from width[0] to width[1] and its strength from amp[0] to amp[1]. The centre wanders by a
    smooth random walk (`wobble` pixels per `step`), and `beads` modulates the strength
    along the run, as water leaves heavier deposits where it pauses. A `neck` below 1
    narrows the width quickly just below the head, as a stain leaving its source does."""
    n = max(round(length), 1)
    s = np.arange(n, dtype=np.float32) / n
    k = int(n / step) + 2
    walk = np.cumsum([0.0] + [rng.gauss(0.0, wobble) for _ in range(k)]).astype(
        np.float32
    )
    centre = col + np.interp(np.arange(n) / step, np.arange(k + 1), walk)
    half = np.maximum(0.5 * (width[0] + (width[1] - width[0]) * s**neck), 0.5)
    strength = amp[0] + (amp[1] - amp[0]) * s
    if beads:
        strength = strength * (1 + beads * (noise_1d(n, max(step / 4, 2.0), rng) - 0.5))
    r0 = math.floor(row)
    a, b, c, d = _window(
        dst, r0, r0 + n - 1, float((centre - half).min()), float((centre + half).max())
    )
    if a >= b or c >= d:
        return
    rows = slice(a - r0, b - r0)
    xx = np.arange(c, d, dtype=np.float32) + 0.5
    t = (xx[None, :] - centre[rows, None]) / half[rows, None]
    value = strength[rows, None] * np.clip(1 - t * t, 0, None) ** 2
    combine(dst[a:b, c:d], np.clip(value, 0, 1).astype(np.float32), mode)


def _shifts(mask: np.ndarray, r: int, op) -> np.ndarray:
    """`op` (np.logical_and or np.logical_or) of `mask` over its (2r+1)^2 neighbourhood;
    outside the array counts as False."""
    h, w = mask.shape
    pad = np.pad(mask, r, constant_values=False)
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out = op(out, pad[r + dy : r + dy + h, r + dx : r + dx + w])
    return out


def erode(mask: np.ndarray, r: int) -> np.ndarray:
    return _shifts(mask, r, np.logical_and) if r > 0 else mask.copy()


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    return _shifts(mask, r, np.logical_or) if r > 0 else mask.copy()


def specks(mask: np.ndarray, r: int) -> np.ndarray:
    """The parts of `mask` too small to survive an opening of radius r: isolated specks
    rather than parts of a shape."""
    return mask & ~dilate(erode(mask, r), 2 * r)


def coverage_threshold(
    field: np.ndarray, fraction: float, eligible: np.ndarray | None = None
) -> float:
    """The value above which `fraction` of the (eligible) pixels lie."""
    values = field[eligible] if eligible is not None else field.ravel()
    return float(np.quantile(values, 1.0 - fraction))


def to_u8(*channels: np.ndarray) -> np.ndarray:
    """Float channels in 0..1 as one uint8 image (h, w, n)."""
    out = np.empty((*channels[0].shape, len(channels)), np.uint8)
    for i, c in enumerate(channels):
        out[..., i] = np.clip(c * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


def to_u16(values: np.ndarray) -> np.ndarray:
    return np.round(np.clip(values, 0, 1) * 65535).astype(np.uint16)


def write_png(path: Path, image: np.ndarray, level: int = 6) -> None:
    """Grey, RGB or RGBA PNG from uint8 or uint16 pixels (rows top first), Sub-filtered
    so smooth masks compress well."""
    if image.ndim == 2:
        image = image[..., None]
    h, w, channels = image.shape
    bits = 16 if image.dtype == np.uint16 else 8
    colour = {1: 0, 3: 2, 4: 6}[channels]
    data = (
        image.astype(">u2").view(np.uint8) if bits == 16 else image.astype(np.uint8)
    ).reshape(h, -1)
    bpp = channels * bits // 8
    sub = data.astype(np.int16)
    sub[:, bpp:] -= data[:, :-bpp]
    rows = np.concatenate(
        [np.ones((h, 1), np.uint8), (sub % 256).astype(np.uint8)], axis=1
    )

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", w, h, bits, colour, 0, 0, 0)
    Path(path).write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows.tobytes(), level))
        + chunk(b"IEND", b"")
    )
