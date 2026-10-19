# SPDX-License-Identifier: MIT
"""Dents and rippled plating for the worn NB57 'Wren' (numpy, no bpy).

Steel narrowboats take their knocks from lock walls, gates, bridge holes and other boats:
the bow and the stern quarters get dented most, the cabin sides only low down and near
the ends, where the boat rubs the lock. Years of that, and of the plate shrinking onto the
frames when it was welded, also leave the plating rippled between the frames, which shows
as a gentle wave along the hull and cabin sides ("hungry horse").

Dents live in the side map's (u, z) frame (girth mm, height mm), so the build can push the
mesh in by `depth()` and the wear maps can put the paint damage on the same spots.
"""

from __future__ import annotations

from dataclasses import dataclass

import narrowboat_design as D
import narrowboat_lines as L
import narrowboat_wear as W
import numpy as np
from procgen_mesh import seeded

NAMESPACE = "nbdents"
HULL, CABIN = "hull", "cabin"
HULL_FRAMES = 500.0  # mm between the hull's side frames
CABIN_FRAMES = 610.0  # and the cabin's
OPENING_CLEAR = 70.0  # mm of flat plate kept round every window, porthole and hatch


@dataclass(frozen=True)
class Dent:
    """A smooth dish pushed into the plate at (u, z) mm, radius `r` mm, `depth` mm deep at
    its centre; `kind` says what made it."""

    side: int
    surface: str
    u: float
    z: float
    r: float
    depth: float
    kind: str


def wl_mm(u: float) -> float:
    return 1000 * L.wl_z(float(W.b_of_u(u)) / 1000)


def _clear_of_openings(side: int, u: float, z: float, r: float) -> bool:
    for op in W.openings(side):
        u0, u1, z0, z1 = W.frame_box(op)
        if u0 - r - OPENING_CLEAR < u < u1 + r + OPENING_CLEAR and (
            z0 - r - OPENING_CLEAR < z < z1 + r + OPENING_CLEAR
        ):
            return False
    if D.SIDE_HATCH.side == side:
        h0, h1, z0, z1 = W.hatch_box()
        if h0 - r - OPENING_CLEAR < u < h1 + r + OPENING_CLEAR and (
            z0 - r - OPENING_CLEAR < z < z1 + r + OPENING_CLEAR
        ):
            return False
    for u0, u1, z0, z1 in W.decal_rects(side):
        if u0 - r < u < u1 + r and z0 - r < z < z1 + r:
            return False
    return True


def dents(wear: W.Wear, seed: int) -> list[Dent]:
    """Every dent on the boat, from the seed: knocks on the hull sides (thickest at the bow
    and round the stern quarters), and dings low on the cabin sides near its ends."""
    out: list[Dent] = []
    u_stem = float(W.u_wall(D.STEM - 150))
    u_tail = float(W.u_wall(W.X_TIP * 1000 + 150))
    u_cab0, u_cab1 = (float(W.u_wall(b)) for b in W.CABIN_B)
    for side, where in W.SIDES:
        rng = seeded(NAMESPACE, seed, f"hull-{where}")
        zones = (
            (u_stem - 3500, u_stem, wear.dents_bow, (140, 380), (10, 26), "bow"),
            (u_tail, u_tail + 2500, wear.dents_stern, (120, 320), (8, 20), "quarter"),
            (u_tail + 2500, u_stem - 3500, wear.dents_mid, (90, 260), (4, 12), "lock"),
        )
        for u0, u1, count, rr, dd, kind in zones:
            for _ in range(rng.randint(*count)):
                u = rng.uniform(u0, u1)
                wl = wl_mm(u)
                top = 1000 * L.gunwale_z(float(W.b_of_u(u)) / 1000) - 40
                r = rng.uniform(*rr)
                z = rng.uniform(wl + 40, max(top - 0.3 * r, wl + 60))
                out.append(Dent(side, HULL, u, z, r, rng.uniform(*dd), kind))
        rng = seeded(NAMESPACE, seed, f"cabin-{where}")
        placed = 0
        want = rng.randint(*wear.dents_cabin)
        for _ in range(200):
            if placed >= want:
                break
            r = rng.uniform(70, 230)
            if rng.random() < 0.65:
                # Near the ends, where the cabin meets the lock wall and the gates.
                end = rng.choice((u_cab0, u_cab1))
                u = end + (1 if end == u_cab0 else -1) * rng.uniform(r + 40, 1600)
            else:
                u = rng.uniform(u_cab0 + r + 40, u_cab1 - r - 40)
            z = rng.uniform(W.GUNWALE + r * 0.6 + 30, W.GUNWALE + 520)
            if z + r > W.SIDE_TOP - 40 or not _clear_of_openings(side, u, z, r):
                continue
            out.append(Dent(side, CABIN, u, z, r, rng.uniform(3.5, 11.0), "ding"))
            placed += 1
    return out


def _dish(du, dz, r):
    """A smooth dish: 1 at the centre, 0 with zero slope at radius r."""
    q = np.clip(1.0 - (du * du + dz * dz) / (r * r), 0.0, 1.0)
    return q * q


def _opening_fade(side: int, u, z) -> np.ndarray:
    """1 on open plate, 0 close round the openings, so frames still sit flush."""
    fade = np.ones(np.broadcast(u, z).shape)
    for op in W.openings(side):
        d = W.frame_distance(op, u, z)
        fade = np.minimum(fade, np.clip((d - 15.0) / OPENING_CLEAR, 0.0, 1.0))
    if D.SIDE_HATCH.side == side:
        h0, h1, z0, z1 = W.hatch_box()
        d = np.maximum(np.maximum(h0 - u, u - h1), np.maximum(z0 - z, z - z1))
        fade = np.minimum(fade, np.clip((d - 15.0) / OPENING_CLEAR, 0.0, 1.0))
    return fade


def ripple(wear: W.Wear, side: int, surface: str, u, z) -> np.ndarray:
    """The plate sagging inwards between the frames (mm), zero on each frame line and at
    the plate's welded edges (chine and gunwale on the hull, gunwale and roof on the
    cabin), with an uneven amount from bay to bay."""
    u, z = np.asarray(u, float), np.asarray(z, float)
    if surface == HULL:
        spacing, amp = HULL_FRAMES, wear.ripple_hull
        lo, hi = 150.0, W.GUNWALE - 30.0
    else:
        spacing, amp = CABIN_FRAMES, wear.ripple_cabin
        lo, hi = W.GUNWALE + 10.0, W.SIDE_TOP - 10.0
    across = np.sin(np.pi * np.clip((z - lo) / (hi - lo), 0.0, 1.0))
    bay = np.floor(u / spacing)
    # Each bay sags its own amount (a cheap hash of the bay number and side).
    k = np.sin(bay * 12.9898 + side * 78.233) * 43758.5453
    vary = 0.55 + 0.45 * (k - np.floor(k))
    along = np.sin(np.pi * (u / spacing - bay)) ** 2
    out = amp * vary * along * across
    if surface == CABIN:
        out *= _opening_fade(side, u, z)
    return out


def depth(wear: W.Wear, items: list[Dent], side: int, surface: str, u, z):
    """How far (mm) the plate is pushed in at (u, z) on one side: the dents plus the
    ripple."""
    u, z = np.asarray(u, float), np.asarray(z, float)
    out = ripple(wear, side, surface, u, z)
    for d in items:
        if d.side != side or d.surface != surface:
            continue
        near = (np.abs(u - d.u) < d.r) & (np.abs(z - d.z) < d.r)
        if not near.any():
            continue
        out = out + d.depth * _dish(u - d.u, z - d.z, d.r) * near
    return out


def max_depth(items: list[Dent]) -> float:
    return max((d.depth for d in items), default=0.0)


def dent_sources(wear: W.Wear, items: list[Dent], seed: int) -> W.Sources:
    """The paint damage the dents leave: on the hull the blacking scraped off across the
    dent and rust in its deepest part; on the cabin the paint cracked off at its centre,
    with rust in the deeper dings; a stain running down from each rusty one."""
    src = W.Sources()
    rng = seeded(NAMESPACE, seed, "damage")
    for d in items:
        where = "port" if d.side > 0 else "starboard"
        k = d.depth / 20.0
        if d.surface == HULL:
            src.blobs.append(
                W.Blob(where, "chip", "dent", d.u, d.z, 0.7 * d.r, 0.35 * d.r, 0.9)
            )
            rusty = rng.random() < 0.4 + k
        else:
            src.blobs.append(
                W.Blob(where, "chip", "dent", d.u, d.z, 0.35 * d.r, 0.3 * d.r, 0.85)
            )
            rusty = d.depth > 6.0 and rng.random() < 0.7
        if rusty:
            rr = (0.25 + 0.15 * rng.random()) * d.r
            src.blobs.append(W.Blob(where, "rust", "dent", d.u, d.z, rr, 0.8 * rr, 0.9))
            src.runs.append(
                W.Run(
                    where,
                    "stain",
                    "dent",
                    d.u + rng.uniform(-0.2, 0.2) * rr,
                    d.z - 0.5 * rr,
                    rng.uniform(150, 450),
                    (0.8 * rr, 0.3 * rr),
                    (0.8, 0.2),
                )
            )
    return src


def summary(items: list[Dent]) -> dict:
    """For the manifest: counts and depths by surface."""
    out = {}
    for surface in (HULL, CABIN):
        ds = [d for d in items if d.surface == surface]
        out[surface] = {
            "count": len(ds),
            "max_depth_mm": round(max((d.depth for d in ds), default=0.0), 1),
            "mean_radius_mm": round(sum(d.r for d in ds) / len(ds) if ds else 0.0, 1),
        }
    return out
