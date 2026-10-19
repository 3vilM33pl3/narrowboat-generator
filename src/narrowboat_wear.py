# SPDX-License-Identifier: MIT
"""Wear maps for the weathered NB57 'Wren' (no bpy): where rust, stains, chips, dirt,
touch-ups, verdigris, moss, polish and soot go, worked out from the design.

Damage starts at *sources* taken from narrowboat_design:
- weld clusters along the cabin-side foot;
- the lower corners of the window frames, porthole rims and hatch leaves;
- rust spots on the roof corner, knocks on the gunwale edge, scuffs on the strakes;
- the counter's deck edge, the handrail legs, vents and chimney.

Every layer comes from that one list and follows the water. A window corner weeps 90 mm
down across the coachlines into the weld below it; the stain pools on the gunwale and runs
over its edge down the hull, picking up fresh rust under each scuffed strake.

Channels hold smooth potentials in 0..1. The shaders threshold them, with a little noise,
at the wear level's `threshold`. Colour, roughness and bump therefore all come from the
same masks, and edges stay crisp at any distance.

The maps are 2 mm per pixel, rows top first, with data rather than colour in them:
- **Side A/B:** both cabin sides and hull sides as elevations, port band above starboard.
  u is the hull girth: it equals b along the parallel body and is unrolled round the
  counter's curved quarters and end plate, and round the bow. v is z.
  A = (rust potential, rust stain, chip potential); B = (dirt runs, touch-up field, verdigris).
- **Plan A/B:** roof and decks from above. u is b and v is y, except over the roof's corner
  radius, where v is the section's arc length so the roof wraps on into the sides.
  A = (rust potential, moss potential, walking polish); B = (soot, pooled stain, verdigris).
- **Ends A/B:** the rear and front bulkheads with their doors in (y, z), and the semi-trad
  shelter plates' inner faces in (b, z), side by side; channels as Side A/B.
- **Girth:** a 16-bit 1-D lookup from world x to the hull's u, for the counter and bow.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import narrowboat_common as N
import narrowboat_design as D
import narrowboat_lines as L
import numpy as np
import procgen_masks as M
from procgen_mesh import seeded

WEAR_VERSION = 2  # 2: heavy wear, dents and rippled plating
PX_MM = 2.0
NAMESPACE = "nbwear"


@dataclass(frozen=True)
class Wear:
    """Everything that sets how worn the boat is. Lengths in mm; shares are fractions.
    The defaults are the weathered level: a hard-used boat some years past its last
    repaint, rusting through at the welds and edges, knocked about in the locks."""

    name: str
    threshold: float = 0.6  # shaders show potentials above this
    # Cabin sides.
    weld_clusters: tuple[int, int] = (8, 11)  # rust clusters along each cabin foot
    weld_len: tuple[float, float] = (450, 1200)
    weld_fill: tuple[float, float] = (0.8, 1.0)  # share of a cluster that is rusty
    weld_bead: tuple[float, float] = (30, 75)  # rust band height above the gunwale
    weld_blister: tuple[float, float] = (40, 90)
    corner_weep: float = 1.0  # share of window corners that weep rust
    sill_weep: float = 0.8  # chance of a run off the middle of a window's sill
    porthole_weep: float = 1.0
    hatch_runs: tuple[int, int] = (3, 4)
    drip_share: float = 0.9  # share of the roof's drip points that leave a dirt run
    dirt_run: tuple[float, float] = (300, 1100)
    roof_spots: tuple[int, int] = (
        9,
        13,
    )  # per side: chips in the roof edge, each running
    roof_run: tuple[float, float] = (400, 1100)
    # Chips on the cabin sides' paint, as a share of each area: the bottom coachline, the
    # lowest 150 mm, the cabin's end corners, rings round the frames and the panel
    # corners; open panels only get a few separate dings.
    chip_line: float = 0.08
    chip_foot: float = 0.06
    chip_ends: float = 0.15
    chip_rings: float = 0.12
    chip_corners: float = 0.12
    chip_hatch: float = 0.2  # round the side hatch's leaves
    chip_top: float = 0.08  # along the top edge under the roof
    dings: tuple[int, int] = (18, 32)  # per side
    touchups: tuple[int, int] = (1, 2)  # per side, each over part of the damage
    verdigris: float = 1.0  # share of porthole rims with a green run below
    # Hull.
    hull_runs: tuple[int, int] = (2, 4)  # per weld cluster, from the gunwale edge down
    hull_run: tuple[float, float] = (250, 550)
    knocks: int = 24  # per side, chips on the gunwale edge, mostly near the ends
    shoulder_scrapes: tuple[int, int] = (
        9,
        14,
    )  # per side, on the bow and stern plating
    scuff_ends: tuple[float, float] = (0.55, 0.75)  # strake length scuffed at the ends
    scuff_mid: tuple[float, float] = (0.25, 0.4)
    scuff_rust: float = 0.7  # the rest of a scuff is ground to bright steel
    counter_clusters: tuple[int, int] = (6, 9)  # per side, of 3-8 runs each
    # Roof and decks.
    moss_edge: float = 0.04  # of the roof, in clumps along its edges: moss is starting
    algae: float = 0.16
    leg_rust: float = 0.85  # share of handrail feet with rust
    leg_moss: float = 0.7  # share of handrail feet with a moss cushion
    soot_len: tuple[float, float] = (2500, 4500)
    soot_w: tuple[float, float] = (800, 1300)
    # Shader ageing.
    chalk: tuple[float, float] = (0.11, 0.18)  # lower panel, top 300 mm
    # Broad paint failure: flaking patches over a share of the cabin paint (most low
    # down), rusting in the middle of the larger ones; blacking lost from the hull sides;
    # rust stain dragged down from everything rusty; grime and green low on the sides.
    flake: float = 0.07
    hull_flake: float = 0.12
    stain_drop: float = 650.0
    grime: float = 0.7
    roof_rust: float = 0.08
    roof_centre: float = 0.05  # rust and standing water across the roof's middle too
    ends_flake: float = 0.12  # of the bulkheads and doors, mostly round the leaves
    # Dents (narrowboat_dents) and the plating rippled between the frames (mm).
    dents_bow: tuple[int, int] = (7, 10)
    dents_stern: tuple[int, int] = (4, 6)
    dents_mid: tuple[int, int] = (6, 9)
    dents_cabin: tuple[int, int] = (5, 8)
    ripple_hull: float = 4.5
    ripple_cabin: float = 2.2


WEARS = {"weathered": Wear("weathered")}
LEVELS = ("clean", *WEARS)

# --------------------------------------------------------------------------- geometry (mm)

GUNWALE, SIDE_TOP = float(D.Z_GUNWALE), float(D.Z_SIDE_TOP)
# Amidships the hull's top edge is at the cabin foot's height, so the two share (u, z) in
# the side map: the cabin foot's rust stops just above it rather than ringing the hull.
FOOT = GUNWALE + 1.0
CABIN_B = (float(D.SHELTER_AFT), float(D.CABIN_LEN))
F, RIM = float(D.FRAME_FLANGE), float(D.RIM_W)
ROOF_YC, CORNER_R = 1000 * L.ROOF_YC, float(D.CORNER_R)
V_ROOF = (
    ROOF_YC + CORNER_R * math.pi / 2
)  # arc length from the centreline round the corner
HANDRAIL_Y = 1000 * L.handrail_y()
CABIN_HALF = float(D.CABIN_BASE_HALF)
PANEL_B = tuple(float(v) for v in D.PANEL_B)
PANEL_Z = tuple(float(v) for v in D.PANEL_Z)

# The hull girth: arc length along the plan outline, from the corner of the counter's flat
# end plate forward round the counter quarter, along the parallel body and round the bow.
X_TIP = L.STERN_TIP + 0.02  # the end plate (metres from the rear bulkhead)
W_END = max(L.plan_half(X_TIP), 0.09)  # its half width
_GB = np.linspace(X_TIP, L.STEM, 6001)
_GY = np.array([L.plan_half(b) for b in _GB])
_GY[0] = W_END
_GG = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(_GB), np.diff(_GY)))])
# u (metres) = U0 + girth from the stern centreline, so that u = b along the parallel body.
U0 = -(W_END + float(np.interp(0.0, _GB, _GG)))


def u_wall(b_mm):
    """Girth u (mm) of a point on the cabin or hull side at b (mm)."""
    return 1000.0 * (U0 + W_END + np.interp(np.asarray(b_mm, float) / 1000.0, _GB, _GG))


def u_plate(y_mm):
    """Girth u (mm) of a point on the counter's end plate at y (mm)."""
    return 1000.0 * U0 + np.abs(np.asarray(y_mm, float))


def b_of_u(u_mm):
    """b (mm) of the side point with girth u (end-plate points map to its b)."""
    u = np.asarray(u_mm, float) / 1000.0 - U0 - W_END
    return 1000.0 * np.interp(u, _GG, _GB)


def girth_lut(n: int = 2048) -> tuple[np.ndarray, dict]:
    """The girth lookup for the shaders: world x across the hull -> u (m) without the
    end-plate term, as 0..1 values, and the affine numbers that decode it."""
    b = np.linspace(X_TIP, L.STEM, n)
    lut = U0 + np.interp(b, _GB, _GG)
    lo, hi = float(lut[0]), float(lut[-1])
    info = {
        "x0": N.wx(1000 * X_TIP),
        "x1": N.wx(1000 * L.STEM),
        "u0": lo,
        "u1": hi,
        "w_end": W_END,
        "size": n,
    }
    return (lut - lo) / (hi - lo), info


def roof_v(y_mm, z_mm):
    """Plan v (mm) of a roof point: y on the camber, the arc length round the corner."""
    y, z = np.asarray(y_mm, float), np.asarray(z_mm, float)
    ay = np.abs(y)
    theta = np.arctan2(np.maximum(z - SIDE_TOP, 0.0), np.maximum(ay - ROOF_YC, 1e-6))
    corner = ROOF_YC + CORNER_R * (np.pi / 2 - theta)
    v = np.where((ay > ROOF_YC) & (z > SIDE_TOP - 5), corner, ay)
    return np.sign(y) * v


# --------------------------------------------------------------------------- map frames


@dataclass(frozen=True)
class MapFrame:
    """A raster over a rectangle of map millimetres: column = (u - u0) / px and row =
    (v1 - v) / px, so row 0 is the top edge."""

    name: str
    u0: float
    v1: float
    px: float
    w: int
    h: int

    @classmethod
    def covering(cls, name, u0, u1, v0, v1, px):
        return cls(
            name, u0, v1, px, math.ceil((u1 - u0) / px), math.ceil((v1 - v0) / px)
        )

    @property
    def u1(self) -> float:
        return self.u0 + self.w * self.px

    @property
    def v0(self) -> float:
        return self.v1 - self.h * self.px

    def col(self, u):
        return (np.asarray(u, float) - self.u0) / self.px

    def row(self, v):
        return (self.v1 - np.asarray(v, float)) / self.px

    def u(self, col):
        return self.u0 + np.asarray(col, float) * self.px

    def v(self, row):
        return self.v1 - np.asarray(row, float) * self.px

    def grid(self):
        """Pixel-centre coordinates: u as a (1, w) row, v as an (h, 1) column."""
        u = self.u(np.arange(self.w) + 0.5).astype(np.float32)
        v = self.v(np.arange(self.h) + 0.5).astype(np.float32)
        return u[None, :], v[:, None]

    def record(self) -> dict:
        return {
            "u0": self.u0,
            "u1": self.u1,
            "v0": self.v0,
            "v1": self.v1,
            "px_mm": self.px,
        }


SIDE_U = (1000 * U0 - 60.0, float(u_wall(D.STEM)) + 60.0)
SIDE_Z = (-20.0, 2100.0)
PLAN_U = (D.STERN_TIP - D.FENDER_AFT - 150.0, D.STEM + D.FENDER_FWD + 150.0)
PLAN_V = (-1100.0, 1100.0)
END_Z = (740.0, 2140.0)


@dataclass(frozen=True)
class Tile:
    """One face group in the ends atlas: its horizontal axis ("y" for the bulkheads, "b"
    for the shelter plates' inner faces) from lo to hi (mm), placed at `offset`."""

    name: str
    axis: str
    lo: float
    hi: float
    offset: float

    def a(self, coord):
        return self.offset + (np.asarray(coord, float) - self.lo)


TILES = {
    "Rear": Tile("Rear", "y", -1000.0, 1000.0, 0.0),
    "Front": Tile("Front", "y", -1000.0, 1000.0, 2100.0),
    "ShelterPort": Tile("ShelterPort", "b", -950.0, 50.0, 4200.0),
    "ShelterStarboard": Tile("ShelterStarboard", "b", -950.0, 50.0, 5300.0),
}
ENDS_U = (0.0, 6400.0)


def frames(px: float = PX_MM) -> dict[str, MapFrame]:
    return {
        "side": MapFrame.covering("side", *SIDE_U, *SIDE_Z, px),
        "plan": MapFrame.covering("plan", *PLAN_U, *PLAN_V, px),
        "ends": MapFrame.covering("ends", *ENDS_U, *END_Z, px),
    }


# --------------------------------------------------------------------------- features


def openings(side: int) -> list:
    return [op for op in D.OPENINGS if op.side == side]


def frame_box(op) -> tuple[float, float, float, float]:
    """(u0, u1, z0, z1) of a window frame's flange or a porthole's rim (mm)."""
    rim = RIM if op.kind == "porthole" else F
    return (
        op.b - op.w / 2 - rim,
        op.b + op.w / 2 + rim,
        op.zc - op.h / 2 - rim,
        op.zc + op.h / 2 + rim,
    )


def _frame_shape(op) -> tuple[float, float, float]:
    """(half width, half height, corner radius) of the frame's outline (mm)."""
    if op.kind == "porthole":
        r = op.w / 2 + RIM
        return r, r, r
    return op.w / 2 + F, op.h / 2 + F, op.r + F


def frame_edge(op, u: float, top: bool = False) -> float:
    """z of the frame's lower (or upper) edge at u, round its corners."""
    hw, hh, r = _frame_shape(op)
    dx = min(max(abs(u - op.b) - (hw - r), 0.0), r)
    sag = r - math.sqrt(max(r * r - dx * dx, 0.0))
    return op.zc + hh - sag if top else op.zc - hh + sag


def inside_frame(op, u, z, grow: float = 0.0):
    """True where (u, z) lies inside the frame outline grown by `grow` mm."""
    hw, hh, r = _frame_shape(op)
    hw, hh, r = hw + grow, hh + grow, r + grow
    dx = np.maximum(np.abs(np.asarray(u, float) - op.b) - (hw - r), 0.0)
    dz = np.maximum(np.abs(np.asarray(z, float) - op.zc) - (hh - r), 0.0)
    return dx * dx + dz * dz <= r * r


def frame_distance(op, u, z):
    """Distance (mm) outside the frame outline (0 inside)."""
    hw, hh, r = _frame_shape(op)
    dx = np.maximum(np.abs(np.asarray(u, float) - op.b) - (hw - r), 0.0)
    dz = np.maximum(np.abs(np.asarray(z, float) - op.zc) - (hh - r), 0.0)
    return np.maximum(np.hypot(dx, dz) - r, 0.0)


def hatch_box() -> tuple[float, float, float, float]:
    h = D.SIDE_HATCH
    return h.b - h.clear / 2, h.b + h.clear / 2, float(h.z0), float(h.z0 + h.leaf_h)


def obstacle_top(side: int, u: float, z: float) -> float:
    """The highest frame or hatch edge below z at u, or the gunwale: where water running
    down the side from z stops."""
    stop = GUNWALE
    for op in openings(side):
        u0, u1, _, _ = frame_box(op)
        if u0 < u < u1:
            top = frame_edge(op, u, top=True)
            if top < z:
                stop = max(stop, top)
    if D.SIDE_HATCH.side == side:
        u0, u1, _, z1 = hatch_box()
        if u0 < u < u1 and z1 < z:
            stop = max(stop, z1)
    return stop


def decal_rects(side: int) -> list[tuple[float, float, float, float]]:
    """(u0, u1, z0, z1) of every decal on one cabin side (mm)."""
    out = []
    for p in N.decal_panels():
        x, y, z = N.panel_centre(p)
        if y * side <= 0 or not p.part.startswith(("DecalCabin", "DecalHatch")):
            continue
        b = 1000 * N.bx(x)
        out.append(
            (
                b - 500 * p.width,
                b + 500 * p.width,
                1000 * z - 500 * p.height,
                1000 * z + 500 * p.height,
            )
        )
    return out


@dataclass(frozen=True)
class Run:
    """A downward run of stain (or a fore-and-aft scratch when `horizontal`) from (u, v),
    `length` mm long, its width and strength tapering from head to tail."""

    where: str  # port | starboard | ends
    layer: str  # stain | dirt | verd | chip
    kind: str
    u: float
    v: float
    length: float
    width: tuple[float, float]
    amp: tuple[float, float]
    horizontal: bool = False
    wobble: float = 0.8  # how far the run wanders, pixels per 40
    beads: float = 0.6  # how unevenly it deposits along its length (1.5 breaks it up)
    neck: float = 1.0  # below 1 it narrows quickly just below its head


@dataclass(frozen=True)
class Blob:
    """A compact bump of potential centred on (u, v) with radii (ru, rv) mm; with `ring` > 0
    it is a ring of that width round radius ru instead."""

    where: str  # port | starboard | plan | ends
    layer: str
    kind: str
    u: float
    v: float
    ru: float
    rv: float
    amp: float
    ring: float = 0.0
    down: bool = False  # a ring heavier at the bottom, where water collects
    floor: float = -math.inf  # nothing is drawn below this v (mm)


@dataclass(frozen=True)
class Patch:
    """A rectangle of potential (mm), edges ramped over `soft` mm. `breakup` multiplies it
    by noise `cell` mm across so a bead or moss patch is ragged rather than ruled; `jitter`
    instead gives a crisp edge that wanders by about that much, like a brushed touch-up.
    `ragged` moves its upper edge up and down along its length (a weld bead's rust creeps
    up the plate unevenly), and `taper` lets it thin out towards its ends."""

    where: str
    layer: str
    kind: str
    u0: float
    u1: float
    v0: float
    v1: float
    amp: float
    soft: float = 2.0
    breakup: float = 0.0
    cell: float = 6.0
    jitter: float = 0.0
    stretch: float = 1.0  # the breakup's cells this many times longer along u
    taper: float = 0.0  # share of its length over which it fades out at each end
    ragged: float = 0.0  # its upper edge wanders up and down by up to this much (mm)
    floor: float = -math.inf  # nothing is drawn below this v (mm)


@dataclass
class Sources:
    runs: list[Run] = field(default_factory=list)
    blobs: list[Blob] = field(default_factory=list)
    patches: list[Patch] = field(default_factory=list)
    welds: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    # (u, half span, weight, top) where a touch-up may go: over damage below `top`.
    sites: dict[str, list[tuple[float, float, float, float]]] = field(
        default_factory=dict
    )
    scuffs: dict[str, list[tuple[float, float, float, bool]]] = field(
        default_factory=dict
    )

    def of(self, where: str, kind: type):
        items = {Run: self.runs, Blob: self.blobs, Patch: self.patches}[kind]
        return [i for i in items if i.where == where]


SIDES = ((1, "port"), (-1, "starboard"))
STRAKES = tuple(
    (float(z), w) for z, w in zip(D.Z_STRAKES, (0.45, 0.6, 1.0))
)  # bottom to top, with how much each is scuffed relative to the top one
STRAKE_HALF = D.STRAKE_H / 2


def gunwale_edge(u: float) -> float:
    """z (mm) of the top edge of the hull side at girth u."""
    return 1000 * L.gunwale_z(float(b_of_u(u)) / 1000)


def _cluster_merge(clusters, u, length, lo, hi):
    a, b = max(u - length / 2, lo), min(u + length / 2, hi)
    for k, (c, l) in enumerate(clusters):
        if a < c + l / 2 + 60 and b > c - l / 2 - 60:
            a, b = min(a, c - l / 2), max(b, c + l / 2)
            clusters[k] = ((a + b) / 2, b - a)
            return
    clusters.append(((a + b) / 2, b - a))


def _ramp(x, a, b):
    """0 at a, 1 at b (either way round), clamped."""
    return min(max((x - a) / (b - a), 0.0), 1.0)


def end_weight(u: float) -> float:
    """How near u is to either end of the hull: 1 round the counter and the last 1.5 m of
    the bow, 0 amidships, ramping over 2 m; boats meet lock walls and gates at the ends."""
    stern = 1.0 - _ramp(u, float(u_wall(-500.0)), float(u_wall(1500.0)))
    bow = _ramp(u, float(u_wall(D.STEM - 3500.0)), float(u_wall(D.STEM - 1500.0)))
    return max(stern, bow)


def porthole_picks(wear: Wear, seed: int) -> dict[str, set[tuple[int, float]]]:
    """Which portholes, as (side, b), have verdigris and which weep rust: the level's share
    of all of them, rounded, rather than a coin toss each, which with four portholes too
    often gives none. Drawn from their own stream so other changes don't move them."""
    rng = seeded(NAMESPACE, seed, "portholes")
    ports = [
        (side, float(op.b))
        for side, _ in SIDES
        for op in openings(side)
        if op.kind == "porthole"
    ]
    return {
        "verd": set(rng.sample(ports, round(wear.verdigris * len(ports)))),
        "weep": set(rng.sample(ports, round(wear.porthole_weep * len(ports)))),
    }


def _side_sources(
    src: Sources, wear: Wear, side: int, where: str, seed: int, legs: list[float]
) -> None:
    """Every wear source on one side: weeping frames and the weld clusters below them,
    dirt from the roof's drip points, knocks and scuffs gathering towards the ends, the
    fender's rub, counter runs and a few dings; and where touch-ups may go."""
    rng = seeded(NAMESPACE, seed, f"side-{where}")
    u_lo, u_hi = float(u_wall(CABIN_B[0])), float(u_wall(CABIN_B[1]))
    feed: list[float] = []  # weeping sources whose runs reach the weld
    sites: list[tuple[float, float, float, float]] = []  # see Sources.sites

    def stain(kind, u, z, length, w, a, layer="stain", wobble=0.8, neck=1.0, beads=0.6):
        """A run down from (u, z), w = (width at its source, width at its end) and a the
        same for its strength: narrow and strong where it leaves its source, spreading
        and fading as it goes. Down the hull it breaks at each strake, which catches the
        water and lets it go again from its lower lip, narrow and weaker; it is washed
        away above the waterline band."""
        if z >= GUNWALE:
            if length > 4:
                src.runs.append(
                    Run(
                        where,
                        layer,
                        kind,
                        u,
                        z,
                        length,
                        w,
                        a,
                        wobble=wobble,
                        beads=beads,
                        neck=neck,
                    )
                )
            return
        wl = 1000 * L.wl_z(float(b_of_u(u)) / 1000)
        end = max(z - length, wl + 60)
        top, (w0, w1), (a0, a1) = z, w, a
        for zk in sorted((s for s, _ in STRAKES), reverse=True):
            lip_top, lip_bottom = zk + STRAKE_HALF, zk - STRAKE_HALF
            if lip_bottom >= top or lip_top <= end:
                continue
            if top - lip_top > 4:
                src.runs.append(
                    Run(
                        where,
                        layer,
                        kind,
                        u,
                        top,
                        top - lip_top,
                        (w0, w0 * 1.25),
                        (a0, a0 * 0.45),
                        wobble=wobble,
                        beads=beads,
                    )
                )
            # The strake catches it: a little pools along its top, and some runs end
            # there; the rest drip again from elsewhere along its lower lip, weaker.
            src.blobs.append(
                Blob(
                    where,
                    layer,
                    kind,
                    u,
                    lip_top + 2,
                    rng.uniform(1.0, 2.5) * w0,
                    rng.uniform(3, 5),
                    0.6 * a0,
                )
            )
            if rng.random() < 0.35:
                return
            top = lip_bottom - 1
            u += rng.uniform(-30, 30)
            w0, a0 = w0 * rng.uniform(0.6, 0.85), a0 * rng.uniform(0.5, 0.7)
        if top - end > 4:
            src.runs.append(
                Run(
                    where,
                    layer,
                    kind,
                    u,
                    top,
                    top - end,
                    (w0, max(w1, w0)),
                    (a0, a1),
                    wobble=wobble,
                    beads=beads,
                    neck=neck,
                )
            )

    def weep(kind, u, z0, bloom, head, amp, runs=1):
        """A teardrop of rust hanging from (u, z0) and the stains it sends down to the
        gunwale, wide where they leave it."""
        rb, rv = bloom
        src.blobs.append(
            Blob(where, "rust", kind, u, z0 - 0.45 * rv, 0.6 * rb, 0.6 * rv, 0.92)
        )
        for k in range(runs):
            du = 0.0 if k == 0 else rng.uniform(-0.3, 0.3) * rb
            h = head * (1.0 if k == 0 else rng.uniform(0.4, 0.7))
            stain(
                kind,
                u + du,
                z0 - 0.6 * rv,
                z0 - 0.6 * rv - GUNWALE,
                (max(h * rng.uniform(0.45, 0.65), 4.0), h * rng.uniform(1.1, 1.6)),
                (amp[0] * (1.0 if k == 0 else 0.7), amp[1]),
                wobble=rng.uniform(1.2, 2.2),
                beads=1.0,
            )

    # Window corners weep into the weld 90 mm below; a few sills drip from a small bloom.
    picks = porthole_picks(wear, seed)
    for op in openings(side):
        if op.kind == "porthole":
            z0 = frame_edge(op, op.b)
            u = op.b + rng.uniform(-12, 12)
            if (side, float(op.b)) in picks["weep"]:
                weep(
                    "porthole",
                    u,
                    z0,
                    (rng.uniform(8, 14), rng.uniform(14, 22)),
                    rng.uniform(22, 38),
                    (rng.uniform(0.8, 0.95), rng.uniform(0.4, 0.6)),
                )
                feed.append(u)
                sites.append((u, 120.0, 0.3, z0 + 10))
            if (side, float(op.b)) in picks["verd"]:
                # Verdigris on the lower part of the rim, where water sits, and the paint
                # just below it: a few soft tendrils fanning down from the bottom, broad
                # where they leave the rim and gone within a hand's breadth or two.
                hw, _, _ = _frame_shape(op)
                src.blobs.append(
                    Blob(
                        where,
                        "verd",
                        "rim",
                        op.b,
                        op.zc,
                        hw - 18,
                        hw - 18,
                        rng.uniform(0.8, 0.95),
                        ring=26.0,
                        down=True,
                    )
                )
                src.blobs.append(
                    Blob(
                        where,
                        "verd",
                        "rim",
                        op.b,
                        op.zc,
                        hw - RIM - 6,
                        hw - RIM - 6,
                        rng.uniform(0.6, 0.8),
                        ring=10.0,
                        down=True,
                    )
                )
                for _ in range(1 if rng.random() < 0.6 else 2):
                    du = rng.uniform(-0.12, 0.12) * hw
                    head = rng.uniform(12, 18)
                    stain(
                        "rim",
                        op.b + du,
                        frame_edge(op, op.b + du) + 4,
                        rng.uniform(100, 260),
                        (head, head * rng.uniform(2.0, 3.2)),
                        (rng.uniform(0.65, 0.85), 0.04),
                        layer="verd",
                        wobble=rng.uniform(1.0, 2.0),
                        beads=1.3,
                    )
            continue
        hw, hh, r = _frame_shape(op)
        z_bot = op.zc - hh
        for s in (-1, 1):
            u = op.b + s * (hw - r + 0.342 * r)
            if rng.random() < wear.corner_weep:
                z0 = frame_edge(op, u)
                weep(
                    "corner",
                    u,
                    z0,
                    (rng.uniform(9, 16), rng.uniform(16, 28)),
                    rng.uniform(25, 45),
                    (rng.uniform(0.85, 1.0), rng.uniform(0.5, 0.7)),
                    runs=1 if rng.random() < 0.7 else 2,
                )
                feed.append(u)
                sites.append((u, 120.0, 0.25, z0 + 10))
        if rng.random() < wear.sill_weep:
            u = op.b + rng.uniform(-0.35, 0.35) * (2 * hw - 2 * r)
            weep(
                "sill",
                u,
                z_bot,
                (rng.uniform(6, 10), rng.uniform(10, 16)),
                rng.uniform(15, 28),
                (rng.uniform(0.6, 0.8), rng.uniform(0.3, 0.5)),
            )
            feed.append(u)
    if D.SIDE_HATCH.side == side:
        h0, h1, z0, _ = hatch_box()
        gap = 1000 * N.HATCH_GAP
        hb = float(D.SIDE_HATCH.b)
        for u in rng.sample(
            [h0, hb - gap, hb + gap, h1], rng.randint(*wear.hatch_runs)
        ):
            u += rng.uniform(-10, 10)
            weep(
                "hatch",
                u,
                z0,
                (rng.uniform(7, 16), rng.uniform(12, 26)),
                rng.uniform(18, 40),
                (rng.uniform(0.75, 0.95), rng.uniform(0.4, 0.6)),
            )
            feed.append(u)
            sites.append((u, 150.0, 0.4, z0 + 10))

    # Weld clusters: one under every weeping source, then random ones up to the count;
    # long and mostly continuous, so they read from a distance.
    clusters: list[tuple[float, float]] = []
    for u in sorted(feed):
        _cluster_merge(clusters, u, rng.uniform(250, 500), u_lo + 80, u_hi - 80)
    target = rng.randint(*wear.weld_clusters)
    for _ in range(400):
        if len(clusters) >= target:
            break
        length = rng.uniform(*wear.weld_len)
        u = rng.uniform(u_lo + 80 + length / 2, u_hi - 80 - length / 2)
        if all(abs(u - c) > (ln + length) / 2 + 300 for c, ln in clusters):
            clusters.append((u, length))
    clusters.sort()
    src.welds[where] = clusters
    for c, length in clusters:
        n = rng.randint(1, 3)
        rusty = rng.uniform(*wear.weld_fill) * length
        segs = [rng.uniform(0.5, 1.5) for _ in range(n)]
        gaps = [rng.uniform(0.5, 1.5) for _ in range(n + 1)]
        ks, kg = rusty / sum(segs), (length - rusty) / sum(gaps)
        u = c - length / 2 + gaps[0] * kg
        for k in range(n):
            u0, u1 = u, u + segs[k] * ks
            bead = rng.uniform(*wear.weld_bead)
            src.patches.append(
                Patch(
                    where,
                    "rust",
                    "weld",
                    u0,
                    u1,
                    GUNWALE - 3,
                    GUNWALE + bead,
                    0.95,
                    2.0,
                    0.45,
                    cell=10.0,
                    stretch=6.0,
                    taper=rng.uniform(0.12, 0.25),
                    ragged=rng.uniform(3, 7),
                    floor=FOOT,
                )
            )
            for _ in range(rng.randint(1, 3)):
                rv = rng.uniform(25, 55)
                src.blobs.append(
                    Blob(
                        where,
                        "stain",
                        "creep",
                        rng.uniform(u0, u1),
                        GUNWALE + 0.3 * rv,
                        rng.uniform(0.25, 0.6) * (u1 - u0) + 40,
                        rv,
                        rng.uniform(0.45, 0.65),
                        floor=FOOT,
                    )
                )
            # Blisters along the bead, each a cluster of pits with a ragged outline.
            for _ in range(max(1, round((u1 - u0) / rng.uniform(90, 160)))):
                rb = rng.uniform(*wear.weld_blister)
                ub = rng.uniform(u0, u1)
                wb, hb = rng.uniform(0.6, 1.3) * rb, 0.35 * rb
                for _ in range(rng.randint(5, 12)):
                    a = rng.uniform(0, math.pi)
                    d = rng.uniform(0, 0.75)
                    pr = rng.uniform(0.15, 0.45) * rb
                    src.blobs.append(
                        Blob(
                            where,
                            "rust",
                            "blister",
                            ub + d * wb * math.cos(a),
                            GUNWALE + d * hb * math.sin(a) + 0.2 * hb,
                            pr,
                            pr * rng.uniform(0.45, 0.75),
                            rng.uniform(0.72, 0.92),
                            floor=FOOT,
                        )
                    )
            u = u1 + gaps[k + 1] * kg
        sites.append((c, length / 2, 0.6, GUNWALE + 90))
        # Off the gunwale edge and down the hull: a few runs, wide where they leave it.
        for _ in range(rng.randint(*wear.hull_runs)):
            u = rng.uniform(c - length / 2, c + length / 2)
            head = rng.uniform(12, 35)
            stain(
                "hull",
                u,
                gunwale_edge(u) - 8,
                rng.uniform(*wear.hull_run),
                (head * rng.uniform(0.4, 0.6), head * rng.uniform(1.1, 1.5)),
                (rng.uniform(0.55, 0.85), rng.uniform(0.08, 0.2)),
                wobble=rng.uniform(1.2, 2.2),
                beads=1.0,
            )

    # Dirt runs from some of the roof's drip points: under handrail feet, at the cabin's
    # end corners and, beside the plank, its blocks. Soft and broad; they stop on frame
    # tops.
    drips = [1000 * b for b in N.handrail_legs()] + [80.0, CABIN_B[1] - 80.0]
    if side < 0:
        drips += [float(b) for b in D.PLANK_BLOCKS]
    drips = [b for b in drips if rng.random() < wear.drip_share]
    for b in drips:
        u = float(u_wall(b)) + rng.uniform(-60, 60)
        z0 = SIDE_TOP - rng.uniform(2, 10)
        length = min(rng.uniform(*wear.dirt_run), z0 - obstacle_top(side, u, z0) - 2)
        if length > 40:
            head = rng.uniform(25, 70)
            stain(
                "dirt",
                u,
                z0,
                length,
                (head, head * rng.uniform(0.4, 0.7)),
                (rng.uniform(0.2, 0.45), 0.02),
                layer="dirt",
                wobble=rng.uniform(0.6, 1.8),
            )
    # Rain-wash: faint grey streaks from the roof edge where it drips, a few together
    # where it drips most and then a gap, spreading as they fade; they stop on frame tops.
    u = u_lo + rng.uniform(40, 300)
    while u < u_hi - 40:
        for _ in range(rng.randint(1, 4)):
            us = min(max(u + rng.uniform(-60, 60), u_lo + 20), u_hi - 20)
            z0 = SIDE_TOP - rng.uniform(2, 8)
            length = min(rng.uniform(120, 600), z0 - obstacle_top(side, us, z0) - 2)
            if length > 40:
                head = rng.uniform(12, 40)
                stain(
                    "wash",
                    us,
                    z0,
                    length,
                    (head, head * rng.uniform(1.4, 2.4)),
                    (rng.uniform(0.08, 0.3), 0.0),
                    layer="dirt",
                    wobble=rng.uniform(0.5, 1.4),
                    beads=1.0,
                )
        u += rng.uniform(250, 900)
    # Rust where ropes and the pole have chipped the roof edge, each with a long run.
    for _ in range(rng.randint(*wear.roof_spots)):
        u = rng.uniform(u_lo + 100, u_hi - 100)
        z0 = SIDE_TOP - rng.uniform(3, 10)
        src.blobs.append(
            Blob(
                where,
                "rust",
                "roofspot",
                u,
                z0,
                rng.uniform(6, 14),
                rng.uniform(5, 10),
                0.9,
            )
        )
        length = min(
            rng.uniform(*wear.roof_run), z0 - 4 - obstacle_top(side, u, z0) - 2
        )
        if length > 40:
            head = rng.uniform(20, 35)
            stain(
                "roofspot",
                u,
                z0 - 4,
                length,
                (head * rng.uniform(0.4, 0.55), head * rng.uniform(1.2, 1.6)),
                (rng.uniform(0.7, 0.9), rng.uniform(0.1, 0.2)),
                wobble=rng.uniform(1.0, 2.0),
                beads=1.0,
            )
    # Fainter dirt off the gunwale edge down the hull, here and there.
    u = float(u_wall(1000 * L.STERN_TIP + 60)) + rng.uniform(0, 900)
    u_bow = float(u_wall(D.STEM - 400))
    while u < u_bow:
        head = rng.uniform(15, 45)
        stain(
            "hulldirt",
            u,
            gunwale_edge(u) - 8,
            rng.uniform(80, 300),
            (head, head * rng.uniform(0.3, 0.6)),
            (rng.uniform(0.15, 0.3), 0.02),
            layer="dirt",
            wobble=rng.uniform(0.5, 1.5),
        )
        u += rng.uniform(700, 2200)

    def chip_cluster(kind, u, z, size, rust_share=0.6, amp=0.95, flat=1.0):
        """A handful of chips of varied size and shape round (u, z), some rusted. With
        `flat` > 1 they are drawn out fore and aft into short scrapes, as a lock wall or
        another boat's rubbing strake leaves them on the hull."""
        k = math.sqrt(flat)
        for _ in range(rng.randint(3, 7)):
            cu, cz = u + rng.gauss(0, 0.5 * size * k), z + rng.gauss(0, 0.3 * size / k)
            ru = rng.uniform(0.15, 0.45) * size * k
            rv = rng.uniform(0.1, 0.35) * size / k
            src.blobs.append(Blob(where, "chip", kind, cu, cz, ru, rv, amp))
            if rng.random() < rust_share:
                src.blobs.append(
                    Blob(where, "rust", kind, cu, cz, 0.6 * ru, 0.6 * rv, 0.85)
                )

    # Knocks on the gunwale edge, gathered towards the ends.
    u_first = float(u_wall(1000 * X_TIP))
    knocks = 0
    while knocks < wear.knocks:
        u = rng.uniform(u_first + 50, u_bow)
        if rng.random() > 0.15 + 0.85 * end_weight(u):
            continue
        knocks += 1
        z = gunwale_edge(u) - 6
        chip_cluster("knock", u, z, rng.uniform(20, 45), flat=3.0)
        if rng.random() < 0.6:
            head = rng.uniform(10, 22)
            stain(
                "knock",
                u,
                z - 8,
                rng.uniform(120, 300),
                (head, head * 0.4),
                (rng.uniform(0.6, 0.8), 0.1),
            )

    # Strake scuffs: broken bands ground into the strakes' outer faces by lock walls and
    # other boats, heaviest round the counter and the bow; some rusted again.
    scuffs = []
    s_lo, s_hi = float(u_wall(1000 * L.STERN_TIP + 60)), float(u_wall(D.STEM - 350))

    def scuff(zk, u, u1):
        """One broken band from u to u1 on the strake at zk, perhaps rusted again, with
        the stains it lets go below and a few scratches along it."""
        rusty = rng.random() < wear.scuff_rust
        scuffs.append((zk, u, u1, rusty))
        half = rng.uniform(7, 15)
        zc = zk + rng.uniform(-6, 2)
        src.patches.append(
            Patch(
                where,
                "chip",
                "scuff",
                u,
                u1,
                zc - half,
                zc + half,
                rng.uniform(0.42, 0.5),
                6.0,
                0.5,
                cell=8.0,
                stretch=10.0,
                taper=rng.uniform(0.2, 0.4),
            )
        )
        for _ in range(rng.randint(6, 14)):
            a0 = rng.uniform(u, u + 0.7 * (u1 - u))
            src.runs.append(
                Run(
                    where,
                    "chip",
                    "scratch",
                    a0,
                    zc + rng.uniform(-0.9, 0.9) * half,
                    min(rng.uniform(0.25, 1.0) * (u1 - u), u1 - a0),
                    (rng.uniform(4, 9), rng.uniform(1.5, 3)),
                    (rng.uniform(0.7, 1.0), rng.uniform(0.45, 0.7)),
                    horizontal=True,
                    beads=1.4,
                )
            )
        if rusty:
            src.patches.append(
                Patch(
                    where,
                    "rust",
                    "scuff",
                    u,
                    u1,
                    zc - 0.6 * half,
                    zc + 0.6 * half,
                    rng.uniform(0.7, 0.85),
                    4.0,
                    0.9,
                    cell=8.0,
                    stretch=6.0,
                    taper=rng.uniform(0.25, 0.45),
                )
            )
            if u1 - u >= 80:
                for _ in range(rng.randint(1, 3)):
                    head = rng.uniform(10, 24)
                    stain(
                        "scuff",
                        rng.uniform(u, u1),
                        zk - STRAKE_HALF - 1,
                        rng.uniform(60, 220),
                        (head, head * 0.4),
                        (rng.uniform(0.6, 0.85), rng.uniform(0.1, 0.2)),
                    )

    def stacked(zk, u, u1):
        """Whether u..u1 would sit mostly straight above or below a scuff already on
        another strake (three pills in a column read as stamped)."""
        for z, a, b, _ in scuffs:
            if z != zk and min(b, u1) - max(a, u) > 0.7 * min(b - a, u1 - u):
                return True
        return False

    shoulder = float(u_wall(D.STEM - 3000))
    for zk, weight in STRAKES:
        u = s_lo + rng.uniform(0, 300)
        while u < s_hi:
            e = end_weight(u)
            frac = weight * (
                (1 - e) * rng.uniform(*wear.scuff_mid)
                + e * rng.uniform(*wear.scuff_ends)
            )
            length = rng.uniform(150, 900) * (1 + 1.2 * e)
            u1 = min(u + length, s_hi)
            if not stacked(zk, u, u1):
                scuff(zk, u, u1)
            u = u1 + length * (1 / max(frac, 0.02) - 1) * rng.uniform(0.5, 1.5)
        # The bow shoulder meets every lock wall first: its upper strakes always carry a
        # scuff there, whatever the spacing left.
        if weight >= 0.6 and not any(z == zk and b > shoulder for z, _, b, _ in scuffs):
            length = rng.uniform(250, 700)
            u0 = rng.uniform(shoulder, s_hi - length)
            scuff(zk, u0, u0 + length)
    src.scuffs[where] = scuffs
    # Fore-and-aft scratches on the plating where the hull rubs lock walls: the bow
    # shoulder and the counter quarter. Broken, tapering strokes.
    u_bow_end = float(u_wall(D.STEM - 3000))
    u_stern = float(u_wall(-200))
    for _ in range(rng.randint(*wear.shoulder_scrapes)):
        bow = rng.random() < 0.65
        u = rng.uniform(u_bow_end, u_bow - 200) if bow else rng.uniform(s_lo, u_stern)
        z = rng.uniform(560, gunwale_edge(u) - 40)
        span = rng.uniform(150, 600)
        src.patches.append(
            Patch(
                where,
                "chip",
                "scrape",
                u,
                u + span,
                z - rng.uniform(20, 40),
                z + rng.uniform(20, 40),
                rng.uniform(0.42, 0.5),
                8.0,
                0.5,
                cell=10.0,
                stretch=8.0,
                taper=0.3,
            )
        )
        for _ in range(rng.randint(2, 6)):
            a0 = rng.uniform(u, u + 0.6 * span)
            src.runs.append(
                Run(
                    where,
                    "chip",
                    "scrape",
                    a0,
                    z + rng.uniform(-35, 35),
                    rng.uniform(0.3, 0.9) * span,
                    (rng.uniform(2, 6), rng.uniform(0.8, 2)),
                    (rng.uniform(0.8, 1.0), rng.uniform(0.4, 0.7)),
                    horizontal=True,
                    beads=1.5,
                )
            )
    # Runs from the gunwale pick up fresh rust under each rusty scuff they pass.
    for run in [
        r for r in src.runs if r.where == where and r.kind in ("hull", "knock")
    ]:
        for zk, u0, u1, rusty in scuffs:
            if (
                rusty
                and u0 - 10 < run.u < u1 + 10
                and run.v - run.length < zk < run.v - 20
            ):
                head = rng.uniform(8, 18)
                stain(
                    "refresh",
                    run.u,
                    zk - STRAKE_HALF - 1,
                    rng.uniform(60, 150),
                    (head, head * 0.4),
                    (0.7, 0.15),
                )

    # The counter: runs from under the deck rim in clusters, like the working boats'
    # sterns.
    c_lo, c_hi = float(u_wall(1000 * X_TIP)), float(u_wall(1000 * L.COUNTER_START))
    p_lo, p_hi = 1000 * U0 + 100.0, 1000 * U0 + 1000 * W_END
    for _ in range(rng.randint(*wear.counter_clusters)):
        wall = rng.random() < (c_hi - c_lo) / (c_hi - c_lo + p_hi - p_lo)
        uc = rng.uniform(c_lo, c_hi) if wall else rng.uniform(p_lo, p_hi)
        spread = rng.uniform(60, 200)
        z0 = GUNWALE - 6
        rv = rng.uniform(8, 18)
        src.blobs.append(
            Blob(
                where,
                "rust",
                "counter",
                uc,
                z0 - rv,
                rng.uniform(0.5, 0.8) * spread + 20,
                rv,
                0.85,
            )
        )
        for _ in range(rng.randint(1, 3)):
            u = uc + rng.uniform(-0.4, 0.4) * spread
            head = rng.uniform(8, 18)
            # Mostly short, a few long; some faint: not a comb of equal teeth.
            stain(
                "counter",
                u,
                z0 - 1.5 * rv,
                min(60 + 360 * rng.random() ** 2, z0 - 1.5 * rv - D.Z_COUNTER - 5),
                (head * rng.uniform(0.4, 0.6), head * rng.uniform(1.2, 1.6)),
                (rng.uniform(0.45, 0.9), rng.uniform(0.08, 0.2)),
                wobble=rng.uniform(1.2, 2.2),
                beads=1.0,
            )

    # A few separate dings on the open panels, from poles, ropes and windlasses.
    for _ in range(rng.randint(*wear.dings)):
        u, z = (
            rng.uniform(u_lo + 40, u_hi - 40),
            rng.uniform(GUNWALE + 150, SIDE_TOP - 25),
        )
        if any(inside_frame(op, u, z, grow=10) for op in openings(side)):
            continue
        if D.SIDE_HATCH.side == side:
            h0, h1, z0, z1 = hatch_box()
            if h0 - 10 < u < h1 + 10 and z0 - 10 < z < z1 + 10:
                continue
        r = rng.uniform(3, 9)
        src.blobs.append(
            Blob(where, "chip", "ding", u, z, r, r * rng.uniform(0.4, 0.9), 0.95)
        )
    src.sites[where] = sites


def _plan_sources(
    src: Sources, wear: Wear, seed: int, rusty: dict[int, list[float]]
) -> None:
    rng = seeded(NAMESPACE, seed, "plan")
    for side in (1, -1):
        y = side * HANDRAIL_Y
        for b in rusty[side]:
            src.blobs.append(
                Blob(
                    "plan",
                    "rust",
                    "leg",
                    b,
                    y,
                    rng.uniform(20, 35),
                    rng.uniform(16, 28),
                    0.85,
                )
            )
            src.blobs.append(
                Blob(
                    "plan",
                    "pool",
                    "leg",
                    b,
                    y,
                    rng.uniform(40, 70),
                    rng.uniform(30, 55),
                    0.5,
                )
            )
            # The stain it sheds runs outboard down the camber and over the roof edge.
            src.blobs.append(
                Blob(
                    "plan",
                    "pool",
                    "leg",
                    b,
                    side * (HANDRAIL_Y + 60),
                    rng.uniform(15, 25),
                    75,
                    0.55,
                )
            )
        for b in handrail_feet():
            if rng.random() < wear.leg_moss:
                r = rng.uniform(25, 50)
                src.blobs.append(
                    Blob("plan", "moss", "leg", b, y, r, r, rng.uniform(0.75, 0.95))
                )
    # Cushions in the roof's end corners, where leaves and dirt collect.
    for b0, b1 in ((40.0, 450.0), (CABIN_B[1] - 450.0, CABIN_B[1] - 40.0)):
        for side in (1, -1):
            for _ in range(rng.randint(1, 3)):
                r = rng.uniform(40, 110)
                src.blobs.append(
                    Blob(
                        "plan",
                        "moss",
                        "corner",
                        rng.uniform(b0, b1),
                        side * rng.uniform(V_ROOF - 90, V_ROOF - 10),
                        r,
                        r * rng.uniform(0.6, 1.0),
                        rng.uniform(0.75, 0.95),
                    )
                )
    vr = D.VENT_D / 2
    for k, b in enumerate(D.VENT_BS):
        if k % 2 == 0:
            src.blobs.append(
                Blob(
                    "plan",
                    "rust",
                    "vent",
                    b,
                    0.0,
                    vr + 4,
                    vr + 4,
                    0.7,
                    ring=rng.uniform(8, 15),
                )
            )
        if rng.random() < 0.75:
            src.blobs.append(
                Blob(
                    "plan",
                    "verd",
                    "vent",
                    b,
                    0.0,
                    vr + 8,
                    vr + 8,
                    0.6,
                    ring=rng.uniform(12, 25),
                )
            )
        src.blobs.append(Blob("plan", "pool", "vent", b, 0.0, vr + 25, vr + 25, 0.35))
        if rng.random() < 0.5:
            for _ in range(rng.randint(1, 3)):
                a = rng.uniform(0, 2 * math.pi)
                d = vr + rng.uniform(10, 35)
                r = rng.uniform(15, 35)
                src.blobs.append(
                    Blob(
                        "plan",
                        "moss",
                        "vent",
                        b + d * math.cos(a),
                        d * math.sin(a),
                        r,
                        r,
                        0.8,
                    )
                )
    for b in D.PLANK_BLOCKS:
        y0, y1 = D.BLOCK_Y
        src.patches.append(
            Patch(
                "plan",
                "moss",
                "block",
                b - 70,
                b + 70,
                y0 - 30,
                y1 + 30,
                0.66,
                8.0,
                0.7,
                cell=30.0,
            )
        )
        if rng.random() < 0.3:
            src.blobs.append(
                Blob(
                    "plan",
                    "rust",
                    "block",
                    b + rng.choice((-40, 40)),
                    rng.uniform(y0, y1),
                    12,
                    10,
                    0.8,
                )
            )
    pw = D.PLANK_W / 2
    src.patches.append(
        Patch(
            "plan",
            "moss",
            "plank",
            D.PLANK_B[0],
            D.PLANK_B[1],
            D.PLANK_Y - pw - 30,
            D.PLANK_Y + pw + 30,
            0.5,
            15.0,
            0.8,
            cell=45.0,
        )
    )
    # Deck fittings: rust and verdigris round the studs and fillers.
    for b, y in D.T_STUDS:
        if rng.random() < 0.5:
            src.blobs.append(Blob("plan", "rust", "stud", b, y, 34, 34, 0.75, ring=10))
        src.blobs.append(Blob("plan", "verd", "stud", b, y, 34, 34, 0.55, ring=14))
    for b, y in (D.DIESEL_FILLER, D.WATER_FILLER):
        src.blobs.append(Blob("plan", "verd", "filler", b, y, 47, 47, 0.5, ring=12))
    w0, w1, wh = D.WEED_HATCH
    for b, y in ((w0, -wh), (w0, wh), (w1, -wh), (w1, wh)):
        if rng.random() < 0.5:
            src.blobs.append(Blob("plan", "rust", "weedhatch", b, y, 15, 15, 0.8))
    # Rust along the gunwale top's inner edge and the stain pooled beside each weld cluster.
    for side, where in SIDES:
        for c, length in src.welds.get(where, []):
            y_in = side * CABIN_HALF
            src.patches.append(
                Patch(
                    "plan",
                    "rust",
                    "weld",
                    c - length / 2,
                    c + length / 2,
                    min(y_in, y_in + side * 8),
                    max(y_in, y_in + side * 8),
                    0.85,
                    2.0,
                    0.4,
                )
            )
            depth = rng.uniform(25, 60)
            src.patches.append(
                Patch(
                    "plan",
                    "pool",
                    "weld",
                    c - 0.6 * length,
                    c + 0.6 * length,
                    min(y_in, y_in + side * depth),
                    max(y_in, y_in + side * depth),
                    rng.uniform(0.5, 0.8),
                    10.0,
                    0.5,
                )
            )
    # Soot streaming aft from the chimney collar, widening and thinning as it drifts.
    cb, cy = D.CHIMNEY
    length, width = rng.uniform(*wear.soot_len), rng.uniform(*wear.soot_w)
    drift = rng.uniform(-0.25, 0.1) * width
    src.blobs.append(Blob("plan", "soot", "collar", cb, cy, 120, 120, 0.55))
    for k in range(24):
        t = (k + 0.5) / 24
        src.blobs.append(
            Blob(
                "plan",
                "soot",
                "plume",
                cb - t * length,
                cy + t * drift,
                90 + 0.12 * length * t,
                70 + 0.5 * width * t,
                0.42 * (1 - t) ** 1.3,
            )
        )
    # Walked smooth where feet go: the steering spot, the door thresholds, the well deck,
    # the boarding points on the gunwales and where the roof is climbed on at the stern.
    feet = [
        (-450.0, 0.0, 330, 260, 0.85),  # steering, on the engine boards
        (-70.0, 0.0, 110, 330, 0.9),  # stern door threshold
        (-1150.0, 0.0, 420, 420, 0.45),  # round the tiller
        (12890.0, 0.0, 110, 330, 0.9),  # bow door threshold
        (13350.0, 0.0, 450, 480, 0.6),  # across the well deck
        (14150.0, 0.0, 250, 300, 0.4),  # onto the foredeck
        (900.0, 250.0, 350, 220, 0.45),  # climbing onto the roof at the stern
    ]
    for side, where in SIDES:
        gun = side * (CABIN_HALF + D.GUNWALE_W / 2)
        feet += [
            (-950.0, side * 680.0, 160, 120, 0.6),  # stepping aboard onto the counter
            (13000.0, side * 720.0, 150, 120, 0.6),  # stepping down into the well
            (rng.uniform(0, 600), gun, 350, 45, 0.7),  # boarding at the stern quarter
            (rng.uniform(12000, 12700), gun, 350, 45, 0.7),  # and at the bow
        ]
        feet += [
            (rng.uniform(2000, 11000), gun, rng.uniform(200, 400), 45, 0.45)
            for _ in range(2)
        ]
        if D.SIDE_HATCH.side == side:
            feet.append(
                (float(D.SIDE_HATCH.b), gun, 450, 45, 0.65)
            )  # handing things in
    for b, y, rb, ry, amp in feet:
        src.blobs.append(Blob("plan", "polish", "feet", b, y, rb, ry, amp))


def _end_sources(src: Sources, wear: Wear, seed: int) -> None:
    rng = seeded(NAMESPACE, seed, "ends")

    def weld(tile: Tile, lo: float, hi: float, z: float, n: tuple[int, int]):
        for _ in range(rng.randint(*n)):
            length = rng.uniform(100, 400)
            c = rng.uniform(lo + length / 2, hi - length / 2)
            bead = rng.uniform(*wear.weld_bead)
            src.patches.append(
                Patch(
                    "ends",
                    "rust",
                    "weld",
                    tile.a(c - length / 2),
                    tile.a(c + length / 2),
                    z - 3,
                    z + bead,
                    0.95,
                    2.0,
                    0.45,
                    cell=10.0,
                    stretch=3.0,
                    taper=rng.uniform(0.15, 0.3),
                    ragged=rng.uniform(3, 6),
                )
            )
            src.patches.append(
                Patch(
                    "ends",
                    "stain",
                    "creep",
                    tile.a(c - length / 2 - 15),
                    tile.a(c + length / 2 + 15),
                    z - 3,
                    z + rng.uniform(25, 50),
                    rng.uniform(0.45, 0.65),
                    4.0,
                    0.5,
                    cell=14.0,
                    stretch=2.0,
                    taper=rng.uniform(0.2, 0.35),
                    ragged=rng.uniform(6, 12),
                )
            )
            rb = rng.uniform(*wear.weld_blister)
            src.blobs.append(
                Blob(
                    "ends",
                    "rust",
                    "blister",
                    tile.a(rng.uniform(c - length / 3, c + length / 3)),
                    z + 0.35 * rb,
                    0.9 * rb,
                    0.55 * rb,
                    0.85,
                )
            )

    def runs(tile: Tile, lo: float, hi: float, top, n_dirt, n_rust, floor: float):
        for _ in range(rng.randint(*n_dirt)):
            c = rng.uniform(lo, hi)
            z0 = top(c) - 4
            src.runs.append(
                Run(
                    "ends",
                    "dirt",
                    "dirt",
                    tile.a(c),
                    z0,
                    min(rng.uniform(150, 600), z0 - floor - 2),
                    (rng.uniform(12, 40), rng.uniform(6, 20)),
                    (rng.uniform(0.35, 0.6), 0.05),
                )
            )
        for _ in range(rng.randint(*n_rust)):
            c = rng.uniform(lo, hi)
            z0 = top(c) - 4
            src.blobs.append(
                Blob(
                    "ends",
                    "rust",
                    "roofspot",
                    tile.a(c),
                    z0,
                    rng.uniform(6, 12),
                    rng.uniform(4, 8),
                    0.85,
                )
            )
            src.runs.append(
                Run(
                    "ends",
                    "stain",
                    "roofspot",
                    tile.a(c),
                    z0,
                    min(rng.uniform(250, 700), z0 - floor - 2),
                    (rng.uniform(8, 16), rng.uniform(4, 8)),
                    (rng.uniform(0.6, 0.9), 0.2),
                )
            )

    def chips(tile: Tile, c: float, z: float, size: float, n=(3, 7)):
        """A handful of chips of varied size and shape round (c, z) on a tile."""
        for _ in range(rng.randint(*n)):
            cc, cz = c + rng.gauss(0, 0.5 * size), z + rng.gauss(0, 0.3 * size)
            ru, rv = rng.uniform(0.15, 0.45) * size, rng.uniform(0.1, 0.35) * size
            src.blobs.append(Blob("ends", "chip", "edge", tile.a(cc), cz, ru, rv, 0.95))
            if rng.random() < 0.6:
                src.blobs.append(
                    Blob(
                        "ends", "rust", "edge", tile.a(cc), cz, 0.6 * ru, 0.6 * rv, 0.85
                    )
                )

    roof_top = lambda y: 1000 * L.roof_z(abs(y) / 1000)
    for name, foot in (("Rear", GUNWALE), ("Front", float(D.WELL_SOLE_Z))):
        tile = TILES[name]
        half = 1000 * L.CABIN_TOP_HALF - 20
        weld(tile, -half, half, foot, (2, 3))
        # Soft dirt runs from the roof's front edge; no rust runs without a source.
        runs(tile, -half, half, roof_top, (1, 3), (0, 0), foot)
        door = next(d for d in D.DOORS if (d.end == "Stern") == (name == "Rear"))
        # The leaves' edges are chipped where they are handled (the meeting edges at
        # handle height) and kicked (every bottom corner), not all along.
        zh = door.z0 + rng.uniform(0.45, 0.55) * (door.z1 - door.z0)
        for y in (-door.half, -5.0, 5.0, float(door.half)):
            spans = [(door.z0, door.z0 + rng.uniform(80, 200))]
            if abs(y) < 10:
                spans.append((zh - rng.uniform(40, 100), zh + rng.uniform(40, 100)))
            for z0, z1 in spans:
                w = rng.uniform(7, 16)
                src.patches.append(
                    Patch(
                        "ends",
                        "chip",
                        "door",
                        tile.a(y - w),
                        tile.a(y + w),
                        z0,
                        z1,
                        0.85,
                        2.0,
                        0.8,
                        cell=12.0,
                        stretch=0.3,
                        taper=0.0,
                    )
                )
        # Boots at the foot of the doors: chip clusters, not a ruled strip.
        for _ in range(rng.randint(2, 4)):
            chips(
                tile,
                rng.uniform(-door.half, door.half),
                door.z0 + rng.uniform(5, 25),
                rng.uniform(25, 60),
            )
        # The bulkhead's outer edges, where the cabin sides meet it, take knocks too.
        for side in (1, -1):
            edge = side * (1000 * L.cabin_side_y(L.Z_GUNWALE) - 12)
            for _ in range(rng.randint(2, 4)):
                chips(
                    tile,
                    edge,
                    rng.uniform(foot + 60, SIDE_TOP - 40),
                    rng.uniform(20, 50),
                )
    for name, side in (("ShelterPort", 1), ("ShelterStarboard", -1)):
        tile = TILES[name]
        weld(tile, CABIN_B[0] + 40, -60.0, GUNWALE, (1, 2))
        runs(tile, CABIN_B[0] + 30, -30.0, lambda b: SIDE_TOP, (1, 2), (0, 0), GUNWALE)
        # The shelter plate's aft edge: chips in a few places along it.
        for _ in range(rng.randint(3, 6)):
            chips(
                tile,
                CABIN_B[0] + 15,
                rng.uniform(GUNWALE + 40, SIDE_TOP - 30),
                rng.uniform(20, 45),
            )


def camera_aims(wear: Wear, seed: int) -> dict[str, float]:
    """b (mm) for the close-ups that look at seeded wear, so they always find some: the
    port porthole with verdigris (one that weeps too, if any), and the starboard window
    with the most weld rust under it."""
    picks = porthole_picks(wear, seed)
    port = sorted(b for side, b in picks["verd"] if side > 0)
    weeping = [b for b in port if (1, b) in picks["weep"]]
    aims = {"porthole": (weeping or port or [4450.0])[0]}
    src = sources(wear, seed)
    best, aims["weld"] = 0.0, 7450.0
    for op in openings(-1):
        if op.kind == "porthole":
            continue
        hw, _, _ = _frame_shape(op)
        rust = sum(
            max(0.0, min(c + ln / 2, op.b + hw) - max(c - ln / 2, op.b - hw))
            for c, ln in src.welds["starboard"]
        )
        if rust > best:
            best, aims["weld"] = rust, float(op.b)
    return aims


def handrail_feet() -> list[float]:
    """b (mm) of every place a handrail comes down to the roof: its legs and its ends."""
    return [1000 * b for b in N.handrail_legs()] + [float(v) for v in D.HANDRAIL_B]


def sources(wear: Wear, seed: int) -> Sources:
    """Every wear source on the boat, from the design and the seed."""
    src = Sources()
    rng = seeded(NAMESPACE, seed, "legs")
    rusty = {
        side: [b for b in handrail_feet() if rng.random() < wear.leg_rust]
        for side in (1, -1)
    }
    for side, where in SIDES:
        _side_sources(src, wear, side, where, seed, rusty[side])
    _plan_sources(src, wear, seed, rusty)
    _end_sources(src, wear, seed)
    import narrowboat_dents as DN  # it builds on this module

    hurt = DN.dent_sources(wear, DN.dents(wear, seed), seed)
    src.blobs += hurt.blobs
    src.runs += hurt.runs
    return src


# --------------------------------------------------------------------------- rasters


def _above(target: np.ndarray, frame: MapFrame, floor: float) -> np.ndarray:
    """The rows of `target` above map height `floor`, as a view (row 0 is the top edge, so
    drawing into the view draws into `target` at the same rows)."""
    if floor == -math.inf:
        return target
    return target[: max(round(float(frame.row(floor))), 0)]


def _draw(arrays: dict, frame: MapFrame, src: Sources, where: str, rng) -> None:
    """Rasterise one surface's patches, blobs and runs into its layer arrays."""
    px = frame.px
    for p in src.of(where, Patch):
        target = _above(arrays[p.layer], frame, p.floor)
        soft = max(p.soft / px, 0.5)
        r0, r1 = sorted((float(frame.row(p.v1)), float(frame.row(p.v0))))
        c0, c1 = float(frame.col(p.u0)), float(frame.col(p.u1))
        a, b = max(int(r0 - soft) - 2, 0), min(int(r1 + soft) + 3, target.shape[0])
        c, d = max(int(c0 - soft) - 2, 0), min(int(c1 + soft) + 3, target.shape[1])
        if a >= b or c >= d:
            continue
        tmp = np.zeros((b - a, d - c), np.float32)
        if p.jitter:
            # A crisp edge that wanders: threshold a soft rectangle plus noise.
            pad = int(p.jitter / px) + 2
            a, b = max(a - pad, 0), min(b + pad, target.shape[0])
            c, d = max(c - pad, 0), min(d + pad, target.shape[1])
            tmp = np.zeros((b - a, d - c), np.float32)
            M.rect(tmp, r0 - a, r1 - a, c0 - c, c1 - c, 1.0, p.jitter / px, mode="max")
            n = M.value_noise(tmp.shape, max(1.5 * p.jitter / px, 1.5), rng, octaves=2)
            tmp = (p.amp * ((tmp + 0.7 * (n - 0.5)) > 0.5)).astype(np.float32)
        elif p.ragged:
            # The upper edge (the lower row) wanders along the patch's length.
            pad = int(p.ragged / px) + 2
            a = max(a - pad, 0)
            cell = max(p.cell * p.stretch / px, 1.5)
            wander = (p.ragged / px) * (2 * M.noise_1d(d - c, cell, rng) - 1)
            yy = np.arange(a, b, dtype=np.float32)[:, None] + 0.5
            xx = np.arange(c, d, dtype=np.float32)[None, :] + 0.5
            fy = np.minimum(yy - (r0 + wander[None, :]), r1 - yy)
            fx = np.minimum(xx - c0, c1 - xx)
            tmp = (
                p.amp * np.clip(fy / soft + 0.5, 0, 1) * np.clip(fx / soft + 0.5, 0, 1)
            ).astype(np.float32)
        else:
            M.rect(tmp, r0 - a, r1 - a, c0 - c, c1 - c, p.amp, soft, mode="max")
        if p.taper:
            cols = np.arange(c, d, dtype=np.float32) + 0.5
            ramp = max(p.taper * (c1 - c0), 1.0)
            fade = np.clip(np.minimum(cols - c0, c1 - cols) / ramp, 0, 1)
            tmp *= fade[None, :] ** 1.5
        if p.breakup:
            n = M.value_noise(
                tmp.shape,
                max(p.cell * p.stretch / px, 1.5),
                rng,
                octaves=3,
                cell_y=max(p.cell / px, 1.5),
            )
            tmp *= 1 - p.breakup + p.breakup * 1.6 * n
        M.combine(target[a:b, c:d], np.clip(tmp, 0, 1), "max")
    for bl in src.of(where, Blob):
        target = _above(arrays[bl.layer], frame, bl.floor)
        mode = "screen" if bl.layer in ("stain", "pool", "soot") else "max"
        r, c = float(frame.row(bl.v)), float(frame.col(bl.u))
        if bl.ring:
            outer = (bl.ru + bl.ring) / px
            a, b = max(int(r - outer) - 1, 0), min(int(r + outer) + 2, target.shape[0])
            cc, d = max(int(c - outer) - 1, 0), min(int(c + outer) + 2, target.shape[1])
            if a >= b or cc >= d:
                continue
            yy = (np.arange(a, b) + 0.5 - r)[:, None]
            xx = (np.arange(cc, d) + 0.5 - c)[None, :]
            dist = np.hypot(yy, xx) * px
            t = np.clip(1 - np.abs(dist - bl.ru - bl.ring / 2) / (bl.ring / 2), 0, 1)
            if bl.down:
                # Only the lower part, where water sits, and unevenly.
                t = t * np.clip(1.6 * yy * px / (bl.ru + bl.ring), 0, 1) ** 3
                t = t * (
                    0.35
                    + 0.9 * M.value_noise(t.shape, max(15 / px, 1.5), rng, octaves=2)
                )
            M.combine(
                target[a:b, cc:d],
                (bl.amp * t * t * (3 - 2 * t)).astype(np.float32),
                mode,
            )
        else:
            M.splat(target, r, c, bl.rv / px, bl.ru / px, bl.amp, mode=mode)
    for run in src.of(where, Run):
        target = arrays[run.layer]
        mode = "max" if run.layer == "chip" else "screen"
        if run.horizontal:
            M.run(
                target.T,
                float(frame.col(run.u)),
                float(frame.row(run.v)),
                run.length / px,
                (run.width[0] / px, run.width[1] / px),
                run.amp,
                rng,
                wobble=0.3,
                beads=run.beads,
                mode=mode,
                step=60,
            )
        else:
            M.run(
                target,
                float(frame.row(run.v)),
                float(frame.col(run.u)),
                run.length / px,
                (run.width[0] / px, run.width[1] / px),
                run.amp,
                rng,
                wobble=run.wobble,
                beads=run.beads,
                mode=mode,
                neck=run.neck,
            )


def _cabin_mask(frame: MapFrame, side: int):
    """(paint, cabin) masks of one side band: the cabin side's paint (frames and hatch
    excluded) and the whole cabin side rectangle."""
    uu, zz = frame.grid()
    cabin = (
        (zz >= GUNWALE)
        & (zz <= SIDE_TOP)
        & (uu >= u_wall(CABIN_B[0]))
        & (uu <= u_wall(CABIN_B[1]))
    )
    paint = cabin.copy()
    for op in openings(side):
        paint &= ~inside_frame(op, uu, zz)
    if D.SIDE_HATCH.side == side:
        h0, h1, z0, z1 = hatch_box()
        paint &= ~((uu > h0) & (uu < h1) & (zz > z0) & (zz < z1))
    return paint, cabin


def coachline_mask(uu, zz):
    """Where the cream and red coachlines run on a cabin side (mm), with a little margin."""
    cream, inset, red = 1000 * N.CREAM_W, 1000 * N.RED_INSET, 1000 * N.RED_W
    (b0, b1), (z0, z1) = PANEL_B, PANEL_Z
    inside = (uu > b0 - 3) & (uu < b1 + 3) & (zz > z0 - 3) & (zz < z1 + 3)
    out = np.zeros(np.broadcast(uu, zz).shape, bool)
    for a, w in ((0.0, cream), (inset, red)):
        for lo, hi in ((z0 + a, z0 + a + w), (z1 - a - w, z1 - a)):
            out |= (zz > lo - 3) & (zz < hi + 3)
        for lo, hi in ((b0 + a, b0 + a + w), (b1 - a - w, b1 - a)):
            out |= (uu > lo - 3) & (uu < hi + 3)
    return out & inside


def _grow_touchups(arrays, frame, src, wear, side, where, rng) -> None:
    """Touch-ups over some of the damage. An owner brushes paint over a rust patch with a
    margin: a rough rectangle round the damage (or round one stretch of a long weld
    cluster), its ends ragged along the horizontal brush strokes and its top edge wavering.
    It stops at the coachlines (which keep their own wear), the frames, the hatch and the
    art, and no two touch."""
    sites = src.sites.get(where, [])
    if not sites:
        return
    t, px = wear.threshold, frame.px
    uu, zz = frame.grid()
    _, cabin = _cabin_mask(frame, side)
    keep_out = coachline_mask(uu, zz) | ~cabin
    for op in openings(side):
        keep_out |= inside_frame(op, uu, zz, grow=4)
    if D.SIDE_HATCH.side == side:
        h0, h1, z0, z1 = hatch_box()
        keep_out |= (uu > h0 - 4) & (uu < h1 + 4) & (zz > z0 - 4) & (zz < z1 + 4)
    for a0, a1, b0, b1 in decal_rects(side):
        keep_out |= (uu > a0 - 10) & (uu < a1 + 10) & (zz > b0 - 10) & (zz < b1 + 10)
    damage = (
        (arrays["rust"] > t - 0.12)
        | (arrays["stain"] > 0.35)
        | (arrays["chip"] > t - 0.1)
    )
    pool = list(sites)
    chosen: list[tuple[float, float]] = []
    n = rng.randint(*wear.touchups)
    while pool and len(chosen) < n:
        total = sum(item[2] for item in pool)
        pick = rng.uniform(0, total)
        for k, item in enumerate(pool):
            pick -= item[2]
            if pick <= 0:
                break
        u, half, _, z_top = pool.pop(k)
        # The owner paints round the lining rather than over it, and no higher than the
        # red line: the weep above it is left.
        z_top = min(z_top, PANEL_Z[0] + 1000 * N.RED_INSET - 3)
        if half > 200:
            # Part of a long weld cluster: the owner got to one stretch of it.
            u += rng.uniform(-0.5, 0.5) * half
            half *= rng.uniform(0.25, 0.5)
        if any(abs(u - cu) < half + ch + 300 for cu, ch in chosen):
            continue
        grow = rng.uniform(10, 25)
        ends = rng.uniform(40, 70)  # how far the brush strokes straggle at each end
        pad = grow + 2 * ends
        c0 = max(int(frame.col(u - half - pad)), 0)
        c1 = min(int(frame.col(u + half + pad)) + 1, frame.w)
        r0 = max(int(frame.row(min(z_top + grow + 10, SIDE_TOP))), 0)
        r1 = min(int(frame.row(GUNWALE - 10)) + 1, frame.h)
        win = (slice(r0, r1), slice(c0, c1))
        inner = (uu[:, c0:c1] > u - half) & (uu[:, c0:c1] < u + half)
        dmg = damage[win] & cabin[win] & inner & (zz[r0:r1] <= z_top)
        if dmg.sum() < 20:
            continue
        rows = np.nonzero(dmg.any(axis=1))[0]
        top = float(zz[r0 + rows.min(), 0]) + grow
        bottom = float(zz[r0 + rows.max(), 0]) - grow
        uw, zw = uu[:, c0:c1], zz[r0:r1]
        fx = np.clip(
            np.minimum(uw - (u - half - grow), (u + half + grow) - uw) / ends, -1, 1
        )
        fz = np.clip(np.minimum(zw - bottom, top - zw) / 6.0, -1, 1)
        strokes = M.value_noise(
            (r1 - r0, c1 - c0),
            max(60 / px, 1.5),
            rng,
            octaves=2,
            cell_y=max(5 / px, 1.0),
        )
        region = (np.minimum(fx, fz) + 1.1 * (strokes - 0.5)) > 0
        region &= ~keep_out[win]
        arrays["touch"][win][region] = rng.uniform(0.3, 1.0)
        chosen.append((u, half))


def _drop_specks(chip: np.ndarray, where, t: float, px: float) -> None:
    """Chips under about 6 mm that belong to no cluster read as sprinkled: push them
    below the threshold (in place)."""
    lone = M.specks((chip > t) & where, int(3.0 / px))
    chip[lone] = np.minimum(chip[lone], t - 0.15)


def _drag_down(source: np.ndarray, frame: MapFrame, length: float, rng) -> np.ndarray:
    """Stain carried down the plate from `source` (0-1): each column fades over about
    `length` mm below what feeds it, some columns further than others, as water does."""
    px = frame.px
    reach = 0.5 + M.noise_1d(source.shape[1], max(30 / px, 1.5), rng)
    decay = np.exp(-px / (length * reach)).astype(np.float32)
    out = np.zeros_like(source)
    acc = np.zeros(source.shape[1], np.float32)
    for r in range(source.shape[0]):  # row 0 is the top edge
        acc = np.maximum(acc * decay, source[r])
        out[r] = acc
    return out


def _heavy_side(arrays, frame: MapFrame, wear: Wear, side: int, rng, paint, cabin):
    """Broad wear over one side band, beyond the damage at its sources: flaking patches
    of cabin paint (more of them low down), rusting through in the middle of the bigger
    ones; blacking flaked off the hull side above the water, rusty underneath; rust
    stain dragged down from everything rusty; grime low on the plate and streaks of dirt
    from the roof edge."""
    if wear.flake <= 0 and wear.hull_flake <= 0:
        return
    t = wear.threshold
    px = frame.px
    shape = arrays["chip"].shape
    uu, zz = frame.grid()
    failing = np.clip(
        (M.value_noise(shape, 450 / px, rng, octaves=2) - 0.35) * 2.2, 0, 1
    )
    flakes = M.value_noise(shape, 28 / px, rng, octaves=3)
    low = np.clip(1.15 - (zz - GUNWALE) / 900, 0.35, 1.0)
    field_ = (failing * flakes * low * paint).astype(np.float32)
    if wear.flake > 0 and paint.any():
        q = M.coverage_threshold(field_, wear.flake, paint)
        flake = field_ * (t / max(q, 1e-6))
        arrays["chip"] = np.maximum(arrays["chip"], flake)
        # The bigger flakes' middles: through the primer to rust.
        core = M.value_noise(shape, 90 / px, rng, octaves=2)
        arrays["rust"] = np.maximum(
            arrays["rust"], (flake - 0.12) * (core > 0.45) * paint
        )
    wl = 1000 * np.interp(b_of_u(uu) / 1000, _GB, [L.wl_z(b) for b in _GB])
    hull = (zz < GUNWALE - 5) & (zz > wl + 20)
    if wear.hull_flake > 0 and hull.any():
        lost = (
            np.clip((M.value_noise(shape, 350 / px, rng, octaves=2) - 0.3) * 2, 0, 1)
            * M.value_noise(shape, 22 / px, rng, octaves=3)
            * hull
        ).astype(np.float32)
        q = M.coverage_threshold(lost, wear.hull_flake, hull)
        lost *= t / max(q, 1e-6)
        arrays["chip"] = np.maximum(arrays["chip"], lost)
        arrays["rust"] = np.maximum(arrays["rust"], (lost - 0.05) * hull)
    rusty = np.clip((arrays["rust"] - t) / 0.1, 0, 1).astype(np.float32)
    stain = _drag_down(rusty, frame, wear.stain_drop, rng)
    stain *= 0.35 + 0.65 * M.value_noise(shape, 60 / px, rng, octaves=2)
    arrays["stain"] = np.maximum(arrays["stain"], stain * (paint | hull))
    if wear.grime > 0:
        broad = M.value_noise(shape, 300 / px, rng, octaves=2)
        foot = np.clip(1 - (zz - GUNWALE) / 700, 0, 1) * paint
        streaks = M.noise_1d(shape[1], max(25 / px, 1.5), rng)[None, :]
        top = np.clip((zz - (SIDE_TOP - 900)) / 900, 0, 1) * paint
        dirt = wear.grime * (
            0.6 * foot * (0.4 + 0.6 * broad)
            + 0.7 * top * np.clip((streaks - 0.45) * 2.5, 0, 1)
        )
        arrays["dirt"] = np.maximum(arrays["dirt"], dirt.astype(np.float32))


def side_layers(
    wear: Wear, src: Sources, side: int, seed: int, px: float = PX_MM
) -> dict:
    """Layers of one side band: rust, stain, chip, dirt, touch, verd (each (h, w))."""
    where = dict(SIDES)[side]
    frame = frames(px)["side"]
    shape = (frame.h, frame.w)
    arrays = {
        k: np.zeros(shape, np.float32)
        for k in ("rust", "stain", "chip", "dirt", "touch", "verd")
    }
    rng = seeded(NAMESPACE, seed, f"raster-{where}")
    _draw(arrays, frame, src, where, rng)
    uu, zz = frame.grid()
    paint, cabin = _cabin_mask(frame, side)
    # Chips on the cabin sides, area by area, in clusters: some stretches stay clean.
    ends = np.minimum(np.abs(uu - u_wall(CABIN_B[0])), np.abs(uu - u_wall(CABIN_B[1])))
    line = (
        (zz > PANEL_Z[0] - 3)
        & (zz < PANEL_Z[0] + 17)
        & (uu > PANEL_B[0])
        & (uu < PANEL_B[1])
    )
    corners = np.zeros(shape, np.float32)
    for bp in (PANEL_B[0], PANEL_B[1]):
        for bz in (PANEL_Z[0], PANEL_Z[1]):
            corners = np.maximum(
                corners, np.clip(1 - np.hypot(uu - bp, zz - bz) / 80, 0, 1) ** 2
            )
    rings = np.zeros(shape, np.float32)
    for op in openings(side):
        u0, u1, z0, z1 = frame_box(op)
        c0, c1 = (
            max(int(frame.col(u0 - 20)), 0),
            min(int(frame.col(u1 + 20)) + 1, frame.w),
        )
        r0, r1 = (
            max(int(frame.row(z1 + 20)), 0),
            min(int(frame.row(z0 - 20)) + 1, frame.h),
        )
        d = frame_distance(op, uu[:, c0:c1], zz[r0:r1])
        rings[r0:r1, c0:c1] = np.maximum(
            rings[r0:r1, c0:c1], np.clip(1 - d / 12, 0, 1) ** 2 * (d > 0)
        )
    cluster = np.clip(
        (M.value_noise(shape, 250 / px, rng, octaves=2) - 0.45) * 3.0, 0, 1
    )
    # Separate flakes of 6-20 mm rather than speckle.
    base = cluster * M.value_noise(shape, 16 / px, rng, octaves=2)
    hatch = np.zeros(shape, np.float32)
    if D.SIDE_HATCH.side == side:
        # The leaves' edges and the gap between them, where hands and the deck pole go.
        h0, h1, z0, z1 = hatch_box()
        hb, gap = float(D.SIDE_HATCH.b), 1000 * N.HATCH_GAP
        d = np.minimum(np.minimum(uu - h0, h1 - uu), np.minimum(zz - z0, z1 - zz))
        d = np.minimum(d, np.abs(uu - hb) - gap)
        hatch = np.clip(1 - d / 12, 0, 1) ** 2 * (d > -2)
    top = np.clip(1 - (SIDE_TOP - zz) / 15, 0, 1) ** 2
    areas = (
        (line & paint, wear.chip_line),
        (np.clip(1 - (zz - GUNWALE) / 60, 0, 1) * paint * ~line, wear.chip_foot),
        (np.clip(1 - ends / 20, 0, 1) ** 2 * paint, wear.chip_ends),
        (rings * paint, wear.chip_rings),
        (corners * paint, wear.chip_corners),
        (hatch * cabin, wear.chip_hatch),
        (top * paint, wear.chip_top),
    )
    t = wear.threshold
    for weight, share in areas:
        field_ = np.asarray(weight, np.float32) * base
        eligible = np.asarray(weight, np.float32) > 0.05
        if share <= 0 or not eligible.any():
            continue
        q = M.coverage_threshold(field_, share, eligible)
        arrays["chip"] = np.maximum(arrays["chip"], field_ * (t / max(q, 1e-6)))
    _drop_specks(arrays["chip"], paint, t, px)
    _heavy_side(arrays, frame, wear, side, rng, paint, cabin)
    _grow_touchups(arrays, frame, src, wear, side, where, rng)
    # Fresh paint over the touch-ups hides the damage under them.
    hidden = (arrays["touch"] > 0.05) & cabin
    for k in ("rust", "stain", "chip", "verd"):
        arrays[k][hidden] = 0.0
    for value in arrays.values():
        np.clip(value, 0, 1, out=value)
    return arrays


def plan_layers(
    wear: Wear, src: Sources, seed: int, px: float = PX_MM
) -> tuple[dict, dict]:
    """Layers of the plan map (rust, moss, polish, soot, pool, verd) and the moss and algae
    thresholds that give the wear level's coverage."""
    frame = frames(px)["plan"]
    shape = (frame.h, frame.w)
    arrays = {
        k: np.zeros(shape, np.float32)
        for k in ("rust", "moss", "polish", "soot", "pool", "verd")
    }
    rng = seeded(NAMESPACE, seed, "raster-plan")
    _draw(arrays, frame, src, "plan", rng)
    bb, vv = frame.grid()
    av = np.abs(vv)
    roof = (bb >= 0) & (bb <= CABIN_B[1]) & (av <= V_ROOF)
    # Moss outboard of the handrails, where water lingers before it runs off the edge: a
    # few small cushions where the clumping allows, more on the port side (which the
    # plank and pole leave to the weather less often).
    band = np.clip((av - HANDRAIL_Y - 15) / 40, 0, 1) * np.clip(
        (V_ROOF + 5 - av) / 30, 0, 1
    )
    where_ = np.clip(
        (M.value_noise(shape, 900 / px, rng, octaves=2) - 0.45) * 3.0, 0, 1
    )
    cushions = M.value_noise(shape, 60 / px, rng, octaves=3)
    bias = np.where(vv > 0, 1.0, 0.5).astype(np.float32)
    edge_moss = band * roof * where_ * cushions * bias
    t = wear.threshold
    q = M.coverage_threshold(edge_moss, wear.moss_edge, roof)
    arrays["moss"] = np.maximum(arrays["moss"], edge_moss * (t / max(q, 1e-6)))
    # Algae: a thinner film smeared round the moss and along the lowest edge. The film
    # stays below the moss threshold, so it never reads as moss; its own threshold is
    # chosen for the wear level's coverage.
    smear = M.gauss_blur(arrays["moss"], 25 / px)
    smear = np.maximum(
        smear, 0.5 * band * where_ * M.value_noise(shape, 120 / px, rng, octaves=2)
    )
    arrays["moss"] = np.maximum(
        arrays["moss"], 0.85 * t * np.clip(smear / (0.85 * t), 0, 1) ** 0.7
    )
    t_algae = min(
        M.coverage_threshold(arrays["moss"] * roof, wear.algae, roof), 0.95 * t
    )
    # Feet polish the deck, feathered by scuffing; along the gunwales a lighter, patchy
    # wear from walking the length of the boat, following the actual gunwale.
    scuff = 0.55 + 0.45 * M.value_noise(shape, 40 / px, rng, octaves=3)
    half = 1000 * np.interp(bb / 1000, _GB, _GY, left=np.nan, right=np.nan)
    gun = (
        (av > half - D.GUNWALE_W + 5)
        & (av < half - 5)
        & (bb > CABIN_B[0])
        & (bb < D.STEM - 300)
    )
    walk = 0.3 * gun * M.value_noise(shape, 500 / px, rng, octaves=2)
    arrays["polish"] = (np.maximum(arrays["polish"], walk) * scuff).astype(np.float32)
    # Damp corners on the decks.
    clumps = M.value_noise(shape, 80 / px, rng, octaves=3)
    wet = ((bb > CABIN_B[0]) & (bb < -100) & (av > 780) & (av < 900)) | (
        (bb > 13900) & (bb < 14100) & (av > 600) & (av < 800)
    )
    arrays["moss"] = np.maximum(arrays["moss"], 0.5 * wet * clumps)
    # The roof's paint failing to rust in patches, mostly towards its edges and corners
    # where water stands and ropes and boots wear it.
    if wear.roof_rust > 0:
        edgeward = np.clip((av - 250) / 450, 0.2 + 3 * wear.roof_centre, 1.0)
        patch = (
            np.clip((M.value_noise(shape, 500 / px, rng, octaves=2) - 0.35) * 2, 0, 1)
            * M.value_noise(shape, 35 / px, rng, octaves=3)
            * edgeward
            * roof
        ).astype(np.float32)
        q = M.coverage_threshold(patch, wear.roof_rust, roof)
        patch *= t / max(q, 1e-6)
        arrays["rust"] = np.maximum(arrays["rust"], patch)
        arrays["pool"] = np.maximum(
            arrays["pool"], 0.8 * M.gauss_blur(np.clip((patch - t) * 6, 0, 1), 30 / px)
        )
        # Water standing in the roof's dips: rust-brown tide marks in shallow puddles
        # along the middle, where the roof sags between its carlines.
        dips = (
            np.clip((M.value_noise(shape, 700 / px, rng, octaves=2) - 0.55) * 4, 0, 1)
            * np.clip(1 - av / 500, 0, 1)
            * roof
        )
        arrays["pool"] = np.maximum(arrays["pool"], 0.7 * dips.astype(np.float32))
    # Soot settles as a fine speckle, not a smooth blur.
    speckle = M.value_noise(shape, 12 / px, rng, octaves=2)
    arrays["soot"] *= 0.25 + 1.1 * speckle
    for value in arrays.values():
        np.clip(value, 0, 1, out=value)
    return arrays, {"moss": t, "algae": float(t_algae)}


def end_layers(wear: Wear, src: Sources, seed: int, px: float = PX_MM) -> dict:
    """Layers of the ends atlas: rust, stain, chip, dirt, touch, verd."""
    frame = frames(px)["ends"]
    shape = (frame.h, frame.w)
    arrays = {
        k: np.zeros(shape, np.float32)
        for k in ("rust", "stain", "chip", "dirt", "touch", "verd")
    }
    rng = seeded(NAMESPACE, seed, "raster-ends")
    _draw(arrays, frame, src, "ends", rng)
    cluster = np.clip(
        (M.value_noise(shape, 200 / px, rng, octaves=2) - 0.3) * 2.2, 0, 1
    )
    fine = M.value_noise(shape, 5 / px, rng, octaves=3)
    arrays["chip"] *= cluster * (0.4 + 0.9 * fine)
    _drop_specks(arrays["chip"], np.ones(shape, bool), wear.threshold, px)
    _heavy_ends(arrays, frame, wear, rng)
    for value in arrays.values():
        np.clip(value, 0, 1, out=value)
    return arrays


def _heavy_ends(arrays, frame: MapFrame, wear: Wear, rng) -> None:
    """Broad wear on the bulkheads and doors, heaviest where they are used: paint flaked
    off round the leaves' edges, the handles and the bottom rails (kicked, scraped by
    boots and fenders), rust in the larger flakes, stain dragged down from it, and grime
    up from the deck."""
    if wear.ends_flake <= 0:
        return
    t = wear.threshold
    px = frame.px
    shape = arrays["chip"].shape
    aa, zz = frame.grid()
    aa, zz = np.broadcast_to(aa, shape), np.broadcast_to(zz, shape)
    half = 1000 * L.CABIN_TOP_HALF
    for name, foot in (("Rear", GUNWALE), ("Front", float(D.WELL_SOLE_Z))):
        tile = TILES[name]
        y = aa - tile.offset + tile.lo
        ys = np.linspace(0, half, 200)
        roof = np.interp(np.abs(y), ys, [1000 * L.roof_z(v / 1000) for v in ys])
        plate = (np.abs(y) < half) & (zz > foot) & (zz < roof) & (aa >= tile.offset)
        plate &= aa <= tile.offset + (tile.hi - tile.lo)
        if not plate.any():
            continue
        door = next(d for d in D.DOORS if (d.end == "Stern") == (name == "Rear"))
        on_door = (np.abs(y) < door.half) & (zz > door.z0) & (zz < door.z1)
        edges = np.minimum.reduce(
            [
                np.abs(np.abs(y) - door.half),
                np.abs(np.abs(y) - 5.0),
                np.abs(zz - door.z0),
                np.abs(zz - door.z1),
            ]
        )
        handle = door.z0 + 0.5 * (door.z1 - door.z0)
        near = np.clip(1 - edges / 140, 0, 1) * (on_door | (edges < 60))
        grip = np.clip(1 - np.hypot(np.abs(y) - 60, zz - handle) / 200, 0, 1)
        low = np.clip(1.1 - (zz - foot) / 700, 0.3, 1.0)
        failing = np.clip(
            (M.value_noise(shape, 300 / px, rng, octaves=2) - 0.3) * 2.0, 0, 1
        )
        weight = np.maximum.reduce([near, grip, 0.6 * low]) * (0.3 + 0.7 * failing)
        flakes = M.value_noise(shape, 22 / px, rng, octaves=3)
        field_ = (weight * flakes * plate).astype(np.float32)
        q = M.coverage_threshold(field_, wear.ends_flake, plate)
        flake = field_ * (t / max(q, 1e-6))
        arrays["chip"] = np.maximum(arrays["chip"], flake)
        core = M.value_noise(shape, 70 / px, rng, octaves=2)
        arrays["rust"] = np.maximum(
            arrays["rust"], (flake - 0.1) * (core > 0.42) * plate
        )
        rusty = np.clip((arrays["rust"] - t) / 0.1, 0, 1) * plate
        stain = _drag_down(rusty.astype(np.float32), frame, wear.stain_drop, rng)
        stain *= 0.35 + 0.65 * M.value_noise(shape, 50 / px, rng, octaves=2)
        arrays["stain"] = np.maximum(arrays["stain"], stain * plate)
        dirt = wear.grime * np.clip(1 - (zz - foot) / 600, 0, 1) * plate
        dirt *= 0.4 + 0.6 * M.value_noise(shape, 200 / px, rng, octaves=2)
        arrays["dirt"] = np.maximum(arrays["dirt"], dirt.astype(np.float32))


@dataclass
class Layers:
    side: dict  # "port"/"starboard" -> layer arrays
    plan: dict
    ends: dict
    thresholds: dict
    frames: dict


def layers(wear: Wear, seed: int, px: float = PX_MM) -> Layers:
    src = sources(wear, seed)
    plan, thresholds = plan_layers(wear, src, seed, px)
    return Layers(
        {where: side_layers(wear, src, side, seed, px) for side, where in SIDES},
        plan,
        end_layers(wear, src, seed, px),
        thresholds,
        frames(px),
    )


MAPS = {
    "SideA": ("rust", "stain", "chip"),
    "SideB": ("dirt", "touch", "verd"),
    "PlanA": ("rust", "moss", "polish"),
    "PlanB": ("soot", "pool", "verd"),
    "EndsA": ("rust", "stain", "chip"),
    "EndsB": ("dirt", "touch", "verd"),
}


def wear_maps(lay: Layers) -> dict[str, np.ndarray]:
    """The map images: uint8 RGB per map (side maps with the port band above starboard),
    and the 16-bit girth lookup."""
    out = {}
    for name, channels in MAPS.items():
        if name.startswith("Side"):
            bands = [
                M.to_u8(*(lay.side[w][c] for c in channels))
                for w in ("port", "starboard")
            ]
            out[name] = np.concatenate(bands, axis=0)
        else:
            source = lay.plan if name.startswith("Plan") else lay.ends
            out[name] = M.to_u8(*(source[c] for c in channels))
    lut, _ = girth_lut()
    out["Girth"] = M.to_u16(lut)[None, :]
    return out


def coverage(wear: Wear, lay: Layers) -> dict:
    """Share of each region above the shaders' thresholds (stains: above 0.5)."""
    t = wear.threshold
    out = {}
    for side, where in SIDES:
        a = lay.side[where]
        paint, _ = _cabin_mask(lay.frames["side"], side)
        n = max(int(paint.sum()), 1)
        out[where] = {
            "cabin_rust": float((a["rust"][paint] > t).sum() / n),
            "cabin_stain": float((a["stain"][paint] > 0.5).sum() / n),
            "cabin_chip": float((a["chip"][paint] > t).sum() / n),
            "cabin_touch": float((a["touch"][paint] > 0.05).sum() / n),
        }
    frame = lay.frames["plan"]
    bb, vv = frame.grid()
    roof = (bb >= 0) & (bb <= CABIN_B[1]) & (np.abs(vv) <= V_ROOF)
    n = max(int(roof.sum()), 1)
    out["roof"] = {
        "moss": float((lay.plan["moss"][roof] > lay.thresholds["moss"]).sum() / n),
        "algae": float((lay.plan["moss"][roof] > lay.thresholds["algae"]).sum() / n),
    }
    return out


def write_maps(
    folder: Path, wear: Wear, lay: Layers, maps: dict[str, np.ndarray]
) -> dict:
    """Write the maps as PNGs into `folder`; returns the manifest record."""
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, image in maps.items():
        path = folder / f"wear_{name}.png"
        M.write_png(path, image)
        files[name] = {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": [int(image.shape[1]), int(image.shape[0])],
            "channels": list(MAPS.get(name, ("u",))),
        }
    _, lut = girth_lut()
    return {
        "level": wear.name,
        "version": WEAR_VERSION,
        "params": asdict(wear),
        "frames": {k: f.record() for k, f in lay.frames.items()},
        "tiles": {k: asdict(t) for k, t in TILES.items()},
        "girth": lut,
        "thresholds": {"potential": wear.threshold, **lay.thresholds},
        "maps": files,
        "coverage": coverage(wear, lay),
    }
