# SPDX-License-Identifier: MIT
"""Blender side of the weathered NB57 'Wren'. bpy is passed in, never imported.

This module loads the wear maps from narrowboat_wear and samples them through the boat's
object coordinates. It also provides the aged materials, a false-colour view of the masks
for placement reviews, and the close-up cameras.

A material reads the maps through three projections:
- the side map over (girth u, z), on faces that look out of their own side of the boat;
- the plan map over (b, roof arc v), on faces that look up;
- the ends atlas, on the bulkheads and the semi-trad shelter plates' inner faces.

Object coordinates are boat coordinates because every part hangs off the "Boat" empty with
an identity transform, so the masks follow the boat when it is trimmed on the lake.
Potentials are thresholded here, so colour, roughness and bump share one mask.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import narrowboat_common as N
import narrowboat_design as D
import narrowboat_lines as L
import narrowboat_wear as W
from procgen_bpy import Graph

CHALK = (0.26, 0.27, 0.25)  # what weathered enamel chalks towards
PAINT_USES = frozenset({"rust", "stain", "chip", "dirt", "touch", "verd"})
# The masks each slot's weathered material reads (the false-colour view shows only these).
USES = {
    "HullBlack": frozenset({"rust", "stain", "chip", "dirt", "polish", "pool"}),
    "CabinGreen": PAINT_USES,
    "CoachCream": PAINT_USES,
    "CoachRed": PAINT_USES,
    "RoofOxide": frozenset({"rust", "moss", "polish", "soot", "pool", "verd"}),
    "DeckNonSlip": frozenset({"rust", "moss", "polish", "pool", "verd"}),
    "Brass": frozenset({"verd", "polish"}),
    "FrameGold": frozenset({"verd"}),
    "CastIron": frozenset({"soot"}),
    "TillerWood": frozenset({"polish"}),
    "Rope": frozenset(),
}
DECAL_USES = frozenset({"rust", "stain", "dirt"})
GRIME = (0.012, 0.014, 0.010)
DIRT = (0.075, 0.08, 0.07)  # the pale grey of dirt runs and rain-wash on dark paint
Z_TOP = L.Z_SIDE_TOP


@dataclass
class WearCtx:
    """The wear maps loaded in Blender, the record they were written with and the level."""

    images: dict
    record: dict
    wear: W.Wear

    def frame(self, name: str) -> dict:
        return self.record["frames"][name]


def load_maps(bpy, folder: Path, record: dict, wear: W.Wear) -> WearCtx:
    images = {}
    for name, info in record["maps"].items():
        image = bpy.data.images.load(str(folder / info["file"]), check_existing=False)
        image.name = info["file"]
        image.colorspace_settings.name = "Non-Color"
        images[name] = image
    return WearCtx(images, record, wear)


def _affine(lo: float, hi: float) -> tuple[float, float]:
    """Scale and offset taking lo..hi to 0..1."""
    return 1.0 / (hi - lo), -lo / (hi - lo)


class Wearing:
    """The wear masks of one material graph: the three projections, their weights, and the
    maps' channels sampled lazily and combined."""

    def __init__(self, g: Graph, ctx: WearCtx):
        self.g, self.ctx = g, ctx
        self.x, self.y, self.z = g.separate(g.coords.outputs["Object"])
        self.nx, self.ny, self.nz = g.separate(g.coords.outputs["Normal"])
        self.port = g.math("GREATER_THAN", self.y, 0.0)
        self.sy = g.madd(self.port, 2.0, -1.0)  # +1 on the port half, -1 to starboard
        self.ay = g.math("ABSOLUTE", self.y)
        self.zmm = g.math("MULTIPLY", self.z, 1000.0)
        self.bmm = g.madd(self.x, 1000.0, 1000.0 * N.X_MID)
        self._samples: dict = {}
        self._uv: dict = {}

    # ------------------------------------------------------------------ projections

    def side_weight(self):
        g = self.g
        outward = g.math("MULTIPLY", self.ny, self.sy)
        return g.map_range(outward, 0.35, 0.55)

    def plan_weight(self):
        """Up-facing faces at roof height (the plan map's roof) or deck height (its decks
        and gunwales), so no other up-facing face picks the plan's features up."""
        g = self.g
        level = g.math(
            "MAXIMUM",
            g.math("GREATER_THAN", self.zmm, W.SIDE_TOP - 5),
            g.math("LESS_THAN", self.zmm, 1400.0),
        )
        return g.math("MULTIPLY", g.map_range(self.nz, 0.6, 0.8), level)

    def side_uv(self):
        if "side" in self._uv:
            return self._uv["side"]
        g, ctx = self.g, self.ctx
        info = ctx.record["girth"]
        k, c = _affine(info["x0"], info["x1"])
        lut, _, _ = g.image_rgb(
            ctx.images["Girth"],
            g.combine(g.madd(self.x, k, c), 0.5),
            interpolation="Linear",
            extension="EXTEND",
        )
        u = g.madd(lut, info["u1"] - info["u0"], info["u0"])
        # Aft of amidships the counter's end plate unrolls from the centreline; elsewhere
        # the girth includes the plate's full half width.
        aft = g.math("LESS_THAN", self.x, 0.0)
        plate = g.math(
            "SUBTRACT", g.math("MINIMUM", self.ay, info["w_end"]), info["w_end"]
        )
        u = g.math("ADD", u, g.madd(aft, plate, info["w_end"]))
        f = ctx.frame("side")
        ku, cu = _affine(f["u0"], f["u1"])
        kv, cv = _affine(f["v0"], f["v1"])
        s = g.madd(g.math("MULTIPLY", u, 1000.0), ku, cu)
        band = g.math(
            "MINIMUM", g.math("MAXIMUM", g.madd(self.zmm, kv, cv), 0.001), 0.999
        )
        t = g.madd(band, 0.5, g.math("MULTIPLY", self.port, 0.5))
        self._uv["side"] = g.combine(s, t)
        return self._uv["side"]

    def plan_uv(self):
        if "plan" in self._uv:
            return self._uv["plan"]
        g, ctx = self.g, self.ctx
        aymm = g.math("MULTIPLY", self.ay, 1000.0)
        theta = g.math(
            "ARCTAN2",
            g.math("MAXIMUM", g.math("SUBTRACT", self.zmm, W.SIDE_TOP), 0.0),
            g.math("MAXIMUM", g.math("SUBTRACT", aymm, W.ROOF_YC), 1e-3),
        )
        corner = g.madd(theta, -W.CORNER_R, W.ROOF_YC + W.CORNER_R * math.pi / 2)
        use = g.math(
            "MULTIPLY",
            g.math("GREATER_THAN", aymm, W.ROOF_YC),
            g.math("GREATER_THAN", self.zmm, W.SIDE_TOP - 5),
        )
        v = g.madd(use, g.math("SUBTRACT", corner, aymm), aymm)
        v = g.math("MULTIPLY", v, self.sy)
        f = ctx.frame("plan")
        ku, cu = _affine(f["u0"], f["u1"])
        kv, cv = _affine(f["v0"], f["v1"])
        self._uv["plan"] = g.combine(g.madd(self.bmm, ku, cu), g.madd(v, kv, cv))
        return self._uv["plan"]

    def ends_uv(self):
        """(uv, weight) of the ends atlas: the bulkheads seen from outside, and the shelter
        plates' faces that look in towards the centreline."""
        if "ends" in self._uv:
            return self._uv["ends"]
        g, ctx = self.g, self.ctx
        ymm = g.math("MULTIPLY", self.y, 1000.0)
        rear = g.math(
            "MULTIPLY",
            g.math("LESS_THAN", self.nx, -0.5),
            g.math("LESS_THAN", self.x, N.wx(0) + 0.06),
        )
        front = g.math(
            "MULTIPLY",
            g.math("GREATER_THAN", self.nx, 0.5),
            g.math("GREATER_THAN", self.x, N.wx(D.CABIN_LEN) - 0.06),
        )
        inward = g.math("LESS_THAN", g.math("MULTIPLY", self.ny, self.sy), -0.35)
        shelter = g.math(
            "MULTIPLY",
            g.math("MULTIPLY", inward, g.math("LESS_THAN", self.x, N.wx(0) + 0.01)),
            g.math("GREATER_THAN", self.zmm, W.GUNWALE - 20),
        )
        sp = g.math("MULTIPLY", shelter, self.port)
        ss = g.math("SUBTRACT", shelter, sp)
        tiles = W.TILES
        a = g.math(
            "MULTIPLY",
            rear,
            g.math("ADD", ymm, tiles["Rear"].offset - tiles["Rear"].lo),
        )
        a = g.math(
            "ADD",
            a,
            g.math(
                "MULTIPLY",
                front,
                g.math("ADD", ymm, tiles["Front"].offset - tiles["Front"].lo),
            ),
        )
        for sel, name in ((sp, "ShelterPort"), (ss, "ShelterStarboard")):
            t = tiles[name]
            a = g.math(
                "ADD",
                a,
                g.math("MULTIPLY", sel, g.math("ADD", self.bmm, t.offset - t.lo)),
            )
        f = ctx.frame("ends")
        ku, cu = _affine(f["u0"], f["u1"])
        kv, cv = _affine(f["v0"], f["v1"])
        uv = g.combine(g.madd(a, ku, cu), g.madd(self.zmm, kv, cv))
        weight = g.math(
            "MINIMUM", g.math("ADD", g.math("ADD", rear, front), shelter), 1.0
        )
        self._uv["ends"] = (uv, weight)
        return self._uv["ends"]

    # ------------------------------------------------------------------ channels

    def sample(self, name: str):
        """(r, g, b) of one map at this shading point's projection."""
        if name not in self._samples:
            if name.startswith("Side"):
                uv = self.side_uv()
            elif name.startswith("Plan"):
                uv = self.plan_uv()
            else:
                uv = self.ends_uv()[0]
            self._samples[name] = self.g.image_rgb(
                self.ctx.images[name], uv, interpolation="Cubic"
            )
        return self._samples[name]

    def masks(
        self, gunwale_only: bool = False, planes=("side", "ends", "plan")
    ) -> dict:
        """Every channel, weighted by its projection and combined across the maps, from
        the projections in `planes` only (unused maps then cost nothing). With
        `gunwale_only`, the plan map only reaches up-facing faces at gunwale height or
        above (the hull's own ledges, such as the strake tops, are not decks)."""
        key = ("masks", gunwale_only, tuple(planes))
        if key in self._samples:
            return self._samples[key]
        g = self.g
        pairs: dict[str, list] = {}

        def add(channel, weight, value):
            pairs.setdefault(channel, []).append((weight, value))

        if "side" in planes:
            ws = self.side_weight()
            sa, sb = self.sample("SideA"), self.sample("SideB")
            for ch, v in zip(
                ("rust", "stain", "chip", "dirt", "touch", "verd"), (*sa, *sb)
            ):
                add(ch, ws, v)
        if "ends" in planes:
            we = self.ends_uv()[1]
            ea, eb = self.sample("EndsA"), self.sample("EndsB")
            for ch, v in zip(
                ("rust", "stain", "chip", "dirt", "touch", "verd"), (*ea, *eb)
            ):
                add(ch, we, v)
        if "plan" in planes:
            wp = self.plan_weight()
            if gunwale_only:
                wp = g.math(
                    "MULTIPLY", wp, g.math("GREATER_THAN", self.zmm, W.GUNWALE - 15)
                )
            pa, pb = self.sample("PlanA"), self.sample("PlanB")
            for ch, v in zip(
                ("rust", "moss", "polish", "soot", "pool", "verd"), (*pa, *pb)
            ):
                add(ch, wp, v)
        masks = {}
        for ch in (
            "rust",
            "stain",
            "chip",
            "dirt",
            "touch",
            "verd",
            "moss",
            "polish",
            "soot",
            "pool",
        ):
            out = None
            for w, v in pairs.get(ch, ()):
                m = g.math("MULTIPLY", w, v)
                out = m if out is None else g.math("MAXIMUM", out, m)
            masks[ch] = out if out is not None else 0.0
        self._samples[key] = masks
        return masks

    def dirt(self):
        """Just the dirt runs (for base ageing, which needs nothing else from the maps)."""
        g = self.g
        sb, eb = self.sample("SideB"), self.sample("EndsB")
        return g.math(
            "MAXIMUM",
            g.math("MULTIPLY", self.side_weight(), sb[0]),
            g.math("MULTIPLY", self.ends_uv()[1], eb[0]),
        )


# --------------------------------------------------------------------------- helpers


def distance_fade(g: Graph, near: float = 5.0, far: float = 12.0):
    """1 close to the camera, 0 far away: fine noise fades out where the GPU, which has no
    mipmaps, would only make it sparkle."""
    dist = g.node("ShaderNodeCameraData").outputs["View Distance"]
    return g.map_range(dist, near, far, 1.0, 0.0)


def chalk_amount(g: Graph, wear: W.Wear, z, obj, scale: float = 1.0, dirt=None):
    """How far paint has chalked: more in the top 300 mm under the roof edge, unevenly
    over metres, and washed paler along the dirt runs from the roof's drip points."""
    lo, hi = wear.chalk
    top = g.map_range(z, Z_TOP - 0.3, Z_TOP, 0.0, 1.0)
    k = g.madd(top, hi - lo, lo)
    k = g.math("ADD", k, g.madd(g.noise(obj, 0.45, detail=2.0), 0.05, -0.025))
    if dirt is not None:
        k = g.madd(dirt, 0.1, k)
    return g.math("MAXIMUM", g.math("MULTIPLY", k, scale), 0.0)


def grime(g: Graph, distance: float = 0.035):
    """Dirt in the corners and joints, from ambient occlusion."""
    return g.map_range(g.occlusion(distance), 0.55, 1.0, 0.5, 0.0)


def waterline_height(g: Graph, x, z):
    """Height (m) above the trimmed design waterline, which is linear along the boat:
    wl(b) = wl(0) + slope * b with b = x + X_MID."""
    slope = (L.wl_z(L.STEM) - L.wl_z(L.STERN_POST)) / (L.STEM - L.STERN_POST)
    return g.math("SUBTRACT", z, g.madd(x, slope, L.wl_z(0.0) + slope * N.X_MID))


def peel(g: Graph, obj, strength: float = 0.02):
    fade = distance_fade(g)
    return g.bump(
        g.noise(obj, 260.0, detail=3.0), g.math("MULTIPLY", fade, strength), 0.001
    )


def _vec(g: Graph, op: str, a, b=None, scale=None):
    n = g.node("ShaderNodeVectorMath", operation=op)
    g.set(n.inputs[0], a)
    if b is not None:
        g.set(n.inputs[1], b)
    if scale is not None:
        g.set(n.inputs[3], scale)
    return n


def handled_capsules() -> list[tuple[tuple, tuple, float]]:
    """Where hands keep the brass and timber bright: (start, end, radius) capsules in boat
    metres round the swan neck's crown, the tiller's grip, the T-stud heads, the chimney
    bands and the filler caps."""
    caps = []
    neck = L.swan_neck()
    zmax = max(z for _, z in neck)
    crown = [(N.wx(1000 * b), 0.0, z) for b, z in neck if z > zmax - 0.07]
    caps.append((crown[0], crown[-1], 0.045))
    (b0, z0), (b1, z1) = L.tiller_bar()
    t = 0.25 / math.dist((b0, z0), (b1, z1))
    grip0 = (N.wx(1000 * (b1 + (b0 - b1) * t)), 0.0, z1 + (z0 - z1) * t)
    caps.append((grip0, (N.wx(1000 * b1), 0.0, z1), 0.04))
    for b, y in D.T_STUDS:
        z = L.gunwale_z(L.m(b)) if b > 0 else N.DECK_Z
        x, ym = N.wx(b), L.m(y)
        caps.append(((x, ym - 0.08, z + 0.133), (x, ym + 0.08, z + 0.133), 0.035))
    x, y = N.wx(D.CHIMNEY[0]), N.CHIMNEY_Y
    z0 = L.roof_z(y) - 0.01
    caps.append(((x, y, z0 + 0.15), (x, y, z0 + 0.375), L.m(D.CHIMNEY_D) / 2 + 0.03))
    for (b, y), z in (
        (D.DIESEL_FILLER, N.DECK_Z),
        (D.WATER_FILLER, L.m(D.WELL_SOLE_Z)),
    ):
        p = (N.wx(b), L.m(y), z)
        caps.append((p, (p[0], p[1], z + 0.03), 0.06))
    return caps


def handled(g: Graph):
    """1 inside the handled capsules, fading out at their rims."""
    p = g.coords.outputs["Object"]
    out = None
    for a, b, r in handled_capsules():
        ab = tuple(bb - aa for aa, bb in zip(a, b))
        l2 = max(sum(c * c for c in ab), 1e-9)
        ap = _vec(g, "SUBTRACT", p, a).outputs["Vector"]
        t = g.math(
            "MULTIPLY", _vec(g, "DOT_PRODUCT", ap, ab).outputs["Value"], 1.0 / l2
        )
        t = g.math("MINIMUM", g.math("MAXIMUM", t, 0.0), 1.0)
        near = _vec(g, "ADD", _vec(g, "SCALE", ab, scale=t).outputs["Vector"], a)
        d = _vec(g, "DISTANCE", p, near.outputs["Vector"]).outputs["Value"]
        m = g.map_range(d, r, 0.6 * r)
        out = m if out is None else g.math("MAXIMUM", out, m)
    return out


# --------------------------------------------------------------------------- damage

RUST_FLAT = (0.06, 0.027, 0.011)  # scale on vertical plate: dark brown
RUST_EDGE = (
    0.20,
    0.075,
    0.02,
)  # its fresh edge and faces that look up, where it is dry
RUST_PIT = (0.028, 0.015, 0.008)  # deep in the pits
PRIMER = (0.11, 0.05, 0.03)  # red-oxide primer under the topcoat
STAIN_CORE = (0.16, 0.07, 0.025)  # lighter than the dark green: it reads at a distance
STAIN_HALO = (0.13, 0.08, 0.035)
VERDIGRIS = (0.05, 0.20, 0.16)  # the crust on brass: a saturated blue-green
VERD_STAIN = (0.02, 0.12, 0.10)  # washed down paint: teal, a little lighter than it
EPS = 0.025  # half width of a threshold's ramp, in potential


def edge_noise(g: Graph, obj, fray: bool = True):
    """Noise that breaks up threshold edges: a ragged outline at about 15 mm, and a
    fine 3 mm fray close to the camera only (no mipmaps on the GPU); without `fray` for
    edges that should stay soft."""
    coarse = g.math("SUBTRACT", g.noise(obj, 60.0, detail=3.0), 0.5)
    if not fray:
        return g.math("MULTIPLY", coarse, 0.4)
    fine = g.math(
        "MULTIPLY",
        g.math("SUBTRACT", g.noise(obj, 300.0, detail=2.0), 0.5),
        g.math("MULTIPLY", distance_fade(g), 0.25),
    )
    return g.madd(coarse, 0.4, fine)


def above(g: Graph, potential, level: float, jitter):
    """The mask where a potential, its edge broken by `jitter`, passes `level`."""
    p = g.math("ADD", potential, jitter)
    return g.map_range(p, level - EPS, level + EPS)


def rust_colour(g: Graph, obj, depth=None):
    """Rust in three tones: orange at its edges and where a face looks up, mid brown on
    vertical plate, dark umber in the pits where it is deepest; with a coarse grain."""
    _, _, nz = g.separate(g.coords.outputs["Normal"])
    up = g.map_range(nz, 0.2, 0.7)
    grain = g.noise(obj, 180.0, detail=4.0)
    col = g.mix(g.math("ADD", up, g.madd(grain, 0.6, -0.3)), RUST_FLAT, RUST_EDGE)
    if depth is not None:
        edge = g.map_range(depth, 0.06, 0.0)
        col = g.mix(g.math("MULTIPLY", edge, 0.7), col, RUST_EDGE)
        pits = g.math(
            "MULTIPLY",
            g.map_range(depth, 0.1, 0.3),
            g.map_range(g.noise(obj, 90.0, detail=3.0), 0.45, 0.65),
        )
        col = g.mix(pits, col, RUST_PIT)
    return col, grain


@dataclass
class Damage:
    """Colour, roughness and height sockets after the damage layers."""

    colour: object
    rough: object
    height: object


def paint_damage(
    g: Graph,
    m: dict,
    t: float,
    col,
    rough,
    under=PRIMER,
    chips: bool = True,
    touch_colour=None,
    stain_k: float = 1.0,
) -> Damage:
    """The damage chain on paint, in the order it happens: touch-ups over old damage,
    chips through the topcoat to `under` and deeper to rust, the stains rust and
    verdigris send down the paint, and on top the rust itself, its dark scale ringed
    with blistered paint; one height for bump. Colour, roughness and height all come
    from the same thresholded masks. `stain_k` weakens stains on pale paint, where the
    same deposit shows far more than it does on dark green."""
    obj = g.coords.outputs["Object"]
    jit = edge_noise(g, obj)
    height = 0.0
    if touch_colour is not None:
        # Fresh paint, a shade off and glossier; brushed on thin, so no ridge.
        touch = g.map_range(m["touch"], 0.03, 0.07)
        col = g.mix(touch, col, touch_colour)
        rough = g.madd(touch, g.math("SUBTRACT", 0.32, rough), rough)
    rdepth = g.math("SUBTRACT", g.math("ADD", m["rust"], jit), t)
    rcol, grain = rust_colour(g, obj, rdepth)
    if chips:
        # Through the topcoat to the primer, and in the middle down to rust.
        chip = above(g, m["chip"], t, jit)
        deep = above(g, m["chip"], t + 0.06, jit)
        ccol = g.mix(deep, under, rcol)
        col = g.mix(chip, col, ccol)
        rough = g.madd(chip, g.math("SUBTRACT", g.madd(deep, 0.3, 0.6), rough), rough)
        height = g.math("SUBTRACT", height, g.math("MULTIPLY", chip, 0.5))
    stain = m["stain"]
    core = g.map_range(stain, 0.4, 0.7)
    scol = g.mix(core, STAIN_HALO, STAIN_CORE)
    smix = g.math("MINIMUM", g.math("MULTIPLY", stain, 1.1 * stain_k), 0.85 * stain_k)
    col = g.mix(smix, col, scol)
    rough = g.madd(stain, 0.1, rough)
    # Verdigris dries as a mottled, granular deposit, not an even wash.
    streak = g.noise(g.mapping(obj, (60.0, 60.0, 15.0)), 1.0, detail=3.0)
    grains = g.noise(obj, 260.0, detail=2.0)
    verd = g.math("MULTIPLY", m["verd"], g.map_range(streak, 0.3, 0.7, 0.15, 1.2))
    verd = g.math("MULTIPLY", verd, g.map_range(grains, 0.35, 0.65, 0.5, 1.2))
    col = g.mix(g.math("MINIMUM", g.math("MULTIPLY", verd, 0.7), 0.75), col, VERD_STAIN)
    rough = g.madd(verd, 0.25, rough)
    # The rust last, over its own stain: paint lifted by the rust swelling under it (a
    # raised ring, a little darker where dirt catches under its lip, here and there
    # rather than all the way round), then the scale.
    rust = above(g, m["rust"], t, jit)
    ring = g.math("SUBTRACT", above(g, m["rust"], t - 0.1, jit), rust)
    ring = g.math(
        "MULTIPLY", ring, g.map_range(g.noise(obj, 40.0, detail=2.0), 0.45, 0.6)
    )
    lifted = g.node("ShaderNodeHueSaturation", {"Value": 0.9, "Color": col}).outputs[
        "Color"
    ]
    col = g.mix(g.math("MULTIPLY", ring, 0.5), col, lifted)
    col = g.mix(rust, col, rcol)
    rough = g.madd(rust, g.math("SUBTRACT", 0.9, rough), rough)
    rough = g.madd(ring, 0.15, rough)
    height = g.math("ADD", height, g.math("MULTIPLY", ring, 0.45))
    height = g.math("ADD", height, g.math("MULTIPLY", rust, g.madd(grain, 0.6, 0.2)))
    return Damage(col, g.math("MINIMUM", rough, 0.95), height)


def finish_damaged(g: Graph, obj, d: Damage, strength: float = 0.5):
    """Close the graph: bump from the damage height over the paint's orange peel."""
    fine = peel(g, obj)
    normal = g.bump(d.height, strength, 0.0015, fine)
    return g.finish(d.colour, d.rough, normal)


# --------------------------------------------------------------------------- aged materials


def roof_fittings(g: Graph, wearing: Wearing, m: dict, z) -> dict:
    """Masks for paint above the cabin sides (the handrails, the rear hatch): the side and
    ends maps stop at the side top, so above it they give way to rust from the plan map
    at the handrail feet, where water sits round the leg, and a few chips along the rail
    where ropes and the pole knock it."""
    roof = g.map_range(z, Z_TOP + 0.003, Z_TOP + 0.008)
    keep = g.math("SUBTRACT", 1.0, roof)
    out = {
        ch: (v if isinstance(v, float) else g.math("MULTIPLY", v, keep))
        for ch, v in m.items()
    }
    rail_y = L.handrail_y()
    near = g.map_range(
        g.math("ABSOLUTE", g.math("SUBTRACT", wearing.ay, rail_y)), 0.03, 0.015
    )
    foot = g.map_range(z, L.roof_z(rail_y) + 0.035, L.roof_z(rail_y) + 0.015)
    legs = g.math(
        "MULTIPLY",
        wearing.sample("PlanA")[0],
        g.math("MULTIPLY", roof, g.math("MULTIPLY", near, foot)),
    )
    out["rust"] = (
        legs if isinstance(m["rust"], float) else g.math("MAXIMUM", out["rust"], legs)
    )
    obj = g.coords.outputs["Object"]
    knocks = g.map_range(g.noise(obj, 9.0, detail=3.0), 0.66, 0.74, 0.0, 0.75)
    knocks = g.math("MULTIPLY", knocks, g.math("MULTIPLY", roof, near))
    out["chip"] = (
        knocks
        if isinstance(m["chip"], float)
        else g.math("MAXIMUM", out["chip"], knocks)
    )
    return out


def aged_paint(
    bpy,
    name: str,
    colour,
    ctx: WearCtx,
    rough: float = 0.30,
    chalk: float = 1.0,
    under=PRIMER,
    touch: bool = True,
    stain_k: float = 1.0,
):
    """Owner-maintained enamel some years on: chalked lighter and flatter, unevenly and
    more under the roof edge, washed along the dirt runs from the roof's drip points, dirt
    in the joints; then the damage chain (see paint_damage). Touch-ups are the same
    colour a shade off, freshly painted so barely chalked. `under` is what a chip shows
    first: grey primer, or the green a coachline is painted over."""
    g = Graph(bpy, name, colour)
    obj = g.coords.outputs["Object"]
    _, _, z = g.separate(obj)
    wearing = Wearing(g, ctx)
    m = roof_fittings(g, wearing, wearing.masks(planes=("side", "ends")), z)
    k = chalk_amount(g, ctx.wear, z, obj, chalk, dirt=m["dirt"])
    col = g.mix(k, colour, CHALK)
    col = g.mix(g.math("MULTIPLY", m["dirt"], 0.85), col, DIRT)
    dirt = grime(g)
    col = g.mix(dirt, col, GRIME)
    r = g.math("ADD", g.madd(k, 1.8, rough), g.math("MULTIPLY", dirt, 0.15))
    r = g.madd(m["dirt"], 0.08, r)
    # Each touch-up's plateau value (0.3-1) sets how far off the original it came out.
    shade = g.math("MAXIMUM", m["touch"], 0.3)
    fresh = g.node(
        "ShaderNodeHueSaturation",
        {
            "Hue": g.madd(shade, 0.03, 0.48),
            "Value": g.madd(shade, 0.26, 0.83),
            "Color": colour,
        },
    ).outputs["Color"]
    fresh = g.mix(g.math("MULTIPLY", k, 0.5), fresh, CHALK)
    d = paint_damage(
        g,
        m,
        ctx.wear.threshold,
        col,
        r,
        under,
        touch_colour=fresh if touch else None,
        stain_k=stain_k,
    )
    return finish_damaged(g, obj, d)


def crazing(g: Graph, obj):
    """Varnish crazing over old canal art: fine cracks at two scales, running mostly one
    way, over about a fifth of it, faint, and only close to the camera."""
    flat = g.mapping(obj, (1.0, 1.0, 2.4))
    crack = None
    for scale, width in ((120.0, 0.05), (260.0, 0.04)):
        cells = g.node(
            "ShaderNodeTexVoronoi",
            {"Vector": flat, "Scale": scale, "Randomness": 1.0},
            feature="DISTANCE_TO_EDGE",
        ).outputs["Distance"]
        c = g.map_range(cells, width, 0.0)
        crack = (
            c if crack is None else g.math("MAXIMUM", crack, g.math("MULTIPLY", c, 0.7))
        )
    patch = g.map_range(g.noise(obj, 6.0, detail=2.0), 0.6, 0.66)
    return g.math(
        "MULTIPLY", g.math("MULTIPLY", crack, patch), distance_fade(g, 2.0, 5.0)
    )


def aged_decal(bpy, name: str, design: str, ctx: WearCtx):
    """Canal art under yellowed varnish, a little chalked, on the same enamel: the same
    dirt runs, stains and rust cross it with the same relief; no chips (the art keeps
    clear of the leaf edges) and no touch-ups (none are painted over the art). The door
    and hatch art, oldest and most handled, has crazed varnish."""
    g = Graph(bpy, name, (0.6, 0.25, 0.15))
    obj = g.coords.outputs["Object"]
    _, _, z = g.separate(obj)
    tex = g.node(
        "ShaderNodeTexImage",
        {"Vector": g.coords.outputs["UV"]},
        image=bpy.data.images[N.texture_name(design)],
        extension="CLIP",
        interpolation="Linear",
    )
    art = g.node(
        "ShaderNodeHueSaturation", {"Saturation": 0.9, "Color": tex.outputs["Color"]}
    ).outputs["Color"]
    art = g.mix(1.0, art, (1.0, 0.94, 0.78), blend="MULTIPLY")
    m = Wearing(g, ctx).masks(planes=("side", "ends"))
    k = chalk_amount(g, ctx.wear, z, obj, 0.35, dirt=m["dirt"])
    col = g.mix(k, art, CHALK)
    col = g.mix(g.math("MULTIPLY", m["dirt"], 0.55), col, DIRT)
    r = g.madd(m["dirt"], 0.08, 0.42)
    # Every panel's varnish has yellowed a little, most low down where the sun and
    # splashes reach.
    low = g.map_range(z, 1.9, 1.1)
    col = g.mix(g.madd(low, 0.1, 0.12), col, (0.62, 0.42, 0.18), blend="MULTIPLY")
    if "Door" in design or "Hatch" in design:
        # Older, more handled art: the varnish has gone further amber, and crazed.
        col = g.mix(g.madd(low, 0.25, 0.05), col, (0.62, 0.42, 0.18), blend="MULTIPLY")
        crack = crazing(g, obj)
        col = g.mix(g.math("MULTIPLY", crack, 0.3), col, (0.03, 0.02, 0.01))
        r = g.madd(crack, 0.15, r)
    # The door and hatch art is handled and kicked: it chips with the paint under it.
    worn = "Door" in design or "Hatch" in design
    d = paint_damage(g, m, ctx.wear.threshold, col, r, chips=worn)
    g.set(g.bsdf.inputs["Alpha"], tex.outputs["Alpha"])
    return finish_damaged(g, obj, d)


def aged_hull(bpy, name: str, ctx: WearCtx):
    """Blacking gone matt with a warm cast: drier and lighter towards the gunwale, darker
    near the water, dusty on the ledges. A dull grey-brown band above the waterline,
    higher at the bow wave and the propeller wash, its top wandering and soft, with faint
    tide lines; a wet slime lip at the water that climbs as dry streaks in places, and
    dull weed below."""
    blacking = (0.019, 0.017, 0.014)
    g = Graph(bpy, name, blacking)
    obj = g.coords.outputs["Object"]
    x, _, z = g.separate(obj)
    _, _, nz = g.separate(g.coords.outputs["Normal"])
    up = g.map_range(nz, 0.6, 0.9)
    tone = g.map_range(g.noise(obj, 0.9, detail=3.0), 0.3, 0.7, 0.88, 1.12)
    col = g.node("ShaderNodeHueSaturation", {"Value": tone, "Color": blacking}).outputs[
        "Color"
    ]
    dry = g.map_range(z, L.Z_GUNWALE - 0.35, L.Z_GUNWALE - 0.02)
    col = g.mix(g.math("MULTIPLY", dry, 0.9), col, (0.05, 0.045, 0.036))
    col = g.mix(up, col, (0.07, 0.064, 0.052))
    rough = g.madd(up, 0.08, g.madd(tone, 0.25, 0.4))
    # Height above the trimmed design waterline. The band starts at the water; only its
    # top wanders, over metres (30-110 mm with the bow and stern lift) and frayed by the
    # wash into lobes a few centimetres across; no spikes.
    h0 = waterline_height(g, x, z)
    h = g.math("SUBTRACT", h0, g.madd(g.noise(obj, 0.6, detail=2.0), 0.05, -0.025))
    h = g.math("SUBTRACT", h, g.madd(g.noise(obj, 3.0, detail=3.0), 0.016, -0.008))
    fray = g.noise(g.mapping(obj, (14.0, 14.0, 5.0)), 1.0, detail=3.0)
    h = g.math("SUBTRACT", h, g.madd(fray, 0.014, -0.007))
    b = g.math("ADD", x, N.X_MID)
    bow = g.map_range(b, L.STEM - 3.5, L.STEM - 1.0)
    stern = g.map_range(b, 1.0, -1.0)
    top = g.madd(bow, 0.06, g.madd(stern, 0.04, 0.12))
    band = g.math(
        "MULTIPLY",
        g.map_range(g.math("SUBTRACT", h, top), -0.014, 0.004, 1.0, 0.0),
        g.map_range(h0, -0.004, 0.004),
    )
    # The band is uneven along its length and patchy within it.
    band = g.math(
        "MULTIPLY",
        band,
        g.math(
            "MULTIPLY",
            g.map_range(g.noise(obj, 1.5, detail=2.0), 0.3, 0.7, 0.7, 1.0),
            g.map_range(g.noise(obj, 9.0, detail=3.0), 0.3, 0.7, 0.75, 1.0),
        ),
    )
    # Tide lines of dried scum where the level has stood, broken in places.
    tide = None
    for share in (0.3, 0.55, 0.8):
        line = g.map_range(
            g.math("ABSOLUTE", g.math("SUBTRACT", h, g.math("MULTIPLY", top, share))),
            0.0,
            0.0025,
            1.0,
            0.0,
        )
        tide = line if tide is None else g.math("MAXIMUM", tide, line)
    tide = g.math(
        "MULTIPLY", tide, g.map_range(g.noise(obj, 4.0, detail=2.0), 0.35, 0.5)
    )
    # Slime right at the water, densest there and gone within a few centimetres.
    lip = g.math(
        "MULTIPLY",
        g.map_range(h0, -0.065, -0.05),
        g.map_range(h0, 0.015, 0.03, 1.0, 0.0),
    )
    lip = g.math(
        "MULTIPLY", lip, g.map_range(g.noise(obj, 6.0, detail=4.0), 0.38, 0.52)
    )
    lip = g.math("MULTIPLY", lip, g.map_range(h0, 0.025, 0.0, 0.3, 1.0))
    # Slime climbing the band in places: streaks only in some stretches of the hull.
    streaks = g.math(
        "MULTIPLY",
        g.map_range(
            g.noise(g.mapping(obj, (45.0, 45.0, 1.5)), 1.0, detail=3.0), 0.7, 0.78
        ),
        g.map_range(g.noise(obj, 0.8, detail=2.0), 0.52, 0.6),
    )
    # Splashes dried above the band, reaching higher at the bow wave and the wash.
    climb = g.math(
        "MULTIPLY",
        g.map_range(h, 0.03, 0.04),
        g.map_range(h, 0.05, g.madd(bow, 0.06, g.madd(stern, 0.04, 0.11)), 1.0, 0.0),
    )
    dried = g.math("MULTIPLY", streaks, climb)
    weed = g.map_range(h0, -0.055, -0.075)
    # Green-brown algae low in the band, grading up to pale dried scum; about a quarter
    # lighter than the blacking, not a painted stripe.
    scum = g.map_range(h0, 0.0, 0.07)
    bcol = g.mix(scum, (0.035, 0.045, 0.02), (0.075, 0.07, 0.055))
    col = g.mix(band, col, bcol)
    col = g.mix(
        g.math("MULTIPLY", band, g.math("MULTIPLY", tide, 0.4)),
        col,
        (0.056, 0.053, 0.045),
    )
    col = g.mix(g.math("MULTIPLY", dried, 0.7), col, (0.046, 0.043, 0.036))
    col = g.mix(lip, col, (0.022, 0.03, 0.012))
    col = g.mix(weed, col, (0.02, 0.025, 0.014))
    rough = g.math("ADD", rough, g.math("MULTIPLY", band, 0.12))
    rough = g.madd(dried, 0.08, rough)
    rough = g.madd(lip, -0.3, rough)
    rough = g.math("MAXIMUM", g.madd(weed, 0.25, rough), 0.3)
    dirt = grime(g, 0.04)
    col = g.mix(dirt, col, (0.01, 0.009, 0.008))
    # The gunwale top (from the plan map): walked dark and smooth, stains pooled beside
    # the weld clusters, rust along its inner edge.
    m = Wearing(g, ctx).masks(gunwale_only=True, planes=("side", "plan"))
    polish = m["polish"]
    col = g.mix(g.math("MULTIPLY", polish, 0.7), col, (0.012, 0.011, 0.01))
    rough = g.madd(polish, g.math("SUBTRACT", 0.4, rough), rough)
    # Dirt runs show grey on black.
    col = g.mix(g.math("MULTIPLY", m["dirt"], 0.6), col, (0.075, 0.072, 0.064))
    t = ctx.wear.threshold
    jit = edge_noise(g, obj)
    # Rust stain pooled on the gunwale beside the weld clusters: a dull brown with a
    # tide-marked edge, not a glow.
    pool = g.map_range(
        g.math("ADD", m["pool"], g.math("MULTIPLY", jit, 0.5)), 0.2, 0.42
    )
    col = g.mix(g.math("MULTIPLY", pool, 0.75), col, (0.085, 0.036, 0.013))
    rcol, grain = rust_colour(
        g, obj, g.math("SUBTRACT", g.math("ADD", m["rust"], jit), t)
    )
    # Blacking worn thin round scrapes and where the fender rubs: paler and satin, with a
    # soft edge (no fray, which would speckle it).
    soft = edge_noise(g, obj, fray=False)
    worn = g.math(
        "SUBTRACT", above(g, m["chip"], t - 0.2, soft), above(g, m["chip"], t, soft)
    )
    col = g.mix(g.math("MULTIPLY", worn, 0.45), col, (0.045, 0.042, 0.037))
    rough = g.madd(worn, g.math("SUBTRACT", 0.45, rough), rough)
    # Scrapes and knocks through the blacking: old ones rusted dark brown, the newest
    # grey steel (dull, not mirror bright: it is scored).
    chip = above(g, m["chip"], t, jit)
    fresh = above(g, m["chip"], t + 0.3, jit)
    steel = g.mix(fresh, (0.085, 0.048, 0.026), (0.19, 0.185, 0.17))
    col = g.mix(chip, col, steel)
    rough = g.madd(chip, g.math("SUBTRACT", g.madd(fresh, 0.08, 0.6), rough), rough)
    metallic = g.math("MULTIPLY", chip, g.math("MULTIPLY", fresh, 0.4))
    rust = above(g, m["rust"], t, jit)
    col = g.mix(rust, col, rcol)
    rough = g.madd(rust, g.math("SUBTRACT", 0.9, rough), rough)
    metallic = g.madd(rust, g.math("MULTIPLY", metallic, -1.0), metallic)
    # Rust stain down the black: dark umber, low contrast, warmer where thickest.
    stain = m["stain"]
    scol = g.mix(
        g.map_range(stain, 0.4, 0.75), (0.05, 0.026, 0.012), (0.09, 0.04, 0.015)
    )
    col = g.mix(g.math("MINIMUM", g.math("MULTIPLY", stain, 1.0), 0.8), col, scol)
    height = g.math(
        "ADD",
        g.math("MULTIPLY", chip, -0.6),
        g.math("MULTIPLY", rust, g.madd(grain, 0.8, 0.2)),
    )
    normal = g.bump(
        height, 0.4, 0.0015, g.bump(g.noise(obj, 40.0, detail=6.0), 0.05, 0.002)
    )
    return g.finish(col, rough, normal, metallic=metallic)


MOSS = (0.045, 0.085, 0.018)  # cushion moss: dark olive, a little yellow in the sun
MOSS_TIP = (0.10, 0.14, 0.03)
ALGAE = (0.035, 0.05, 0.02)  # the green-black film round it
SOOT = (0.012, 0.011, 0.010)
POOL = (0.09, 0.045, 0.02)  # rust-brown where water stands and dries


def plan_wear(g: Graph, ctx: WearCtx, col, rough, height, wet_rough: float = 0.35):
    """The plan map's wear on an up-facing surface (roof or deck), over its aged colour:
    walking polish darker and smoother, standing-water stains, the algae film and moss
    cushions, soot from the chimney, verdigris washed from the brass, and rust at the
    handrail feet and the vents. Returns (colour, roughness, height)."""
    obj = g.coords.outputs["Object"]
    m = Wearing(g, ctx).masks(planes=("plan",))
    t = ctx.wear.threshold
    th = ctx.record["thresholds"]
    jit = edge_noise(g, obj)
    # Walked smooth: the grit worn down, so darker and less rough.
    polish = g.map_range(m["polish"], 0.05, 0.35)
    col = g.mix(
        g.math("MULTIPLY", polish, 0.45), col, (0.6, 0.6, 0.6), blend="MULTIPLY"
    )
    rough = g.madd(polish, -0.25, rough)
    height = g.math("SUBTRACT", height, g.math("MULTIPLY", polish, 0.2))
    # Standing water: a darker, rust-tinged tide mark that dries at its rim.
    pool = g.map_range(
        g.math("ADD", m["pool"], g.math("MULTIPLY", jit, 0.3)), 0.25, 0.6
    )
    col = g.mix(g.math("MULTIPLY", pool, 0.55), col, POOL)
    rough = g.madd(pool, -0.1, rough)
    # Verdigris washed out of the brass.
    col = g.mix(
        g.math("MINIMUM", g.math("MULTIPLY", m["verd"], 0.6), 0.6), col, VERD_STAIN
    )
    # Soot: a speckled grey-black that dulls the paint.
    soot = g.math("MINIMUM", g.math("MULTIPLY", m["soot"], 1.3), 0.8)
    col = g.mix(soot, col, SOOT)
    rough = g.madd(soot, 0.1, rough)
    # Algae: a thin film, wet-looking, below the moss threshold.
    algae = g.map_range(
        g.math("ADD", m["moss"], g.math("MULTIPLY", jit, 0.5)),
        th["algae"] - EPS,
        th["algae"] + EPS,
    )
    col = g.mix(g.math("MULTIPLY", algae, 0.7), col, ALGAE)
    rough = g.madd(algae, g.math("SUBTRACT", wet_rough, rough), rough)
    # Moss cushions: raised, fibrous, lighter where the tips catch the light.
    moss = above(g, m["moss"], th["moss"], jit)
    fibre = g.noise(obj, 400.0, detail=3.0)
    cushion = g.noise(obj, 50.0, detail=3.0)
    mcol = g.mix(g.map_range(g.math("ADD", fibre, cushion), 0.9, 1.3), MOSS, MOSS_TIP)
    col = g.mix(moss, col, mcol)
    rough = g.madd(moss, g.math("SUBTRACT", 0.95, rough), rough)
    height = g.math(
        "ADD",
        height,
        g.math("MULTIPLY", moss, g.madd(cushion, 0.8, g.madd(fibre, 0.4, 0.2))),
    )
    # Rust: at the handrail feet and round the vents, over its own pooled stain.
    rdepth = g.math("SUBTRACT", g.math("ADD", m["rust"], jit), t)
    rcol, grain = rust_colour(g, obj, rdepth)
    rust = above(g, m["rust"], t, jit)
    col = g.mix(rust, col, rcol)
    rough = g.madd(rust, g.math("SUBTRACT", 0.9, rough), rough)
    height = g.math("ADD", height, g.math("MULTIPLY", rust, g.madd(grain, 0.5, 0.15)))
    return col, g.math("MINIMUM", g.math("MAXIMUM", rough, 0.2), 0.97), height


def aged_roof(
    bpy, name: str, colour, fade_to, k_centre: float, k_edge: float, ctx=None
):
    """Non-slip paint faded by the sky: most on the outer edges and the corner radius,
    more where it faces straight up, less in sheltered corners, unevenly; then the plan
    map's wear (see plan_wear)."""
    g = Graph(bpy, name, colour)
    obj = g.coords.outputs["Object"]
    _, y, _ = g.separate(obj)
    _, _, nz = g.separate(g.coords.outputs["Normal"])
    grit = g.noise(obj, 700.0, detail=2.0)
    col = g.mix(
        g.math("MULTIPLY", g.map_range(grit, 0.4, 0.7), 0.25),
        colour,
        tuple(c * 1.5 for c in colour),
    )
    edge = g.map_range(g.math("ABSOLUTE", y), 0.62, 0.86)
    k = g.madd(edge, k_edge - k_centre, k_centre)
    k = g.math("MULTIPLY", k, g.madd(g.math("MAXIMUM", nz, 0.0), 0.35, 0.65))
    shelter = grime(g, 0.05)
    k = g.math("MULTIPLY", k, g.madd(shelter, -1.0, 1.0))
    k = g.math("ADD", k, g.madd(g.noise(obj, 0.8, detail=3.0), 0.08, -0.04))
    col = g.mix(k, col, fade_to)
    col = g.mix(shelter, col, tuple(c * 0.6 for c in colour))
    rough = g.madd(k, 0.15, 0.82)
    if ctx is None:
        return g.finish(col, rough, g.bump(grit, 0.35, 0.0015))
    col, rough, height = plan_wear(g, ctx, col, rough, 0.0)
    fine = g.bump(grit, 0.35, 0.0015)
    return g.finish(col, rough, g.bump(height, 0.6, 0.002, fine))


def aged_non_slip(bpy, name: str, colour, ctx=None):
    """Black non-slip deck paint gone dusty and uneven, chalked grey where the sun gets
    at it; then the plan map's wear (see plan_wear): the steering spot and thresholds
    walked smooth, wet low points, a little rust."""
    g = Graph(bpy, name, colour)
    obj = g.coords.outputs["Object"]
    grit = g.noise(obj, 700.0, detail=2.0)
    tone = g.map_range(g.noise(obj, 1.2, detail=3.0), 0.3, 0.7, 0.85, 1.15)
    base = g.node("ShaderNodeHueSaturation", {"Value": tone, "Color": colour}).outputs[
        "Color"
    ]
    col = g.mix(
        g.math("MULTIPLY", g.map_range(grit, 0.4, 0.7), 0.25),
        base,
        tuple(c * 1.5 for c in colour),
    )
    dirt = grime(g, 0.04)
    chalk = g.math(
        "MULTIPLY",
        g.madd(dirt, -1.0, 1.0),
        g.map_range(g.noise(obj, 0.7, detail=2.0), 0.3, 0.7, 0.12, 0.28),
    )
    col = g.mix(chalk, col, (0.075, 0.073, 0.068))
    col = g.mix(dirt, col, (0.012, 0.011, 0.01))
    rough = g.madd(chalk, 0.1, 0.8)
    if ctx is None:
        return g.finish(col, rough, g.bump(grit, 0.35, 0.0015))
    col, rough, height = plan_wear(g, ctx, col, rough, 0.0, wet_rough=0.3)
    fine = g.bump(grit, 0.35, 0.0015)
    return g.finish(col, rough, g.bump(height, 0.6, 0.002, fine))


def aged_metal(
    bpy,
    name: str,
    colour,
    rough: float,
    patch,
    cavity,
    metallic: float = 0.85,
    hands: bool = False,
    ctx: WearCtx | None = None,
    frame: bool = False,
):
    """Tarnished brass: brown-olive, in patches of darker tarnish, darkest and dullest in
    its cavities; with `hands`, bright where hands polish it. Window frames (`frame`)
    tarnish by where water goes rather than in patches."""
    g = Graph(bpy, name, colour)
    obj = g.coords.outputs["Object"]
    blotch = g.map_range(g.noise(obj, 9.0, detail=4.0), 0.42, 0.68)
    if frame:
        blotch = g.math("MULTIPLY", blotch, 0.3)
    col = g.mix(blotch, colour, patch)
    r = g.madd(blotch, 0.12, rough)
    dirt = grime(g, 0.02)
    col = g.mix(dirt, col, cavity)
    r = g.madd(dirt, 0.2, r)
    m = g.madd(dirt, -0.35, metallic)
    if ctx is not None:
        # Verdigris as a crust in chalky clumps, heaviest in the joints and at the
        # lowest point (the masks carry where; the clumps are its texture).
        verd = Wearing(g, ctx).masks(planes=("side", "plan"))["verd"]
        crust = g.map_range(g.noise(obj, 45.0, detail=3.0), 0.42, 0.58, 0.0, 1.4)
        verd = g.math("MINIMUM", g.math("MULTIPLY", verd, crust), 1.0)
        col = g.mix(g.math("MULTIPLY", verd, 0.85), col, VERDIGRIS)
        r = g.madd(verd, g.math("SUBTRACT", 0.9, r), r)
        m = g.madd(verd, g.math("MULTIPLY", m, -1.0), m)
    if frame:
        # Window frames: tarnish follows the water. It deepens smoothly down the frame
        # to the bottom rail, where water sits, with a few marks running down the rail's
        # face; the joints are darkest (the grime above); the open face stays a broad,
        # even, duller brass.
        _, _, z = g.separate(obj)
        hh = L.m(D.WINDOW_H) / 2
        zc = L.m(D.WINDOW_ZC)
        low = g.map_range(z, zc + 0.2 * hh, zc - hh - 0.03)
        bottom = g.map_range(z, zc - hh + 0.02, zc - hh - 0.02)
        marks = g.map_range(
            g.noise(g.mapping(obj, (40.0, 40.0, 3.0)), 1.0, detail=2.0), 0.55, 0.7
        )
        broad = g.map_range(g.noise(obj, 4.0, detail=2.0), 0.3, 0.7, 0.8, 1.0)
        fail = g.math(
            "MULTIPLY",
            g.math(
                "ADD",
                g.math("MULTIPLY", low, 0.55),
                g.math("MULTIPLY", marks, g.math("MULTIPLY", bottom, 0.3)),
            ),
            broad,
        )
        col = g.mix(fail, col, (0.15, 0.1, 0.042))
        r = g.madd(fail, 0.2, r)
        m = g.madd(fail, -0.05, m)
    if hands:
        bright = handled(g)
        col = g.mix(bright, col, (0.80, 0.62, 0.33))
        r = g.madd(bright, g.math("SUBTRACT", 0.18, r), r)
        m = g.madd(bright, g.math("SUBTRACT", 1.0, m), m)
    return g.finish(
        col,
        g.math("ADD", r, g.math("MULTIPLY", g.noise(obj, 30.0), 0.05)),
        g.geometry.outputs["Normal"],
        metallic=m,
        specular=0.5,
    )


def aged_glass(bpy, name: str, tint=(0.92, 0.96, 0.94)):
    """Window glass some weeks from its last clean: a faint film of dust and dried
    splashes along the bottom of the pane with a few water spots in it, clear above.
    Shadow rays still pass straight through, as with the clean glass, so the cabin is
    lit the same."""
    g = Graph(bpy, name, tint)
    obj = g.coords.outputs["Object"]
    _, _, z = g.separate(obj)
    zc = L.m(D.WINDOW_ZC)
    low = g.map_range(z, zc - 0.12, zc - 0.21)
    film = g.math(
        "MULTIPLY",
        low,
        g.map_range(g.noise(obj, 18.0, detail=3.0), 0.3, 0.7, 0.3, 1.0),
    )
    spots = g.math(
        "MULTIPLY",
        g.map_range(g.noise(obj, 240.0, detail=1.0), 0.72, 0.75),
        g.math("MULTIPLY", low, distance_fade(g, 2.0, 5.0)),
    )
    # No occlusion term: it found the pane's own inner face and veiled all of it.
    dirt = g.math(
        "MINIMUM",
        g.math("ADD", g.math("MULTIPLY", film, 0.15), g.math("MULTIPLY", spots, 0.1)),
        0.3,
    )
    glass = g.bsdf
    g.set(glass.inputs["Base Color"], tint)
    g.set(glass.inputs["Roughness"], 0.0)
    g.set(glass.inputs["Transmission Weight"], 1.0)
    g.set(glass.inputs["IOR"], 1.5)
    film_bsdf = g.node(
        "ShaderNodeBsdfPrincipled",
        {"Base Color": (0.16, 0.15, 0.13), "Roughness": 0.85},
    )
    surface = g.node(
        "ShaderNodeMixShader",
        {0: dirt, 1: glass.outputs["BSDF"], 2: film_bsdf.outputs["BSDF"]},
    )
    path = g.node("ShaderNodeLightPath")
    clear = g.node("ShaderNodeBsdfTransparent")
    shadow = g.node(
        "ShaderNodeMixShader",
        {
            0: path.outputs["Is Shadow Ray"],
            1: surface.outputs["Shader"],
            2: clear.outputs["BSDF"],
        },
    )
    out = next(n for n in g.nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
    g.nt.links.new(shadow.outputs["Shader"], out.inputs["Surface"])
    return g.mat


GREY_WOOD = (0.15, 0.12, 0.085)  # bare oak gone silver-brown in the weather


def aged_timber(bpy, name: str, base, rough: float = 0.35):
    """Weathered varnished timber: the grain and per-board tint of the clean oak, dirt in
    the joints; on faces that look up the varnish has gone in patches (a quarter to a
    third of them) to grey, rougher wood with a darker rim where it lifts; darker and
    smoother where hands grip it (the tiller)."""
    g = Graph(bpy, name, base)
    obj = g.coords.outputs["Object"]
    grain = g.grain(26.0, 10.0, detail=6.0)
    big = g.noise(obj, 0.9, detail=2.0)
    plank = g.math("SUBTRACT", g.attribute("plank_rand"), 0.5)
    value = g.math(
        "ADD", g.map_range(big, 0.3, 0.7, 0.9, 1.08), g.math("MULTIPLY", plank, 0.25)
    )
    tinted = g.node("ShaderNodeHueSaturation", {"Value": value, "Color": base}).outputs[
        "Color"
    ]
    col = g.mix(g.math("MULTIPLY", grain, 0.55), tinted, tuple(c * 0.55 for c in base))
    _, _, nz = g.separate(g.coords.outputs["Normal"])
    up = g.map_range(nz, 0.5, 0.8)
    # The varnish fails along the grain: patches stretched with the boards, their edges
    # soft where it has worn through gradually and fraying into the grain.
    along = g.mapping(g.coords.outputs["UV"], (0.25, 1.0, 1.0))
    patches = g.math(
        "ADD",
        g.noise(along, 9.0, detail=4.0),
        g.math("MULTIPLY", g.math("SUBTRACT", grain, 0.5), 0.12),
    )
    lost = g.math("MULTIPLY", up, g.map_range(patches, 0.55, 0.62))
    rim = g.math(
        "MULTIPLY", up, g.math("SUBTRACT", g.map_range(patches, 0.52, 0.56), lost)
    )
    grey = g.mix(
        g.math("MULTIPLY", grain, 0.6), GREY_WOOD, tuple(c * 0.55 for c in GREY_WOOD)
    )
    col = g.mix(g.math("MULTIPLY", lost, 0.85), col, grey)
    col = g.mix(g.math("MULTIPLY", rim, 0.35), col, tuple(c * 0.6 for c in base))
    r = g.madd(lost, 0.7 - rough, rough)
    dirt = grime(g, 0.05)
    col = g.mix(dirt, col, tuple(c * 0.3 for c in base))
    hands = handled(g)
    col = g.mix(g.math("MULTIPLY", hands, 0.7), col, (0.17, 0.11, 0.065))
    r = g.madd(hands, g.math("SUBTRACT", 0.4, r), r)
    height = g.madd(grain, 0.06, g.math("MULTIPLY", lost, -0.15))
    return g.finish(col, r, g.bump(height, 0.3, 0.001))


def aged_iron(bpy, name: str):
    """Black stove enamel, the chimney's middle chalked by heat and weather; soot caked on
    its top 120 mm and running down from the lip in streaks; the cap and the roof collar
    rusting, over a third or so of each. Only the chimney: other cast iron (the engine)
    keeps the chalk alone."""
    base = (0.02, 0.02, 0.02)
    g = Graph(bpy, name, base)
    obj = g.coords.outputs["Object"]
    x, y, z = g.separate(obj)
    heat = g.math(
        "MULTIPLY",
        g.map_range(z, 1.9, 2.3),
        g.map_range(g.noise(obj, 4.0, detail=3.0), 0.3, 0.7, 0.4, 1.0),
    )
    col = g.mix(heat, base, (0.05, 0.045, 0.04))
    r = g.madd(heat, 0.1, 0.65)
    # Within the chimney's footprint (the collar's radius plus a little).
    cx, cy = N.wx(D.CHIMNEY[0]), N.CHIMNEY_Y
    d = g.math(
        "SQRT",
        g.math(
            "ADD",
            g.math("POWER", g.math("SUBTRACT", x, cx), 2.0),
            g.math("POWER", g.math("SUBTRACT", y, cy), 2.0),
        ),
    )
    stack = g.map_range(d, L.m(D.COLLAR_D) / 2 + 0.03, L.m(D.COLLAR_D) / 2 + 0.01)
    z0 = L.roof_z(N.CHIMNEY_Y) - 0.01
    top = z0 + L.m(D.CHIMNEY_H)
    jit = edge_noise(g, obj)
    streaks = g.noise(g.mapping(obj, (90.0, 90.0, 4.0)), 1.0, detail=3.0)
    reach = g.madd(g.map_range(streaks, 0.4, 0.75), 0.25, 0.1)  # m below the lip
    caked = g.map_range(z, top - 0.13, top - 0.11)
    runs = g.math(
        "MULTIPLY",
        g.map_range(
            g.math("SUBTRACT", top, z),
            g.math("ADD", reach, 0.0),
            g.math("MULTIPLY", reach, 0.5),
        ),
        g.map_range(streaks, 0.45, 0.6),
    )
    soot = g.math(
        "MULTIPLY", stack, g.math("MAXIMUM", caked, g.math("MULTIPLY", runs, 0.7))
    )
    col = g.mix(soot, col, (0.006, 0.0055, 0.005))
    r = g.madd(soot, g.math("SUBTRACT", 0.92, r), r)
    cap = g.map_range(z, top - 0.055, top - 0.045)
    collar = g.map_range(z, z0 + 0.04, z0 + 0.025)
    patchy = g.math("ADD", g.noise(obj, 30.0, detail=4.0), g.math("MULTIPLY", jit, 0.5))
    rust = g.math(
        "MULTIPLY",
        g.math("MULTIPLY", stack, g.math("MAXIMUM", cap, collar)),
        g.map_range(patchy, 0.5, 0.55),
    )
    rcol, grain = rust_colour(g, obj)
    col = g.mix(rust, col, rcol)
    r = g.madd(rust, g.math("SUBTRACT", 0.9, r), r)
    height = g.math("MULTIPLY", rust, g.madd(grain, 0.6, 0.2))
    return g.finish(col, r, g.bump(height, 0.5, 0.0015))


def aged_rope(bpy, name: str):
    """Rope gone grey and hairy; the fender's lower part wet from the water it hangs in
    (darker and less fuzzy in its lowest 45%), and green with algae in the lowest 20%."""
    base = (0.20, 0.18, 0.15)
    g = Graph(bpy, name, base)
    obj = g.coords.outputs["Object"]
    _, _, gz = g.separate(g.coords.outputs["Generated"])
    twist = g.node(
        "ShaderNodeTexWave",
        {"Vector": obj, "Scale": 14.0, "Distortion": 1.0, "Detail": 3.0},
        wave_type="BANDS",
        bands_direction="DIAGONAL",
    ).outputs["Factor"]
    fuzz = g.map_range(g.noise(obj, 180.0, detail=2.0), 0.55, 0.8)
    wobble = g.madd(g.noise(obj, 8.0, detail=2.0), 0.08, -0.04)
    h = g.math("ADD", gz, wobble)
    wet = g.map_range(h, 0.47, 0.43)
    slime = g.map_range(h, 0.22, 0.18)
    col = g.mix(g.math("MULTIPLY", twist, 0.55), (0.07, 0.065, 0.055), base)
    col = g.mix(
        g.math("MULTIPLY", fuzz, g.madd(wet, -0.35, 0.5)), col, (0.34, 0.32, 0.28)
    )
    col = g.mix(g.math("MULTIPLY", wet, 0.8), col, (0.05, 0.043, 0.033))
    col = g.mix(g.math("MULTIPLY", slime, 0.75), col, (0.04, 0.06, 0.02))
    r = g.madd(wet, -0.35, 0.95)
    return g.finish(col, r, g.bump(twist, 0.6, 0.004), specular=0.1)


# --------------------------------------------------------------------------- false colour

FALSE = {
    "dirt": (0.6, 0.6, 0.55),
    "touch": (0.15, 0.75, 0.85),
    "stain": (0.45, 0.2, 0.05),
    "verd": (0.15, 0.85, 0.6),
    "chip": (1.0, 1.0, 1.0),
    "rust": (1.0, 0.42, 0.02),
    "polish": (0.25, 0.4, 1.0),
    "algae": (0.45, 0.6, 0.15),
    "moss": (0.05, 0.85, 0.05),
    "soot": (0.0, 0.0, 0.0),
    "pool": (0.55, 0.28, 0.06),
    "waterline": (0.6, 0.9, 1.0),
}


def false_colour(
    bpy,
    name: str,
    base,
    ctx: WearCtx,
    uses=PAINT_USES,
    hull: bool = False,
    design: str | None = None,
):
    """Flat, self-lit view of the masks a slot uses, over a grey version of its colour:
    where the wear lands, for placement reviews."""
    g = Graph(bpy, name, base)
    m = dict(Wearing(g, ctx).masks(gunwale_only=hull))
    if "polish" in uses and name.endswith(("Brass", "TillerWood")):
        m["polish"] = g.math("MAXIMUM", m["polish"], handled(g))
    t = ctx.wear.threshold
    th = ctx.record["thresholds"]
    lum = 0.2126 * base[0] + 0.7152 * base[1] + 0.0722 * base[2]
    grey = 0.18 + 0.45 * min(lum * 4, 1.0)
    col = tuple(0.8 * grey + 0.2 * c for c in base)

    def step(p, level):
        return g.math("GREATER_THAN", p, level)

    def clamp(v):
        return g.math("MINIMUM", v, 1.0)

    layers = (
        ("dirt", lambda: g.math("MULTIPLY", m["dirt"], 0.8)),
        ("touch", lambda: g.math("MULTIPLY", step(m["touch"], 0.05), 0.7)),
        ("stain", lambda: clamp(g.math("MULTIPLY", m["stain"], 1.2))),
        ("verd", lambda: m["verd"]),
        ("polish", lambda: g.math("MULTIPLY", step(m["polish"], 0.05), m["polish"])),
        ("algae", lambda: g.math("MULTIPLY", step(m["moss"], th["algae"]), 0.6)),
        ("moss", lambda: step(m["moss"], th["moss"])),
        ("soot", lambda: clamp(g.math("MULTIPLY", m["soot"], 1.8))),
        ("pool", lambda: g.math("MULTIPLY", m["pool"], 0.8)),
    )
    for key, amount in layers:
        if (key if key != "algae" else "moss") in uses:
            col = g.mix(amount(), col, FALSE[key])
    if "rust" in uses or "chip" in uses:
        rust = step(m["rust"], t) if "rust" in uses else 0.0
        if "chip" in uses:
            chip = step(m["chip"], t)
            if "rust" in uses:
                chip = g.math("MULTIPLY", chip, g.math("SUBTRACT", 1.0, rust))
            col = g.mix(chip, col, FALSE["chip"])
        if "rust" in uses:
            col = g.mix(rust, col, FALSE["rust"])
    if hull:
        x, _, z = g.separate(g.coords.outputs["Object"])
        h = waterline_height(g, x, z)
        band = g.math(
            "MULTIPLY",
            g.map_range(h, 0.105, 0.115, 1.0, 0.0),
            g.map_range(h, -0.005, 0.005),
        )
        col = g.mix(g.math("MULTIPLY", band, 0.6), col, FALSE["waterline"])
    if design is not None:
        tex = g.node(
            "ShaderNodeTexImage",
            {"Vector": g.coords.outputs["UV"]},
            image=bpy.data.images[N.texture_name(design)],
            extension="CLIP",
        )
        g.set(g.bsdf.inputs["Alpha"], tex.outputs["Alpha"])
    g.set(g.bsdf.inputs["Emission Color"], col)
    g.set(g.bsdf.inputs["Emission Strength"], 1.0)
    return g.finish((0.0, 0.0, 0.0), 1.0, g.geometry.outputs["Normal"], specular=0.0)


# --------------------------------------------------------------------------- cameras


def wear_camera_specs(aims: dict | None = None) -> dict:
    """Close-ups for judging wear, lakeside. `sun` is the sun's azimuth (degrees), chosen
    for each so the sun rakes across the surface in view instead of leaving it in shade.
    `aims` (narrowboat_wear.camera_aims) sets b (mm) of the starboard window and the port
    porthole the weld and porthole close-ups look at, so they find the seeded wear."""
    wx = N.wx
    aims = {"weld": 7450.0, "porthole": 4450.0, **(aims or {})}
    weld, port = aims["weld"], aims["porthole"]

    def cam(lens, location, target, sun):
        return {
            "lens": lens,
            "location": location,
            "target": target,
            "resolution": (1600, 1000),
            "sun": sun,
        }

    return {
        "wear_weld": cam(
            50, (wx(weld) + 0.55, -2.25, 1.40), (wx(weld), -0.95, 1.10), -150
        ),
        "wear_porthole": cam(
            50, (wx(port) + 0.35, 1.75, 1.55), (wx(port), 0.92, 1.38), 150
        ),
        "wear_counter": cam(
            35, (wx(-2950), -1.85, 1.55), (wx(-1450), -0.55, 0.85), -110
        ),
        "wear_bow": cam(
            35, (wx(D.STEM) + 1.5, -1.6, 1.55), (wx(14200), -0.45, 0.85), -20
        ),
        "wear_doors": cam(
            35, (wx(D.CABIN_LEN) + 1.7, -0.75, 1.65), (wx(D.CABIN_LEN), 0.0, 1.45), -50
        ),
        "wear_roof": cam(35, (wx(9600), -1.25, 2.75), (wx(8300), -0.2, 2.10), -40),
    }
