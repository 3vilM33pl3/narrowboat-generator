# SPDX-License-Identifier: MIT
"""Canal-art castle scenes as layered SVG, the "castles" of narrowboat roses and castles.

A canal castle is a naive, romantic landscape painted in a shaped panel, most often on the
cabin doors: a white castle with red cone-roofed towers on a hill above a lake, an arched
stone bridge, a tall tree in the foreground and a bright sky fading to the horizon. Like
`canal_roses`, everything is drawn as flat painted shapes with one-sided shading, built from
that module's brush primitives, so the scene stays clean, editable vector art.

The scene is clipped to a frame (arched top, oval or rounded rectangle) edged with a cream
and a red line; outside the frame the SVG is transparent, ready to use as a decal. `side`
puts the castle on the left (-1) or right (+1); a door pair uses one panel and its mirror.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from xml.sax.saxutils import escape

import canal_roses as C

Pt = tuple[float, float]
fmt = C.fmt

SKIES = {
    # zenith, middle, horizon
    "day": ("#2c5dab", "#7eaedf", "#f5e3b3"),
    "sunset": ("#294b97", "#d68aa3", "#f7c665"),
}
FRAMES = ("arch", "oval", "rect")

WALL, WALL_SHADE, WALL_DARK = "#f3ead6", "#cdb88f", "#9a8460"
ROOF, ROOF_SHADE = "#cc3a2a", "#8c2519"
WINDOW, GATE = "#2b2433", "#3a2c24"
FLAG, POLE = "#d63a2c", "#4a3a2a"
HILL_FAR, HILL_FAR_LIGHT = "#8e93c4", "#b2b5da"
HILL, HILL_LIGHT, HILL_SHADE = "#5a8c40", "#94c35a", "#3d6a2c"
WATER_TOP, WATER_DEEP, RIPPLE = "#9cc4ea", "#2e5c9c", "#eef5ff"
STONE, STONE_SHADE, ARCH = "#e0cda4", "#b49f76", "#2f3140"
BANK, BANK_DARK, BANK_LIGHT, PATH = "#4c7d2e", "#2e5520", "#8fbf4f", "#e9d9aa"
TREE_DARK, TREE_MID, TREE_LIGHT, TRUNK = "#20401d", "#3b762d", "#8dc24f", "#5a3a22"
CLOUD, CLOUD_SHADE, BIRD = "#ffffff", "#e9cfd2", "#3a3340"
FRAME_CREAM, FRAME_RED = "#efe2b8", "#b8322a"


@dataclass(frozen=True)
class Spec:
    name: str
    width: float = 400
    height: float = 560
    frame: str = "arch"
    side: int = 1  # the castle stands on the right (+1) or the left (-1)
    towers: int = 3  # round towers round the keep, 2-4
    arches: int = 3  # bridge arches; 0 leaves the bridge out
    sky: str = "day"
    seed: int = 1
    description: str = ""


def validate(spec: Spec) -> list[str]:
    errors = []
    if spec.frame not in FRAMES:
        errors.append(f"unknown frame {spec.frame!r}")
    if spec.sky not in SKIES:
        errors.append(f"unknown sky {spec.sky!r}")
    if spec.side not in (-1, 1):
        errors.append("side must be -1 or 1")
    if not 2 <= spec.towers <= 4:
        errors.append("towers must be 2-4")
    if not 0 <= spec.arches <= 5:
        errors.append("arches must be 0-5")
    if spec.width < 100 or spec.height < 100:
        errors.append("canvas must be at least 100 × 100")
    return errors


# ---------------------------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------------------------


def path(d: str, fill: str, extra: str = "") -> str:
    return f'<path fill="{fill}"{extra} d="{d}"/>'


def poly(pts: list[Pt], fill: str, extra: str = "") -> str:
    """A straight-edged polygon: masonry wants hard corners, not Catmull-Rom curves."""
    d = "M" + "L".join(f"{fmt(x)} {fmt(y)}" for x, y in pts) + "Z"
    return path(d, fill, extra)


def rect(x0: float, y0: float, x1: float, y1: float, fill: str) -> str:
    return poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], fill)


def arch_window(cx: float, base: float, w: float, h: float, fill: str) -> str:
    """A round-headed opening: a rectangle under a semicircle."""
    r = w / 2
    top = base - h + r
    d = (
        f"M{fmt(cx - r)} {fmt(base)}L{fmt(cx - r)} {fmt(top)}"
        f"A{fmt(r)} {fmt(r)} 0 0 1 {fmt(cx + r)} {fmt(top)}"
        f"L{fmt(cx + r)} {fmt(base)}Z"
    )
    return path(d, fill)


def frame_path(spec: Spec, inset: float) -> str:
    W, H, i = spec.width, spec.height, inset
    if spec.frame == "oval":
        rx, ry = W / 2 - i, H / 2 - i
        return (
            f"M{fmt(W / 2 - rx)} {fmt(H / 2)}"
            f"A{fmt(rx)} {fmt(ry)} 0 1 1 {fmt(W / 2 + rx)} {fmt(H / 2)}"
            f"A{fmt(rx)} {fmt(ry)} 0 1 1 {fmt(W / 2 - rx)} {fmt(H / 2)}Z"
        )
    if spec.frame == "rect":
        r = 0.08 * W
        x0, y0, x1, y1 = i, i, W - i, H - i
        return (
            f"M{fmt(x0 + r)} {fmt(y0)}L{fmt(x1 - r)} {fmt(y0)}"
            f"A{fmt(r)} {fmt(r)} 0 0 1 {fmt(x1)} {fmt(y0 + r)}L{fmt(x1)} {fmt(y1 - r)}"
            f"A{fmt(r)} {fmt(r)} 0 0 1 {fmt(x1 - r)} {fmt(y1)}L{fmt(x0 + r)} {fmt(y1)}"
            f"A{fmt(r)} {fmt(r)} 0 0 1 {fmt(x0)} {fmt(y1 - r)}L{fmt(x0)} {fmt(y0 + r)}"
            f"A{fmt(r)} {fmt(r)} 0 0 1 {fmt(x0 + r)} {fmt(y0)}Z"
        )
    r = W / 2 - i
    return (
        f"M{fmt(i)} {fmt(H - i)}L{fmt(i)} {fmt(W / 2)}"
        f"A{fmt(r)} {fmt(r)} 0 0 1 {fmt(W - i)} {fmt(W / 2)}"
        f"L{fmt(W - i)} {fmt(H - i)}Z"
    )


def ridge(x0: float, x1: float, base: float, rng: random.Random, amp: float, n=40):
    """A hill line from x0 to x1: a few summed sines over `base` (y down)."""
    waves = [
        (rng.uniform(0.6, 2.2), rng.uniform(0, math.tau), rng.uniform(0.3, 1.0))
        for _ in range(3)
    ]
    pts = []
    for k in range(n):
        t = k / (n - 1)
        h = sum(a * math.sin(f * math.pi * t * 2 + p) for f, p, a in waves) / 2.3
        pts.append((x0 + (x1 - x0) * t, base - amp * (0.5 + 0.5 * h)))
    return pts


def cloud(cx: float, cy: float, w: float, rng: random.Random) -> list[str]:
    """A fair-weather cloud: a flat-bottomed row of puffs over a pinkish underside."""
    out = [
        C.Shape(
            C.smooth_path(C.blob((cx, cy + w * 0.06), w * 0.5, w * 0.1, rng, 0.05)),
            CLOUD_SHADE,
        ).svg()
    ]
    puffs = rng.randint(4, 6)
    for k in range(puffs):
        t = k / (puffs - 1) - 0.5
        r = w * (0.17 - 0.1 * abs(t)) * rng.uniform(0.85, 1.15)
        x = cx + t * w * 0.8
        y = cy - r * 0.55 + abs(t) * w * 0.05
        out.append(
            C.Shape(
                C.smooth_path(C.blob((x, y), r, r * 0.85, rng, 0.04, 14)), CLOUD
            ).svg()
        )
    return out


def tree(
    x: float, ground: float, height: float, rng: random.Random, light: int
) -> list[str]:
    """A canal-painting tree: a tapering trunk under a crown of stacked round clumps, dark
    below and picked out with light dabs on the lit side (`light` = +1 from the right)."""
    out = []
    crown_w = height * 0.36
    top = ground - height
    trunk = [
        (x - height * 0.035, ground),
        (x - height * 0.012, top + height * 0.45),
        (x + height * 0.012, top + height * 0.45),
        (x + height * 0.035, ground),
    ]
    out.append(poly(trunk, TRUNK))
    clumps = []
    for k in range(7):
        t = k / 6
        cy = top + height * (0.12 + 0.42 * t) + rng.uniform(-6, 6)
        spread = crown_w * (0.25 + 0.3 * math.sin(math.pi * min(1.0, t * 1.15)))
        cx = x + rng.uniform(-spread, spread)
        r = crown_w * rng.uniform(0.26, 0.34) * (0.75 + 0.35 * math.sin(math.pi * t))
        clumps.append((cx, cy, r))
    for cx, cy, r in clumps:
        out.append(
            C.Shape(
                C.smooth_path(C.blob((cx, cy + r * 0.12), r * 1.05, r, rng, 0.07)),
                TREE_DARK,
            ).svg()
        )
    for cx, cy, r in clumps:
        out.append(
            C.Shape(
                C.smooth_path(
                    C.blob(
                        (cx + light * r * 0.12, cy - r * 0.08),
                        r * 0.8,
                        r * 0.72,
                        rng,
                        0.08,
                    )
                ),
                TREE_MID,
            ).svg()
        )
    for cx, cy, r in clumps:
        for _ in range(2):
            dx = light * r * rng.uniform(0.2, 0.5)
            dy = -r * rng.uniform(0.1, 0.45)
            out.append(
                C.Shape(
                    C.smooth_path(
                        C.leaf_stroke(
                            (cx + dx - r * 0.18, cy + dy + 2),
                            (cx + dx + r * 0.18, cy + dy - 2),
                            r * 0.22,
                            0.1,
                        )
                    ),
                    TREE_LIGHT,
                ).svg()
            )
    return out


def poplar(
    x: float, ground: float, height: float, rng: random.Random, light: int
) -> list[str]:
    w = height * 0.16
    body = C.blob((x, ground - height * 0.5), w, height * 0.5, rng, 0.05, 20)
    lit = C.blob(
        (x + light * w * 0.25, ground - height * 0.55),
        w * 0.5,
        height * 0.4,
        rng,
        0.06,
        16,
    )
    return [
        C.Shape(C.smooth_path(body), TREE_DARK).svg(),
        C.Shape(C.smooth_path(lit), TREE_MID).svg(),
    ]


# ---------------------------------------------------------------------------------------------
# The castle
# ---------------------------------------------------------------------------------------------


def _merlons(x0: float, x1: float, top: float, size: float, light: int) -> list[str]:
    """Crenellations along a wall top: merlons with their shaded side away from the light."""
    n = max(2, round((x1 - x0) / (size * 1.8)))
    w = (x1 - x0) / (2 * n - 1)
    out = []
    for k in range(n):
        a = x0 + 2 * k * w
        out.append(rect(a, top - size, a + w, top + 0.5, WALL))
        sx = a + w * 0.65 if light < 0 else a
        out.append(rect(sx, top - size, sx + w * 0.35, top + 0.5, WALL_SHADE))
    return out


def _shaded_block(
    x0: float, x1: float, top: float, base: float, light: int, frac=0.32
) -> list[str]:
    """A wall or tower face: lit colour with a shaded strip on the side away from the light."""
    w = x1 - x0
    out = [rect(x0, top, x1, base, WALL)]
    if light > 0:
        out.append(rect(x0, top, x0 + w * frac, base, WALL_SHADE))
    else:
        out.append(rect(x1 - w * frac, top, x1, base, WALL_SHADE))
    return out


def _cone(cx: float, top: float, w: float, h: float, light: int) -> list[str]:
    """A conical roof with a slight overhang, shaded on its dark half, with a pennant."""
    l, r = cx - w / 2 - w * 0.12, cx + w / 2 + w * 0.12
    apex = (cx, top - h)
    out = [
        path(
            f"M{fmt(l)} {fmt(top)}Q{fmt(cx)} {fmt(top + w * 0.16)} {fmt(r)} {fmt(top)}"
            f"L{fmt(apex[0])} {fmt(apex[1])}Z",
            ROOF,
        )
    ]
    dark_edge = l if light > 0 else r
    out.append(
        path(
            f"M{fmt(dark_edge)} {fmt(top)}Q{fmt((dark_edge + cx) / 2)} {fmt(top + w * 0.12)} "
            f"{fmt(cx - light * w * 0.08)} {fmt(top + w * 0.15)}L{fmt(apex[0])} {fmt(apex[1])}Z",
            ROOF_SHADE,
        )
    )
    pole_top = apex[1] - h * 0.42
    out.append(rect(cx - 0.9, pole_top, cx + 0.9, apex[1] + 1, POLE))
    fly = (
        -light
    )  # the pennant streams away from the light, i.e. downwind in the picture
    tip = (cx + fly * h * 0.55, pole_top + h * 0.1)
    out.append(
        path(
            f"M{fmt(cx)} {fmt(pole_top)}Q{fmt(cx + fly * h * 0.3)} {fmt(pole_top - h * 0.05)} "
            f"{fmt(tip[0])} {fmt(tip[1])}Q{fmt(cx + fly * h * 0.25)} {fmt(pole_top + h * 0.14)} "
            f"{fmt(cx)} {fmt(pole_top + h * 0.17)}Z",
            FLAG,
        )
    )
    return out


def castle(
    cx: float, base: float, scale: float, spec: Spec, rng: random.Random, light: int
) -> list[str]:
    """Keep, curtain walls and round towers standing on (cx, base); about 170 × 190 at scale 1."""
    s = scale
    out: list[str] = []
    # Tower slots across the front, outermost first; the keep stands just off centre.
    slots = {2: [-1, 1], 3: [-1, 0.55, 1.05], 4: [-1.05, -0.45, 0.5, 1.05]}[spec.towers]
    towers = []
    for k, u in enumerate(slots):
        w = rng.uniform(26, 34) * s
        h = rng.uniform(80, 120) * s * (1.0 if abs(u) > 0.9 else 0.85)
        towers.append((cx + u * 72 * s * rng.uniform(0.95, 1.05), w, h))
    tallest = max(range(len(towers)), key=lambda k: towers[k][2])
    keep_x = cx + rng.uniform(-14, 6) * s
    keep_w, keep_h = rng.uniform(44, 54) * s, rng.uniform(92, 112) * s

    # Curtain wall between the outer towers, with a gate.
    xs = sorted(t[0] for t in towers)
    wall_top = base - rng.uniform(42, 52) * s
    out += _shaded_block(xs[0], xs[-1], wall_top, base, light, 0.0)
    out += _merlons(xs[0], xs[-1], wall_top, 7 * s, light)
    out.append(rect(xs[0], base - 6 * s, xs[-1], base, WALL_SHADE))
    gate_x = cx + rng.uniform(-30, 30) * s
    out.append(arch_window(gate_x, base, 16 * s, 26 * s, GATE))

    # The keep: square, crenellated, with a turret on its lit corner.
    k0, k1 = keep_x - keep_w / 2, keep_x + keep_w / 2
    keep_top = base - keep_h
    out += _shaded_block(k0, k1, keep_top, base - 10 * s, light)
    out += _merlons(k0 - 2 * s, k1 + 2 * s, keep_top, 8 * s, light)
    for row in range(2):
        y = keep_top + (26 + row * 30) * s
        for col in (-1, 1):
            out.append(
                arch_window(keep_x + col * keep_w * 0.22, y, 7 * s, 15 * s, WINDOW)
            )
    turret_x = k1 - 6 * s if light > 0 else k0 + 6 * s
    out += _shaded_block(
        turret_x - 8 * s, turret_x + 8 * s, keep_top - 22 * s, keep_top, light
    )
    out += _cone(turret_x, keep_top - 22 * s, 16 * s, 26 * s, light)

    # Round towers in front, shorter ones first so the tallest overlaps its neighbours.
    for k in sorted(range(len(towers)), key=lambda k: towers[k][2]):
        x, w, h = towers[k]
        top = base - h
        out += _shaded_block(x - w / 2, x + w / 2, top, base, light, 0.38)
        # A corbelled band under the roof, wider than the shaft.
        out.append(
            rect(
                x - w / 2 - 2 * s,
                top - 5 * s,
                x + w / 2 + 2 * s,
                top + 2 * s,
                WALL_DARK,
            )
        )
        for row in range(rng.randint(2, 3)):
            sill = top + (24 + row * 30) * s
            if (
                sill < base - 14 * s
            ):  # short towers get fewer windows, never one below ground
                out.append(arch_window(x, sill, 6 * s, 13 * s, WINDOW))
        roof_h = w * rng.uniform(1.2, 1.6) * (1.15 if k == tallest else 1.0)
        out += _cone(x, top - 5 * s, w + 4 * s, roof_h, light)
    return out


# ---------------------------------------------------------------------------------------------
# The scene
# ---------------------------------------------------------------------------------------------


def view_box(spec: Spec) -> tuple[float, float, float, float]:
    return 0.0, 0.0, spec.width, spec.height


def render_svg(spec: Spec) -> str:
    errors = validate(spec)
    if errors:
        raise ValueError(f"{spec.name}: " + "; ".join(errors))
    rng = random.Random(spec.seed)
    W, H, side = spec.width, spec.height, spec.side
    light = -side  # lit from the open side of the picture, away from the castle
    horizon = 0.57 * H
    shore = horizon + 0.035 * H  # where the castle hill meets the lake
    bank_y = 0.84 * H
    zen, mid, low = SKIES[spec.sky]
    defs = (
        f'<linearGradient id="sky" x1="0" y1="0" x2="0" y2="{fmt(horizon)}" gradientUnits="userSpaceOnUse">'
        f'<stop offset="0" stop-color="{zen}"/><stop offset="0.55" stop-color="{mid}"/>'
        f'<stop offset="1" stop-color="{low}"/></linearGradient>'
        f'<linearGradient id="lake" x1="0" y1="{fmt(horizon)}" x2="0" y2="{fmt(bank_y)}" gradientUnits="userSpaceOnUse">'
        f'<stop offset="0" stop-color="{WATER_TOP}"/><stop offset="1" stop-color="{WATER_DEEP}"/></linearGradient>'
        f'<clipPath id="frame"><path d="{frame_path(spec, 6)}"/></clipPath>'
        f'<clipPath id="water"><rect x="0" y="{fmt(shore)}" width="{fmt(W)}" height="{fmt(bank_y - shore)}"/></clipPath>'
    )
    scene: list[str] = [
        f'<rect width="{fmt(W)}" height="{fmt(horizon + 2)}" fill="url(#sky)"/>'
    ]

    # Clouds on the open side of the sky, birds above them.
    for k in range(rng.randint(2, 3)):
        x = W * (0.5 - side * rng.uniform(0.05, 0.3)) + rng.uniform(-20, 20)
        y = H * rng.uniform(0.14, 0.32) + k * 18
        scene += cloud(x, y, min(W, 0.75 * H) * rng.uniform(0.28, 0.4), rng)
    for _ in range(2):
        x, y = W * (0.5 - side * rng.uniform(0.0, 0.25)), H * rng.uniform(0.08, 0.2)
        b = W * 0.025
        scene.append(
            C.Shape(
                C.smooth_path(C.leaf_stroke((x - b, y - b * 0.4), (x, y), 2.2, 0.15)),
                BIRD,
            ).svg()
        )
        scene.append(
            C.Shape(
                C.smooth_path(C.leaf_stroke((x, y), (x + b, y - b * 0.45), 2.2, 0.15)),
                BIRD,
            ).svg()
        )

    # Far hills: two blue ranges under the horizon haze.
    for amp, colour, lift in ((0.09, HILL_FAR_LIGHT, 0.02), (0.06, HILL_FAR, 0.0)):
        top = ridge(-10, W + 10, horizon - lift * H, rng, amp * H)
        scene.append(poly([*top, (W + 10, horizon + 2), (-10, horizon + 2)], colour))

    # The lake.
    scene.append(
        f'<rect y="{fmt(horizon)}" width="{fmt(W)}" height="{fmt(bank_y - horizon + 4)}" fill="url(#lake)"/>'
    )

    # Castle hill and castle, grouped so the lake can reflect them. The castle is sized to
    # fit between the hill top and the frame's top, and to the panel's width.
    cx = W * (0.5 + side * 0.2)
    hill_top = 0.43 * H
    scale = min(W / 400, (hill_top - 0.1 * H) / 205) * 0.95
    hill = []
    foot = W * (0.5 - side * 0.14)  # where the hill's open-side slope reaches the lake
    beyond = W + 30 if side > 0 else -30
    mound = [
        (foot, shore + 2),
        (foot + side * 0.12 * W, shore - 0.045 * H),
        (cx - side * 100 * scale, hill_top + 0.012 * H),
        (cx, hill_top),
        (cx + side * 100 * scale, hill_top + 0.008 * H),
        (beyond, hill_top + 0.03 * H),
        (beyond, shore + 2),
    ]
    hill.append(path(C.smooth_path(mound), HILL))
    # Lit crest on the open side, shadowed foot along the water.
    crest = C.quad_curve(
        (cx - side * 100 * scale, hill_top + 0.016 * H),
        (cx - side * 10 * scale, hill_top + 0.004 * H),
        0.04 * side,
        16,
    )
    widths = [0.008 * H * math.sin(math.pi * k / 15) for k in range(16)]
    hill.append(C.Shape(C.smooth_path(C.ribbon(crest, widths)), HILL_LIGHT).svg())
    foot_line = C.quad_curve((foot, shore), (beyond, shore - 2), 0.0, 24)
    hill.append(
        C.Shape(C.smooth_path(C.ribbon(foot_line, [0.012 * H] * 24)), HILL_SHADE).svg()
    )
    hill += poplar(cx - side * 82 * scale, hill_top + 0.04 * H, 0.11 * H, rng, light)
    hill += castle(cx, hill_top + 2, scale, spec, rng, light)
    scene.append(f'<g id="castle-hill">{"".join(hill)}</g>')

    # Its reflection: flipped about the shoreline, squashed, faded into the water.
    scene.append(
        f'<g clip-path="url(#water)" opacity="0.32">'
        f'<use xlink:href="#castle-hill" transform="translate(0 {fmt(shore * 2 + 4)}) scale(1 -1)"/></g>'
    )

    # The bridge across the lake from the castle's shore.
    if spec.arches:
        # It leaves the foot of the castle hill and runs off behind the tree on the far bank.
        water_at = shore + 0.05 * H
        deck = water_at - 0.065 * H
        x_castle = foot + side * 0.08 * W
        x_far = W * (0.5 - side * 0.75)
        a, b = sorted((x_castle, x_far))
        scene.append(rect(a, deck, b, water_at, STONE))
        scene.append(rect(a, water_at - 0.012 * H, b, water_at, STONE_SHADE))
        scene.append(rect(a, deck - 0.012 * H, b, deck, STONE_SHADE))
        span = (b - a) / spec.arches
        for k in range(spec.arches):
            mx = a + span * (k + 0.5)
            r = span * 0.34
            scene.append(
                arch_window(mx, water_at, 2 * r, (water_at - deck) * 0.68, ARCH)
            )
            # The arch's reflection closes it into a dark ring on the water.
            scene.append(
                C.Shape(
                    C.smooth_path(
                        C.blob((mx, water_at + r * 0.35), r, r * 0.45, rng, 0.03, 16)
                    ),
                    ARCH,
                    "",
                )
                .svg()
                .replace("<path", '<path opacity="0.45"')
            )
        scene.append(
            f'<rect x="{fmt(a)}" y="{fmt(water_at)}" width="{fmt(b - a)}" height="{fmt(0.02 * H)}" fill="{STONE}" opacity="0.25"/>'
        )

    # Ripples: short pale strokes, denser near the far shore.
    for _ in range(26):
        y = horizon + (bank_y - horizon) * rng.random() ** 1.4
        x = rng.uniform(0, W)
        ln = W * rng.uniform(0.04, 0.12)
        scene.append(
            C.Shape(
                C.smooth_path(
                    C.leaf_stroke(
                        (x - ln / 2, y), (x + ln / 2, y), rng.uniform(1.4, 2.6)
                    )
                ),
                RIPPLE,
            )
            .svg()
            .replace("<path", '<path opacity="0.7"')
        )

    # Foreground bank with a path running down to the water, grass and a tall tree.
    bank_top = ridge(-10, W + 10, bank_y + 0.03 * H, rng, 0.06 * H, 30)
    scene.append(poly([*bank_top, (W + 10, H + 5), (-10, H + 5)], BANK))
    scene.append(
        poly(
            [*[(x, y + 0.02 * H) for x, y in bank_top], (W + 10, H + 5), (-10, H + 5)],
            BANK_DARK,
        ).replace("<path", '<path opacity="0.35"')
    )
    px = W * (0.5 + side * 0.08)
    path_pts = [
        (px - W * 0.05, bank_y + 0.06 * H),
        (px + W * 0.02, bank_y + 0.06 * H),
        (px - side * W * 0.02 + W * 0.12, H + 4),
        (px - side * W * 0.02 - W * 0.14, H + 4),
    ]
    scene.append(C.Shape(C.smooth_path(path_pts), PATH).svg())
    for _ in range(18):
        x = rng.uniform(0, W)
        y = bank_y + rng.uniform(0.08, 0.15) * H
        h = rng.uniform(8, 16)
        lean = rng.uniform(-5, 5)
        scene.append(
            C.Shape(
                C.smooth_path(C.leaf_stroke((x, y), (x + lean, y - h), 2.6, 0.1)),
                BANK_LIGHT,
            ).svg()
        )
    tree_x = W * (0.5 - side * 0.36)
    scene += tree(tree_x, bank_y + 0.07 * H, min(0.5 * H, 1.2 * W), rng, light)

    border = (
        f'<path fill="none" stroke="{FRAME_CREAM}" stroke-width="{fmt(0.025 * W)}" d="{frame_path(spec, 0.0125 * W)}"/>'
        f'<path fill="none" stroke="{FRAME_RED}" stroke-width="{fmt(0.008 * W)}" d="{frame_path(spec, 0.035 * W)}"/>'
    )
    title = escape(f"Canal castle: {spec.name}")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 {fmt(W)} {fmt(H)}" width="{fmt(W * 2)}" height="{fmt(H * 2)}">'
        f"<title>{title}</title><defs>{defs}</defs>"
        f'<g id="scene" clip-path="url(#frame)">{"".join(scene)}</g>{border}</svg>\n'
    )


PRESETS = {
    s.name: s
    for s in (
        Spec(
            "castle_day",
            description="A white castle on a hill above a lake, on the right.",
        ),
        Spec(
            "castle_sunset",
            side=-1,
            sky="sunset",
            towers=4,
            arches=2,
            seed=5,
            description="Four towers against a sunset, castle on the left.",
        ),
        Spec(
            "castle_oval",
            frame="oval",
            width=480,
            height=360,
            towers=2,
            seed=9,
            description="A landscape oval for a cabin-side panel.",
        ),
    )
}
