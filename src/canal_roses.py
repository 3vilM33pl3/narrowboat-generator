# SPDX-License-Identifier: MIT
"""Canal-art rose bouquets as layered SVG brush strokes, after a traditional narrowboat panel.

The reference is the user's photo of a painted panel (Reference/canal_roses_reference.png):
two red roses flank a block of pink and yellow ones, blue-and-white daisies trail off both
ends, and round olive leaves edged with white dots sit behind them, the whole spray fanned
out with slender yellow-green leaf strokes on a green ground. Its corner number is not art
and is left out.

Every motif is built the way a canal painter builds it, as a stack of single brush marks:
a base blob, then tapered C-strokes and commas on top. Motifs are drawn in a local frame
(a rose or daisy has radius 100, a leaf runs 100 along +x from its stem) and placed with an
SVG group transform, so each flower stays one editable group in Inkscape or Illustrator.

A `Spec` picks a layout, palette and seed. `PRESETS` holds the faithful reference redraw and
a set of variations; `random_spec(seed)` samples more. Nothing here needs more than the
standard library; PNG previews use `rsvg-convert` and the contact sheet `magick` when present.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from itertools import pairwise
from pathlib import Path
from xml.sax.saxutils import escape

Pt = tuple[float, float]

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "build" / "CanalRoses"

# A layout's rose radius is its spacing; the painted bloom is drawn a little larger so
# neighbours overlap as they do on the reference instead of sitting apart.
ROSE_SCALE = 1.1

# ---------------------------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RoseTone:
    """Paints for one rose: the outer petals (rim), the base ball under them (body), the
    dark bowl (cup), the inner petals (mid), the calligraphic hairlines (accent), and the
    brightest petal edges (highlight)."""

    rim: str
    body: str
    cup: str
    mid: str
    accent: str
    highlight: str


@dataclass(frozen=True)
class Palette:
    ground: str
    roses: dict[str, RoseTone]
    petal: str
    petal_shade: str
    eye: str
    disc: str
    spark: str
    leaf: str
    leaf_dark: str
    vein: str
    dot: str
    stroke: str


# Sampled from the reference photo, then lifted slightly toward fresh enamel paint.
ROSE_TONES = {
    "red": RoseTone("#c9563f", "#1d0f09", "#120806", "#3a120b", "#120806", "#e3826a"),
    "pink": RoseTone("#efe8d8", "#dea38b", "#9c3c2a", "#e3937a", "#1c0d08", "#ffffff"),
    "yellow": RoseTone(
        "#e9b526", "#e29a45", "#8e2f12", "#d9823a", "#3a0e06", "#f7da64"
    ),
    "white": RoseTone("#f2eee2", "#d9d0bb", "#a08660", "#e6dfcb", "#3a2a1a", "#ffffff"),
    "crimson": RoseTone(
        "#b8262a", "#3a0809", "#1a0405", "#931a1e", "#140304", "#ea6a5e"
    ),
    "apricot": RoseTone(
        "#f0a75a", "#d9733a", "#8a3014", "#e48848", "#2a0b05", "#fbd29a"
    ),
    "blush": RoseTone("#f4c9c0", "#e08f86", "#9c3f3c", "#e9a49b", "#250a0a", "#fff1ec"),
}

TRADITIONAL = Palette(
    ground="#3f5a15",
    roses=ROSE_TONES,
    petal="#eeeadb",
    petal_shade="#7f95c4",
    eye="#0d0f10",
    disc="#cf4a1f",
    spark="#efbd2c",
    leaf="#a29613",
    leaf_dark="#4a520c",
    vein="#c2b532",
    dot="#ece6cc",
    stroke="#ab9d1a",
)

PALETTES = {
    "green": TRADITIONAL,
    "black": replace(TRADITIONAL, ground="#141210"),
    "red": replace(
        TRADITIONAL,
        ground="#7d1716",
        leaf="#9aa21c",
        leaf_dark="#3e4a10",
        stroke="#b6b028",
        vein="#cfd04a",
    ),
    "navy": replace(
        TRADITIONAL, ground="#17233f", petal_shade="#c9d3ea", leaf="#9a9a22"
    ),
    "cream": replace(
        TRADITIONAL,
        ground="#eadfc2",
        petal_shade="#5d79b8",
        dot="#3f5a15",
        leaf="#8c8a18",
        leaf_dark="#3c470b",
        stroke="#6f7a16",
        vein="#c6bb3e",
    ),
    "blue": replace(
        TRADITIONAL, ground="#1f4f6e", petal_shade="#f3c94a", disc="#c7271c"
    ),
}

# ---------------------------------------------------------------------------------------------
# Brush geometry: pure functions returning point lists and SVG path data
# ---------------------------------------------------------------------------------------------


def fmt(v: float) -> str:
    s = f"{v:.1f}"
    return "0" if s in ("0.0", "-0.0") else s.rstrip("0").rstrip(".")


def smooth_path(pts: list[Pt], closed: bool = True) -> str:
    """Catmull-Rom through `pts`, written as cubic Béziers so the outline stays a true curve."""
    n = len(pts)
    if n < 3:
        raise ValueError("a smooth path needs at least three points")
    out = [f"M{fmt(pts[0][0])} {fmt(pts[0][1])}"]
    last = n if closed else n - 1
    for i in range(last):
        p0 = pts[(i - 1) % n] if closed or i > 0 else pts[0]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n] if closed or i + 2 < n else pts[-1]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        out.append(
            f"C{fmt(c1[0])} {fmt(c1[1])} {fmt(c2[0])} {fmt(c2[1])} {fmt(p2[0])} {fmt(p2[1])}"
        )
    if closed:
        out.append("Z")
    return "".join(out)


def ribbon(center: list[Pt], half: list[float]) -> list[Pt]:
    """Outline of a brush mark: the centreline offset by its half-width on both sides."""
    if len(center) != len(half) or len(center) < 3:
        raise ValueError("ribbon needs matching centre and width samples")
    n = len(center)
    left, right = [], []
    for i, (c, w) in enumerate(zip(center, half)):
        a, b = center[max(i - 1, 0)], center[min(i + 1, n - 1)]
        tx, ty = b[0] - a[0], b[1] - a[1]
        d = math.hypot(tx, ty) or 1.0
        nx, ny = -ty / d, tx / d
        left.append((c[0] + nx * w, c[1] + ny * w))
        right.append((c[0] - nx * w, c[1] - ny * w))
    pts = left + right[::-1]
    # Drop coincident points at pointed ends; Catmull-Rom would kink on them.
    clean = [pts[0]]
    for p in pts[1:]:
        if math.dist(p, clean[-1]) > 0.05:
            clean.append(p)
    return clean


def quad_curve(p0: Pt, p1: Pt, bend: float, n: int) -> list[Pt]:
    """Samples of a quadratic curve from p0 to p1 whose middle bows sideways by bend × length."""
    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    c = (mx - dy * bend * 2, my + dx * bend * 2)
    out = []
    for i in range(n):
        t = i / (n - 1)
        u = 1 - t
        out.append(
            (
                u * u * p0[0] + 2 * u * t * c[0] + t * t * p1[0],
                u * u * p0[1] + 2 * u * t * c[1] + t * t * p1[1],
            )
        )
    return out


def comma(
    p0: Pt, p1: Pt, width: float, bend: float = 0.0, head: float = 0.25, n: int = 22
):
    """The canal painter's comma: a round head at p0 thinning to a point at p1.

    `head` is where along the stroke the full width is reached; before it the head is rounded
    off with a quarter-ellipse profile so the start reads as a pressed-down brush tip."""
    center = quad_curve(p0, p1, bend, n)
    half = []
    for i in range(n):
        t = i / (n - 1)
        if t < head:
            w = math.sqrt(max(0.0, 1 - ((head - t) / head) ** 2))
        else:
            w = ((1 - t) / (1 - head)) ** 1.15
        half.append(width / 2 * w)
    return ribbon(center, half)


def leaf_stroke(
    p0: Pt, p1: Pt, width: float, bend: float = 0.0, swell: float = 0.4, n: int = 22
):
    """A slender leaf: pointed at both ends, fullest at `swell` along its length."""
    center = quad_curve(p0, p1, bend, n)
    half = []
    for i in range(n):
        t = i / (n - 1)
        w = (t / swell) if t < swell else (1 - t) / (1 - swell)
        half.append(width / 2 * math.sin(w * math.pi / 2) ** 0.8)
    return ribbon(center, half)


def arc_stroke(
    c: Pt,
    r: float,
    a0: float,
    a1: float,
    width: float,
    taper: tuple[float, float] = (1.0, 1.0),
    squash: float = 1.0,
    n: int = 26,
):
    """A C-stroke: a brush mark along an arc (degrees, SVG y-down), swelling in the middle.

    `taper` sets how sharply each end thins (1 = to a point, 0 = blunt); `squash` flattens
    the arc vertically, as petals seen at an angle."""
    center = []
    half = []
    for i in range(n):
        t = i / (n - 1)
        a = math.radians(a0 + (a1 - a0) * t)
        center.append((c[0] + r * math.cos(a), c[1] + r * squash * math.sin(a)))
        s = math.sin(math.pi * t) ** 0.7
        end = taper[0] if t < 0.5 else taper[1]
        half.append(width / 2 * ((1 - end) + end * s))
    return ribbon(center, half)


def blob(
    c: Pt,
    rx: float,
    ry: float,
    rng: random.Random,
    wobble: float = 0.06,
    n: int = 18,
    rot=0.0,
):
    """An irregular, hand-painted ellipse."""
    phase = [rng.uniform(0, math.tau) for _ in range(3)]
    amp = [rng.uniform(0.3, 1.0) * wobble for _ in range(3)]
    out = []
    cr, sr = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    for i in range(n):
        a = math.tau * i / n
        k = 1 + sum(
            m * math.sin((j + 2) * a + p) for j, (m, p) in enumerate(zip(amp, phase))
        )
        x, y = rx * k * math.cos(a), ry * k * math.sin(a)
        out.append((c[0] + x * cr - y * sr, c[1] + x * sr + y * cr))
    return out


def scallop(
    c: Pt, r: float, lobes: int, rng: random.Random, depth=0.13, squash=1.0, n=96
):
    """A rose silhouette: a circle whose rim is a row of rounded petal lobes with notches."""
    phase = rng.uniform(0, math.tau)
    size = [rng.uniform(0.85, 1.15) for _ in range(lobes)]
    out = []
    for i in range(n):
        a = math.tau * i / n
        u = (a + phase) * lobes / math.tau
        k = int(u) % lobes
        hump = math.sin(math.pi * (u % 1.0)) ** 0.45
        rr = r * (1 - depth + depth * hump * size[k])
        out.append((c[0] + rr * math.cos(a), c[1] + rr * squash * math.sin(a)))
    return out


# ---------------------------------------------------------------------------------------------
# Motifs: each returns SVG elements in its local frame
# ---------------------------------------------------------------------------------------------


@dataclass
class Shape:
    d: str
    fill: str
    role: str = ""

    def svg(self) -> str:
        cls = f' class="{self.role}"' if self.role else ""
        return f'<path{cls} fill="{self.fill}" d="{self.d}"/>'


def rose(tone: RoseTone, rng: random.Random) -> list[Shape]:
    """A cabbage rose of radius 100, seen a little from the front.

    Painted in the canal order: a scalloped silhouette in the rim colour with the body inside
    it; C-stroke petals hugging the rim; the dark cup round an off-centre eye; a spiral of
    petals wound from the rim in to the eye, each turn smaller, drifting toward the eye and
    edged with a calligraphic hairline; then a few smiling front petals across the bottom."""
    out: list[Shape] = []
    ex, ey = rng.uniform(-16, 16), rng.uniform(-28, -14)
    lobes = rng.randint(7, 9)
    squash = rng.uniform(0.78, 0.88)

    def add(pts, colour, role):
        out.append(Shape(smooth_path(pts), colour, role))

    add(scallop((0, 0), 100, lobes, rng, squash=0.95), tone.rim, "rim")
    add(
        scallop((ex * 0.3, 6), 80, lobes, rng, depth=0.08, squash=0.9),
        tone.body,
        "body",
    )

    start = rng.uniform(0, 360)
    count = rng.randint(6, 7)
    for k in range(count):
        a = start + k * 360 / count + rng.uniform(-14, 14)
        span = rng.uniform(48, 72)
        r = rng.uniform(76, 86)
        w = rng.uniform(15, 22)
        colour = tone.rim if rng.random() < 0.7 else tone.highlight
        add(
            arc_stroke(
                (ex * 0.25, 4), r, a - span / 2, a + span / 2, w, (0.9, 0.9), 0.93
            ),
            colour,
            "petal",
        )

    add(blob((ex, ey + 4), 50, 38, rng, 0.08), tone.cup, "cup")

    turns = rng.randint(5, 6)
    a = rng.uniform(0, 360)
    for k in range(turns):
        t = k / (turns - 1)
        r = 68 * (1 - t) ** 1.1 + 11 * t
        cx = ex * (0.3 + 0.7 * t) + rng.uniform(-3, 3)
        cy = 6 * (1 - t) + ey * t + rng.uniform(-3, 3)
        span = rng.uniform(130, 190)
        w = r * rng.uniform(0.34, 0.46)
        colour = (tone.rim, tone.mid, tone.highlight, tone.mid, tone.rim, tone.mid)[k]
        if k == turns - 1:
            colour = tone.cup
        add(
            arc_stroke((cx, cy), r, a, a + span, w, (0.55, 1.0), squash),
            colour,
            "spiral",
        )
        if 0 < k < turns - 1 and rng.random() < 0.85:
            off = r - w * 0.42
            seg = rng.uniform(0.55, 0.85)
            add(
                arc_stroke(
                    (cx, cy),
                    off,
                    a + span * 0.08,
                    a + span * seg,
                    max(2.6, w * 0.24),
                    (1, 1),
                    squash,
                ),
                tone.accent,
                "accent",
            )
        if k < turns - 2 and rng.random() < 0.6:
            add(
                arc_stroke(
                    (cx, cy),
                    r + w * 0.28,
                    a + span * 0.35,
                    a + span * 0.8,
                    w * 0.22,
                    (1, 1),
                    squash,
                ),
                tone.highlight,
                "highlight",
            )
        a += rng.uniform(105, 140)
    add(blob((ex, ey), 6.5, 5.5, rng, 0.1, 10), tone.accent, "eye")

    # Front petals: a couple of loose smiles across the bottom, not concentric with each other.
    for k in range(rng.randint(2, 3)):
        cx = ex * 0.4 + rng.uniform(-18, 18)
        cy = ey + rng.uniform(0, 14)
        r = rng.uniform(62, 74)
        mid = 90 + rng.uniform(-35, 35)
        half = rng.uniform(28, 42)
        w = rng.uniform(13, 18)
        colour = tone.rim if k % 2 == 0 else tone.highlight
        add(
            arc_stroke((cx, cy), r, mid - half, mid + half, w, (1, 1), squash),
            colour,
            "front",
        )
        if rng.random() < 0.8:
            add(
                arc_stroke(
                    (cx, cy),
                    r - w * 0.45,
                    mid - half * 0.8,
                    mid + half * 0.7,
                    max(2.4, w * 0.2),
                    (1, 1),
                    squash,
                ),
                tone.accent,
                "accent",
            )

    # The long calligraphic sweep that wraps the front of the cup, as on the reference's pink
    # and yellow roses: thin at the eye, swelling round the bottom, flicking out at the side.
    side = rng.choice((-1, 1))
    a0 = 90 - side * rng.uniform(70, 100)
    a1 = 90 + side * rng.uniform(30, 60)
    c = (ex * 0.7, ey + rng.uniform(10, 18))
    add(
        arc_stroke(
            c,
            rng.uniform(40, 50),
            min(a0, a1),
            max(a0, a1),
            rng.uniform(5.5, 7.5),
            (1, 1),
            squash,
        ),
        tone.accent,
        "sweep",
    )
    return out


def daisy(pal: Palette, rng: random.Random) -> list[Shape]:
    """A five-petal canal daisy of radius 100: white petals shaded blue at the heart, a black
    eye, an orange-red disc and a yellow spark."""
    out: list[Shape] = []
    rot = rng.uniform(0, 72)
    petals = []
    for i in range(5):
        a = rot + i * 72 + rng.uniform(-7, 7)
        ar = math.radians(a)
        dist = rng.uniform(58, 64)
        c = (dist * math.cos(ar), dist * math.sin(ar))
        petals.append((a, ar, c))
        pts = blob(c, rng.uniform(38, 43), rng.uniform(34, 39), rng, 0.05, 16, a)
        out.append(Shape(smooth_path(pts), pal.petal, "petal"))
    for a, ar, _ in petals:
        side = math.radians(a - 90)
        c = (
            48 * math.cos(ar) + 9 * math.cos(side),
            48 * math.sin(ar) + 9 * math.sin(side),
        )
        pts = blob(c, 24, 19, rng, 0.06, 14, a + 20)
        out.append(Shape(smooth_path(pts), pal.petal_shade, "shade"))
    out.append(Shape(smooth_path(blob((0, 0), 52, 48, rng, 0.1, 20)), pal.eye, "eye"))
    disc = (rng.uniform(-4, 6), rng.uniform(-2, 6))
    out.append(Shape(smooth_path(blob(disc, 33, 29, rng, 0.06, 16)), pal.disc, "disc"))
    sp = (disc[0] - 2, disc[1] - 8)
    pts = comma((sp[0] - 12, sp[1] + 2), (sp[0] + 14, sp[1] - 6), 11, 0.12, 0.3)
    out.append(Shape(smooth_path(pts), pal.spark, "spark"))
    return out


def leaf(pal: Palette, rng: random.Random, dots: int = 1) -> list[Shape]:
    """A round olive leaf, stem at the origin and tip at (100, 0): a dark heart at the stem
    with pale veins fanned through it, and a row of white dots around the tip, leaning to the
    upper edge for `dots` = 1, the lower for -1, or none for 0."""
    out: list[Shape] = []
    width = rng.uniform(40, 46)
    lean = rng.uniform(-0.08, 0.08)

    def profile(u: float) -> float:
        # Round at the stem and nearly round at the tip: the canal leaf is close to a disc.
        return u**0.5 * (1 - u) ** 0.62 / 0.47

    def outline(scale: float, n: int = 30) -> tuple[list[Pt], list[Pt]]:
        upper, lower = [], []
        for i in range(n):
            u = i / (n - 1)
            x = 100 * u * scale
            h = width * scale * profile(u)
            bow = lean * 20 * math.sin(math.pi * u)
            upper.append((x, -h * (1 + lean) + bow))
            lower.append((x, h * (1 - lean) + bow))
        return upper, lower

    upper, lower = outline(1.0)
    ring = upper + lower[::-1][1:-1]
    out.append(Shape(smooth_path(ring), pal.leaf, "leaf"))

    du, dl = outline(0.66)
    dark = [(x + 3, y * 1.05) for x, y in du + dl[::-1][1:-1]]
    out.append(Shape(smooth_path(dark), pal.leaf_dark, "leaf-dark"))

    fan = rng.randint(5, 7)
    for k in range(fan):
        t = k / (fan - 1)
        a = math.radians(-58 + 116 * t + rng.uniform(-5, 5))
        length = rng.uniform(46, 62) * (1 - 0.3 * abs(t - 0.5))
        tip = (4 + length * math.cos(a), length * math.sin(a) * 0.95)
        pts = leaf_stroke(
            (4, 0), tip, rng.uniform(3.6, 4.8), rng.uniform(-0.07, 0.07), 0.72
        )
        out.append(Shape(smooth_path(pts), pal.vein, "vein"))

    if dots:
        n = len(upper)
        # Walk the outline from part-way along one edge, round the tip, a little down the other.
        lead = upper if dots > 0 else lower
        trail = lower if dots > 0 else upper
        path = lead[int(n * 0.28) : n - 1] + trail[::-1][1 : int(n * 0.3)]
        step = 2
        for x, y in path[::step]:
            r = rng.uniform(3.4, 4.3)
            cx, cy = x * 1.02 + 1, y * 1.05
            out.append(
                Shape(smooth_path(blob((cx, cy), r, r * 0.92, rng, 0.05, 8)), pal.dot)
            )
    return out


# ---------------------------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Flower:
    kind: str  # "rose" or "daisy"
    x: float
    y: float
    r: float
    tone: str = ""  # rose tone key
    angle: float = 0.0


@dataclass(frozen=True)
class Leaf:
    base: Pt
    tip: Pt
    dots: int = 1


@dataclass(frozen=True)
class Stroke:
    base: Pt
    tip: Pt
    width: float
    bend: float = 0.0


@dataclass
class Layout:
    width: float
    height: float
    flowers: list[Flower] = field(default_factory=list)
    leaves: list[Leaf] = field(default_factory=list)
    strokes: list[Stroke] = field(default_factory=list)


@dataclass(frozen=True)
class Spec:
    name: str
    layout: str = "reference"  # one of LAYOUTS
    palette: str = "green"
    tones: tuple[str, ...] = ("red", "pink", "yellow")
    roses: int = 6
    daisies: int = (
        3  # per end (spray, column, border) or in total (posy, corner, crescent)
    )
    width: float = 900
    height: float = 380
    seed: int = 1
    description: str = ""


LAYOUTS = (
    "reference",
    "spray",
    "column",
    "posy",
    "border",
    "corner",
    "crescent",
    "single",
)


def reference_layout() -> Layout:
    """The user's panel, measured on its 902 × 379 photo; the corner number is left out."""
    L = Layout(900, 380)
    L.flowers = [
        Flower("rose", 305, 183, 63, "red", 0),
        Flower("rose", 620, 186, 61, "red", 0),
        Flower("rose", 408, 146, 55, "pink", 0),
        Flower("rose", 517, 136, 54, "yellow", 0),
        Flower("rose", 412, 246, 56, "yellow", 0),
        Flower("rose", 523, 242, 56, "pink", 0),
        Flower("daisy", 108, 198, 30),
        Flower("daisy", 185, 146, 31),
        Flower("daisy", 196, 220, 30),
        Flower("daisy", 715, 152, 30),
        Flower("daisy", 782, 192, 32),
        Flower("daisy", 712, 228, 30),
    ]
    L.leaves = [
        Leaf((264, 146), (219, 98)),
        Leaf((372, 128), (320, 76)),
        Leaf((489, 103), (449, 48)),
        Leaf((560, 130), (603, 86), -1),
        Leaf((657, 141), (693, 110), -1),
        Leaf((742, 171), (785, 146), -1),
        Leaf((722, 207), (758, 246), -1),
        Leaf((152, 166), (108, 139)),
        Leaf((178, 214), (145, 238), -1),
        Leaf((272, 222), (226, 262), -1),
        Leaf((366, 252), (327, 310), -1),
        Leaf((466, 284), (452, 336), -1),
        Leaf((584, 244), (623, 300), -1),
    ]
    S = Stroke
    L.strokes = [
        # left trail
        S((95, 200), (16, 190), 9, 0.04),
        S((92, 208), (12, 218), 8, -0.05),
        S((100, 186), (62, 148), 8, 0.08),
        S((104, 214), (110, 250), 7, -0.08),
        S((168, 136), (118, 114), 8, 0.06),
        S((182, 128), (178, 84), 8, -0.1),
        S((200, 232), (205, 280), 8, 0.08),
        S((130, 200), (260, 190), 4, -0.04),
        S((210, 158), (262, 176), 4, 0.05),
        S((215, 222), (262, 206), 4, -0.05),
        S((262, 120), (250, 78), 8, 0.1),
        S((282, 116), (296, 68), 8, -0.08),
        S((330, 92), (352, 50), 8, 0.1),
        S((395, 92), (438, 30), 8, -0.08),
        S((280, 250), (288, 320), 8, 0.08),
        S((340, 290), (358, 345), 7, -0.06),
        S((420, 300), (445, 362), 8, 0.06),
        # right trail
        S((525, 82), (492, 27), 8, 0.08),
        S((540, 82), (582, 54), 8, -0.08),
        S((628, 104), (607, 74), 7, 0.1),
        S((650, 118), (662, 92), 6, -0.08),
        S((700, 128), (707, 92), 7, 0.1),
        S((735, 150), (788, 133), 7, -0.06),
        S((805, 178), (834, 146), 8, 0.08),
        S((808, 196), (888, 196), 9, -0.04),
        S((800, 210), (842, 232), 8, 0.06),
        S((782, 220), (784, 252), 7, -0.08),
        S((672, 186), (762, 190), 4, 0.03),
        S((672, 206), (700, 214), 4, -0.05),
        S((722, 250), (748, 262), 7, 0.06),
        S((660, 240), (677, 302), 8, -0.08),
        S((646, 248), (660, 296), 7, 0.08),
        S((560, 290), (612, 320), 8, -0.06),
        S((520, 300), (512, 356), 8, 0.08),
        S((540, 296), (572, 332), 7, -0.08),
    ]
    return L


def _outward(cx: float, cy: float, x: float, y: float) -> float:
    return math.atan2(y - cy, x - cx)


def _polar(p: Pt, a: float, d: float) -> Pt:
    return (p[0] + d * math.cos(a), p[1] + d * math.sin(a))


def _dress(
    L: Layout, rng: random.Random, centre: Pt, leaves_per_rose=(2, 3), sprays=(2, 4)
):
    """Tuck leaves behind each flower and fan leaf strokes outward, away from `centre`."""
    roses = [f for f in L.flowers if f.kind == "rose"]
    daisies = [f for f in L.flowers if f.kind == "daisy"]
    for f in roses:
        out = _outward(*centre, f.x, f.y)
        for _ in range(rng.randint(*leaves_per_rose)):
            a = out + rng.uniform(-1.3, 1.3)
            base = _polar((f.x, f.y), a, f.r * 0.55)
            tip = _polar(
                base, a + rng.uniform(-0.3, 0.3), f.r * rng.uniform(0.95, 1.15)
            )
            dots = 1 if math.sin(a - out) < 0 else -1
            L.leaves.append(Leaf(base, tip, dots))
        for _ in range(rng.randint(1, 2)):
            a = out + rng.uniform(-1.1, 1.1)
            base = _polar((f.x, f.y), a, f.r * 0.9)
            tip = _polar(
                base, a + rng.uniform(-0.25, 0.25), f.r * rng.uniform(0.7, 1.0)
            )
            L.strokes.append(Stroke(base, tip, f.r * 0.13, rng.uniform(-0.1, 0.1)))
    for f in daisies:
        out = _outward(*centre, f.x, f.y)
        if rng.random() < 0.7:
            a = out + rng.choice((-1, 1)) * rng.uniform(0.9, 1.6)
            base = _polar((f.x, f.y), a, f.r * 0.6)
            tip = _polar(base, a, f.r * rng.uniform(1.4, 1.7))
            L.leaves.append(Leaf(base, tip, 1 if math.sin(a - out) < 0 else -1))
        for _ in range(rng.randint(*sprays)):
            a = out + rng.uniform(-1.2, 1.2)
            base = _polar((f.x, f.y), a, f.r * 0.9)
            tip = _polar(base, a + rng.uniform(-0.2, 0.2), f.r * rng.uniform(1.3, 2.4))
            L.strokes.append(Stroke(base, tip, f.r * 0.26, rng.uniform(-0.1, 0.1)))
        # A fine stem back toward the middle ties each daisy into the spray.
        a = out + math.pi + rng.uniform(-0.3, 0.3)
        tip = _polar((f.x, f.y), a, f.r * rng.uniform(1.6, 2.4))
        L.strokes.append(Stroke((f.x, f.y), tip, f.r * 0.13, rng.uniform(-0.06, 0.06)))


def _rose_tone(spec: Spec, i: int, rng: random.Random) -> str:
    return (
        spec.tones[i % len(spec.tones)]
        if rng.random() < 0.75
        else rng.choice(spec.tones)
    )


def spray_layout(spec: Spec, rng: random.Random, girth: float = 0.17) -> Layout:
    """A horizontal oval spray like the reference: a tight zigzag of roses between two end
    roses in the first tone, with daisies trailing off both ends. Sized to fit the canvas;
    `girth` caps the rose radius as a fraction of the canvas height."""
    W, H = spec.width, spec.height
    L = Layout(W, H)
    cx, cy = W / 2, H / 2
    n = spec.roses
    inner = max(0, n - 2)
    step = 1.1  # zigzag pitch, in rose radii
    half_inner = step * (inner - 1) / 2 if inner else 0.0
    end_x = half_inner + (1.75 if inner else (0.9 if n == 2 else 0.0))
    # Half-width in rose radii: inner roses, end rose, then the daisy trail and its strokes.
    half = end_x + 1.9 + 1.1 * max(0, spec.daisies - 1) + 1.2
    r = min(H * girth, W * 0.48 / half)
    tones_in = spec.tones[1:] or spec.tones
    for i in range(inner):
        x = cx + (-half_inner + step * i) * r
        lift = -0.45 if inner == 1 else (-1) ** i * 0.85
        y = cy + lift * r + rng.uniform(-3, 3)
        L.flowers.append(Flower("rose", x, y, r, tones_in[i % len(tones_in)]))
    ends = (-1, 1) if n >= 2 else (0,)
    for side in ends:
        # A pair alone gets two tones; with a middle, both ends match like the reference.
        tone = spec.tones[1 % len(spec.tones)] if n == 2 and side > 0 else spec.tones[0]
        L.flowers.append(Flower("rose", cx + side * end_x * r, cy, r * 1.1, tone))
    dr = r * 0.52
    for side in (-1, 1):
        for k in range(spec.daisies):
            along = (end_x + 1.9 + 1.1 * k) * r
            lift = 0 if k == 0 else (dr * 1.6 if k % 2 else -dr * 1.5)
            y = cy + lift + rng.uniform(-5, 5)
            L.flowers.append(
                Flower("daisy", cx + side * along, y, dr * rng.uniform(0.95, 1.08))
            )
    _dress(L, rng, (cx, cy))
    return L


def column_layout(spec: Spec, rng: random.Random) -> Layout:
    """A spray turned on end for tall panels such as cabin doors: built as a spray on the
    transposed canvas, then reflected across the diagonal. Leaf dots and stroke bends flip
    with the reflection so every mark keeps its painted shape."""
    # A door leaf is narrow, so the roses may fill more of its width than a spray's height.
    flat = spray_layout(replace(spec, width=spec.height, height=spec.width), rng, 0.23)

    def t(p: Pt) -> Pt:
        return (p[1], p[0])

    L = Layout(spec.width, spec.height)
    L.flowers = [replace(f, x=f.y, y=f.x) for f in flat.flowers]
    L.leaves = [Leaf(t(lf.base), t(lf.tip), -lf.dots) for lf in flat.leaves]
    L.strokes = [Stroke(t(s.base), t(s.tip), s.width, -s.bend) for s in flat.strokes]
    return L


def posy_layout(spec: Spec, rng: random.Random) -> Layout:
    """A round posy for a round panel or a water-can lid: one rose in the middle, a ring of
    roses round it, and daisies and leaf strokes radiating to the rim."""
    W = H = min(spec.width, spec.height)
    L = Layout(W, H)
    cx, cy = W / 2, H / 2
    r = W * 0.13
    ring = spec.roses - 1
    L.flowers.append(Flower("rose", cx, cy, r * 1.08, spec.tones[0]))
    for i in range(ring):
        a = math.tau * i / ring + rng.uniform(-0.1, 0.1) - math.pi / 2
        pos = _polar((cx, cy), a, r * 1.75)
        L.flowers.append(Flower("rose", *pos, r * 0.92, _rose_tone(spec, i + 1, rng)))
    for i in range(spec.daisies):
        a = math.tau * (i + 0.5) / spec.daisies + rng.uniform(-0.15, 0.15) - math.pi / 2
        L.flowers.append(Flower("daisy", *_polar((cx, cy), a, r * 3.0), r * 0.48))
    _dress(L, rng, (cx, cy), (1, 2), (2, 3))
    return L


def border_layout(spec: Spec, rng: random.Random) -> Layout:
    """A long frieze for a cabin side: roses alternating above and below a wandering stem,
    with daisies between them; leaves and strokes fan away from the stem."""
    W, H = spec.width, spec.height
    L = Layout(W, H)
    cy = H / 2
    r = H * 0.27
    gap = W / spec.roses
    for i in range(spec.roses):
        x = gap * (i + 0.5)
        y = cy + (-1) ** i * r * 0.25
        L.flowers.append(Flower("rose", x, y, r, _rose_tone(spec, i, rng)))
        if i == spec.roses - 1:
            break
        for k in range(spec.daisies):
            t = (k + 1) / (spec.daisies + 1)
            dy = cy + (-1) ** (i + k) * r * 0.7
            L.flowers.append(Flower("daisy", x + gap * t, dy, r * 0.45))
    for f in list(L.flowers):
        # Dress each flower on its own so its foliage fans away from the stem line.
        away = (f.x, cy + H) if f.y < cy else (f.x, cy - H)
        sub = Layout(W, H, [f])
        _dress(sub, rng, away, (2, 2), (1, 2))
        L.leaves += sub.leaves
        L.strokes += [
            s for s in sub.strokes if s.width > f.r * 0.15 or f.kind == "rose"
        ]
    stem = sorted(L.flowers, key=lambda f: f.x)
    for a, b in pairwise(stem):
        L.strokes.append(
            Stroke((a.x, a.y), (b.x, b.y), r * 0.09, rng.uniform(-0.12, 0.12))
        )
    return L


def corner_layout(spec: Spec, rng: random.Random) -> Layout:
    """A corner spray for the top-left of a panel: one rose near the corner, the rest in an
    arc round it kept clear of both edges, and daisies fanning out between them."""
    W, H = spec.width, spec.height
    L = Layout(W, H)
    r = min(W, H) * 0.13
    origin = (r * 0.35, r * 0.35)
    L.flowers.append(
        Flower(
            "rose", *_polar(origin, math.radians(45), r * 1.55), r * 1.12, spec.tones[0]
        )
    )
    rest = spec.roses - 1
    for i in range(rest):
        t = (i + 0.5) / rest
        a = math.radians(16 + 58 * t + rng.uniform(-3, 3))
        pos = _polar(origin, a, r * 3.0)
        L.flowers.append(Flower("rose", *pos, r * 0.96, _rose_tone(spec, i + 1, rng)))
    for i in range(spec.daisies):
        t = i / max(1, spec.daisies - 1)
        a = math.radians(14 + t * 62)
        d = r * (4.3 + 0.7 * (i % 2))
        L.flowers.append(Flower("daisy", *_polar(origin, a, d), r * 0.46))
    _dress(L, rng, origin, (2, 3), (2, 3))
    return L


def crescent_layout(spec: Spec, rng: random.Random) -> Layout:
    """A crescent swag: roses shoulder to shoulder along a shallow downward arc, daisies
    trailing up off the horns."""
    W, H = spec.width, spec.height
    L = Layout(W, H)
    n = spec.roses
    R = W * 0.4
    sweep = math.radians(76)
    centre = (W / 2, H * 0.62 - R)
    r = min(H * 0.19, R * sweep / max(1, n - 1) / 1.7 if n > 1 else H * 0.19)
    for i in range(n):
        t = i / (n - 1) - 0.5 if n > 1 else 0
        a = math.pi / 2 + t * sweep
        pos = _polar(centre, a, R + (-1) ** i * r * 0.12)
        L.flowers.append(Flower("rose", *pos, r, _rose_tone(spec, i, rng)))
    for side in (-1, 1):
        for k in range(spec.daisies):
            dr = R + (-1) ** k * r * (0.35 + 0.45 * (k // 2))
            off = sweep / 2 + r * (1.7 + 1.0 * k) / R
            # Keep the trail on the canvas: stop climbing once a daisy would touch the top.
            off = min(off, math.acos(max(-1.0, min(1.0, (r * 0.85 - centre[1]) / dr))))
            a = math.pi / 2 + side * off
            L.flowers.append(Flower("daisy", *_polar(centre, a, dr), r * 0.5))
    _dress(L, rng, (W / 2, centre[1]), (2, 3), (2, 3))
    return L


def single_layout(spec: Spec, rng: random.Random) -> Layout:
    """One rose with its leaves and a daisy either side, for small panels, stools and cans."""
    W, H = spec.width, spec.height
    L = Layout(W, H)
    cx, cy = W / 2, H / 2
    r = min(W, H) * 0.27
    L.flowers.append(Flower("rose", cx, cy, r, spec.tones[0]))
    for side in (-1, 1):
        L.flowers.append(Flower("daisy", cx + side * r * 1.85, cy + r * 0.25, r * 0.4))
    for a in (-2.75, -1.75, -0.4, 0.7, 2.45):
        a += rng.uniform(-0.1, 0.1)
        base = _polar((cx, cy), a, r * 0.55)
        tip = _polar(base, a, r * rng.uniform(0.8, 0.95))
        L.leaves.append(Leaf(base, tip, 1 if math.sin(a) < 0 else -1))
    for a in (-2.2, -1.2, -0.95, 1.6, 1.85):
        base = _polar((cx, cy), a, r * 0.9)
        L.strokes.append(
            Stroke(base, _polar(base, a, r * rng.uniform(0.5, 0.8)), r * 0.11)
        )
    for f in L.flowers[1:]:
        out = math.atan2(f.y - cy, f.x - cx)
        for _ in range(3):
            a = out + rng.uniform(-1.0, 1.0)
            base = _polar((f.x, f.y), a, f.r * 0.9)
            tip = _polar(base, a, f.r * rng.uniform(1.2, 1.9))
            L.strokes.append(Stroke(base, tip, f.r * 0.26, rng.uniform(-0.1, 0.1)))
        L.strokes.append(
            Stroke((f.x, f.y), (cx, cy), f.r * 0.12, rng.uniform(-0.08, 0.08))
        )
    return L


BUILDERS = {
    "spray": spray_layout,
    "column": column_layout,
    "posy": posy_layout,
    "border": border_layout,
    "corner": corner_layout,
    "crescent": crescent_layout,
    "single": single_layout,
}


def validate(spec: Spec) -> list[str]:
    errors = []
    if spec.layout not in LAYOUTS:
        errors.append(f"unknown layout {spec.layout!r}")
    if spec.palette not in PALETTES:
        errors.append(f"unknown palette {spec.palette!r}")
    if not spec.tones:
        errors.append("at least one rose tone is needed")
    errors += [f"unknown rose tone {t!r}" for t in spec.tones if t not in ROSE_TONES]
    if not 1 <= spec.roses <= 12:
        errors.append("roses must be 1-12")
    if spec.layout == "posy" and spec.roses < 3:
        errors.append("a posy needs at least 3 roses")
    if spec.layout == "single" and spec.roses != 1:
        errors.append("a single layout has exactly 1 rose")
    if not 0 <= spec.daisies <= 8:
        errors.append("daisies must be 0-8")
    if spec.width < 100 or spec.height < 100:
        errors.append("canvas must be at least 100 × 100")
    return errors


def compose(spec: Spec) -> Layout:
    errors = validate(spec)
    if errors:
        raise ValueError(f"{spec.name}: " + "; ".join(errors))
    if spec.layout == "reference":
        return reference_layout()
    return BUILDERS[spec.layout](spec, random.Random(spec.seed))


# ---------------------------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------------------------


def _group(
    gid: str, x: float, y: float, rot: float, scale: float, shapes: list[Shape]
) -> str:
    t = f"translate({fmt(x)} {fmt(y)}) rotate({fmt(rot)}) scale({scale:.4f})"
    body = "".join(s.svg() for s in shapes)
    return f'<g id="{gid}" transform="{t}">{body}</g>'


def view_box(spec: Spec, fit: bool = False) -> tuple[float, float, float, float]:
    """(x, y, width, height) of the drawing: the whole canvas, or with `fit` the painted
    extent plus a 2% transparent margin, so a decal texture never cuts a leaf off."""
    L = compose(spec)
    if not fit:
        return 0.0, 0.0, L.width, L.height
    x0, y0, x1, y1 = extent(L)
    pad = 0.02 * max(x1 - x0, y1 - y0)
    return x0 - pad, y0 - pad, x1 - x0 + 2 * pad, y1 - y0 + 2 * pad


def render_svg(spec: Spec, background: bool = True, fit: bool = False) -> str:
    """The panel as SVG. Without `background` the ground rect is left out; `fit` crops the
    view box to the painted extent (see `view_box`)."""
    pal = PALETTES[spec.palette]
    L = compose(spec)
    rng = random.Random(spec.seed * 7919 + 13)
    parts: list[str] = []
    if background:
        parts.append(
            f'<rect id="ground" width="{fmt(L.width)}" height="{fmt(L.height)}" '
            f'fill="{pal.ground}"/>'
        )
    strokes = []
    for i, s in enumerate(L.strokes):
        pts = leaf_stroke(s.base, s.tip, s.width, s.bend, rng.uniform(0.3, 0.45))
        strokes.append(f'<path id="stroke-{i}" d="{smooth_path(pts)}"/>')
    parts.append(f'<g id="leaf-strokes" fill="{pal.stroke}">{"".join(strokes)}</g>')

    leaves = []
    for i, lf in enumerate(L.leaves):
        length = math.dist(lf.base, lf.tip)
        rot = math.degrees(math.atan2(lf.tip[1] - lf.base[1], lf.tip[0] - lf.base[0]))
        leaves.append(
            _group(f"leaf-{i}", *lf.base, rot, length / 100, leaf(pal, rng, lf.dots))
        )
    parts.append(f'<g id="leaves">{"".join(leaves)}</g>')

    flowers = []
    # Daisies first, then roses smallest to largest so big blooms sit on top.
    order = sorted(L.flowers, key=lambda f: (f.kind == "rose", f.r))
    for i, f in enumerate(order):
        if f.kind == "rose":
            shapes = rose(pal.roses[f.tone], rng)
            angle = f.angle + rng.uniform(-25, 25)
        else:
            shapes = daisy(pal, rng)
            angle = 0.0
        scale = f.r / 100 * (ROSE_SCALE if f.kind == "rose" else 1.0)
        flowers.append(_group(f"{f.kind}-{i}", f.x, f.y, angle, scale, shapes))
    parts.append(f'<g id="flowers">{"".join(flowers)}</g>')

    title = escape(f"Canal roses: {spec.name}")
    vx, vy, vw, vh = view_box(spec, fit)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{fmt(vx)} {fmt(vy)} {fmt(vw)} {fmt(vh)}" '
        f'width="{fmt(vw * 2)}" height="{fmt(vh * 2)}">'
        f"<title>{title}</title>{''.join(parts)}</svg>\n"
    )


PRESETS = {
    s.name: s
    for s in (
        Spec(
            "reference",
            description="The user's panel redrawn as vectors on its green ground, number omitted.",
        ),
        Spec(
            "reference_black",
            palette="black",
            description="The same panel on the black ground of a traditional cabin-block panel.",
        ),
        Spec(
            "spray_cream",
            layout="spray",
            palette="cream",
            tones=("crimson", "white", "apricot"),
            roses=5,
            seed=11,
            description="A shorter spray of crimson, white and apricot roses on cream.",
        ),
        Spec(
            "spray_blue",
            layout="spray",
            palette="blue",
            tones=("yellow", "white"),
            roses=3,
            daisies=2,
            seed=4,
            description="Three yellow and white roses on a deep canal blue.",
        ),
        Spec(
            "posy_round",
            layout="posy",
            tones=("red", "yellow", "pink"),
            roses=7,
            daisies=6,
            width=640,
            height=640,
            seed=7,
            description="A round posy for a water-can lid or a round panel.",
        ),
        Spec(
            "border_frieze",
            layout="border",
            palette="red",
            tones=("yellow", "white", "blush"),
            roses=6,
            daisies=1,
            width=1800,
            height=300,
            seed=3,
            description="A long frieze for a cabin side, on a coach red ground.",
        ),
        Spec(
            "corner_spray",
            layout="corner",
            palette="black",
            tones=("red", "pink", "yellow"),
            roses=4,
            daisies=4,
            width=640,
            height=640,
            seed=5,
            description="A corner spray for the top-left of a panel; mirror it for the others.",
        ),
        Spec(
            "crescent_swag",
            layout="crescent",
            palette="navy",
            tones=("pink", "yellow", "red"),
            roses=5,
            daisies=2,
            width=900,
            height=480,
            seed=9,
            description="A crescent swag of five roses on navy.",
        ),
        Spec(
            "single_rose",
            layout="single",
            tones=("red",),
            roses=1,
            width=480,
            height=320,
            seed=2,
            description="One red rose with leaves and two daisies, for small panels and stools.",
        ),
        Spec(
            "door_column",
            layout="column",
            palette="black",
            tones=("red", "yellow", "pink"),
            roses=3,
            daisies=2,
            width=400,
            height=1100,
            seed=31,
            description="A tall spray for a cabin door leaf; flip it for the other leaf.",
        ),
    )
}

# The random sampler predates the column layout and leaves it out, so published random
# variants keep their seeds.
RANDOM_SIZES = {
    "spray": (900, 380),
    "posy": (640, 640),
    "border": (1800, 300),
    "corner": (640, 640),
    "crescent": (900, 480),
    "single": (480, 320),
}


def random_spec(seed: int) -> Spec:
    """A valid variation drawn from the preset ranges; the same seed gives the same panel."""
    rng = random.Random(seed)
    layout = rng.choice(list(RANDOM_SIZES))
    w, h = RANDOM_SIZES[layout]
    tones = tuple(rng.sample(list(ROSE_TONES), rng.randint(1, 3)))
    roses = {
        "spray": rng.randint(3, 7),
        "posy": rng.randint(5, 8),
        "border": rng.randint(4, 7),
        "corner": rng.randint(3, 5),
        "crescent": rng.randint(3, 6),
        "single": 1,
    }[layout]
    return Spec(
        f"random_{seed:03d}",
        layout=layout,
        palette=rng.choice(list(PALETTES)),
        tones=tones,
        roses=roses,
        daisies=rng.randint(1, 3)
        if layout in ("spray", "border", "crescent")
        else rng.randint(3, 6),
        width=w,
        height=h,
        seed=seed,
        description=f"Random {layout} (seed {seed}).",
    )


def counts(spec: Spec) -> dict[str, int]:
    L = compose(spec)
    return {
        "roses": sum(f.kind == "rose" for f in L.flowers),
        "daisies": sum(f.kind == "daisy" for f in L.flowers),
        "leaves": len(L.leaves),
        "leaf_strokes": len(L.strokes),
    }


def extent(L: Layout) -> tuple[float, float, float, float]:
    """Approximate painted bounds (x0, y0, x1, y1): bloom discs at their drawn size, leaves
    as discs round their midpoints, and stroke ends. Decals use it to prove nothing is cut
    off at the texture edge."""
    xs: list[float] = []
    ys: list[float] = []

    def disc(x: float, y: float, r: float) -> None:
        xs.extend((x - r, x + r))
        ys.extend((y - r, y + r))

    for f in L.flowers:
        # Daisy petals reach a little past the nominal radius.
        disc(f.x, f.y, f.r * (ROSE_SCALE if f.kind == "rose" else 1.1))
    for lf in L.leaves:
        mx, my = (lf.base[0] + lf.tip[0]) / 2, (lf.base[1] + lf.tip[1]) / 2
        disc(mx, my, math.dist(lf.base, lf.tip) * 0.55)
    for st in L.strokes:
        for x, y in (st.base, st.tip):
            disc(x, y, st.width / 2)
    return min(xs), min(ys), max(xs), max(ys)


def _png(svg: Path, png: Path) -> bool:
    if not shutil.which("rsvg-convert"):
        return False
    subprocess.run(["rsvg-convert", str(svg), "-o", str(png)], check=True)
    return True


def _contact_sheet(pngs: list[Path], out: Path) -> bool:
    if not shutil.which("magick") or not pngs:
        return False
    cmd = ["magick", "montage", *map(str, pngs), "-geometry", "600x400>+12+12"]
    cmd += ["-tile", "3x", "-background", "#2b2b2b", "-label", "", str(out)]
    subprocess.run(cmd, check=True)
    return True


def build(specs: list[Spec], out: Path, reference: Path | None, render: bool) -> dict:
    from procgen_bpy import fresh_staging, publish, sha256, write_json

    staging = fresh_staging(out)
    (staging / "SVG").mkdir()
    (staging / "Previews").mkdir()
    entries = []
    pngs = []
    for spec in specs:
        files = {}
        for suffix, ground in (("", True), ("_transparent", False)):
            path = staging / "SVG" / f"{spec.name}{suffix}.svg"
            path.write_text(render_svg(spec, ground))
            files[f"svg{suffix}"] = {"path": f"SVG/{path.name}", "sha256": sha256(path)}
        if render:
            png = staging / "Previews" / f"{spec.name}.png"
            if _png(staging / "SVG" / f"{spec.name}.svg", png):
                files["preview"] = {"path": f"Previews/{png.name}"}
                pngs.append(png)
        entries.append(
            {
                "name": spec.name,
                "description": spec.description,
                "spec": {
                    k: list(v) if isinstance(v, tuple) else v
                    for k, v in spec.__dict__.items()
                    if k not in ("name", "description")
                },
                "counts": counts(spec),
                "files": files,
            }
        )
    sheet = None
    if render and _contact_sheet(pngs, staging / "Previews" / "contact_sheet.png"):
        sheet = "Previews/contact_sheet.png"
    ref = None
    if reference and reference.is_file():
        (staging / "Reference").mkdir()
        target = staging / "Reference" / f"canal_roses_reference{reference.suffix}"
        shutil.copy2(reference, target)
        ref = {"path": f"Reference/{target.name}", "sha256": sha256(target)}
    manifest = {
        "generator": "src/canal_roses.py",
        "reference": ref,
        "contact_sheet": sheet,
        "variants": entries,
    }
    write_json(staging / "canal_roses_manifest.json", manifest)
    publish(staging, out)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUTPUT_DIR))
    parser.add_argument("--preset", action="append", choices=sorted(PRESETS))
    parser.add_argument(
        "--random", type=int, default=0, help="add this many random variants"
    )
    parser.add_argument(
        "--random-seed", type=int, default=100, help="first random seed"
    )
    parser.add_argument(
        "--reference", type=Path, help="copy the reference photo alongside"
    )
    parser.add_argument("--no-render", action="store_true", help="skip PNG previews")
    args = parser.parse_args()
    specs = [PRESETS[n] for n in (args.preset or PRESETS)]
    specs += [random_spec(args.random_seed + i) for i in range(args.random)]
    manifest = build(specs, Path(args.out), args.reference, not args.no_render)
    for v in manifest["variants"]:
        print(f"{v['name']:<18} {json.dumps(v['counts'])}")


if __name__ == "__main__":
    main()
