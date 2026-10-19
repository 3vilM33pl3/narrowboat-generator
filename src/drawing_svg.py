# SPDX-License-Identifier: MIT
"""A small technical-drawing kit that writes SVG in sheet millimetres (stdlib only).

Conventions follow the ISO drawing standards:
- ISO 5457 sheet: A3 landscape, 20 mm filing margin on the left and 10 mm elsewhere, 0.7 mm
  frame, 50 mm grid reference fields counted from the centring marks.
- ISO 7200 title block, 180 mm wide, in the bottom right corner.
- ISO 128 lines in the 0.5 line group (wide 0.5, narrow 0.25, extra wide 0.7). Dashes are
  multiples of the line width d: dashed 12d/3d, long-dash dotted 24d/3d/0.5d/3d, and the
  double-dotted phantom line.
- ISO 3098 lettering heights (1.8, 2.5, 3.5, 5, 7 mm).
- ISO 129-1 dimensioning: values in millimetres without units, written above the dimension
  line and read from the bottom or the right; chains with oblique strokes, single dimensions
  with filled arrowheads, running dimensions from an origin circle.

A `Sheet` collects elements into named groups and keeps each group's bounding box so a
layout can be tested for overlaps. A `View` maps model millimetres (b along, v up) onto the
sheet at a scale, so every dimension value is computed from the same numbers as the lines.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import pairwise
from xml.sax.saxutils import escape

A3 = (420.0, 297.0)
FRAME = (20.0, 10.0, 410.0, 287.0)  # x0, y0, x1, y1
WIDTHS = {"extra": 0.7, "wide": 0.5, "medium": 0.35, "narrow": 0.25, "fine": 0.18}
TEXT_HEIGHTS = (1.8, 2.5, 3.5, 5.0, 7.0)
FONT = "DejaVu Sans Condensed"
CHAR_W = 0.6  # average advance of the condensed face, as a fraction of the text height
ARROW_L, ARROW_W = 2.5, 0.9  # filled arrowhead length and half-width
TICK = 1.6  # oblique stroke half-length
EXT_GAP, EXT_OVER = (
    1.0,
    1.5,
)  # extension lines: gap from the object, overshoot of the line
INK = "#000"

# Line types: (width key, dash pattern in multiples of the width).
LINE_TYPES = {
    "visible": ("wide", None),
    "cut": ("extra", None),
    "thin": ("narrow", None),
    "fine": ("fine", None),
    "hidden": ("narrow", (12, 3)),
    "centre": ("narrow", (24, 3, 0.5, 3)),
    "phantom": ("narrow", (24, 3, 0.5, 3, 0.5, 3)),
    "frame": ("extra", None),
    "medium": ("medium", None),
}

Pt = tuple[float, float]


def fmt(v: float) -> str:
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def text_width(s: str, h: float) -> float:
    return CHAR_W * h * len(s)


@dataclass
class Sheet:
    """One drawing sheet; elements go into the current group (see `group`)."""

    title: str
    size: Pt = A3
    groups: dict[str, list[str]] = field(default_factory=dict)
    boxes: dict[str, list[float]] = field(default_factory=dict)
    widths: set = field(default_factory=set)
    heights: set = field(default_factory=set)
    defs: list[str] = field(default_factory=list)
    current: str = "frame"
    # (group, text, x0, y0, x1, y1) of every text, for collision tests.
    texts: list[tuple] = field(default_factory=list)

    @contextmanager
    def group(self, name: str):
        prev, self.current = self.current, name
        self.groups.setdefault(name, [])
        try:
            yield self
        finally:
            self.current = prev

    # ------------------------------------------------------------------ primitives

    def _emit(self, element: str, xs, ys) -> None:
        self.groups.setdefault(self.current, []).append(element)
        if xs:
            box = self.boxes.setdefault(
                self.current, [math.inf, math.inf, -math.inf, -math.inf]
            )
            box[0], box[1] = min(box[0], *xs), min(box[1], *ys)
            box[2], box[3] = max(box[2], *xs), max(box[3], *ys)

    def _stroke(self, style: str) -> str:
        key, dash = LINE_TYPES[style]
        w = WIDTHS[key]
        self.widths.add(w)
        attrs = f'stroke="{INK}" stroke-width="{fmt(w)}"'
        if dash:
            attrs += f' stroke-dasharray="{" ".join(fmt(d * w) for d in dash)}"'
        return attrs

    def polyline(self, pts, style="thin", *, closed=False, fill="none") -> None:
        if len(pts) < 2:
            return
        tag = "polygon" if closed else "polyline"
        d = " ".join(f"{fmt(x)},{fmt(y)}" for x, y in pts)
        stroke = self._stroke(style) if style else 'stroke="none"'
        self._emit(
            f'<{tag} points="{d}" fill="{fill}" {stroke} stroke-linejoin="round"/>',
            [p[0] for p in pts],
            [p[1] for p in pts],
        )

    def line(self, p: Pt, q: Pt, style="thin") -> None:
        self.polyline([p, q], style)

    def rect(self, x0, y0, x1, y1, style="thin", fill="none") -> None:
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        self.polyline(
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], style, closed=True, fill=fill
        )

    def circle(self, c: Pt, r: float, style="thin", fill="none") -> None:
        stroke = self._stroke(style) if style else 'stroke="none"'
        self._emit(
            f'<circle cx="{fmt(c[0])}" cy="{fmt(c[1])}" r="{fmt(r)}" fill="{fill}" {stroke}/>',
            [c[0] - r, c[0] + r],
            [c[1] - r, c[1] + r],
        )

    def arc(self, c: Pt, r: float, a0: float, a1: float, style="thin", n=24) -> None:
        """Arc through angles a0..a1 (degrees, counter-clockwise on the sheet)."""
        pts = []
        for k in range(n + 1):
            a = math.radians(a0 + (a1 - a0) * k / n)
            pts.append((c[0] + r * math.cos(a), c[1] - r * math.sin(a)))
        self.polyline(pts, style)

    def text(
        self,
        p: Pt,
        s: str,
        h: float = 2.5,
        *,
        anchor: str = "start",
        rot: float = 0.0,
        bold: bool = False,
        middle: bool = False,
        knockout: bool = False,
    ) -> None:
        """Text with its baseline at p (or its middle, with `middle`); rot in degrees,
        counter-clockwise (90 reads from the right-hand side)."""
        if not s:
            return
        self.heights.add(h)
        x, y = p
        dy = 0.35 * h if middle else 0.0
        w = text_width(s, h)
        off = {"start": 0.0, "middle": -w / 2, "end": -w}[anchor]
        corners = [(off, dy), (off + w, dy), (off, dy - h), (off + w, dy - h)]
        a = math.radians(-rot)
        pts = [
            (
                x + u * math.cos(a) - v * math.sin(a),
                y + u * math.sin(a) + v * math.cos(a),
            )
            for u, v in corners
        ]
        weight = ' font-weight="bold"' if bold else ""
        transform = f' transform="rotate({fmt(-rot)} {fmt(x)} {fmt(y)})"' if rot else ""
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        self.texts.append((self.current, s, min(xs), min(ys), max(xs), max(ys)))
        if knockout:
            # A white ground so lettering stays legible over linework.
            pad = 0.35
            box = (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
            self.groups.setdefault(self.current, []).append(
                f'<rect x="{fmt(box[0])}" y="{fmt(box[1])}" width="{fmt(box[2] - box[0])}" '
                f'height="{fmt(box[3] - box[1])}" fill="#fff"/>'
            )
        self._emit(
            f'<text x="{fmt(x)}" y="{fmt(y + dy)}" font-size="{fmt(h)}" text-anchor="{anchor}"'
            f"{weight}{transform}>{escape(s)}</text>",
            [q[0] for q in pts],
            [q[1] for q in pts],
        )

    def hatch(self, pts, pattern: str) -> None:
        self.polyline(pts, None, closed=True, fill=f"url(#{pattern})")

    def fill(self, pts, colour: str = INK) -> None:
        self.polyline(pts, None, closed=True, fill=colour)

    # ------------------------------------------------------------------ output

    def svg(self, only=None, *, text: bool = True) -> str:
        """The sheet as SVG; `only` limits it to some groups and `text=False` drops the
        lettering (the lines-only overlays keep the sheet's page geometry)."""
        w, h = self.size
        body = []
        for name, items in self.groups.items():
            if only is not None and name not in only:
                continue
            body.append(f'<g id="{escape(name)}">')
            body.extend(e for e in items if text or not e.startswith("<text"))
            body.append("</g>")
        defs = "".join(self.defs + list(PATTERNS.values()))
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{fmt(w)}mm" height="{fmt(h)}mm" '
            f'viewBox="0 0 {fmt(w)} {fmt(h)}" font-family="{FONT}, DejaVu Sans, sans-serif" '
            f'fill="{INK}">\n<title>{escape(self.title)}</title>\n<defs>{defs}</defs>\n'
            f'<rect width="{fmt(w)}" height="{fmt(h)}" fill="#fff"/>\n'
            + "\n".join(body)
            + "\n</svg>\n"
        )


def _pattern(pid: str, w: float, h: float, body: str, angle: float = 0.0) -> str:
    rot = f' patternTransform="rotate({fmt(angle)})"' if angle else ""
    return f'<pattern id="{pid}" patternUnits="userSpaceOnUse" width="{fmt(w)}" height="{fmt(h)}"{rot}>{body}</pattern>'


def _hline(y: float, w: float, sw: float = 0.13) -> str:
    return f'<line x1="0" y1="{fmt(y)}" x2="{fmt(w)}" y2="{fmt(y)}" stroke="{INK}" stroke-width="{fmt(sw)}"/>'


# Hatching for sections (ISO 128-50 general hatch at 45 degrees) and finishes in plan.
PATTERNS = {
    "hatch45": _pattern("hatch45", 1.2, 1.2, _hline(0.6, 1.2), 45),
    "hatch45wide": _pattern("hatch45wide", 2.0, 2.0, _hline(1.0, 2.0), 45),
    "cross": _pattern(
        "cross",
        1.2,
        1.2,
        _hline(0.6, 1.2)
        + f'<line x1="0.6" y1="0" x2="0.6" y2="1.2" stroke="{INK}" stroke-width="0.13"/>',
        45,
    ),
    "insulation": _pattern(
        "insulation",
        1.6,
        1.6,
        f'<path d="M0,1.2 L0.4,0.4 L0.8,1.2 L1.2,0.4 L1.6,1.2" fill="none" stroke="{INK}" stroke-width="0.13"/>',
    ),
    "boards": _pattern("boards", 3.6, 3.6, _hline(0.0, 3.6, 0.1)),
    "vinyl": _pattern("vinyl", 1.6, 1.6, _hline(0.8, 1.6, 0.1), 45),
    "carpet": _pattern(
        "carpet", 1.4, 1.4, f'<circle cx="0.7" cy="0.7" r="0.12" fill="{INK}"/>'
    ),
    "concrete": _pattern(
        "concrete",
        3.0,
        3.0,
        f'<circle cx="0.6" cy="0.8" r="0.18" fill="{INK}"/><circle cx="2.1" cy="2.2" r="0.12" fill="{INK}"/>'
        f'<path d="M1.6,0.5 l0.5,0.3 l-0.5,0.2 z" fill="none" stroke="{INK}" stroke-width="0.1"/>',
    ),
    "water": _pattern("water", 3.0, 1.2, _hline(0.6, 1.6, 0.1)),
}


# --------------------------------------------------------------------------- views


@dataclass
class View:
    """Model millimetres (b along, v up or across) onto the sheet: x = ox + sb*b/scale,
    y = oy - sv*v/scale."""

    sheet: Sheet
    name: str
    ox: float
    oy: float
    scale: float = 50.0
    b_sign: int = 1
    v_sign: int = 1

    def P(self, b: float, v: float) -> Pt:
        return (
            self.ox + self.b_sign * b / self.scale,
            self.oy - self.v_sign * v / self.scale,
        )

    def x(self, b: float) -> float:
        return self.ox + self.b_sign * b / self.scale

    def y(self, v: float) -> float:
        return self.oy - self.v_sign * v / self.scale

    def poly(self, pts, style="thin", *, closed=False, fill="none") -> None:
        self.sheet.polyline(
            [self.P(b, v) for b, v in pts], style, closed=closed, fill=fill
        )

    def line(self, p, q, style="thin") -> None:
        self.sheet.line(self.P(*p), self.P(*q), style)

    def rect(self, b0, v0, b1, v1, style="thin", fill="none") -> None:
        x0, y0 = self.P(b0, v0)
        x1, y1 = self.P(b1, v1)
        self.sheet.rect(x0, y0, x1, y1, style, fill)

    def circle(self, c, r, style="thin", fill="none") -> None:
        self.sheet.circle(self.P(*c), r / self.scale, style, fill)

    def hatch(self, pts, pattern: str) -> None:
        self.sheet.hatch([self.P(b, v) for b, v in pts], pattern)

    def fill(self, pts, colour: str = INK) -> None:
        self.sheet.fill([self.P(b, v) for b, v in pts], colour)

    def text(self, p, s, h=2.5, **kw) -> None:
        self.sheet.text(self.P(*p), s, h, **kw)


# --------------------------------------------------------------------------- dimensions


@dataclass(frozen=True)
class Dim:
    """A dimension as drawn: its value and the model points it measures between."""

    value: int
    a: float
    b: float
    axis: str  # "b" (along the view) or "v"


def arrowhead(sheet: Sheet, tip: Pt, toward: Pt) -> None:
    """Filled arrowhead at `tip`, pointing away from `toward`."""
    dx, dy = tip[0] - toward[0], tip[1] - toward[1]
    n = math.hypot(dx, dy) or 1.0
    ux, uy = dx / n, dy / n
    bx, by = tip[0] - ux * ARROW_L, tip[1] - uy * ARROW_L
    sheet.fill(
        [
            tip,
            (bx - uy * ARROW_W, by + ux * ARROW_W),
            (bx + uy * ARROW_W, by - ux * ARROW_W),
        ]
    )


def tick(sheet: Sheet, p: Pt, horizontal: bool) -> None:
    """Oblique stroke at 45 degrees across a dimension line, drawn in the wide line."""
    t = TICK / math.sqrt(2)
    if horizontal:
        sheet.line((p[0] - t, p[1] + t), (p[0] + t, p[1] - t), "medium")
    else:
        sheet.line((p[0] - t, p[1] + t), (p[0] + t, p[1] - t), "medium")


def label(value: float) -> str:
    return str(round(value))


def dim_h(
    view: View,
    b0: float,
    b1: float,
    y_line: float,
    *,
    from_v: tuple[float, float] | None = None,
    style: str = "arrow",
    h: float = 2.5,
    text: str | None = None,
    raise_text: float = 0.0,
) -> Dim:
    """Horizontal dimension between b0 and b1 drawn at sheet height y_line. Extension lines
    run from the model heights `from_v` (if given) to the dimension line."""
    s = view.sheet
    x0, x1 = sorted((view.x(b0), view.x(b1)))
    if from_v is not None:
        for b, v in zip((b0, b1), from_v):
            if v is None:
                continue
            x, y = view.P(b, v)
            sgn = 1 if y_line > y else -1
            s.line((x, y + sgn * EXT_GAP), (x, y_line + sgn * EXT_OVER), "thin")
    s.line((x0, y_line), (x1, y_line), "thin")
    if style == "arrow":
        arrowhead(s, (x0, y_line), (x1, y_line))
        arrowhead(s, (x1, y_line), (x0, y_line))
    else:
        tick(s, (x0, y_line), True)
        tick(s, (x1, y_line), True)
    value = abs(b1 - b0)
    s.text(
        ((x0 + x1) / 2, y_line - 0.8 - raise_text),
        text or label(value),
        h,
        anchor="middle",
    )
    return Dim(round(value), min(b0, b1), max(b0, b1), "b")


def dim_v(
    view: View,
    v0: float,
    v1: float,
    x_line: float,
    *,
    from_b: tuple[float, float] | None = None,
    style: str = "arrow",
    h: float = 2.5,
    text: str | None = None,
    text_left: bool = True,
) -> Dim:
    """Vertical dimension between model heights v0 and v1 at sheet x_line."""
    s = view.sheet
    y0, y1 = sorted((view.y(v0), view.y(v1)))
    if from_b is not None:
        for v, b in zip((v0, v1), from_b):
            if b is None:
                continue
            x, y = view.P(b, v)
            sgn = 1 if x_line > x else -1
            s.line((x + sgn * EXT_GAP, y), (x_line + sgn * EXT_OVER, y), "thin")
    s.line((x_line, y0), (x_line, y1), "thin")
    if style == "arrow":
        arrowhead(s, (x_line, y0), (x_line, y1))
        arrowhead(s, (x_line, y1), (x_line, y0))
    else:
        tick(s, (x_line, y0), False)
        tick(s, (x_line, y1), False)
    value = abs(v1 - v0)
    tx = x_line - 0.8 if text_left else x_line + 0.8 + h
    s.text((tx, (y0 + y1) / 2), text or label(value), h, anchor="middle", rot=90)
    return Dim(round(value), min(v0, v1), max(v0, v1), "v")


def chain_h(
    view: View,
    bs,
    y_line: float,
    *,
    from_v=None,
    h: float = 2.5,
) -> list[Dim]:
    """Chain of horizontal dimensions through the b stations, with oblique strokes; text too
    long for its space is raised to a second level, alternating."""
    s = view.sheet
    out = []
    if from_v is not None:
        for b, v in zip(bs, from_v):
            if v is None:
                continue
            x, y = view.P(b, v)
            sgn = 1 if y_line > y else -1
            s.line((x, y + sgn * EXT_GAP), (x, y_line + sgn * EXT_OVER), "thin")
    s.line((view.x(bs[0]), y_line), (view.x(bs[-1]), y_line), "thin")
    raised = False
    for b0, b1 in pairwise(bs):
        x0, x1 = sorted((view.x(b0), view.x(b1)))
        tick(s, (x0, y_line), True)
        tick(s, (x1, y_line), True)
        value = abs(b1 - b0)
        fits = text_width(label(value), h) + 1.0 < x1 - x0
        raised = (not fits) and not raised
        s.text(
            ((x0 + x1) / 2, y_line - 0.8 - (h + 0.6 if raised else 0)),
            label(value),
            h,
            anchor="middle",
        )
        out.append(Dim(round(value), min(b0, b1), max(b0, b1), "b"))
    return out


def running_h(
    view: View,
    origin: float,
    targets,
    y_line: float,
    *,
    from_v=None,
    h: float = 1.8,
    above: bool = True,
) -> list[Dim]:
    """Running (ordinate) dimensions along b from an origin circle: one dimension line, an
    arrowhead at each target, the value written along its extension line."""
    s = view.sheet
    xo = view.x(origin)
    xs = [view.x(b) for b in targets]
    if from_v is not None:
        for b, v in zip([origin, *targets], from_v):
            if v is None:
                continue
            x, y = view.P(b, v)
            sgn = 1 if y_line > y else -1
            s.line((x, y + sgn * EXT_GAP), (x, y_line + sgn * EXT_OVER), "thin")
    s.line((min(xo, *xs), y_line), (max(xo, *xs), y_line), "thin")
    s.circle((xo, y_line), 0.9, "thin", fill="#fff")
    out = []
    for b, x in zip(targets, xs):
        arrowhead(s, (x, y_line), (xo, y_line))
        value = abs(b - origin)
        ty = y_line - 1.2 if above else y_line + 1.2 + text_width(label(value), h)
        s.text((x - 0.5, ty), label(value), h, anchor="start", rot=90)
        out.append(Dim(round(value), min(origin, b), max(origin, b), "b"))
    return out


def running_v(
    view: View,
    origin: float,
    targets,
    x_line: float,
    *,
    from_b=None,
    h: float = 1.8,
    left: bool = True,
) -> list[Dim]:
    """Running dimensions up a view from an origin circle, values written beside the line."""
    s = view.sheet
    yo = view.y(origin)
    ys = [view.y(v) for v in targets]
    if from_b is not None:
        for v, b in zip([origin, *targets], from_b):
            if b is None:
                continue
            x, y = view.P(b, v)
            sgn = 1 if x_line > x else -1
            s.line((x + sgn * EXT_GAP, y), (x_line + sgn * EXT_OVER, y), "thin")
    s.line((x_line, min(yo, *ys)), (x_line, max(yo, *ys)), "thin")
    s.circle((x_line, yo), 0.9, "thin", fill="#fff")
    out = []
    last_y, shift = math.inf, 0.0
    for v, y in zip(targets, ys):
        arrowhead(s, (x_line, y), (x_line, yo))
        value = abs(v - origin)
        text = label(value)
        # Values too close to the previous one step outwards so they never overlap.
        shift = shift + text_width(text, h) + 1.0 if abs(y - last_y) < h + 0.4 else 0.0
        last_y = y
        gap = 1.0 + shift
        s.text(
            (x_line - gap if left else x_line + gap, y + 0.35 * h),
            text,
            h,
            anchor="end" if left else "start",
        )
        out.append(Dim(round(value), min(origin, v), max(origin, v), "v"))
    return out


def leader(
    sheet: Sheet, target: Pt, at: Pt, note: str, h: float = 1.8, *, dot: bool = True
) -> None:
    """A leader from a note to the object: a dot on a surface (or an arrow on an edge)."""
    sheet.line(at, target, "thin")
    if dot:
        sheet.circle(target, 0.35, None, fill=INK)
    else:
        arrowhead(sheet, target, at)
    left = target[0] < at[0]
    shelf = 1.2 if left else -1.2
    sheet.line(at, (at[0] + shelf, at[1]), "thin")
    sheet.text(
        (at[0] + shelf + (0.4 if left else -0.4), at[1] + 0.35 * h),
        note,
        h,
        anchor="start" if left else "end",
    )


# --------------------------------------------------------------------------- sheet furniture


def frame(sheet: Sheet) -> None:
    """ISO 5457 frame, centring marks and 50 mm grid reference fields (A3: 8 x 6)."""
    w, h = sheet.size
    x0, y0, x1, y1 = FRAME
    with sheet.group("frame"):
        sheet.rect(x0, y0, x1, y1, "frame")
        cx, cy = w / 2, h / 2
        for p, q in (
            ((cx, 0), (cx, y0 + 5)),
            ((cx, h), (cx, y1 - 5)),
            ((0, cy), (x0 + 5, cy)),
            ((w, cy), (x1 - 5, cy)),
        ):
            sheet.line(p, q, "frame")
        cols = sorted({x for k in range(-5, 6) if x0 < (x := cx + 50 * k) < x1})
        edges = [x0, *cols, x1]
        for i, (a, b) in enumerate(pairwise(edges)):
            for ym in (y0 / 2, (y1 + h) / 2):
                sheet.text(
                    ((a + b) / 2, ym), str(i + 1), 3.5, anchor="middle", middle=True
                )
        for x in cols:
            sheet.line((x, y0), (x, y0 - 5), "medium")
            sheet.line((x, y1), (x, y1 + 5), "medium")
        rows = sorted({y for k in range(-5, 6) if y0 < (y := cy + 50 * k) < y1})
        edges = [y0, *rows, y1]
        for i, (a, b) in enumerate(pairwise(edges)):
            letter = "ABCDEFGH"[i]
            sheet.text(
                (x0 / 2 + 3, (a + b) / 2), letter, 3.5, anchor="middle", middle=True
            )
            sheet.text(
                ((x1 + w) / 2, (a + b) / 2), letter, 3.5, anchor="middle", middle=True
            )
        for y in rows:
            sheet.line((x0, y), (x0 - 5, y), "medium")
            sheet.line((x1, y), (x1 + 5, y), "medium")


def first_angle_symbol(sheet: Sheet, at: Pt, h: float = 3.5) -> None:
    """ISO 5456-2 first-angle projection symbol: the truncated cone's side view on the left,
    its end view (seen from the small end) on the right."""
    big, small = 2 * h, h
    x, y = at
    sheet.polyline(
        [
            (x, y - small / 2),
            (x + big, y - big / 2),
            (x + big, y + big / 2),
            (x, y + small / 2),
        ],
        "medium",
        closed=True,
    )
    cx = x + big + 0.6 * h + big / 2
    sheet.circle((cx, y), big / 2, "medium")
    sheet.circle((cx, y), small / 2, "medium")
    sheet.line((x - 1, y), (cx + big / 2 + 1, y), "centre")
    sheet.line((cx, y - big / 2 - 1), (cx, y + big / 2 + 1), "centre")


def scale_bar(
    sheet: Sheet, at: Pt, scale: float, metres: int = 5, h: float = 1.8
) -> None:
    """Graphic scale: alternate filled metre blocks, labelled 0..metres."""
    x, y = at
    step = 1000 / scale
    for k in range(metres):
        sheet.rect(
            x + k * step,
            y,
            x + (k + 1) * step,
            y + 1.2,
            "thin",
            INK if k % 2 == 0 else "#fff",
        )
    for k in range(metres + 1):
        sheet.text((x + k * step, y - 0.8), str(k), h, anchor="middle")
    sheet.text((x + metres * step + 1.5, y + 1.2), f"m   SCALE 1:{round(scale)}", h)


@dataclass(frozen=True)
class TitleBlock:
    owner: str
    number: str
    revision: str
    date: str
    sheet_no: str
    title: str
    subtitle: str
    doc_type: str
    status: str
    creator: str
    approver: str
    scale: str
    units: str = "mm"
    language: str = "en"


def title_block(
    sheet: Sheet, tb: TitleBlock, revisions
) -> tuple[float, float, float, float]:
    """ISO 7200 title block (180 mm wide) in the bottom right corner, with the revision table
    above it. Returns its (x0, y0, x1, y1), revision table included."""
    x1, y1 = FRAME[2], FRAME[3]
    x0 = x1 - 180
    rows = (y1 - 38, y1 - 27, y1 - 13, y1)
    top = rows[0] - 4.5 * (len(revisions) + 1)
    with sheet.group("titleblock"):
        sheet.rect(x0, top, x1, y1, "frame")

        def cell(
            cx0, cy0, cx1, cy1, label_text, value, h=2.5, bold=False, anchor="start"
        ):
            sheet.rect(cx0, cy0, cx1, cy1, "thin")
            sheet.text((cx0 + 1, cy0 + 2.2), label_text, 1.8)
            if value:
                vx = cx0 + 1.5 if anchor == "start" else (cx0 + cx1) / 2
                sheet.text((vx, cy1 - 1.6), value, h, bold=bold, anchor=anchor)

        # Revision table.
        ry = top
        cols = (x0, x0 + 12, x0 + 128, x0 + 156, x1)
        heads = ("REV", "DESCRIPTION", "DATE", "BY")
        for i, head in enumerate(heads):
            sheet.rect(cols[i], ry, cols[i + 1], ry + 4.5, "thin")
            sheet.text((cols[i] + 1, ry + 3.3), head, 1.8, bold=True)
        for r, row in enumerate(revisions):
            yy = ry + 4.5 * (r + 1)
            for i, val in enumerate(row):
                sheet.rect(cols[i], yy, cols[i + 1], yy + 4.5, "thin")
                sheet.text((cols[i] + 1, yy + 3.3), val, 1.8)
        # Row 1: document type, status, scale, projection, units.
        a, b = rows[0], rows[1]
        cell(x0, a, x0 + 52, b, "DOCUMENT TYPE", tb.doc_type)
        cell(x0 + 52, a, x0 + 120, b, "DOCUMENT STATUS", tb.status, 2.5, True)
        cell(x0 + 120, a, x0 + 140, b, "SCALE", tb.scale)
        sheet.rect(x0 + 140, a, x0 + 166, b, "thin")
        sheet.text((x0 + 141, a + 2.2), "PROJECTION", 1.8)
        first_angle_symbol(sheet, (x0 + 144.5, (a + b) / 2 + 1.2), 2.5)
        cell(x0 + 166, a, x1, b, "UNITS", tb.units)
        # Row 2: title.
        a, b = rows[1], rows[2]
        sheet.rect(x0, a, x1, b, "thin")
        sheet.text((x0 + 1, a + 2.2), "TITLE", 1.8)
        sheet.text((x0 + 3, a + 7.6), tb.title, 5.0, bold=True)
        sheet.text((x0 + 3, b - 1.3), tb.subtitle, 2.5)
        # Row 3: owner, people, identification number, revision, date, sheet, language.
        a, b = rows[2], rows[3]
        cell(x0, a, x0 + 34, b, "LEGAL OWNER", tb.owner, 2.5, True)
        sheet.rect(x0, a, x0 + 34, b, "thin")
        cell(x0 + 34, a, x0 + 70, (a + b) / 2, "CREATED BY", tb.creator, 1.8)
        cell(x0 + 34, (a + b) / 2, x0 + 70, b, "APPROVED BY", tb.approver, 1.8)
        cell(x0 + 70, a, x0 + 124, b, "IDENTIFICATION NUMBER", tb.number, 5.0, True)
        cell(x0 + 124, a, x0 + 134, b, "REV", tb.revision, 3.5, True, "middle")
        cell(x0 + 134, a, x0 + 158, b, "DATE OF ISSUE", tb.date)
        cell(x0 + 158, a, x0 + 172, b, "SHEET", tb.sheet_no, 2.5, False, "middle")
        cell(x0 + 172, a, x1, b, "LANG", tb.language, 2.5, False, "middle")
    return (x0, top, x1, y1)


def table(
    sheet: Sheet,
    at: Pt,
    title: str,
    rows,
    widths,
    *,
    h: float = 1.8,
    row_h: float = 3.6,
    head: tuple | None = None,
    align: tuple | None = None,
) -> tuple[float, float, float, float]:
    """A simple ruled table with a bold title row. Returns its box."""
    x, y = at
    total = sum(widths)
    sheet.rect(x, y, x + total, y + row_h + 0.6, "medium")
    sheet.text((x + 1, y + row_h - 0.6), title, 2.5, bold=True)
    yy = y + row_h + 0.6
    body = ([head] if head else []) + list(rows)
    align = align or ("start",) * len(widths)
    for r, row in enumerate(body):
        xx = x
        for i, (w, val) in enumerate(zip(widths, row)):
            sheet.rect(xx, yy, xx + w, yy + row_h, "thin")
            bold = head is not None and r == 0
            if align[i] == "end" and not bold:
                sheet.text((xx + w - 1, yy + row_h - 0.9), str(val), h, anchor="end")
            else:
                sheet.text((xx + 1, yy + row_h - 0.9), str(val), h, bold=bold)
            xx += w
        yy += row_h
    sheet.rect(x, y, x + total, yy, "medium")
    return (x, y, x + total, yy)


def notes(
    sheet: Sheet,
    at: Pt,
    title: str,
    lines,
    width: float,
    *,
    h: float = 1.8,
    gap: float = 3.0,
):
    """Numbered notes under a bold heading; long lines wrap to `width`."""
    x, y = at
    sheet.text((x, y), title, 2.5, bold=True)
    y += 1.5
    per_line = max(10, int(width / (CHAR_W * h)))
    for i, raw in enumerate(lines, 1):
        words, cur, out = raw.split(), "", []
        for w in words:
            if len(cur) + len(w) + 1 > per_line - 4 and cur:
                out.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        out.append(cur)
        for j, chunk in enumerate(out):
            y += gap
            if j == 0:
                sheet.text((x, y), f"{i}.", h)
            sheet.text((x + 3.2, y), chunk, h)
    return y
