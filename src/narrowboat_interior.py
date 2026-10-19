# SPDX-License-Identifier: MIT
"""Interior fit-out of narrowboat NB57 'Wren', built from narrowboat_design (no bpy).

Rooms, partitions and every piece of furniture come from the design's ZONES, PARTITIONS and
FIXTURES tables, the same data the drawing set is drawn from. Positions are metres from the
rear cabin bulkhead (`b`), +Y to port, z above the baseplate underside. Each fixture kind has
a builder in BUILDERS; furniture is grouped into multi-material parts by zone. Every solid is
closed and outward-wound, and the main pieces are registered as labelled boxes so the tests
can check clearances.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import narrowboat_common as N
import narrowboat_design as D
import narrowboat_lines as L
from procgen_mesh import (
    MeshData,
    Part,
    box,
    lathe,
    linspace,
    loft,
    offset_polyline,
    ring,
    ring_profile,
    rounded_rect,
    seeded,
    sweep,
)

m = L.m
LINING_T = m(D.PLY)
FLOOR_T = 0.02
B_IN0, B_IN1 = m(D.LINING_AFT), m(D.LINING_FWD)
Z_GUNWALE = N.Z_GUNWALE
HULL_LINING = m(D.LINING_IN)  # inner face of the hull lining, floor to gunwale


def lining_y(z: float) -> float:
    """Inner face of the port lining at height z."""
    return L.lining_in(z)


def ceiling_z(y: float) -> float:
    """Underside of the ceiling at y."""
    return L.ceiling_z(y)


def _lining_top() -> float:
    """Height where the cabin lining meets the cambered ceiling."""
    z = 1.95
    for _ in range(40):
        z = ceiling_z(lining_y(z))
    return z


Z_TOP = _lining_top()
CEILING_HALF = lining_y(Z_TOP)


@dataclass(frozen=True)
class Zone:
    name: str
    b0: float
    b1: float
    floor_z: float
    floor: str
    walls: str
    ceiling: str


ZONES = tuple(
    Zone(z.name, m(z.b0), m(z.b1), m(z.floor_z), z.floor, z.walls, z.ceiling)
    for z in D.ZONES
)
ZONE = {z.name: z for z in ZONES}

# Wall lights, (side, b), set in the paint between openings (see test_lights_clear_openings),
# and ceiling downlights spaced along each room.
SCONCES = ((1, 9.45), (1, 10.85), (-1, 8.40), (-1, 11.35))
SCONCE_Z = 1.69


def downlights() -> list[float]:
    out = []
    for z in ZONES:
        if z.name == "Lobby":
            continue
        n = max(1, round((z.b1 - z.b0) / 1.3))
        out += [z.b0 + (z.b1 - z.b0) * (k + 0.5) / n for k in range(n)]
    return out


@dataclass
class Kit:
    """Accumulates solids for one multi-material part and registers labelled boxes."""

    name: str
    slots: list = field(default_factory=list)
    mesh: MeshData = field(default_factory=MeshData)
    items: list = field(default_factory=list)

    def slot(self, name: str) -> int:
        if name not in self.slots:
            self.slots.append(name)
        return self.slots.index(name)

    def add(self, md: MeshData, slot, label: str | None = None) -> MeshData:
        if isinstance(slot, str):
            md.mats = [self.slot(slot)] * len(md.faces)
        else:
            md.mats = [self.slot(slot[k]) for k in md.mats]
        self.mesh.extend(md)
        if label:
            self.items.append((label, md.bounds()))
        return md

    def part(self) -> Part:
        return Part(self.name, tuple(self.slots), self.mesh)


def W(b: float) -> float:
    return N.wx(1000 * b)


def bbox(b0, b1, y0, y1, z0, z1, *, grain="x", plank=0.0) -> MeshData:
    """Box in b-coordinates (metres from the rear bulkhead)."""
    return box(
        W(min(b0, b1)),
        W(max(b0, b1)),
        min(y0, y1),
        max(y0, y1),
        min(z0, z1),
        max(z0, z1),
        grain=grain,
        plank=plank,
    )


def rounded_box(b0, b1, y0, y1, z0, z1, r, plank=0.0) -> MeshData:
    yc, zc = (y0 + y1) / 2, (z0 + z1) / 2
    prof = [(yc + u, zc + v) for u, v in rounded_rect(abs(y1 - y0), abs(z1 - z0), r, 4)]
    return loft([W(b0), W(b1)], lambda x: prof, plank=plank)


def upright(md: MeshData, b: float, y: float, z: float = 0.0) -> MeshData:
    """Stand a loft made along x (as a height) upright at (b, y)."""
    return md.transformed(((0, 0, -1), (0, 1, 0), (1, 0, 0)), (W(b), y, z))


def placed(md: MeshData, b: float, y: float, z: float) -> MeshData:
    return md.transformed(((1, 0, 0), (0, 1, 0), (0, 0, 1)), (W(b), y, z))


def rand(seed: int, key: str) -> float:
    return seeded("nb-int", seed, key).random()


# --------------------------------------------------------------------------- shell


def lining_outer(floor_z: float):
    """Outer face of the port lining (against the insulation): up the hull side, in under
    the gunwale, then up the cabin side's tumblehome to the ceiling."""
    out = HULL_LINING + LINING_T
    return [
        (out, floor_z),
        (out, Z_GUNWALE),
        (lining_y(Z_GUNWALE) + LINING_T, Z_GUNWALE),
        (lining_y(Z_TOP) + LINING_T, Z_TOP),
    ]


def lining_profile(floor_z: float):
    """Port lining strip: the outer face, then the inner face back down. Under the gunwale
    the ply turns inboard as a soffit, since the cabin side stands inboard of the hull."""
    outer = lining_outer(floor_z)
    inner = offset_polyline(outer, [LINING_T] * (len(outer) - 1))
    return outer + list(reversed(inner))


def linings(seed: int) -> dict[str, MeshData]:
    """Wall linings per side and finish, one loft per room at its floor height (cut by
    the window cutters in Blender)."""
    out: dict[str, MeshData] = {}
    for zone in ZONES:
        prof = lining_profile(zone.floor_z)
        for side, key in ((1, "Port"), (-1, "Starboard")):
            md = loft(
                [W(zone.b0), W(zone.b1)],
                lambda x, p=prof: p,
                plank=rand(seed, f"lining{zone.name}{side}"),
            )
            out.setdefault(f"Lining{key}{zone.walls}", MeshData()).extend(
                md if side > 0 else md.mirrored_y()
            )
    return out


def clip_y(poly, y_min: float, y_max: float):
    """Clip a closed (y, z) polygon to y_min <= y <= y_max (Sutherland-Hodgman)."""

    def clip(points, keep, edge):
        out = []
        for i, cur in enumerate(points):
            prev = points[i - 1]
            if keep(cur):
                if not keep(prev):
                    out.append(edge(prev, cur))
                out.append(cur)
            elif keep(prev):
                out.append(edge(prev, cur))
        return out

    def at(y):
        def edge(p, q):
            t = (y - p[0]) / (q[0] - p[0])
            return (y, p[1] + (q[1] - p[1]) * t)

        return edge

    poly = clip(poly, lambda p: p[0] >= y_min, at(y_min))
    poly = clip(poly, lambda p: p[0] <= y_max, at(y_max))
    out = []
    for p in poly:
        if not out or math.dist(p, out[-1]) > 1e-6:
            out.append(p)
    if len(out) > 2 and math.dist(out[0], out[-1]) < 1e-6:
        out.pop()
    return out


def room_half(floor_z: float):
    """Port half of the room's outline (inner faces), floor to the ceiling's crown."""
    soffit = Z_GUNWALE - LINING_T
    pts = [
        (HULL_LINING, floor_z),
        (HULL_LINING, soffit),
        (lining_y(Z_GUNWALE), soffit),
        (CEILING_HALF, Z_TOP),
    ]
    arc = [
        (CEILING_HALF * (1 - k / 16), ceiling_z(CEILING_HALF * (1 - k / 16)) - LINING_T)
        for k in range(1, 16)
    ]
    return pts + arc


def section_polygon(floor_z: float, y_min: float = -1.0, y_max: float = 1.0):
    """Interior cross-section (floor to ceiling, lining to lining) clipped to [y_min, y_max]."""
    half = room_half(floor_z)
    full = (
        half + [(0.0, ceiling_z(0.0) - LINING_T)] + [(-y, z) for y, z in reversed(half)]
    )
    return clip_y(full, y_min, y_max)


def wall_strip(
    side: int, b0: float, b1: float, z0: float, z1: float, t: float
) -> MeshData:
    """A panel of thickness t on the lining (following the tumblehome), e.g. wall tiles."""
    zs = sorted({z for z in (z0, Z_GUNWALE, z1) if z0 <= z <= z1})
    face = [
        (lining_y(min(z, Z_GUNWALE - 1e-6) if z == Z_GUNWALE else z), z) for z in zs
    ]
    if len(face) == 3:
        face.insert(2, (lining_y(Z_GUNWALE), Z_GUNWALE))
    inner = offset_polyline(face, [t] * (len(face) - 1))
    prof = face + list(reversed(inner))
    md = loft([W(b0), W(b1)], lambda x: prof)
    return md if side > 0 else md.mirrored_y()


def ceiling(seed: int) -> dict[str, MeshData]:
    top = [
        (CEILING_HALF * (1 - 2 * k / 32), ceiling_z(CEILING_HALF * (1 - 2 * k / 32)))
        for k in range(33)
    ]
    under = offset_polyline(top, [LINING_T] * (len(top) - 1))
    prof = top + list(reversed(under))
    out: dict[str, MeshData] = {}
    for zone in ZONES:
        md = loft(
            [W(zone.b0), W(zone.b1)],
            lambda x: prof,
            grain="across" if zone.ceiling.endswith("TG") else "x",
            plank=rand(seed, f"ceiling{zone.name}"),
        )
        out.setdefault(f"Ceiling{zone.ceiling}", MeshData()).extend(md)
    return out


def floors(seed: int) -> dict[str, MeshData]:
    """Floor per room at its height. Where a flight of steps drops below a raised floor (the
    stern steps in the lobby), the floor stops either side of it."""
    out: dict[str, MeshData] = {}
    for zone in ZONES:
        cuts = []
        for f in D.FIXTURES:
            if f.zone == zone.name and f.kind == "steps" and m(f.z0) < zone.floor_z:
                _, _, y0, y1 = f.footprint()
                cuts.append((m(y0), m(y1)))
        spans, cur = [], -HULL_LINING
        for y0, y1 in sorted(cuts):
            spans.append((cur, y0))
            cur = y1
        spans.append((cur, HULL_LINING))
        for y0, y1 in spans:
            md = bbox(
                zone.b0,
                zone.b1,
                y0,
                y1,
                zone.floor_z - FLOOR_T,
                zone.floor_z,
                plank=rand(seed, f"floor{zone.name}{y0:.2f}"),
            )
            out.setdefault(f"Floor{zone.floor}", MeshData()).extend(md)
    return out


def end_linings(seed: int) -> MeshData:
    """Linings on the inside of both steel bulkheads, with panels on the closed doors."""
    md = MeshData()
    t = m(D.T_BULKHEAD)
    for b0, b1, zone in ((t, B_IN0, ZONES[0]), (B_IN1, N.CABIN_LEN - t, ZONES[-1])):
        poly = section_polygon(zone.floor_z)
        md.extend(
            loft(
                [W(b0), W(b1)],
                lambda x, p=poly: p,
                grain="across",
                plank=rand(seed, f"end{b0}"),
            )
        )
    for door, b, sign in ((D.DOORS[0], B_IN0, 1), (D.DOORS[1], B_IN1, -1)):
        half = m(door.half)
        for y0, y1 in ((-half, -0.01), (0.01, half)):
            md.extend(
                bbox(
                    b,
                    b + sign * 0.015,
                    y0,
                    y1,
                    m(door.z0) + 0.06,
                    m(door.z1) - 0.01,
                    grain="across",
                )
            )
    return md


def partitions(seed: int) -> tuple[MeshData, MeshData]:
    """(partition walls with their doorways, door leaves standing open) from the design."""
    walls, leaves = MeshData(), MeshData()
    t = m(D.PARTITION_T) / 2
    head = m(D.FLOOR_Z + D.DOOR_H)
    for p in D.PARTITIONS:
        b = m(p.b)
        floor_z = max(z.floor_z for z in ZONES if z.b0 <= b <= z.b1)
        d0, d1 = m(p.door_y[0]), m(p.door_y[1])
        for poly in (
            section_polygon(floor_z, y_min=d1),
            section_polygon(floor_z, y_max=d0),
        ):
            if len(poly) >= 3:
                walls.extend(
                    loft(
                        [W(b - t), W(b + t)],
                        lambda x, q=poly: q,
                        grain="across",
                        plank=rand(seed, f"wall{b}"),
                    )
                )
        header = [
            (d1, head),
            (d1, ceiling_z(d1) - LINING_T),
            (d0, ceiling_z(d0) - LINING_T),
            (d0, head),
        ]
        walls.extend(loft([W(b - t), W(b + t)], lambda x, q=header: q, grain="across"))
        # The leaf stands open at 90 degrees, along the boat from its hinge jamb.
        hinge = d1 if p.hinge > 0 else d0
        width = d1 - d0 - 0.01
        b_far = b + p.opens * (t + width)
        y_leaf = hinge - p.hinge * 0.035
        leaves.extend(
            bbox(
                b + p.opens * t,
                b_far,
                hinge - p.hinge * 0.005,
                y_leaf,
                floor_z + 0.01,
                head - 0.01,
                grain="across",
                plank=rand(seed, f"door{b}"),
            )
        )
    return walls, leaves


# --------------------------------------------------------------------------- joinery


def door_front(
    kit: Kit,
    b0,
    b1,
    y_face,
    z0,
    z1,
    side: int,
    seed: int,
    key: str,
    knob=True,
    slot="Ash",
):
    """Shaker door or drawer front on a carcass face (side = +1 port, faces starboard)."""
    t = 0.018
    yf = y_face - side * t
    kit.add(
        bbox(
            b0 + 0.002,
            b1 - 0.002,
            y_face,
            yf,
            z0 + 0.002,
            z1 - 0.002,
            grain="across",
            plank=rand(seed, key),
        ),
        slot,
    )
    f = 0.045
    yr = yf - side * 0.006
    for c0, c1, e0, e1 in (
        (b0 + 0.002, b0 + f, z0 + 0.002, z1 - 0.002),
        (b1 - f, b1 - 0.002, z0 + 0.002, z1 - 0.002),
        (b0 + f, b1 - f, z0 + 0.002, z0 + f),
        (b0 + f, b1 - f, z1 - f, z1 - 0.002),
    ):
        kit.add(bbox(c0, c1, yf, yr, e0, e1, plank=rand(seed, key + "r")), slot)
    if knob:
        k = lathe([(0.012, 0.0), (0.012, 0.012), (0.016, 0.022), (0.012, 0.03)], 12)
        rot = ((1, 0, 0), (0, 0, -side), (0, side, 0))
        kz = z1 - 0.07 if z1 - z0 > 0.25 else (z0 + z1) / 2
        kit.add(k.transformed(rot, (W(b1 - 0.06), yr, kz)), "Stainless")


def split(b0: float, b1: float, width: float = 0.5) -> list[tuple[float, float]]:
    n = max(1, round((b1 - b0) / width))
    return [(b0 + (b1 - b0) * k / n, b0 + (b1 - b0) * (k + 1) / n) for k in range(n)]


def fixture_y(f: D.Fixture) -> tuple[float, float, float]:
    """(wall-side y at floor level, front-face y, side) of a fixture against a lining."""
    side = f.side
    return side * HULL_LINING, side * (HULL_LINING - m(f.depth)), side


def tall_carcass(kit: Kit, f: D.Fixture, seed: int, top: float, slot="Ash") -> float:
    """Full-height carcass whose back follows the hull lining, then the tumblehome."""
    b0, b1 = m(f.b0), m(f.b1)
    y_wall, y_front, side = fixture_y(f)
    z0 = m(f.z0)
    lower = Z_GUNWALE - LINING_T - 0.002
    kit.add(
        bbox(b0, b1, y_wall, y_front, z0, lower, plank=rand(seed, f.name)),
        slot,
        label=f.name,
    )
    kit.add(
        bbox(
            b0,
            b1,
            side * lining_y(top),
            y_front,
            lower,
            top,
            plank=rand(seed, f.name + "u"),
        ),
        slot,
    )
    return y_front


def unit_top(f: D.Fixture) -> float:
    """Top of a tall unit: the design height, held clear of the ceiling at its back."""
    y_back = lining_y(m(f.z1))
    return min(m(f.z1), ceiling_z(y_back) - 0.01)


def build_cupboard(kit: Kit, f: D.Fixture, seed: int) -> None:
    top = unit_top(f)
    y_front = tall_carcass(kit, f, seed, top)
    z0 = m(f.z0) + (0.1 if f.z0 == D.FLOOR_Z else 0.0)
    if f.kind == "wardrobe" or top - z0 > 1.2:
        mid = Z_GUNWALE + 0.15
        for k, (c0, c1) in enumerate(split(m(f.b0), m(f.b1), 0.35)):
            door_front(
                kit,
                c0,
                c1,
                y_front,
                z0,
                mid if f.kind != "wardrobe" else top - 0.01,
                f.side,
                seed,
                f"{f.name}d{k}",
            )
            if f.kind != "wardrobe":
                door_front(
                    kit,
                    c0,
                    c1,
                    y_front,
                    mid + 0.01,
                    top - 0.01,
                    f.side,
                    seed,
                    f"{f.name}u{k}",
                )
    else:
        for k, (c0, c1) in enumerate(split(m(f.b0), m(f.b1), 0.35)):
            door_front(
                kit, c0, c1, y_front, z0, top - 0.01, f.side, seed, f"{f.name}d{k}"
            )
    if f.kind == "wardrobe":
        kit.add(
            bbox(
                m(f.b0) + 0.02,
                m(f.b1) - 0.02,
                y_front - f.side * 0.01,
                y_front - f.side * 0.02,
                top - 0.12,
                top - 0.04,
            ),
            "Ash",
        )


WORKTOP_T = 0.03


def build_base_run(kit: Kit, f: D.Fixture, seed: int) -> None:
    """Base units against a lining: plinth, carcass, fronts round the appliances set into
    the run, and a solid oak worktop with the sink let into it."""
    b0, b1 = m(f.b0), m(f.b1)
    y_wall, y_front, side = fixture_y(f)
    z0 = m(f.z0)
    top = m(f.z1) - WORKTOP_T
    kit.add(bbox(b0, b1, y_wall, y_front + side * 0.06, z0, z0 + 0.1), "Ash")
    kit.add(
        bbox(b0, b1, y_wall, y_front, z0 + 0.1, top, plank=rand(seed, f.name)),
        "Ash",
        label=f.name,
    )
    hosted = sorted((h for h in D.FIXTURES if h.host == f.name), key=lambda h: h.b0)
    taken = [(m(h.b0), m(h.b1)) for h in hosted if h.kind in ("fridge", "cooker")]
    edges, cur = [], b0
    for a, c in taken:
        if a - cur > 0.15:
            edges.append((cur, a))
        cur = c
    if b1 - cur > 0.15:
        edges.append((cur, b1))
    for e0, e1 in edges:
        for k, (c0, c1) in enumerate(split(e0, e1, 0.5)):
            door_front(
                kit,
                c0,
                c1,
                y_front,
                z0 + 0.1,
                top - 0.01,
                side,
                seed,
                f"{f.name}{e0:.2f}d{k}",
            )
    sink = next((h for h in hosted if h.kind == "sink"), None)
    yf = y_front - side * 0.025
    t0, t1 = m(f.z1) - WORKTOP_T, m(f.z1)
    if sink is None:
        kit.add(
            bbox(b0, b1, y_wall, yf, t0, t1), "OakWorktop", label=f"{f.name}.worktop"
        )
        return
    s0, s1 = (m(v) for v in sink.opt("bowl"))
    wall_edge, front_edge = y_wall - side * 0.07, y_front + side * 0.06
    kit.add(bbox(b0, s0, y_wall, yf, t0, t1), "OakWorktop")
    kit.add(bbox(s1, b1, y_wall, yf, t0, t1), "OakWorktop")
    kit.add(bbox(s0, s1, y_wall, wall_edge, t0, t1), "OakWorktop")
    kit.add(bbox(s0, s1, front_edge, yf, t0, t1), "OakWorktop")
    tub(kit, s0, s1, front_edge, wall_edge, t1, 0.18, "Stainless")
    tap(kit, (s0 + s1) / 2, side * tap_y(t1), t1, side)


def tap_y(z_top: float) -> float:
    """|y| for a tap at the back of a worktop: its swan neck rises above the gunwale, so it
    stands clear of the cabin lining there rather than at the hull lining below."""
    return lining_y(z_top + 0.3) - 0.045


def tub(kit: Kit, b0, b1, y0, y1, z_top, depth, slot):
    """An open basin (bottom and four walls), e.g. the sink."""
    t = 0.004
    zb = z_top - depth
    kit.add(bbox(b0, b1, y0, y1, zb, zb + t), slot)
    kit.add(bbox(b0, b0 + t, y0, y1, zb, z_top), slot)
    kit.add(bbox(b1 - t, b1, y0, y1, zb, z_top), slot)
    kit.add(bbox(b0, b1, y0, y0 + math.copysign(t, y1 - y0), zb, z_top), slot)
    kit.add(bbox(b0, b1, y1 - math.copysign(t, y1 - y0), y1, zb, z_top), slot)


def tap(kit: Kit, b, y, z, side, slot="Stainless"):
    path = [(W(b), y, z), (W(b), y, z + 0.22)]
    path += [
        (
            W(b),
            y - side * 0.08 * math.sin(math.pi / 2 * k / 6),
            z + 0.22 + 0.06 * math.sin(math.pi * k / 12),
        )
        for k in range(1, 7)
    ]
    path.append((W(b), y - side * 0.17, z + 0.2))
    kit.add(sweep(path, ring_profile(0.011, 0.011, 10), up=(1, 0, 0)), slot)
    kit.add(placed(lathe([(0.025, 0.0), (0.025, 0.03)], 14), b, y, z), slot)


def build_fridge(kit: Kit, f: D.Fixture, seed: int) -> None:
    _, y_front, side = fixture_y(D.FIXTURE[f.host])
    b0, b1 = m(f.b0), m(f.b1)
    kit.add(
        bbox(
            b0 + 0.005,
            b1 - 0.005,
            y_front,
            y_front - side * 0.02,
            m(f.z0) + 0.1,
            m(f.z1),
        ),
        "Ceramic",
        label=f.name,
    )
    kit.add(
        bbox(
            b1 - 0.07,
            b1 - 0.05,
            y_front - side * 0.02,
            y_front - side * 0.045,
            m(f.z1) - 0.25,
            m(f.z1) - 0.05,
        ),
        "Stainless",
    )


def build_cooker(kit: Kit, f: D.Fixture, seed: int) -> None:
    """Built-in LPG cooker: oven door and grill below the worktop, four burners on top."""
    _, y_front, side = fixture_y(D.FIXTURE[f.host])
    b0, b1 = m(f.b0), m(f.b1)
    top = m(f.z1)
    kit.add(
        bbox(
            b0 + 0.01,
            b1 - 0.01,
            y_front,
            y_front - side * 0.02,
            m(f.z0) + 0.18,
            top - 0.12,
        ),
        "BlackGlass",
        label=f.name,
    )
    kit.add(
        sweep(
            [
                (W(b0 + 0.05), y_front - side * 0.05, top - 0.15),
                (W(b1 - 0.05), y_front - side * 0.05, top - 0.15),
            ],
            ring_profile(0.009, 0.009, 10),
        ),
        "Stainless",
    )
    y_wall = side * HULL_LINING
    hob_y = (y_wall + y_front) / 2
    kit.add(
        bbox(b0 + 0.01, b1 - 0.01, hob_y - 0.25, hob_y + 0.25, top, top + 0.006),
        "BlackGlass",
    )
    for i, (db, dy) in enumerate(
        ((0.13, 0.12), (0.37, 0.12), (0.13, -0.12), (0.37, -0.12))
    ):
        r = 0.045 if i % 2 else 0.035
        kit.add(
            placed(
                lathe([(r, 0.0), (r, 0.012), (r * 0.8, 0.02)], 18),
                b0 + db,
                hob_y + dy,
                top + 0.006,
            ),
            "CastIron",
        )
    for k in range(4):
        kit.add(
            placed(
                lathe([(0.012, 0.0), (0.012, 0.02)], 10).transformed(
                    ((1, 0, 0), (0, 0, -side), (0, side, 0)), (0, 0, 0)
                ),
                b0 + 0.09 + k * 0.1,
                y_front - side * 0.02,
                top - 0.07,
            ),
            "Stainless",
        )


def build_sink(kit: Kit, f: D.Fixture, seed: int) -> None:
    """The sink is let into its base run's worktop (build_base_run)."""


def build_wall_units(kit: Kit, f: D.Fixture, seed: int) -> None:
    b0, b1, side = m(f.b0), m(f.b1), f.side
    z0, z1 = m(f.z0), unit_top(f)
    y_wall = side * lining_y(z1)
    y_front = side * (HULL_LINING - m(f.depth))
    kit.add(
        bbox(b0, b1, y_wall, y_front, z0, z1, plank=rand(seed, f.name)),
        "Ash",
        label=f.name,
    )
    for k, (c0, c1) in enumerate(split(b0, b1, 0.5)):
        door_front(kit, c0, c1, y_front, z0, z1, side, seed, f"{f.name}{k}")


def build_steps(kit: Kit, f: D.Fixture, seed: int) -> None:
    """Oak treads, closed risers and side panels, from the foot of the flight to the sill."""
    n = f.opt("risers")
    b0, b1, z0, z1 = m(f.b0), m(f.b1), m(f.z0), m(f.z1)
    _, _, y0, y1 = (m(v) for v in f.footprint())
    rise, going = (z1 - z0) / n, (b1 - b0) / (n - 1)
    aft = f.opt("rise_to") < 0

    def b_at(k: int) -> float:
        return b1 - k * going if aft else b0 + k * going

    for k in range(1, n):
        z = z0 + k * rise
        a, c = sorted((b_at(k), b_at(k - 1)))
        nose = 0.02 if aft else -0.02
        kit.add(
            bbox(
                a - (nose if not aft else 0),
                c + (nose if aft else 0),
                y0,
                y1,
                z - 0.03,
                z,
                grain="across",
                plank=rand(seed, f"{f.name}t{k}"),
            ),
            "Oak",
            label=f"{f.name}.tread{k}",
        )
        rb = b_at(k - 1)
        kit.add(
            bbox(
                rb - 0.009,
                rb + 0.009,
                y0 + 0.03,
                y1 - 0.03,
                z - rise,
                z - 0.03,
                grain="across",
            ),
            "Ash",
        )
    top = z0 + (n - 1) * rise
    for y in (y0, y1 - 0.03):
        kit.add(
            bbox(b0, b1, y, y + 0.03, z0, top - 0.03, plank=rand(seed, f"{f.name}s")),
            "Ash",
        )


def build_bed(kit: Kit, f: D.Fixture, seed: int) -> None:
    """Fixed double: drawer base facing the corridor, mattress, duvet, pillows, headboard."""
    b0, b1 = m(f.b0), m(f.b1)
    _, _, y0, y1 = (m(v) for v in f.footprint())
    base_top = m(f.z0) + 0.37
    face, face_side = (y1, 1) if f.side < 0 else (y0, -1)
    kit.add(
        bbox(b0, b1, y0, y1, m(f.z0), base_top, plank=rand(seed, "bedbase")),
        "Ash",
        label=f.name,
    )
    for k, (c0, c1) in enumerate(split(b0 + 0.08, b1 - 0.08, 0.62)):
        door_front(
            kit,
            c0,
            c1,
            face,
            m(f.z0) + 0.08,
            base_top - 0.04,
            -face_side,
            seed,
            f"beddrawer{k}",
        )
    kit.add(
        rounded_box(
            b0 + 0.01, b1 - 0.01, y0 + 0.01, y1 - 0.01, base_top, base_top + 0.16, 0.05
        ),
        "Bedding",
    )
    head = f.opt("head")
    hb0, hb1 = (b0, b1 - 0.45) if head > 0 else (b0 + 0.45, b1)
    kit.add(
        rounded_box(
            hb0 - 0.02 * head,
            hb1 + 0.02 * head,
            y0 + 0.005,
            y1 + 0.03 * face_side,
            base_top + 0.12,
            base_top + 0.22,
            0.06,
        ),
        "Duvet",
    )
    for y in (y0 + 0.31, y1 - 0.31):
        pb = (b1 - 0.42, b1 - 0.05) if head > 0 else (b0 + 0.05, b0 + 0.42)
        kit.add(
            rounded_box(
                pb[0], pb[1], y - 0.26, y + 0.26, base_top + 0.16, base_top + 0.3, 0.06
            ),
            "Bedding",
        )
    hb = b1 if head > 0 else b0
    kit.add(
        bbox(
            hb - 0.03 * head,
            hb,
            y0,
            y1,
            base_top,
            base_top + 0.5,
            grain="across",
            plank=rand(seed, "headboard"),
        ),
        "Ash",
    )


def _quadrant_shower(
    r: float, z0: float, z1: float
) -> tuple[MeshData, MeshData, MeshData]:
    """(tray, screen, top rail) of a quadrant shower in a local frame: the corner at the
    origin, the walls along +x and -y (port-aft corner layout)."""
    tray = [(0.0, 0.0)] + [
        (r * math.cos(math.pi / 2 * k / 12), -r * math.sin(math.pi / 2 * k / 12))
        for k in range(13)
    ]
    tray_md = loft([z0 - 0.07, z0], lambda z: [(y, -x) for x, y in tray]).transformed(
        ((0, 0, -1), (0, 1, 0), (1, 0, 0)), (0, 0, 0)
    )
    arc = [
        ((r - 0.02) * math.cos(a), -(r - 0.02) * math.sin(a), z0)
        for a in (math.pi / 2 * k / 16 for k in range(17))
    ]
    screen = sweep(
        arc,
        [(0.0, -0.004), (0.0, 0.004), (z1 - z0, 0.004), (z1 - z0, -0.004)],
        up=(0, 0, 1),
    )
    rail = sweep(
        [(x, y, z1) for x, y, _ in arc], ring_profile(0.012, 0.012, 8), up=(0, 0, 1)
    )
    return tray_md.orient_outward(), screen, rail


def build_shower(kit: Kit, f: D.Fixture, seed: int) -> None:
    """Quadrant tray, PVC wall panels on the lining and partition, curved screen, riser."""
    b0, b1, side = m(f.b0), m(f.b1), f.side
    r = b1 - b0
    z0, z1 = m(f.z0) + 0.07, m(f.z1)
    # Built in the port-aft frame, then mirrored to starboard: the corner sits where the
    # lining meets the aft wall; the screen's top meets the tumbled-home lining.
    tray, screen, rail = _quadrant_shower(r, z0, z1)
    corner = (W(b0), HULL_LINING)
    top_corner = (W(b0), lining_y(z1) - 0.005)
    for md, at, slot, label in (
        (tray, corner, "Ceramic", f.name),
        (screen, top_corner, "Glass", None),
        (rail, top_corner, "Stainless", None),
    ):
        md = md.transformed(((1, 0, 0), (0, 1, 0), (0, 0, 1)), (at[0], at[1], 0.0))
        kit.add(md if side > 0 else md.mirrored_y(), slot, label=label)
    kit.add(wall_strip(side, b0, b1 + 0.02, z0, z1, 0.008), "WhitePanel")
    face = clip_y(
        [
            (HULL_LINING - r - 0.02, z0),
            (HULL_LINING, z0),
            (HULL_LINING, Z_GUNWALE - LINING_T),
            (lining_y(Z_GUNWALE), Z_GUNWALE - LINING_T),
            (lining_y(z1), z1),
            (HULL_LINING - r - 0.02, z1),
        ],
        -1.0,
        1.0,
    )
    panel = loft([W(b0), W(b0 + 0.008)], lambda x: face, grain="across")
    kit.add(panel if side > 0 else panel.mirrored_y(), "WhitePanel")
    yr = side * (lining_y(1.85) - 0.03)
    head = [
        (W(b0 + 0.3), yr, z0 + 0.9),
        (W(b0 + 0.3), yr, z0 + 1.7),
        (W(b0 + 0.3), yr - side * 0.14, z0 + 1.76),
    ]
    kit.add(sweep(head, ring_profile(0.012, 0.012, 10), up=(1, 0, 0)), "Stainless")
    kit.add(
        lathe([(0.07, 0.0), (0.07, 0.015), (0.02, 0.03)], 18).transformed(
            ((1, 0, 0), (0, 1, 0), (0, 0, -1)), head[-1]
        ),
        "Stainless",
    )


def build_vanity(kit: Kit, f: D.Fixture, seed: int) -> None:
    b0, b1 = m(f.b0), m(f.b1)
    y_wall, y_front, side = fixture_y(f)
    top = m(f.z1) - 0.03
    kit.add(
        bbox(b0, b1, y_wall, y_front, m(f.z0) + 0.1, top, plank=rand(seed, f.name)),
        "Ash",
        label=f.name,
    )
    kit.add(bbox(b0, b1, y_wall, y_front + side * 0.06, m(f.z0), m(f.z0) + 0.1), "Ash")
    door_front(
        kit, b0, b1, y_front, m(f.z0) + 0.1, top - 0.01, side, seed, f"{f.name}d"
    )
    kit.add(
        bbox(b0 - 0.01, b1 + 0.01, y_wall, y_front - side * 0.01, top, top + 0.03),
        "OakWorktop",
    )
    bc = ((b0 + b1) / 2, (y_wall + y_front) / 2)
    bowl = lathe([(0.06, 0.0), (0.11, 0.03), (0.15, 0.09), (0.16, 0.12)], 28)
    kit.add(placed(bowl, bc[0], bc[1], top + 0.03), "Ceramic")
    ys = side * tap_y(top)
    spout = [
        (W(bc[0]), ys, top + 0.03),
        (W(bc[0]), ys, top + 0.25),
        (W(bc[0]), ys - side * 0.09, top + 0.25),
    ]
    kit.add(sweep(spout, ring_profile(0.01, 0.01, 10), up=(1, 0, 0)), "Stainless")
    mirror_z = top + 0.35
    kit.add(
        bbox(
            b0 + 0.04,
            b1 - 0.04,
            side * lining_y(mirror_z + 0.5),
            side * (lining_y(mirror_z + 0.5) - 0.012),
            mirror_z,
            mirror_z + 0.5,
        ),
        "Glass",
    )


def build_toilet(kit: Kit, f: D.Fixture, seed: int) -> None:
    """Cassette toilet facing across the boat: moulded body, bowl, seat and lid."""
    b0, b1 = m(f.b0), m(f.b1)
    y_wall, _, side = fixture_y(f)
    z0, z1 = m(f.z0), m(f.z1)
    kit.add(
        rounded_box(b0, b1, y_wall, y_wall - side * 0.22, z0, z1 - 0.02, 0.03),
        "Ceramic",
        label=f.name,
    )
    bc = (b0 + b1) / 2
    yb = y_wall - side * 0.36
    body = lathe([(0.15, 0.0), (0.16, 0.2), (0.17, 0.38), (0.18, 0.43)], 24)
    sx = ((1, 0, 0), (0, 1.2, 0), (0, 0, 1))
    kit.add(body.transformed(sx, (W(bc), yb, z0)), "Ceramic")
    seat = ring(
        [
            (
                0.18 * math.cos(2 * math.pi * k / 24),
                0.21 * math.sin(2 * math.pi * k / 24),
            )
            for k in range(24)
        ],
        [
            (
                0.11 * math.cos(2 * math.pi * k / 24),
                0.14 * math.sin(2 * math.pi * k / 24),
            )
            for k in range(24)
        ],
        0.02,
    )
    kit.add(
        seat.transformed(((1, 0, 0), (0, 1, 0), (0, 0, 1)), (W(bc), yb, z0 + 0.45)),
        "Ceramic",
    )


def build_towel_rail(kit: Kit, f: D.Fixture, seed: int) -> None:
    b0, b1, side = m(f.b0), m(f.b1), f.side
    z0, z1 = m(f.z0), m(f.z1)
    ys = side * (HULL_LINING - 0.04)
    for b in (b0, b1):
        kit.add(
            upright(loft([z0, z1], lambda z: ring_profile(0.014, 0.014, 10)), b, ys),
            "Stainless",
        )
    for z in linspace(z0 + 0.05, z1 - 0.05, 7):
        kit.add(
            sweep([(W(b0), ys, z), (W(b1), ys, z)], ring_profile(0.009, 0.009, 8)),
            "Stainless",
        )
    kit.items.append((f.name, bbox(b0, b1, ys - 0.02, ys + 0.02, z0, z1).bounds()))


def build_hearth(kit: Kit, f: D.Fixture, seed: int) -> None:
    b0, b1, y0, y1 = (m(v) for v in f.footprint())
    kit.add(bbox(b0, b1, y0, y1, m(f.z0), m(f.z1)), "TileWall", label=f.name)


def build_heat_shield(kit: Kit, f: D.Fixture, seed: int) -> None:
    """Calcium silicate board on its air gap, faced with tiles (BS 8511)."""
    b0, b1, y0, y1 = (m(v) for v in f.footprint())
    if f.side and m(f.depth) < 0.1:
        # On the lining: from 10 mm off it, following the tumblehome above the gunwale.
        kit.add(
            wall_strip(f.side, b0, b1, m(f.z0), m(f.z1), m(f.depth) - 0.01), "TileWall"
        )
        kit.items.append((f.name, bbox(b0, b1, y0, y1, m(f.z0), m(f.z1)).bounds()))
    else:
        kit.add(bbox(b0, b1, y0, y1, m(f.z0), m(f.z1)), "TileWall", label=f.name)


def build_stove(kit: Kit, f: D.Fixture, seed: int) -> None:
    """4.5 kW multi-fuel stove on legs, its glass door, and the flue up to the ceiling."""
    b0, b1, y0, y1 = (m(v) for v in f.footprint())
    zb = m(f.z0)
    for b in (b0 + 0.04, b1 - 0.04):
        for y in (y0 + 0.05, y1 - 0.05):
            kit.add(
                bbox(b - 0.02, b + 0.02, y - 0.02, y + 0.02, zb, zb + 0.06), "CastIron"
            )
    kit.add(
        rounded_box(b0, b1, y0, y1, zb + 0.06, m(f.z1), 0.02, rand(seed, "stove")),
        "CastIron",
        label=f.name,
    )
    kit.add(
        bbox(b0 + 0.06, b1 - 0.06, y0 - 0.006, y0, zb + 0.16, m(f.z1) - 0.1), "Glass"
    )
    fx, fy = N.CHIMNEY_B, N.CHIMNEY_Y
    r = m(f.opt("flue_d")) / 2
    top = ceiling_z(fy) + 0.02
    kit.add(
        upright(loft([m(f.z1), top], lambda z: ring_profile(r, r, 18)), fx, fy),
        "CastIron",
    )
    kit.add(
        placed(
            lathe([(r + 0.04, -0.02), (r + 0.04, 0.0)], 20),
            fx,
            fy,
            ceiling_z(fy) - LINING_T,
        ),
        "Brass",
    )


def build_dinette(kit: Kit, f: D.Fixture, seed: int) -> None:
    """Pullman dinette: two facing benches with backs, cushions and a pedestal table."""
    y_wall, y_aisle, side = fixture_y(f)
    seat_z = m(f.opt("seat_z"))
    back_top = m(f.z1)
    for (s0, s1), back_aft in zip(f.opt("benches"), (True, False)):
        s0, s1 = m(s0), m(s1)
        kit.add(
            bbox(
                s0,
                s1,
                y_wall,
                y_aisle,
                m(f.z0),
                seat_z - 0.08,
                plank=rand(seed, f"bench{s0}"),
            ),
            "Ash",
            label=f"{f.name}.bench{s0:.2f}",
        )
        cush = (s0 + 0.1, s1) if back_aft else (s0, s1 - 0.1)
        kit.add(
            rounded_box(
                cush[0] + 0.01,
                cush[1] - 0.01,
                y_wall - side * 0.01,
                y_aisle + side * 0.01,
                seat_z - 0.08,
                seat_z,
                0.03,
            ),
            "Upholstery",
        )
        back = (s0, s0 + 0.1) if back_aft else (s1 - 0.1, s1)
        kit.add(bbox(back[0], back[1], y_wall, y_aisle, seat_z - 0.08, back_top), "Ash")
        cb = (back[1], back[1] + 0.09) if back_aft else (back[0] - 0.09, back[0])
        kit.add(
            rounded_box(
                cb[0],
                cb[1],
                y_wall - side * 0.02,
                y_aisle + side * 0.02,
                seat_z + 0.02,
                back_top - 0.03,
                0.035,
            ),
            "Upholstery",
        )
    t0, t1 = (m(v) for v in f.opt("table"))
    top_z = m(D.FLOOR_Z) + 0.72
    y_tab = y_aisle + side * 0.04
    kit.add(
        rounded_box(
            t0 + 0.03,
            t1 - 0.03,
            y_wall - side * 0.01,
            y_tab,
            top_z - 0.03,
            top_z,
            0.012,
            rand(seed, "table"),
        ),
        "OakWorktop",
        label=f"{f.name}.table",
    )
    tc = ((t0 + t1) / 2, (y_wall + y_tab) / 2)
    kit.add(
        upright(
            loft([m(f.z0), top_z - 0.03], lambda z: ring_profile(0.035, 0.035, 14)),
            tc[0],
            tc[1],
        ),
        "Stainless",
    )
    kit.add(
        placed(
            lathe([(0.18, 0.0), (0.18, 0.012), (0.04, 0.02)], 24), tc[0], tc[1], m(f.z0)
        ),
        "Stainless",
    )


def build_seat(kit: Kit, f: D.Fixture, seed: int) -> None:
    """Sofa or armchair against the lining: plinth, seat and back cushions, arms."""
    b0, b1 = m(f.b0), m(f.b1)
    y_wall, y_front, side = fixture_y(f)
    z0, seat_z, top = m(f.z0), m(f.opt("seat_z")), m(f.z1)
    arm = 0.15
    kit.add(
        bbox(b0 + 0.02, b1 - 0.02, y_wall, y_front + side * 0.03, z0, z0 + 0.08), "Ash"
    )
    kit.add(
        rounded_box(b0, b1, y_wall, y_front, z0 + 0.08, seat_z - 0.12, 0.03),
        "Upholstery",
        label=f.name,
    )
    n = 1 if f.kind == "armchair" else 2
    for k, (c0, c1) in enumerate(split(b0 + arm, b1 - arm, (b1 - b0 - 2 * arm) / n)):
        kit.add(
            rounded_box(
                c0 + 0.005,
                c1 - 0.005,
                y_wall - side * 0.18,
                y_front,
                seat_z - 0.12,
                seat_z,
                0.04,
            ),
            "Upholstery",
        )
        kit.add(
            rounded_box(
                c0 + 0.01,
                c1 - 0.01,
                y_wall,
                y_wall - side * 0.2,
                seat_z,
                top,
                0.06,
                rand(seed, f"{f.name}b{k}"),
            ),
            "Upholstery",
        )
    for a in (b0, b1 - arm):
        kit.add(
            rounded_box(a, a + arm, y_wall, y_front, z0 + 0.08, seat_z + 0.18, 0.05),
            "Upholstery",
        )


def build_shelves(kit: Kit, f: D.Fixture, seed: int) -> None:
    """Low bookcase, kept under the gunwale ledge, with books on its shelves."""
    b0, b1 = m(f.b0), m(f.b1)
    y_wall, y_front, side = fixture_y(f)
    z0, z1 = m(f.z0), m(f.z1)
    t = 0.018
    for b in (b0, b1 - t):
        kit.add(bbox(b, b + t, y_wall, y_front, z0, z1), "Ash")
    for z in (z0, z0 + 0.1, z0 + 0.42, z0 + 0.74, z1 - t):
        kit.add(bbox(b0 + t, b1 - t, y_wall, y_front, z, z + t), "Ash")
    kit.add(bbox(b0 + t, b1 - t, y_wall, y_wall - side * 0.012, z0 + 0.1, z1), "Ash")
    kit.items.append((f.name, bbox(b0, b1, y_wall, y_front, z0, z1).bounds()))
    for k, z in enumerate((z0 + 0.12, z0 + 0.44)):
        cur = b0 + t + 0.01
        while cur < b1 - t - 0.05:
            w = 0.025 + 0.02 * rand(seed, f"book{k}{cur:.3f}")
            h = 0.2 + 0.08 * rand(seed, f"bookh{k}{cur:.3f}")
            kit.add(
                bbox(
                    cur,
                    cur + w,
                    y_wall - side * 0.02,
                    y_wall - side * 0.22,
                    z + t,
                    z + t + h,
                ),
                "Upholstery",
            )
            cur += w + 0.004


BUILDERS = {
    "steps": build_steps,
    "cupboard": build_cupboard,
    "wardrobe": build_cupboard,
    "tall_unit": build_cupboard,
    "base_run": build_base_run,
    "fridge": build_fridge,
    "cooker": build_cooker,
    "sink": build_sink,
    "wall_units": build_wall_units,
    "bed": build_bed,
    "shower": build_shower,
    "vanity": build_vanity,
    "toilet": build_toilet,
    "towel_rail": build_towel_rail,
    "hearth": build_hearth,
    "heat_shield": build_heat_shield,
    "stove": build_stove,
    "dinette": build_dinette,
    "sofa": build_seat,
    "armchair": build_seat,
    "shelves": build_shelves,
}


def lamps(seed: int) -> tuple[Kit, list]:
    """Brass wall sconces and ceiling downlights; returns the kit and light positions."""
    kit = Kit("Lamps")
    lights = []
    for side, b in SCONCES:
        z = SCONCE_Z
        y_wall = side * lining_y(z)
        arm = [
            (W(b), y_wall, z - 0.05),
            (W(b), y_wall - side * 0.08, z - 0.05),
            (W(b), y_wall - side * 0.12, z + 0.02),
        ]
        kit.add(sweep(arm, ring_profile(0.008, 0.008, 8), up=(1, 0, 0)), "Brass")
        kit.add(
            lathe([(0.03, 0.0), (0.03, 0.01)], 14).transformed(
                ((1, 0, 0), (0, 0, -side), (0, side, 0)), (W(b), y_wall, z - 0.05)
            ),
            "Brass",
        )
        shade = lathe(
            [(0.045, 0.0), (0.06, 0.05), (0.07, 0.11), (0.012, 0.11), (0.012, 0.005)],
            18,
        )
        kit.add(placed(shade, b, y_wall - side * 0.12, z + 0.02), "LampShade")
        lights.append(("sconce", (W(b), y_wall - side * 0.12, z + 0.08), 6.0))
    for b in downlights():
        z = ceiling_z(0.0) - LINING_T
        kit.add(
            placed(lathe([(0.045, -0.006), (0.045, 0.0)], 18), b, 0.0, z), "Stainless"
        )
        kit.add(
            placed(lathe([(0.03, -0.008), (0.03, -0.006)], 18), b, 0.0, z), "LampShade"
        )
        lights.append(("downlight", (W(b), 0.0, z - 0.03), 9.0))
    return kit, lights


def interior_parts(seed: int = N.SEED) -> tuple[list[Part], list, list]:
    """(parts, labelled furniture boxes, lights) for the whole fit-out."""
    parts: list[Part] = []
    for name, md in linings(seed).items():
        parts.append(
            Part(
                name,
                (name.removeprefix("LiningPort").removeprefix("LiningStarboard"),),
                md,
            )
        )
    for name, md in ceiling(seed).items():
        parts.append(Part(name, (name.removeprefix("Ceiling"),), md))
    for name, md in floors(seed).items():
        parts.append(Part(name, (name.removeprefix("Floor"),), md))
    parts.append(Part("EndLinings", ("Ash",), end_linings(seed)))
    walls, leaves = partitions(seed)
    parts.append(Part("Partitions", ("Ash",), walls))
    parts.append(Part("InteriorDoors", ("Ash",), leaves))
    kits: dict[str, Kit] = {}
    for f in D.FIXTURES:
        key = "Steps" if f.kind == "steps" else f.zone
        kit = kits.setdefault(key, Kit(key))
        BUILDERS[f.kind](kit, f, seed)
    lamp_kit, lights = lamps(seed)
    items = []
    for kit in [*kits.values(), lamp_kit]:
        parts.append(kit.part())
        items += [(f"{kit.name}.{label}", bounds) for label, bounds in kit.items]
    return parts, items, lights


def interior_cutters() -> dict[str, list]:
    """Window cutters again for the linings (same apertures)."""
    return N.cutters()
