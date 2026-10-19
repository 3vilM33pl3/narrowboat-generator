# SPDX-License-Identifier: MIT
"""Hull lines, sections and hydrostatics of the NB57 design (no bpy).

Turns the millimetre tables in narrowboat_design into continuous functions in metres, shared
by the drawing set and the 3D model: plan half-breadth, sheer, keel and stem profile, swim,
the hull and cabin sections, the waterline, and from them tank volumes, displacement, a
steel-weight estimate, the ballast it implies, air draught and the Standedge profile check.

b is metres from the aft face of the rear cabin bulkhead (+ towards the bow), y metres to
port, z metres above the underside of the baseplate.
"""

from __future__ import annotations

import math
from itertools import pairwise

import narrowboat_design as D
from procgen_mesh import Pchip, linspace

MM = 0.001


def m(v: float) -> float:
    return v * MM


STERN_TIP, STEM = m(D.STERN_TIP), m(D.STEM)
STERN_POST, SWIM_START = m(D.STERN_POST), m(D.SWIM_START)
CABIN_LEN = m(D.CABIN_LEN)
HALF_BEAM, GUNWALE_W = m(D.HALF_BEAM), m(D.GUNWALE_W)
Z_GUNWALE, Z_SIDE_TOP = m(D.Z_GUNWALE), m(D.Z_SIDE_TOP)
Z_ROOF_EDGE, Z_ROOF_CROWN = m(D.Z_ROOF_EDGE), m(D.Z_ROOF_CROWN)
CORNER_R, CAMBER = m(D.CORNER_R), m(D.CAMBER)
CABIN_BASE_HALF, CABIN_TOP_HALF = m(D.CABIN_BASE_HALF), m(D.CABIN_TOP_HALF)
Z_COUNTER, CHINE = m(D.Z_COUNTER), m(D.CHINE)

BOW_START = STEM - m(D.BOW_LEN)
SHEER_START = STEM - m(D.SHEER_LEN)
FOREFOOT_START = STEM - m(D.FOREFOOT_LEN)
# The counter closes on an ellipse from just aft of the stern post to the tip.
COUNTER_START = STERN_POST + 0.05
COUNTER_A = COUNTER_START - STERN_TIP


def _curve(knots) -> Pchip:
    return Pchip([(m(d), m(v)) for d, v in knots])


_BOW = _curve(D.BOW_PLAN)
_SHEER = _curve(D.SHEER)
_FOREFOOT = _curve(D.FOREFOOT)
_SWIM = _curve(D.SWIM)
_SWIM_Z = _curve(D.SWIM_Z)
SWIM_FLARE = m(D.SWIM_FLARE)


def plan_half(b: float) -> float:
    """Half-breadth over the plating at the gunwale."""
    if b >= BOW_START:
        return _BOW(b - BOW_START)
    if b <= COUNTER_START:
        t = (b - COUNTER_START) / COUNTER_A
        return HALF_BEAM * math.sqrt(max(0.0, 1 - t * t))
    return HALF_BEAM


def gunwale_z(b: float) -> float:
    return _SHEER(b - SHEER_START) if b > SHEER_START else Z_GUNWALE


def bottom_z(b: float) -> float:
    """Height of the keel line: flat, then rising round the forefoot to the stem."""
    return _FOREFOOT(b - FOREFOOT_START) if b > FOREFOOT_START else 0.0


def swim_half(b: float) -> float:
    """Half-breadth of the bottom: full beam forward of the swim, closing to the stern post."""
    if b >= SWIM_START:
        return plan_half(b)
    return min(_SWIM(SWIM_START - b), plan_half(b))


def swim_full_z(b: float) -> float:
    """Height above which the hull is full breadth again at b."""
    return _SWIM_Z(SWIM_START - b) if b < SWIM_START else _SWIM_Z(0.0)


def hull_half_section(b: float):
    """Outer half-section (+Y): keel centre, chine, swim flare, side, gunwale's inner edge.

    Every station has the same seven points so the hull can be lofted; forward of the swim
    the flare points fall on the vertical side.
    """
    zg = gunwale_z(b)
    zb = min(bottom_z(b), zg - 0.12)
    w = plan_half(b)
    low = min(swim_half(b), w)
    z_low = zb + CHINE
    z1 = min(max(swim_full_z(b), z_low + 0.02), zg - 0.03)
    z0 = min(max(z1 - SWIM_FLARE, z_low + 0.01), z1 - 0.01)
    return [
        (0.0, zb),
        (max(low - CHINE, 0.004), zb),
        (low, z_low),
        (low, z0),
        (w, z1),
        (w, zg),
        (max(w - GUNWALE_W, 0.012), zg),
    ]


def outer_half_at(b: float, z: float) -> float:
    """Outer half-breadth of the hull plating at (b, z); 0 below the keel or the counter."""
    if b < STERN_POST:
        return plan_half(b) if Z_COUNTER <= z <= Z_GUNWALE else 0.0
    sec = hull_half_section(b)[:6]
    if z < sec[0][1]:
        return 0.0
    for (y0, z0), (y1, z1) in pairwise(sec[1:]):
        if z0 <= z <= z1:
            return y0 if z1 == z0 else y0 + (y1 - y0) * (z - z0) / (z1 - z0)
    return sec[-1][0] if z <= sec[-1][1] else 0.0


def inner_half_at(b: float, z: float) -> float:
    """Inside face of the side plating at (b, z)."""
    return max(outer_half_at(b, z) - m(D.T_HULL), 0.0)


# --------------------------------------------------------------------------- cabin

SIDE_SLOPE = (CABIN_BASE_HALF - CABIN_TOP_HALF) / (Z_SIDE_TOP - Z_GUNWALE)
ROOF_YC = CABIN_TOP_HALF - CORNER_R  # where the corner radius meets the camber


def cabin_side_y(z: float) -> float:
    """Outer face of the port cabin side at height z (tumblehome)."""
    return CABIN_BASE_HALF - SIDE_SLOPE * (z - Z_GUNWALE)


def roof_z(y: float) -> float:
    """Top of the roof plate at y: cambered top, then round the corner."""
    y = abs(y)
    if y <= ROOF_YC:
        return Z_ROOF_EDGE + CAMBER * (1 - (y / ROOF_YC) ** 2)
    dy = min(y - ROOF_YC, CORNER_R)
    return Z_SIDE_TOP + math.sqrt(max(CORNER_R * CORNER_R - dy * dy, 0.0))


def roof_outer(segments: int = 28):
    """Port-to-starboard roof outline: rounded corners into a cambered top."""
    pts = []
    for k in range(7):
        a = math.pi / 2 * k / 6
        pts.append(
            (ROOF_YC + CORNER_R * math.cos(a), Z_SIDE_TOP + CORNER_R * math.sin(a))
        )
    for k in range(1, segments):
        y = ROOF_YC - 2 * ROOF_YC * k / segments
        pts.append((y, Z_ROOF_EDGE + CAMBER * (1 - (y / ROOF_YC) ** 2)))
    for k in range(7):
        a = math.pi / 2 + math.pi / 2 * k / 6
        pts.append(
            (-ROOF_YC + CORNER_R * math.cos(a), Z_SIDE_TOP + CORNER_R * math.sin(a))
        )
    return pts


def lining_in(z: float) -> float:
    """Inner face of the port lining at height z: the hull lining up to the gunwale, then
    the cabin lining following the tumblehome."""
    if z < Z_GUNWALE:
        return m(D.LINING_IN)
    return cabin_side_y(z) - m(D.T_CABIN + D.INSULATION_SIDE + D.PLY)


def ceiling_z(y: float) -> float:
    """Underside of the ceiling ply at y, parallel to the roof camber."""
    return roof_z(y) - m(D.T_ROOF + D.INSULATION_ROOF + D.PLY)


def handrail_y() -> float:
    return CABIN_TOP_HALF - m(D.HANDRAIL_INSET)


def handrail_top() -> float:
    return roof_z(handrail_y()) + m(D.HANDRAIL_H + D.HANDRAIL_D / 2)


# --------------------------------------------------------------------------- waterline


def wl_z(b: float) -> float:
    """Design waterline height at b, trimmed by the stern."""
    t = (b - STERN_POST) / (STEM - STERN_POST)
    return m(D.DRAUGHT_AFT + (D.DRAUGHT_FWD - D.DRAUGHT_AFT) * t)


def _clip_area_below(poly, z_cut: float) -> float:
    """Area of a closed polygon below z = z_cut (Sutherland-Hodgman, then shoelace)."""
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        pin, qin = p[1] <= z_cut, q[1] <= z_cut
        if pin:
            out.append(p)
        if pin != qin:
            t = (z_cut - p[1]) / (q[1] - p[1])
            out.append((p[0] + (q[0] - p[0]) * t, z_cut))
    return abs(
        sum(a[0] * c[1] - c[0] * a[1] for a, c in zip(out, out[1:] + out[:1])) / 2
    )


def section_area(b: float, z_cut: float) -> float:
    """Immersed area of the full hull section at b below z_cut (m²)."""
    if b < STERN_POST:
        return 2 * plan_half(b) * max(0.0, min(z_cut, Z_GUNWALE) - Z_COUNTER)
    half = hull_half_section(b)[:6]
    poly = half + [(0.0, half[-1][1])]
    return 2 * _clip_area_below(poly, z_cut)


def _integrate(f, a: float, b: float, step: float = 0.01) -> float:
    n = max(2, round((b - a) / step) + 1)
    xs = linspace(a, b, n)
    ys = [f(x) for x in xs]
    return sum((ys[i] + ys[i + 1]) / 2 * (xs[i + 1] - xs[i]) for i in range(n - 1))


def displacement_m3() -> float:
    return _integrate(lambda b: section_area(b, wl_z(b)), STERN_TIP, STEM)


def displacement_kg() -> float:
    return displacement_m3() * D.WATER_DENSITY


def tank_volume_l(tank) -> float:
    """Volume of an integral tank (b0, b1, z0, z1 in mm) bounded by the plating."""
    b0, b1, z0, z1 = (m(v) for v in tank)

    def area(b: float) -> float:
        return 2 * _integrate(lambda z: inner_half_at(b, z), z0, z1, 0.005)

    return _integrate(area, b0, b1, 0.01) * 1000


def locker_lid_z() -> float:
    """Underside of the gas locker lid, level at the lowest point of the sheer over it."""
    return gunwale_z(m(D.GAS_LOCKER[0])) + m(D.LOCKER_COAMING) - 0.01


def locker_sole_z() -> float:
    return locker_lid_z() - m(D.LOCKER_DEPTH)


def locker_drain_margin() -> float:
    """Height of the drain outlet (at the sole) above the waterline at the locker (m)."""
    return locker_sole_z() - wl_z(m(D.GAS_LOCKER[0]))


def cylinder_positions() -> list[tuple[float, float]]:
    """Centres (b, y) of the two propane cylinders, side by side across the locker's aft end
    where it is widest."""
    d, _ = D.GAS_CYLINDER
    b = m(D.GAS_LOCKER[0]) + m(d) / 2 + 0.03
    y = m(d) / 2 + 0.02
    return [(b, y), (b, -y)]


# --------------------------------------------------------------------------- weights

STEEL = D.STEEL_DENSITY


def _section_perimeter(b: float) -> float:
    """Half-section plating length from the chine to the gunwale (m)."""
    sec = hull_half_section(b)[1:6]
    return sum(math.dist(p, q) for p, q in pairwise(sec))


def steel_breakdown_kg() -> dict[str, float]:
    """Plate areas times thickness, plus a framing allowance (kg)."""
    t = {k: m(v) for k, v in (("base", D.T_BASE), ("hull", D.T_HULL))}
    t.update(cabin=m(D.T_CABIN), roof=m(D.T_ROOF), bulkhead=m(D.T_BULKHEAD))

    def kg(area: float, thick: float) -> float:
        return area * thick * STEEL

    base = _integrate(lambda b: 2 * hull_half_section(b)[1][0], STERN_POST, STEM)
    sides = _integrate(lambda b: 2 * _section_perimeter(b), STERN_POST, STEM)
    counter = _integrate(
        lambda b: 2 * (Z_GUNWALE - Z_COUNTER) + 2 * plan_half(b),
        STERN_TIP,
        STERN_POST,
    )
    gunwale = 2 * GUNWALE_W * (STEM - STERN_TIP)
    decks = _integrate(lambda b: 2 * plan_half(b), STERN_TIP, m(D.SHELTER_AFT))
    decks += _integrate(lambda b: 2 * plan_half(b), CABIN_LEN, STEM)
    slant = math.hypot(Z_SIDE_TOP - Z_GUNWALE, CABIN_BASE_HALF - CABIN_TOP_HALF)
    cabin = 2 * slant * (CABIN_LEN - m(D.SHELTER_AFT))
    roof_len = sum(math.dist(p, q) for p, q in pairwise(roof_outer()))
    roof = roof_len * CABIN_LEN
    bulkhead = 2 * (2 * CABIN_BASE_HALF * (Z_ROOF_CROWN - 0.0) * 0.92)
    strakes = (
        len(D.Z_STRAKES) * 2 * (STEM - STERN_TIP) * m(D.STRAKE_H) * m(D.STRAKE_PROUD)
    )
    plates = {
        "Baseplate": kg(base, t["base"]),
        "Hull sides and counter": kg(sides + counter, t["hull"]),
        "Gunwales and decks": kg(gunwale + decks, t["hull"]),
        "Cabin sides and shelter": kg(cabin, t["cabin"]),
        "Roof": kg(roof, t["roof"]),
        "Bulkheads": kg(bulkhead, t["bulkhead"]),
        "Rubbing strakes": strakes * STEEL,
    }
    plates["Framing allowance"] = D.FRAMING_ALLOWANCE * sum(plates.values())
    return plates


def weight_estimate_kg() -> dict[str, float]:
    """Lightship and load items, with the ballast that brings the boat to its waterline."""
    out = {"Steelwork": sum(steel_breakdown_kg().values()), "Fit-out": D.FITOUT_KG}
    out.update(dict(D.ITEMS_KG))
    out["Fresh water"] = tank_volume_l(D.WATER_TANK) * D.WATER_DENSITY / 1000
    out["Diesel"] = tank_volume_l(D.FUEL_TANK) * D.DIESEL_DENSITY / 1000
    out["Ballast"] = displacement_kg() - sum(out.values())
    return out


# --------------------------------------------------------------------------- gauge


def exterior_half_at(z: float) -> float:
    """Half-width of the boat's fixed profile at height z over the cabin: hull and strakes,
    cabin side, roof and handrails (chimney and vents are removable)."""
    if z <= Z_GUNWALE:
        return HALF_BEAM + m(D.STRAKE_PROUD)
    half = 0.0
    if z <= Z_SIDE_TOP:
        half = cabin_side_y(z)
    elif z <= Z_ROOF_CROWN:
        # Invert the roof outline: widest y with roof_z(y) >= z.
        lo, hi = 0.0, CABIN_TOP_HALF
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if roof_z(mid) >= z else (lo, mid)
        half = lo
    if z <= handrail_top():
        half = max(half, handrail_y() + m(D.HANDRAIL_D) / 2)
    return half


def block_top() -> float:
    """Top of the pole-and-plank blocks, which sit on the roof at their outboard end."""
    return roof_z(m(D.BLOCK_Y[0])) + m(D.BLOCK_H)


def plank_top() -> float:
    return block_top() + m(D.PLANK_T)


def pole_top() -> float:
    return block_top() + m(D.POLE_D)


def _air_runs() -> tuple:
    """(top, b at the forward end) of each fixed run on the roof."""
    return (
        (Z_ROOF_CROWN, CABIN_LEN),
        (handrail_top(), m(D.HANDRAIL_B[1])),
        (plank_top(), m(D.PLANK_B[1])),
        (pole_top(), m(D.POLE_B + D.POLE_LEN)),
    )


def air_draught_point() -> tuple[float, float]:
    """(b, z) of the point that sets the air draught: the highest above the local waterline
    with the chimney and vents removed, at the forward end of its run where the boat floats
    highest."""
    top, b = max(_air_runs(), key=lambda r: r[0] - wl_z(r[1]))
    return b, top


def air_draught() -> float:
    b, top = air_draught_point()
    return top - wl_z(b)


def standedge_margins() -> list[tuple[int, float]]:
    """(row height mm, allowed width - our width at that height above water, m) for every
    row of the Standedge profile, at the cabin's highest-floating end."""
    wl = min(wl_z(0.0), wl_z(CABIN_LEN))
    return [
        (h, m(wmax) - 2 * exterior_half_at(wl + m(h)))
        for h, wmax in D.STANDEDGE_PROFILE
    ]


# --------------------------------------------------------------------------- tiller


def swan_neck() -> list[tuple[float, float]]:
    """(b, z) path of the swan neck from the rudder stock up, over and down to the bar."""
    xs, top = m(D.RUDDER_STOCK_B), m(D.DECK_Z + D.STOCK_TOP)
    rise, reach = m(D.NECK_RISE), m(D.NECK_REACH)
    pts = [(xs, top - 0.02)]
    for k in range(1, 13):
        a = math.pi * k / 12
        pts.append(
            (xs + 0.38 * reach * (1 - math.cos(a)), top + rise * math.sin(a) + 0.02 * k)
        )
    pts += [(xs + 0.88 * reach, top + 0.22), (xs + reach, top + 0.12)]
    return pts


def tiller_bar() -> tuple[tuple[float, float], tuple[float, float]]:
    """(b, z) ends of the wooden tiller bar, sloping gently down forward."""
    b0, z0 = swan_neck()[-1]
    return (b0, z0), (b0 + m(D.TILLER_BAR), z0 - 0.1)
