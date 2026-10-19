# SPDX-License-Identifier: MIT
"""Pure geometry for narrowboat NB57 'Wren' (no bpy), built from narrowboat_design.

Every dimension comes from narrowboat_design (millimetres), through narrowboat_lines for the
hull lines and sections, so the model is built from the same numbers as the A3 drawing set
(narrowboat_blueprint). b is metres from the aft face of the rear cabin bulkhead; `wx(mm)`
maps design millimetres to world x. World axes: +X towards the bow, +Y to port, Z up, z = 0
on the underside of the baseplate, origin at mid-hull. Every part is a closed, outward-wound
solid (see procgen_mesh).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import canal_castles as K
import canal_roses as C
import narrowboat_design as D
import narrowboat_lines as L
from procgen_mesh import (
    MeshData,
    Part,
    axis_frame,
    box,
    circle,
    lathe,
    linspace,
    loft,
    mirror_full,
    offset_polyline,
    ring,
    ring_profile,
    rounded_rect,
    seeded,
    sweep,
)

# 2: canal roses; 3: castles on the end doors; 4: and on the hatch; 5: the original NB57
BUILD_VERSION = 5
SEED = 1
# Everything the build writes lands under build/ at the repository root.
ASSET_ROOT = Path(__file__).resolve().parents[1] / "build"
OUTPUT_DIR = ASSET_ROOT / "NB57"
BLEND_NAME = "Narrowboat_NB57.blend"
DRAWINGS = ASSET_ROOT / "Drawings"
# Equirectangular HDRI for the lakeside shots (--hdri); without one they fall back to the
# studio world, still afloat on the water ring.
HDRI: Path | None = None

m = L.m
CABIN_LEN, STERN_TIP, STEM = L.CABIN_LEN, L.STERN_TIP, L.STEM
STERN_POST = L.STERN_POST
X_MID = (STERN_TIP + STEM) / 2  # world origin at mid-hull
LOA = STEM - STERN_TIP


def wx(mm: float) -> float:
    """Design millimetres from the rear bulkhead -> world x (m)."""
    return mm / 1000 - X_MID


def bx(x: float) -> float:
    """World x -> metres from the rear bulkhead."""
    return x + X_MID


# Sections (m).
HALF_BEAM, GUNWALE_W = L.HALF_BEAM, L.GUNWALE_W
PLATE, CABIN_PLATE = m(D.T_BASE), m(D.T_CABIN)
Z_FLOOR, Z_GUNWALE = m(D.FLOOR_Z), L.Z_GUNWALE
Z_STRAKES = tuple(m(z) for z in D.Z_STRAKES)
Z_SIDE_TOP, Z_ROOF_EDGE, Z_ROOF_CROWN = L.Z_SIDE_TOP, L.Z_ROOF_EDGE, L.Z_ROOF_CROWN
Z_COUNTER, DECK_Z = L.Z_COUNTER, m(D.DECK_Z)
CABIN_BASE_HALF, CABIN_TOP_HALF, CORNER_R = (
    L.CABIN_BASE_HALF,
    L.CABIN_TOP_HALF,
    L.CORNER_R,
)
STRAKE = (m(D.STRAKE_H), m(D.STRAKE_PROUD))
SHELTER_AFT = m(D.SHELTER_AFT)
SIDE_SLOPE = L.SIDE_SLOPE

plan_half, gunwale_z, bottom_z = L.plan_half, L.gunwale_z, L.bottom_z
swim_half, hull_half_section = L.swim_half, L.hull_half_section
cabin_side_y, roof_outer = L.cabin_side_y, L.roof_outer


def u_ring(outer_half, thickness):
    """Closed U-shaped plate ring (port tip, round the keel, starboard tip, back inside)."""
    inner_half = offset_polyline(outer_half, [thickness] * (len(outer_half) - 1))
    return mirror_full(outer_half) + list(reversed(mirror_full(inner_half)))


def hull_stations() -> list[float]:
    stations = (
        linspace(STERN_POST, L.SWIM_START, 30)
        + linspace(L.SWIM_START + 0.4, L.BOW_START - 0.4, 18)
        + linspace(L.BOW_START, STEM - 0.28, 34)
        + linspace(STEM - 0.24, STEM - 0.02, 12)
    )
    return [wx(1000 * b) for b in sorted({round(s, 6) for s in stations})]


def hull_shell(seed: int) -> MeshData:
    """Baseplate, swim, sides, bow and gunwale shelf as one plate solid."""
    return loft(
        hull_stations(),
        lambda x: u_ring(hull_half_section(bx(x)), PLATE),
        plank=seeded("nb", seed, "hull").random(),
    )


def counter_shell(seed: int) -> MeshData:
    """The round counter over the propeller, from the stern post to the stern tip."""

    def ring_at(x):
        w = max(plan_half(bx(x)), 0.09)
        return u_ring(
            [
                (0.0, Z_COUNTER),
                (w, Z_COUNTER),
                (w, Z_GUNWALE),
                (w - GUNWALE_W, Z_GUNWALE),
            ],
            PLATE,
        )

    xs = [wx(1000 * b) for b in linspace(STERN_TIP + 0.02, STERN_POST + 0.06, 24)]
    md = loft(xs, ring_at, plank=seeded("nb", seed, "counter").random())
    # Close the end of the round counter with a small flat plate.
    w = max(plan_half(STERN_TIP + 0.02), 0.09)
    end = [(-w, Z_COUNTER), (w, Z_COUNTER), (w, Z_GUNWALE), (-w, Z_GUNWALE)]
    x_end = wx(1000 * (STERN_TIP + 0.02))
    md.extend(loft([x_end - PLATE, x_end + 0.001], lambda x: end, grain="across"))
    return md


def rubbing_strakes(seed: int) -> MeshData:
    """Half-round steel strakes along both sides and round the counter."""
    md = MeshData()
    h, out = STRAKE
    for z in Z_STRAKES:
        bs = linspace(STERN_TIP + 0.06, STEM - 0.35, 140)

        def prof(x, z=z):
            w = plan_half(bx(x))
            return [
                (w - 0.004, z - h / 2),
                (w + out * 0.6, z - h / 2),
                (w + out, z),
                (w + out * 0.6, z + h / 2),
                (w - 0.004, z + h / 2),
            ]

        side = loft(
            [wx(1000 * b) for b in bs],
            prof,
            plank=seeded("nb", seed, f"strake{z}").random(),
        )
        md.extend(side)
        md.extend(side.mirrored_y())
    return md


SEAT_DEPTH, SEAT_H = m(D.SEAT_LOCKERS[2]), m(D.SEAT_LOCKERS[3])
SEAT_B = (m(D.SEAT_LOCKERS[0]), m(D.SEAT_LOCKERS[1]))
BOARDS_HALF = (
    CABIN_BASE_HALF - CABIN_PLATE - SEAT_DEPTH
)  # engine boards between the lockers


def rear_deck(seed: int) -> tuple[MeshData, MeshData]:
    """(steel deck from the counter to the rear bulkhead with the weed hatch lid, timber
    engine boards between the seat lockers)."""
    steel, boards = MeshData(), MeshData()
    bs = linspace(STERN_TIP + 0.02, 0.0, 24)

    def prof(x):
        w = max(plan_half(bx(x)) - GUNWALE_W + 0.002, 0.08)
        return [(-w, DECK_Z - 0.012), (w, DECK_Z - 0.012), (w, DECK_Z), (-w, DECK_Z)]

    steel.extend(loft([wx(1000 * b) for b in bs], prof))
    w0, w1, wh = D.WEED_HATCH
    steel.extend(box(wx(w0), wx(w1), -m(wh), m(wh), DECK_Z, DECK_Z + 0.01))
    b0, b1 = D.ENGINE_BOARDS
    n = round((b1 - b0) / 290)
    for k in range(n):
        c0 = b0 + (b1 - b0) * k / n
        c1 = b0 + (b1 - b0) * (k + 1) / n
        boards.extend(
            box(
                wx(c0 + 3),
                wx(c1 - 3),
                -BOARDS_HALF,
                BOARDS_HALF,
                DECK_Z,
                DECK_Z + 0.022,
                grain="across",
                plank=seeded("nb", seed, f"board{k}").random(),
            )
        )
    return steel, boards


def seat_locker(seed: int, side: int) -> tuple[MeshData, MeshData]:
    """(painted locker box against the shelter side, wooden lid) on one side."""
    # The shelter side leans in (tumblehome), so the box stops inside it at lid height.
    y_out = side * (cabin_side_y(DECK_Z + SEAT_H) - CABIN_PLATE - 0.003)
    y_in = side * BOARDS_HALF
    body = box(
        wx(1000 * SEAT_B[0]),
        wx(1000 * SEAT_B[1]),
        min(y_out, y_in),
        max(y_out, y_in),
        DECK_Z,
        DECK_Z + SEAT_H - 0.025,
    )
    lid = box(
        wx(1000 * SEAT_B[0]) - 0.012,
        wx(1000 * SEAT_B[1]) + 0.012,
        min(y_out, y_in - side * 0.015),
        max(y_out, y_in - side * 0.015),
        DECK_Z + SEAT_H - 0.025,
        DECK_Z + SEAT_H,
        grain="x",
        plank=seeded("nb", seed, f"seatlid{side}").random(),
    )
    return body, lid


# --------------------------------------------------------------------------- cabin


def cabin_side(seed: int, side: int) -> MeshData:
    """One cabin side plate from the gunwale to the roof corner, running aft past the rear
    bulkhead as the semi-trad shelter (cut by the window cutters)."""
    y0, y1 = CABIN_BASE_HALF, cabin_side_y(Z_SIDE_TOP)
    nx, nz = 1.0, SIDE_SLOPE  # inward offset direction for the plate thickness
    t = CABIN_PLATE / math.hypot(nx, nz)
    prof = [
        (y0, Z_GUNWALE),
        (y1, Z_SIDE_TOP),
        (y1 - t, Z_SIDE_TOP),
        (y0 - t, Z_GUNWALE),
    ]
    md = loft(
        [wx(1000 * SHELTER_AFT), wx(1000 * CABIN_LEN)],
        lambda x: prof,
        plank=seeded("nb", seed, f"cabinside{side}").random(),
    )
    return md if side > 0 else md.mirrored_y()


def shelter_rail(side: int) -> MeshData:
    """Capping rail along the top of the semi-trad side plate."""
    y = side * (cabin_side_y(Z_SIDE_TOP) - CABIN_PLATE / 2)
    z = Z_SIDE_TOP + 0.016
    path = [(wx(1000 * SHELTER_AFT) + 0.01, y, z), (wx(0), y, z)]
    return sweep(path, ring_profile(0.016, 0.016, 10), up=(0, 0, 1))


def roof(seed: int) -> MeshData:
    outer = roof_outer()
    # Travelling port -> starboard over the top, the left of travel is inside the cabin.
    inner = offset_polyline(outer, [m(D.T_ROOF)] * (len(outer) - 1))
    ring_pts = outer + list(reversed(inner))
    return loft(
        [wx(0), wx(1000 * CABIN_LEN)],
        lambda x: ring_pts,
        plank=seeded("nb", seed, "roof").random(),
    )


def cabin_section_inner():
    """Inside outline of the cabin above the gunwale (for bulkhead plates)."""
    top = offset_polyline(roof_outer(), [m(D.T_ROOF)] * (len(roof_outer()) - 1))
    yb = CABIN_BASE_HALF - CABIN_PLATE
    return [(yb, Z_GUNWALE), *top, (-yb, Z_GUNWALE)]


def bulkhead_outline(b: float):
    """Inside outline at b: the hull plating's inner face up to the gunwale shelf, then the
    cabin's inside up and over the roof."""
    half = hull_half_section(b)[:6]
    inner = offset_polyline(half, [PLATE] * 5)
    shelf = Z_GUNWALE - PLATE
    inner[-1] = (inner[-1][0], shelf)
    cabin = cabin_section_inner()
    yb = cabin[0][0]
    port = inner + [(yb, shelf)]
    starboard = [(-y, z) for y, z in reversed(port)]
    return port + cabin + starboard[:-1]


def bulkheads(seed: int) -> MeshData:
    """Rear and front cabin bulkheads: steel plates filling the hull and cabin section."""
    md = MeshData()
    t = m(D.T_BULKHEAD)
    for b0, b1, key in ((0.0, t, "stern"), (CABIN_LEN - t, CABIN_LEN, "bow")):
        poly = bulkhead_outline(b1 if key == "stern" else b0)
        md.extend(
            loft(
                [wx(1000 * b0), wx(1000 * b1)],
                lambda x, poly=poly: poly,
                grain="across",
                plank=seeded("nb", seed, f"bulkhead{key}").random(),
            )
        )
    return md


# --------------------------------------------------------------------------- openings


@dataclass(frozen=True)
class Opening:
    """A cabin-side aperture in metres: centre (mm from the rear bulkhead), side (+1 port),
    and the height of its centre."""

    name: str
    side: int
    centre_mm: float
    width: float
    height: float
    radius: float
    kind: str = "window"  # window | porthole
    zc: float = m(D.WINDOW_ZC)


OPENINGS = tuple(
    Opening(op.name, op.side, op.b, m(op.w), m(op.h), m(op.r), op.kind, m(op.zc))
    for op in D.OPENINGS
)
FLANGE, RIM = m(D.FRAME_FLANGE), m(D.RIM_W)
HATCH = D.SIDE_HATCH
HATCH_SIDE = HATCH.side
SIDE_NAMES = {1: "Port", -1: "Starboard"}
HATCH_NAME = SIDE_NAMES[HATCH_SIDE]
CHIMNEY_B, CHIMNEY_Y = m(D.CHIMNEY[0]), m(D.CHIMNEY[1])


def side_frame(side: int, z: float):
    """Rotation and origin for drawing on the cabin side at height z: local X along the
    boat, local Y up the side, local Z into the cabin."""
    up = (0.0, -SIDE_SLOPE * side, 1.0)
    norm = math.hypot(up[1], up[2])
    up = (0.0, up[1] / norm, up[2] / norm)
    along = (1.0, 0.0, 0.0) if side > 0 else (-1.0, 0.0, 0.0)
    into = (
        along[1] * up[2] - along[2] * up[1],
        along[2] * up[0] - along[0] * up[2],
        along[0] * up[1] - along[1] * up[0],
    )
    rot = tuple(tuple((along, up, into)[c][r] for c in range(3)) for r in range(3))
    origin = (0.0, side * cabin_side_y(z), z)
    return rot, origin


def on_side(local: MeshData, side: int, centre_mm: float, z: float) -> MeshData:
    rot, origin = side_frame(side, z)
    return local.transformed(rot, (wx(centre_mm), origin[1], origin[2]))


def opening_outline(op: Opening):
    if op.kind == "porthole":
        return circle(op.width / 2, 32)
    return rounded_rect(op.width, op.height, op.radius, 5)


def _prism_local(poly, z0, z1, mat=0):
    """Closed prism of a 2D (u, v) loop between local z0 and z1 (local Z into the cabin)."""
    md = MeshData()
    n = len(poly)
    a = [md.add_vert((u, v, z0)) for u, v in poly]
    b = [md.add_vert((u, v, z1)) for u, v in poly]
    for j in range(n):
        k = (j + 1) % n
        md.add_face(
            (a[j], a[k], b[k], b[j]), ((0, 0), (1, 0), (1, 1), (0, 1)), mat, 0.0
        )
    md.add_face(tuple(reversed(a)), [p for p in reversed(poly)], mat, 0.0)
    md.add_face(tuple(b), list(poly), mat, 0.0)
    return md.orient_outward()


def opening_cutter(op: Opening) -> MeshData:
    """Boolean cutter through the steel side and the interior lining."""
    return on_side(
        _prism_local(opening_outline(op), -0.05, 0.16), op.side, op.centre_mm, op.zc
    )


def flue_cutter() -> MeshData:
    x = wx(1000 * CHIMNEY_B)
    return box(x - 0.075, x + 0.075, CHIMNEY_Y - 0.075, CHIMNEY_Y + 0.075, 1.8, 2.3)


IDENTITY = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def opening_frames(seed: int) -> tuple[MeshData, MeshData, MeshData]:
    """(gold window frames, brass porthole rims, glass panes) for every opening."""
    frames, rims, glass = MeshData(), MeshData(), MeshData()
    for op in OPENINGS:
        outline = opening_outline(op)
        if op.kind == "porthole":
            rim = ring(circle(op.width / 2 + RIM, 32), outline, 0.025)
            rims.extend(
                on_side(
                    rim.transformed(IDENTITY, (0, 0, -0.025)),
                    op.side,
                    op.centre_mm,
                    op.zc,
                )
            )
        else:
            flange = rounded_rect(
                op.width + 2 * FLANGE, op.height + 2 * FLANGE, op.radius + FLANGE, 5
            )
            frame = ring(flange, outline, 0.018)
            frames.extend(
                on_side(
                    frame.transformed(IDENTITY, (0, 0, -0.018)),
                    op.side,
                    op.centre_mm,
                    op.zc,
                )
            )
        pane = _prism_local(outline, 0.002, 0.008)
        glass.extend(on_side(pane, op.side, op.centre_mm, op.zc))
    return frames, rims, glass


# The cabin-side coachline panel: the cream line's outer edge runs PANEL_B along the side
# (metres from the rear bulkhead) and PANEL_H up the slope, centred at PANEL_ZC; the red
# line sits RED_INSET inside it.
PANEL_B = (m(D.PANEL_B[0]), m(D.PANEL_B[1]))
PANEL_Z = (m(D.PANEL_Z[0]), m(D.PANEL_Z[1]))
PANEL_H = PANEL_Z[1] - PANEL_Z[0]
PANEL_ZC = sum(PANEL_Z) / 2
CREAM_W, RED_INSET, RED_W = 0.014, 0.04, 0.008
PANEL_INNER = (
    RED_INSET + RED_W
)  # from the cream line's outer edge to the red line's inner


def coachlines(seed: int) -> tuple[MeshData, MeshData]:
    """Traditional cream outer and red inner panel lines on both cabin sides."""
    cream, red = MeshData(), MeshData()
    length = PANEL_B[1] - PANEL_B[0]
    centre_mm = 1000 * sum(PANEL_B) / 2
    for side in (1, -1):
        for target, inset, width in ((cream, 0.0, CREAM_W), (red, RED_INSET, RED_W)):
            outer = rounded_rect(
                length - 2 * inset, PANEL_H - 2 * inset, 0.09 - inset, 6
            )
            inner = rounded_rect(
                length - 2 * inset - 2 * width,
                PANEL_H - 2 * inset - 2 * width,
                0.09 - inset - width,
                6,
            )
            line = ring(outer, inner, 0.0015).transformed(IDENTITY, (0, 0, -0.0016))
            target.extend(on_side(line, side, centre_mm, PANEL_ZC))
    return cream, red


# --------------------------------------------------------------------------- deck gear


def _up(md: MeshData, x: float, y: float, z: float) -> MeshData:
    return md.transformed(IDENTITY, (x, y, z))


def handrail_legs() -> list[float]:
    """b (metres from the rear bulkhead) of the intermediate legs under each handrail; the
    rail's own ends also come down to the roof, at HANDRAIL_B."""
    b0, b1 = m(D.HANDRAIL_B[0]), m(D.HANDRAIL_B[1])
    return linspace(b0 + 0.6, b1 - 0.6, 9)


def roof_gear(seed: int) -> tuple[MeshData, MeshData, MeshData]:
    """(painted handrails, brass vents and chimney bands, black chimney) on the roof."""
    rails, brass, chimney = MeshData(), MeshData(), MeshData()
    y_rail = L.handrail_y()
    z_top = L.roof_z(y_rail)
    r = m(D.HANDRAIL_D) / 2
    h = m(D.HANDRAIL_H)
    b0, b1 = m(D.HANDRAIL_B[0]), m(D.HANDRAIL_B[1])
    for side in (1, -1):
        y = side * y_rail
        path = [(wx(1000 * b0), y, z_top), (wx(1000 * b0), y, z_top + h - 0.005)]
        path += [
            (wx(1000 * b), y, z_top + h) for b in linspace(b0 + 0.08, b1 - 0.08, 30)
        ]
        path += [(wx(1000 * b1), y, z_top + h - 0.005), (wx(1000 * b1), y, z_top)]
        rails.extend(sweep(path, ring_profile(r, r, 10), up=(0, 0, 1)))
        for b in handrail_legs():
            rails.extend(
                box(
                    wx(1000 * b) - 0.015,
                    wx(1000 * b) + 0.015,
                    y - 0.011,
                    y + 0.011,
                    z_top - 0.005,
                    z_top + h - 0.005,
                )
            )
    vr, vh = m(D.VENT_D) / 2, m(D.VENT_H)
    dome = [
        (vr, 0.0),
        (vr, 0.35 * vh),
        (0.93 * vr, 0.58 * vh),
        (0.73 * vr, 0.81 * vh),
        (0.4 * vr, 0.95 * vh),
        (0.05 * vr, vh),
    ]
    for b in D.VENT_BS:
        brass.extend(_up(lathe(dome, 24), wx(b), 0.0, Z_ROOF_CROWN - 0.004))
    x = wx(D.CHIMNEY[0])
    z0 = L.roof_z(CHIMNEY_Y) - 0.01
    cr, hh, collar = m(D.CHIMNEY_D) / 2, m(D.CHIMNEY_H), m(D.COLLAR_D) / 2
    chimney.extend(
        _up(
            lathe(
                [(cr, 0.0), (cr, hh - 0.05), (cr + 0.01, hh - 0.03), (cr + 0.01, hh)],
                24,
            ),
            x,
            CHIMNEY_Y,
            z0,
        )
    )
    chimney.extend(
        _up(lathe([(collar, 0.0), (collar, 0.035)], 24), x, CHIMNEY_Y, z0 - 0.01)
    )
    for zb in (0.16, 0.34):
        brass.extend(
            _up(
                lathe([(cr + 0.004, zb), (cr + 0.004, zb + 0.025)], 24),
                x,
                CHIMNEY_Y,
                z0,
            )
        )
    return rails, brass, chimney


def pole_and_plank(seed: int) -> MeshData:
    """Plank and pole on blocks along the starboard roof (the chimney is to port)."""
    md = MeshData()
    top = L.block_top()
    y0, y1 = m(D.BLOCK_Y[0]), m(D.BLOCK_Y[1])
    for b in D.PLANK_BLOCKS:
        md.extend(box(wx(b) - 0.04, wx(b) + 0.04, y0, y1, L.roof_z(y1) - 0.01, top))
    pw = m(D.PLANK_W) / 2
    md.extend(
        box(
            wx(D.PLANK_B[0]),
            wx(D.PLANK_B[1]),
            m(D.PLANK_Y) - pw,
            m(D.PLANK_Y) + pw,
            top,
            top + m(D.PLANK_T),
            plank=seeded("nb", seed, "plank").random(),
        )
    )
    pr = m(D.POLE_D) / 2
    pole = loft(linspace(0, m(D.POLE_LEN), 8), lambda s: ring_profile(pr, pr, 12))
    md.extend(pole.transformed(IDENTITY, (wx(D.POLE_B), m(D.POLE_Y), top + pr)))
    return md


def rear_hatch(seed: int) -> MeshData:
    """Sliding hatch over the stern doors, on runners along the roof."""
    b0, b1, half = D.REAR_HATCH
    z = Z_ROOF_CROWN
    md = box(wx(b0), wx(b1), -m(half), m(half), z - 0.01, z + 0.05)
    for side in (1, -1):
        y = side * (m(half) + 0.025)
        md.extend(
            box(
                wx(b0),
                wx(b1 + 650),
                y - 0.012,
                y + 0.012,
                L.roof_z(abs(y)) - 0.01,
                z + 0.035,
            )
        )
    return md


@dataclass(frozen=True)
class DoorLeaf:
    """One closed leaf of the stern or bow doors; `face_x` is its outer face and `sign` the
    outward direction along X (-1 astern, +1 forward)."""

    name: str
    face_x: float
    sign: int
    y0: float
    y1: float
    z0: float
    z1: float


DOOR_THICK = 0.03


def door_leaves() -> list[DoorLeaf]:
    out = []
    for door in D.DOORS:
        sign = -1 if door.end == "Stern" else 1
        # x = wx(b) is the bulkhead plate's outer face.
        face = wx(door.b) + sign * DOOR_THICK
        half = m(door.half)
        for y0, y1, leaf in ((-half, -0.005, "Starboard"), (0.005, half, "Port")):
            out.append(
                DoorLeaf(
                    f"{door.end}Door{leaf}", face, sign, y0, y1, m(door.z0), m(door.z1)
                )
            )
    return out


def doors(seed: int) -> MeshData:
    """Closed steel double doors on the rear and front bulkheads."""
    md = MeshData()
    for d in door_leaves():
        x0, x1 = sorted([d.face_x, d.face_x - d.sign * DOOR_THICK])
        md.extend(box(x0, x1, d.y0, d.y1, d.z0, d.z1))
    return md


HATCH_Z = m(HATCH.z0)  # bottom of the hatch leaves, where their side frame sits
HATCH_LEAF_H, HATCH_PROUD, HATCH_GAP = m(HATCH.leaf_h), 0.025, 0.004


def hatch_leaves() -> list[tuple[str, float, float]]:
    """(name, u0, u1) of the two side-hatch leaves along the side from the hatch centre.
    Local u runs forward on the port side and aft on starboard, so leaves are named by
    where they actually are along the boat."""
    half = m(HATCH.clear) / 2
    minus, plus = (
        ("HatchAft", "HatchFore") if HATCH_SIDE > 0 else ("HatchFore", "HatchAft")
    )
    return [(minus, -half, -HATCH_GAP), (plus, HATCH_GAP, half)]


def side_hatch(seed: int) -> MeshData:
    """The side hatch: two closed steel leaves proud of the cabin side."""
    md = MeshData()
    for _, u0, u1 in hatch_leaves():
        leaf = box(u0, u1, 0.0, HATCH_LEAF_H, -HATCH_PROUD, 0.0)
        md.extend(on_side(leaf, HATCH_SIDE, HATCH.b, HATCH_Z))
    return md


# --------------------------------------------------------------------------- decal panels

# Canal-art decals, one design per kind of panel: roses from src/canal_roses.py and
# castle scenes from src/canal_castles.py. Cabin sprays sit in the gaps between openings,
# the reference panel taking the widest gap mid-side; each end-door and side-hatch leaf
# carries a castle over a smaller bunch of roses.
ROSE_DESIGNS = {
    "CabinCentre": C.PRESETS["reference"],
    "CabinAft": C.Spec(
        "nb_cabin_aft",
        layout="spray",
        tones=("red", "yellow", "pink"),
        roses=3,
        daisies=2,
        width=840,
        height=400,
        seed=21,
    ),
    "CabinFore": C.Spec(
        "nb_cabin_fore",
        layout="spray",
        tones=("red", "pink", "yellow"),
        roses=3,
        daisies=2,
        width=840,
        height=400,
        seed=22,
    ),
    "SternDoorRoses": C.Spec(
        "nb_stern_door_roses",
        layout="posy",
        tones=("red", "yellow", "pink"),
        roses=4,
        daisies=4,
        width=600,
        height=560,
        seed=31,
    ),
    "BowDoorRoses": C.Spec(
        "nb_bow_door_roses",
        layout="column",
        tones=("red", "pink", "yellow"),
        roses=2,
        daisies=1,
        width=400,
        height=800,
        seed=32,
    ),
    "HatchRoses": C.Spec(
        "nb_hatch_roses",
        layout="posy",
        tones=("red", "pink", "yellow"),
        roses=4,
        daisies=4,
        width=600,
        height=560,
        seed=41,
    ),
}
# Seen from astern the port leaf is on the left, from forward on the right; the unmirrored
# leaf is the port one, so these sides put both castles of a pair toward the outer edges.
CASTLE_DESIGNS = {
    "SternDoorCastle": K.Spec("nb_stern_castle", side=-1, towers=3, arches=3, seed=3),
    "BowDoorCastle": K.Spec(
        "nb_bow_castle", side=1, towers=4, arches=2, sky="sunset", seed=5
    ),
    # Seen from outside, the leaf on local -u is on the viewer's right on either side of the
    # boat and is the unmirrored one, so its castle stands toward the right-hand edge.
    "HatchCastle": K.Spec("nb_hatch_castle", side=1, towers=2, arches=2, seed=7),
}
DECAL_DESIGNS = (*ROSE_DESIGNS, *CASTLE_DESIGNS)
CABIN_SPRAY_MAX = {"CabinCentre": 1.2, "CabinAft": 0.85, "CabinFore": 0.85}
# Bare paint kept round every cabin decal: lines, frames, rims, the hatch.
DECAL_CLEAR = 0.06
DOOR_GAP = 0.025  # between a door leaf's castle and its roses
# Slab depth off the paint (m): inside the 0.1-1.6 mm the coachlines stand proud.
DECAL_PROUD = (0.0002, 0.0012)
LEAF_LINE_INSET, LEAF_LINE_W, LEAF_LINE_R = 0.02, 0.01, 0.03  # door and hatch borders
DECAL_OUTSIDE_UV = (-0.5, -0.5)  # where the slab's edges sample: clipped to transparent


def design_kind(design: str) -> str:
    return "castle" if design in CASTLE_DESIGNS else "roses"


def design_view_box(design: str) -> tuple[float, float, float, float]:
    """The texture's view box: roses cropped to their paint, a castle panel's whole frame."""
    if design in CASTLE_DESIGNS:
        return K.view_box(CASTLE_DESIGNS[design])
    return C.view_box(ROSE_DESIGNS[design], fit=True)


def design_svg(design: str) -> str:
    """A design as a transparent-ground SVG, ready to rasterise as a decal texture."""
    if design in CASTLE_DESIGNS:
        return K.render_svg(CASTLE_DESIGNS[design])
    return C.render_svg(ROSE_DESIGNS[design], background=False, fit=True)


def texture_name(design: str) -> str:
    return f"{design_kind(design)}_{design}.png"


def design_aspect(design: str) -> float:
    """Height / width of a design's texture."""
    _, _, w, h = design_view_box(design)
    return h / w


@dataclass(frozen=True)
class DecalPanel:
    """One decal: `rot` and `origin` place a local frame (X along the surface, Y up it, Z
    into it); the decal is centred at (u, v), on the painted face at local z = `face`."""

    name: str
    design: str
    part: str
    rot: tuple
    origin: tuple
    centre: tuple[float, float]
    face: float
    width: float
    height: float
    mirror: bool = False


def plane_frame(into, up) -> tuple:
    """Rotation rows for a local frame with Z = `into` and Y = `up`; X = up × into keeps it
    right-handed like `side_frame`."""
    along = (
        up[1] * into[2] - up[2] * into[1],
        up[2] * into[0] - up[0] * into[2],
        up[0] * into[1] - up[1] * into[0],
    )
    return tuple(tuple((along, up, into)[c][r] for c in range(3)) for r in range(3))


def opening_spans(side: int) -> list[tuple[float, float]]:
    """What interrupts the paint along one cabin side, as (b0, b1) metres from the rear
    bulkhead: window flanges, porthole rims and the side hatch."""
    spans = []
    for op in OPENINGS:
        if op.side == side:
            rim = RIM if op.kind == "porthole" else FLANGE
            b, half = op.centre_mm / 1000, op.width / 2 + rim
            spans.append((b - half, b + half))
    if HATCH_SIDE == side:
        spans.append((m(HATCH.span[0]), m(HATCH.span[1])))
    return sorted(spans)


def panel_gaps(side: int) -> list[tuple[float, float]]:
    """Free runs of paint inside the red line, between the openings."""
    lo, hi = PANEL_B[0] + PANEL_INNER, PANEL_B[1] - PANEL_INNER
    gaps, cur = [], lo
    for a, b in opening_spans(side):
        if a > cur:
            gaps.append((cur, a))
        cur = max(cur, b)
    if hi > cur:
        gaps.append((cur, hi))
    return gaps


def _fit(design: str, avail_w: float, avail_h: float, max_w: float = math.inf):
    aspect = design_aspect(design)
    w = min(max_w, avail_w, avail_h / aspect)
    return w, w * aspect


def _castle_over_roses(
    prefix: str,
    leaf: str,
    part_suffix: str,
    rot: tuple,
    origin: tuple,
    centre: tuple[float, float],
    area: tuple[float, float],
    face: float,
    mirror: bool,
) -> list[DecalPanel]:
    """A leaf's castle at the top of its inner `area` (width, height, centred on `centre`)
    and its roses centred in the space left below, DOOR_GAP apart."""
    aw, ah = area
    castle_w, castle_h = _fit(f"{prefix}Castle", aw, 0.58 * ah)
    roses_w, roses_h = _fit(f"{prefix}Roses", aw, ah - castle_h - DOOR_GAP)
    below = ah / 2 - castle_h - DOOR_GAP  # top of the space left under the castle
    out = []
    for kind, v, w, h in (
        ("Castle", ah / 2 - castle_h / 2, castle_w, castle_h),
        ("Roses", (below - ah / 2) / 2, roses_w, roses_h),
    ):
        design = f"{prefix}{kind}"
        out.append(
            DecalPanel(
                f"{leaf}{kind}",
                design,
                f"Decal{design}{part_suffix}",
                rot,
                origin,
                (centre[0], centre[1] + v),
                face,
                w,
                h,
                mirror=mirror,
            )
        )
    return out


def spray_gaps(side: int) -> list[tuple[str, tuple[float, float]]]:
    """(design, paint gap) of each cabin-side spray: the gap that holds its design hint."""
    out = []
    for design, s, hint in D.SPRAYS:
        if s != side:
            continue
        gap = next((g for g in panel_gaps(side) if g[0] <= m(hint) <= g[1]), None)
        if gap is None:
            raise ValueError(
                f"{design} hint {hint} is not in a paint gap on side {side}"
            )
        out.append((design, gap))
    return out


def decal_panels() -> list[DecalPanel]:
    """Every decal on the boat: cabin-side roses, and castles over roses on the end-door
    and side-hatch leaves."""
    out: list[DecalPanel] = []
    avail_h = PANEL_H - 2 * PANEL_INNER - 2 * DECAL_CLEAR
    for side, side_name in SIDES:
        rot, (_, y, z) = side_frame(side, PANEL_ZC)
        for design, (b0, b1) in spray_gaps(side):
            w, h = _fit(
                design, b1 - b0 - 2 * DECAL_CLEAR, avail_h, CABIN_SPRAY_MAX[design]
            )
            out.append(
                DecalPanel(
                    f"{design}{side_name}",
                    design,
                    f"Decal{design}{side_name}",
                    rot,
                    (wx(500 * (b0 + b1)), y, z),
                    (0.0, 0.0),
                    0.0,
                    w,
                    h,
                )
            )
    clear = LEAF_LINE_INSET + LEAF_LINE_W + 0.015
    for d in door_leaves():
        out += _castle_over_roses(
            "SternDoor" if d.sign < 0 else "BowDoor",
            d.name,
            "",
            plane_frame((-d.sign, 0.0, 0.0), (0.0, 0.0, 1.0)),
            (d.face_x, (d.y0 + d.y1) / 2, (d.z0 + d.z1) / 2),
            (0.0, 0.0),
            (d.y1 - d.y0 - 2 * clear, d.z1 - d.z0 - 2 * clear),
            0.0,
            # The two leaves of a pair are mirror images, as painters do them.
            mirror=d.y0 < 0,
        )
    rot, (_, y, z) = side_frame(HATCH_SIDE, HATCH_Z)
    origin = (wx(HATCH.b), y, z)
    for name, u0, u1 in hatch_leaves():
        out += _castle_over_roses(
            "Hatch",
            name,
            HATCH_NAME,
            rot,
            origin,
            ((u0 + u1) / 2, HATCH_LEAF_H / 2),
            (u1 - u0 - 2 * clear, HATCH_LEAF_H - 2 * clear),
            -HATCH_PROUD,
            mirror=u0 > 0,
        )
    return out


def panel_centre(p: DecalPanel) -> tuple[float, float, float]:
    """World position of a decal's centre on its painted face."""
    local = (p.centre[0], p.centre[1], p.face)
    return tuple(
        sum(p.rot[r][c] * local[c] for c in range(3)) + p.origin[r] for r in range(3)
    )


def decal_slab(p: DecalPanel) -> MeshData:
    """A closed slab DECAL_PROUD off the face. Only its outer face carries the 0-1 UVs; the
    rest sample outside the texture, which clips to transparent."""
    cu, cv = p.centre
    hw, hh = p.width / 2, p.height / 2
    z_out, z_in = p.face - DECAL_PROUD[1], p.face - DECAL_PROUD[0]
    md = box(cu - hw, cu + hw, cv - hh, cv + hh, z_out, z_in)
    uvs = []
    for face in md.faces:
        if all(abs(md.verts[i][2] - z_out) < 1e-9 for i in face):
            corners = []
            for i in face:
                s = (md.verts[i][0] - cu) / p.width
                # Seen from outside, local +X runs to the viewer's left (Z points away from
                # them), so U runs against it unless the leaf is the mirrored one.
                corners.append(
                    (
                        0.5 + s if p.mirror else 0.5 - s,
                        0.5 + (md.verts[i][1] - cv) / p.height,
                    )
                )
            uvs.append(tuple(corners))
        else:
            uvs.append(tuple(DECAL_OUTSIDE_UV for _ in face))
    md.uvs = uvs
    return md.transformed(p.rot, p.origin)


def panel_lines(seed: int) -> tuple[MeshData, MeshData]:
    """(end-door, side-hatch) cream borders: a line round each leaf, framing its decals."""
    doors_md, hatch_md = MeshData(), MeshData()

    def border(w: float, h: float) -> MeshData:
        ow, oh = w - 2 * LEAF_LINE_INSET, h - 2 * LEAF_LINE_INSET
        outer = rounded_rect(ow, oh, LEAF_LINE_R, 4)
        inner = rounded_rect(
            ow - 2 * LEAF_LINE_W, oh - 2 * LEAF_LINE_W, LEAF_LINE_R - LEAF_LINE_W, 4
        )
        return ring(outer, inner, 0.0015)

    for d in door_leaves():
        line = border(d.y1 - d.y0, d.z1 - d.z0).transformed(
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (0, 0, -0.0016)
        )
        rot = plane_frame((-d.sign, 0.0, 0.0), (0.0, 0.0, 1.0))
        doors_md.extend(
            line.transformed(rot, (d.face_x, (d.y0 + d.y1) / 2, (d.z0 + d.z1) / 2))
        )
    rot, (_, y, z) = side_frame(HATCH_SIDE, HATCH_Z)
    origin = (wx(HATCH.b), y, z)
    for _, u0, u1 in hatch_leaves():
        line = border(u1 - u0, HATCH_LEAF_H).transformed(
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            ((u0 + u1) / 2, HATCH_LEAF_H / 2, -HATCH_PROUD - 0.0016),
        )
        hatch_md.extend(line.transformed(rot, origin))
    return doors_md, hatch_md


def decal_parts(seed: int) -> list[Part]:
    """Decal slabs grouped into one part per design and side, plus the leaf borders."""
    grouped: dict[str, tuple[str, MeshData]] = {}
    for p in decal_panels():
        _, md = grouped.setdefault(p.part, (p.design, MeshData()))
        md.extend(decal_slab(p))
    door_lines, hatch_lines = panel_lines(seed)
    parts = [
        Part(name, (f"Decal{design}",), md) for name, (design, md) in grouped.items()
    ]
    parts += [
        Part("DoorLinesCream", ("CoachCream",), door_lines),
        Part(f"HatchLinesCream{HATCH_NAME}", ("CoachCream",), hatch_lines),
    ]
    return parts


def _vertical(md: MeshData, x: float, y: float) -> MeshData:
    """A loft made along local x (as a height) stood upright at (x, y)."""
    return md.transformed(((0, 0, -1), (0, 1, 0), (1, 0, 0)), (x, y, 0))


def rudder_gear(seed: int) -> tuple[MeshData, MeshData, MeshData]:
    """(steel rudder, stock, skeg and propeller; brass swan neck; wooden tiller bar)."""
    steel, brass, wood = MeshData(), MeshData(), MeshData()
    rb0, rb1, rz0, rz1 = D.RUDDER
    outline = rounded_rect(m(rb1 - rb0), m(rz1 - rz0), 0.06, 4)
    blade = loft([-0.02, 0.02], lambda y: outline)
    steel.extend(
        blade.transformed(
            ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
            (wx((rb0 + rb1) / 2), 0.0, m(rz0 + rz1) / 2),
        ).orient_outward()
    )
    xs = wx(D.RUDDER_STOCK_B)
    sr = m(D.STOCK_D) / 2
    top = DECK_Z + m(D.STOCK_TOP)
    steel.extend(
        _vertical(
            loft([m(rz1) - 0.02, top], lambda z: ring_profile(sr, sr, 12)), xs, 0.0
        )
    )
    steel.extend(box(wx(D.SKEG_END), wx(D.STERN_POST) + 0.03, -0.03, 0.03, 0.0, 0.05))
    pb, pz, pr = wx(D.PROP_B), m(D.PROP_Z), m(D.PROP_D) / 2
    for k in range(D.PROP_BLADES):
        a = 2 * math.pi * k / D.PROP_BLADES
        c, s = math.cos(a), math.sin(a)
        rot = ((1, 0, 0), (0, c, -s), (0, s, c))
        steel.extend(
            box(-0.02, 0.02, -0.06, 0.06, 0.04, pr).transformed(rot, (pb, 0, pz))
        )
    steel.extend(
        loft(
            [pb - 0.07, pb + 0.07], lambda x: ring_profile(0.05, 0.05, 16)
        ).transformed(IDENTITY, (0, 0, pz))
    )
    steel.extend(
        loft(
            [pb + 0.07, wx(D.STERN_POST)], lambda x: ring_profile(0.02, 0.02, 12)
        ).transformed(IDENTITY, (0, 0, pz))
    )
    brass.extend(
        sweep(
            [(wx(1000 * b), 0.0, z) for b, z in L.swan_neck()],
            ring_profile(0.022, 0.022, 12),
            up=(0, 1, 0),
        )
    )
    (b0, z0), (b1, z1) = L.tiller_bar()
    length = math.dist((b0, z0), (b1, z1))
    bar = loft(
        linspace(0, length, 6),
        lambda s: ring_profile(
            0.02 - 0.005 * s / length, 0.02 - 0.005 * s / length, 12
        ),
    )
    rot, h, _ = axis_frame((wx(1000 * b0), 0.0, z0), (wx(1000 * b1), 0.0, z1))
    wood.extend(bar.transformed(rot, h))
    return steel, brass, wood


def bow_gear(seed: int) -> tuple[MeshData, MeshData, MeshData, MeshData]:
    """(steel well deck, gas locker and foredeck, stem bar; rope fenders; brass T-studs
    and fillers; the two propane cylinders in the locker)."""
    steel, rope, brass, gas = MeshData(), MeshData(), MeshData(), MeshData()
    well_z, b_wall, b_end = m(D.WELL_SOLE_Z), m(D.WELL_DECK_END), m(D.LOCKER_END)

    def sole(z):
        def prof(x):
            w = plan_half(bx(x)) - PLATE - 0.003
            return [(-w, z - 0.012), (w, z - 0.012), (w, z), (-w, z)]

        return prof

    steel.extend(
        loft([wx(1000 * b) for b in linspace(CABIN_LEN, b_wall, 8)], sole(well_z))
    )
    lid_z = L.locker_lid_z()
    sole_z = L.locker_sole_z()
    steel.extend(loft([wx(1000 * b) for b in linspace(b_wall, b_end, 6)], sole(sole_z)))
    for b, z0 in ((b_wall, well_z), (b_end - 0.006, sole_z)):
        w = plan_half(b) - PLATE - 0.003
        steel.extend(
            box(wx(1000 * b), wx(1000 * b) + 0.006, -w, w, min(z0, sole_z), lid_z)
        )

    # The lid stands proud of the foredeck on its coaming; the foredeck runs on to the stem.
    def lid(x):
        lw = max(plan_half(bx(x)) - 0.06, 0.05)
        return [(-lw, lid_z), (lw, lid_z), (lw, lid_z + 0.012), (-lw, lid_z + 0.012)]

    steel.extend(
        loft([wx(1000 * b) for b in linspace(b_wall - 0.02, b_end + 0.02, 8)], lid)
    )

    def deck(x):
        b = bx(x)
        lw = max(plan_half(b) - GUNWALE_W + 0.002, 0.05)
        z = gunwale_z(b)
        return [(-lw, z - 0.01), (lw, z - 0.01), (lw, z + 0.002), (-lw, z + 0.002)]

    steel.extend(loft([wx(1000 * b) for b in linspace(b_end, STEM - 0.32, 8)], deck))
    stem_path = [
        (wx(1000 * (b + 0.015)), 0.0, bottom_z(b))
        for b in linspace(L.FOREFOOT_START + 0.08, STEM - 0.005, 14)
    ]
    stem_path.append((wx(1000 * STEM), 0.0, gunwale_z(STEM) - 0.02))
    steel.extend(sweep(stem_path, ring_profile(0.03, 0.03, 12), up=(0, 1, 0)))
    # Rope button fenders on the stem head and the counter.
    hr = m(D.FENDER_H) / 2
    button = lathe(
        [
            (0.04, -hr),
            (0.8 * hr, -0.85 * hr),
            (hr, 0.0),
            (0.8 * hr, 0.85 * hr),
            (0.04, hr),
        ],
        20,
    )
    rope.extend(
        button.transformed(
            ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
            (wx(D.STEM + D.FENDER_FWD) - hr, 0.0, m(D.FENDER_FWD_Z)),
        )
    )
    rope.extend(
        button.transformed(
            ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
            (wx(D.STERN_TIP - D.FENDER_AFT) + hr, 0.0, m(D.FENDER_AFT_Z)),
        )
    )
    tstud = lathe([(0.03, 0.0), (0.03, 0.12), (0.016, 0.12), (0.016, 0.14)], 16)
    for b, y in D.T_STUDS:
        z = gunwale_z(m(b)) if b > 0 else DECK_Z
        brass.extend(_up(tstud, wx(b), m(y), z))
        brass.extend(
            box(
                wx(b) - 0.012,
                wx(b) + 0.012,
                m(y) - 0.08,
                m(y) + 0.08,
                z + 0.12,
                z + 0.145,
            )
        )
    cap = lathe([(0.045, 0.0), (0.045, 0.012), (0.03, 0.02)], 18)
    brass.extend(_up(cap, wx(D.DIESEL_FILLER[0]), m(D.DIESEL_FILLER[1]), DECK_Z))
    brass.extend(_up(cap, wx(D.WATER_FILLER[0]), m(D.WATER_FILLER[1]), well_z))
    d, h = (m(v) for v in D.GAS_CYLINDER)
    bottle = lathe(
        [
            (0.0, 0.0),
            (d / 2, 0.0),
            (d / 2, h - 0.12),
            (0.06, h - 0.05),
            (0.03, h - 0.05),
            (0.03, h),
        ],
        20,
    )
    for b, y in L.cylinder_positions():
        gas.extend(_up(bottle, wx(1000 * b), y, sole_z + 0.002))
    return steel, rope, brass, gas


def machinery(seed: int) -> MeshData:
    """Engine, gearbox and battery box in the engine bay under the boards."""
    md = MeshData()
    for b0, b1, y0, y1, z0, z1 in (D.ENGINE, D.GEARBOX, D.BATTERIES):
        md.extend(box(wx(b0), wx(b1), m(y0), m(y1), m(z0), m(z1)))
    return md


def by_side(md: MeshData, side: int) -> MeshData:
    """The faces of a mesh whose centroid lies on one side (+1 port), re-indexed. Every
    solid in the frame, rim, glazing and coachline meshes lies wholly on one side."""
    out = MeshData()
    remap: dict[int, int] = {}
    for face, uv, mat, plank in zip(md.faces, md.uvs, md.mats, md.plank):
        cy = sum(md.verts[i][1] for i in face) / len(face)
        if cy * side <= 0:
            continue
        for i in face:
            if i not in remap:
                remap[i] = out.add_vert(md.verts[i], md.wear[i])
        out.add_face([remap[i] for i in face], uv, mat, plank)
    return out


SIDES = ((1, "Port"), (-1, "Starboard"))


def exterior_parts(seed: int = SEED) -> list[Part]:
    """Every exterior part with its material slots; cutters are returned separately.
    Parts on one cabin side carry the side's name so the cutaway can hide the port side."""
    frames, rims, glass = opening_frames(seed)
    cream, red = coachlines(seed)
    rails, brass, chimney = roof_gear(seed)
    steel, swan, tiller = rudder_gear(seed)
    well, fender, deck_brass, gas = bow_gear(seed)
    deck_brass.extend(swan)
    deck, boards = rear_deck(seed)
    parts = [
        Part("Hull", ("HullBlack",), hull_shell(seed)),
        Part("Counter", ("HullBlack",), counter_shell(seed)),
        Part("RubbingStrakes", ("HullBlack",), rubbing_strakes(seed)),
        Part("RearDeck", ("DeckNonSlip",), deck),
        Part("EngineBoards", ("TillerWood",), boards),
        Part("WellDeck", ("DeckNonSlip",), well),
        Part("CabinPort", ("CabinGreen",), cabin_side(seed, 1)),
        Part("CabinStarboard", ("CabinGreen",), cabin_side(seed, -1)),
        Part("Roof", ("RoofOxide",), roof(seed)),
        Part("RearHatch", ("CabinGreen",), rear_hatch(seed)),
        Part("Bulkheads", ("CabinGreen",), bulkheads(seed)),
        Part("EndDoors", ("CabinGreen",), doors(seed)),
        Part(f"SideHatch{HATCH_NAME}", ("CabinGreen",), side_hatch(seed)),
        Part("Handrails", ("CoachRed",), rails),
        Part("RoofBrass", ("Brass",), brass),
        Part("Chimney", ("CastIron",), chimney),
        Part("PoleAndPlank", ("TillerWood",), pole_and_plank(seed)),
        Part("SternGear", ("HullBlack",), steel),
        Part("Tiller", ("TillerWood",), tiller),
        Part("DeckBrass", ("Brass",), deck_brass),
        Part("Fender", ("Rope",), fender),
        Part("GasCylinders", ("CoachRed",), gas),
        Part("Machinery", ("CastIron",), machinery(seed)),
    ]
    for side, name in SIDES:
        body, lid = seat_locker(seed, side)
        parts += [
            Part(f"SeatLocker{name}", ("CabinGreen",), body),
            Part(f"SeatLid{name}", ("TillerWood",), lid),
            Part(f"ShelterRail{name}", ("CoachRed",), shelter_rail(side)),
            Part(f"CoachlineCream{name}", ("CoachCream",), by_side(cream, side)),
            Part(f"CoachlineRed{name}", ("CoachRed",), by_side(red, side)),
            Part(f"WindowFrames{name}", ("FrameGold",), by_side(frames, side)),
            Part(f"PortholeRims{name}", ("Brass",), by_side(rims, side)),
            Part(f"Glazing{name}", ("Glass",), by_side(glass, side)),
        ]
    parts += decal_parts(seed)
    return [p for p in parts if p.mesh.faces]


def cutters() -> dict[str, list[MeshData]]:
    """Boolean cutters per target object (cabin sides, linings, roof and ceiling)."""
    port = [opening_cutter(op) for op in OPENINGS if op.side > 0]
    starboard = [opening_cutter(op) for op in OPENINGS if op.side < 0]
    return {"port": port, "starboard": starboard, "roof": [flue_cutter()]}
