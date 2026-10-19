# SPDX-License-Identifier: MIT
"""The NB57 drawing set: an A3 general arrangement drawn from narrowboat_design.

Sheet 1 (NB57-GA-001, 1:50) is the furniture layout. Section A–A, on the centreline looking
to port, sits above the plan, cut 1300 above the floor. Section B–B, looking to starboard,
sits below it. The sections are placed in first-angle projection about the plan, so B–B is
inverted with its baseplate towards the plan, and every length lines up across all three
views. Sheet 2 (NB57-EX-002) has the outboard profile, the roof and deck plan, midship
section C–C at 1:20, and the particulars, tankage, weight, ventilation and openings tables.

Every line and dimension comes from narrowboat_design and narrowboat_lines, so a value
printed on the sheet is the value the model is built from. Output: SVG per sheet, a
two-page PDF, a PNG per sheet at 254 dpi (10 px per sheet millimetre, 5 mm per px at 1:50),
a lines-only PNG per view for model overlays, and a manifest with the overlay calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from itertools import pairwise
from pathlib import Path

import drawing_svg as S
import narrowboat_design as D
import narrowboat_lines as L
from procgen_mesh import offset_polyline

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "build" / "Drawings"
SCALE = 50
DPI = 254  # 10 px per sheet millimetre
B0_X = 79.0  # sheet x of b = 0 on both 1:50 sheets
A_BASE_Y, PLAN_Y, B_BASE_Y = 93.0, 128.0, 160.0
PLAN_CUT_Z = D.FLOOR_Z + 1300
SECTION_C_B = 7100  # midship section C-C, through the galley and both its windows
FOREFOOT_START_MM = D.STEM - D.FOREFOOT_LEN
DATE = "2026-10-03"
REVISIONS = (("A", "First issue: concept general arrangement", DATE, "CL"),)
OWNER = "3vilM33pl3"
CREATOR = "Claude (AI-assisted)"
STATUS = "CONCEPT - NOT FOR CONSTRUCTION"
WHITE = "#fff"
SIDE_NAME = {1: "PORT", -1: "STARBOARD"}


def mm(f, b: float) -> float:
    """Evaluate a narrowboat_lines function of b (metres) in millimetres."""
    return f(b / 1000) * 1000


def samples(a: float, b: float, step: float = 50.0) -> list[float]:
    n = max(2, math.ceil(abs(b - a) / step) + 1)
    return [a + (b - a) * k / (n - 1) for k in range(n)]


def rounded_rect(b0, z0, b1, z1, r, n=6):
    """Closed outline of a rounded rectangle in model coordinates."""
    r = min(r, (b1 - b0) / 2, (z1 - z0) / 2)
    pts = []
    for cx, cz, a0 in (
        (b1 - r, z1 - r, 0),
        (b0 + r, z1 - r, 90),
        (b0 + r, z0 + r, 180),
        (b1 - r, z0 + r, 270),
    ):
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n)
            pts.append((cx + r * math.cos(a), cz + r * math.sin(a)))
    return pts


def ellipse(cb, cz, rb, rz, n=32):
    return [
        (
            cb + rb * math.cos(2 * math.pi * k / n),
            cz + rz * math.sin(2 * math.pi * k / n),
        )
        for k in range(n)
    ]


def on_side(f: D.Fixture, side: int) -> bool:
    _, _, y0, y1 = f.footprint()
    return (y1 > 0) if side > 0 else (y0 < 0)


def crosses_centre(f: D.Fixture) -> bool:
    _, _, y0, y1 = f.footprint()
    return y0 < 0 < y1


# =========================================================================== elevations


def hull_cut(v: S.View) -> None:
    """Keel, stem, counter and skeg on the centreline, and the deck plates."""
    keel = [(b, mm(L.bottom_z, b)) for b in samples(D.STERN_POST, D.STEM, 25)]
    keel.append((D.STEM, mm(L.gunwale_z, D.STEM)))
    v.poly(keel, "visible")
    # Counter: underside over the propeller, the end plate, then the deck.
    v.poly(
        [
            (D.STERN_POST, 0),
            (D.STERN_POST, D.Z_COUNTER),
            (D.STERN_TIP, D.Z_COUNTER),
            (D.STERN_TIP, D.DECK_Z),
        ],
        "visible",
    )
    v.poly([(D.STERN_TIP, D.DECK_Z), (D.ENGINE_BOARDS[0], D.DECK_Z)], "visible")
    b0, b1 = D.ENGINE_BOARDS
    v.rect(
        b0, D.DECK_Z - 25, b1, D.DECK_Z, "thin", WHITE
    )  # lift-out boards over the engine
    for b in samples(b0, b1, 300)[1:-1]:
        v.line((b, D.DECK_Z - 25), (b, D.DECK_Z), "thin")
    v.rect(D.SKEG_END, 0, D.STERN_POST, 50, "visible", WHITE)
    # Forward: well-deck sole, gas locker and foredeck.
    sole, lid = L.locker_sole_z() * 1000, L.locker_lid_z() * 1000
    v.poly(
        [
            (D.CABIN_LEN, D.WELL_SOLE_Z),
            (D.WELL_DECK_END, D.WELL_SOLE_Z),
            (D.WELL_DECK_END, lid),
        ],
        "visible",
    )
    v.poly(
        [(D.WELL_DECK_END, sole), (D.LOCKER_END, sole), (D.LOCKER_END, lid)], "visible"
    )
    v.rect(
        D.WELL_DECK_END - 20, lid, D.LOCKER_END + 20, lid + 12, "visible", WHITE
    )  # lid
    fore = [(b, mm(L.gunwale_z, b)) for b in samples(D.LOCKER_END, D.STEM, 50)]
    v.poly(fore, "visible")
    v.poly(
        [(D.LOCKER_END, lid), (D.LOCKER_END, mm(L.gunwale_z, D.LOCKER_END))], "visible"
    )


def cabin_cut(v: S.View) -> None:
    """Roof, ceiling, floor, bulkheads (with their doorways) and partitions on the centreline."""
    crown, ceiling = D.Z_ROOF_CROWN, D.CEILING_CROWN
    v.poly([(0, crown), (D.CABIN_LEN, crown)], "visible")
    v.poly([(D.LINING_AFT, ceiling), (D.LINING_FWD, ceiling)], "thin")
    v.poly(
        [(D.ZONE["Bedroom"].b0, D.FLOOR_Z), (D.FIXTURE["BowSteps"].b0, D.FLOOR_Z)],
        "medium",
    )
    for door in D.DOORS:
        b0, b1 = (
            (door.b, door.b + D.T_BULKHEAD)
            if door.end == "Stern"
            else (door.b - D.T_BULKHEAD, door.b)
        )
        bottom = 0  # the bulkheads run down to the baseplate (engine bay / bow void)
        v.poly([(b0, bottom), (b0, door.z0), (b1, door.z0), (b1, bottom)], "visible")
        v.poly([(b0, crown), (b0, door.z1), (b1, door.z1), (b1, crown)], "visible")
        out = -1 if door.end == "Stern" else 1
        v.rect(
            door.b + out * 30, door.z0, door.b + out * (30 + 30), door.z1, "thin", WHITE
        )
        lining = D.LINING_AFT if door.end == "Stern" else D.LINING_FWD
        v.line((lining, door.z1), (lining, ceiling), "thin")
    b0, b1, _ = D.REAR_HATCH
    v.rect(b0, crown, b1, crown + 60, "visible", WHITE)
    for p in D.PARTITIONS:
        t = D.PARTITION_T / 2
        head = D.FLOOR_Z + D.DOOR_H
        if p.door_y[0] < 0 < p.door_y[1]:
            v.rect(p.b - t, head, p.b + t, ceiling, "visible", WHITE)
        else:
            v.rect(p.b - t, D.FLOOR_Z, p.b + t, ceiling, "visible", WHITE)


def steps_profile(f: D.Fixture) -> list[tuple[float, float]]:
    """Treads and risers from the foot of the flight to the doorway sill."""
    n = f.opt("risers")
    rise, going = (f.z1 - f.z0) / n, (f.b1 - f.b0) / (n - 1)

    def b_at(k: int) -> float:
        return f.b1 - k * going if f.opt("rise_to") < 0 else f.b0 + k * going

    pts = [(b_at(0), f.z0)]
    for k in range(n):
        z = f.z0 + (k + 1) * rise
        pts.append((b_at(k), z))
        if k < n - 1:
            pts.append((b_at(k + 1), z))
    return pts


def door_split(b0: float, b1: float, width: float = 500) -> list[float]:
    n = max(1, round((b1 - b0) / width))
    return [b0 + (b1 - b0) * k / n for k in range(n + 1)]


def fronts(v: S.View, b0, b1, z0, z1, handle_z, *, plinth=True, width=500) -> None:
    v.rect(b0, z0, b1, z1, "thin", WHITE)
    if plinth:
        v.line((b0, z0 + 100), (b1, z0 + 100), "thin")
    splits = door_split(b0, b1, width)
    for b in splits[1:-1]:
        v.line((b, z0 + (100 if plinth else 0)), (b, z1), "thin")
    for i, (a, c) in enumerate(pairwise(splits)):
        hb = c - 60 if i % 2 == 0 else a + 60
        v.line((hb, handle_z - 60), (hb, handle_z + 60), "medium")


def elevation_fixture(v: S.View, f: D.Fixture, side: int) -> None:
    """One fixture in an inside elevation; ones crossing the cut are drawn as cut."""
    style = "visible" if crosses_centre(f) else "thin"
    k = f.kind
    top = min(f.z1, mm(L.ceiling_z, min(abs(f.footprint()[3 if side > 0 else 2]), 820)))
    if k == "steps":
        pts = steps_profile(f)
        v.poly(pts, "visible")
        v.poly([pts[0], pts[-1]], "thin")
    elif k in ("cupboard", "wardrobe", "tall_unit"):
        fronts(v, f.b0, f.b1, f.z0, top, D.FLOOR_Z + 1000)
    elif k == "base_run":
        carcass = D.WORKTOP_Z - 38
        hosted = [h for h in D.FIXTURES if h.host == f.name]
        v.rect(f.b0, f.z0, f.b1, carcass, "thin", WHITE)
        v.line((f.b0, f.z0 + 100), (f.b1, f.z0 + 100), "thin")
        edges = [f.b0]
        for h in sorted(hosted, key=lambda h: h.b0):
            if h.kind in ("fridge", "cooker"):
                edges += [h.b0, h.b1]
        edges.append(f.b1)
        for a, c in zip(edges[::2], edges[1::2]):
            if c - a > 150:
                splits = door_split(a, c)
                for b in splits:
                    v.line((b, f.z0 + 100), (b, carcass), "thin")
                for s0, s1 in pairwise(splits):
                    v.line(
                        ((s0 + s1) / 2 - 60, carcass - 80),
                        ((s0 + s1) / 2 + 60, carcass - 80),
                        "medium",
                    )
        v.rect(f.b0 - 15, carcass, f.b1, D.WORKTOP_Z, "medium", WHITE)
    elif k == "fridge":
        v.rect(f.b0 + 10, f.z0 + 100, f.b1 - 10, f.z1, "thin", WHITE)
        v.text(
            ((f.b0 + f.b1) / 2, (f.z0 + f.z1) / 2),
            "FRIDGE",
            1.8,
            anchor="middle",
            middle=True,
        )
    elif k == "sink":
        b0, b1 = f.opt("bowl")
        v.rect(b0, f.z0, b1, f.z1, "hidden")
        bc = (b0 + b1) / 2
        v.poly(
            [
                (bc, f.z1),
                (bc, f.z1 + 260),
                (bc + 150, f.z1 + 260),
                (bc + 150, f.z1 + 200),
            ],
            "medium",
        )
    elif k == "cooker":
        v.rect(f.b0 + 10, f.z0 + 100, f.b1 - 10, D.WORKTOP_Z - 150, "medium", WHITE)
        v.rect(f.b0 + 60, f.z0 + 250, f.b1 - 60, D.WORKTOP_Z - 280, "thin")
        for kx in range(f.opt("burners")):
            v.circle(
                (f.b0 + 80 + kx * (f.b1 - f.b0 - 160) / 3, D.WORKTOP_Z - 95), 18, "thin"
            )
        v.text(
            ((f.b0 + f.b1) / 2, f.z0 + 180), "COOKER", 1.8, anchor="middle", middle=True
        )
    elif k == "wall_units":
        fronts(v, f.b0, f.b1, f.z0, top, f.z0 + 80, plinth=False)
    elif k == "bed":
        base = f.z0 + 370
        v.rect(f.b0, f.z0, f.b1, base, style, WHITE)
        for b in door_split(f.b0, f.b1, 640)[1:-1]:
            v.line((b, f.z0 + 40), (b, base), "thin")
        v.poly(
            rounded_rect(f.b0 + 10, base, f.b1 - 10, f.z1, 60),
            style,
            closed=True,
            fill=WHITE,
        )
        head = f.b1 - 60 if f.opt("head") > 0 else f.b0 + 60
        pillow = (head - 450, head) if f.opt("head") > 0 else (head, head + 450)
        v.poly(
            rounded_rect(pillow[0], f.z1, pillow[1], f.z1 + 120, 50),
            "thin",
            closed=True,
            fill=WHITE,
        )
    elif k == "shower":
        v.rect(f.b0, f.z0, f.b1, f.z0 + 100, "thin", WHITE)
        v.rect(f.b0, f.z0 + 100, f.b1, f.z1, "thin")
        v.line((f.b0 + 120, f.z0 + 100), (f.b0 + 120, f.z1), "thin")
        riser = f.b0 + 400
        v.line((riser, f.z0 + 900), (riser, f.z0 + 1750), "medium")
        v.poly([(riser, f.z0 + 1750), (riser + 120, f.z0 + 1700)], "medium")
        v.text(
            ((f.b0 + f.b1) / 2, f.z0 + 650), "SHOWER", 1.8, anchor="middle", middle=True
        )
    elif k == "vanity":
        fronts(v, f.b0, f.b1, f.z0, f.z1 - 30, f.z1 - 150)
        v.rect(f.b0 - 10, f.z1 - 30, f.b1 + 10, f.z1, "medium", WHITE)
        bc = (f.b0 + f.b1) / 2
        v.poly(
            rounded_rect(bc - 170, f.z1, bc + 170, f.z1 + 120, 50),
            "thin",
            closed=True,
            fill=WHITE,
        )
        v.poly(
            [(bc + 150, f.z1 + 120), (bc + 150, f.z1 + 280), (bc + 40, f.z1 + 280)],
            "medium",
        )
    elif k == "toilet":
        v.poly(
            rounded_rect(f.b0, f.z0, f.b1, f.z1 - 60, 40),
            "thin",
            closed=True,
            fill=WHITE,
        )
        v.rect(f.b0 - 10, f.z1 - 60, f.b1 + 10, f.z1 - 30, "thin", WHITE)
        v.rect(f.b0 + 10, f.z1 - 30, f.b1 - 10, f.z1, "thin", WHITE)
        v.text(((f.b0 + f.b1) / 2, f.z0 + 220), "WC", 1.8, anchor="middle", middle=True)
    elif k == "towel_rail":
        v.line((f.b0, f.z0), (f.b0, f.z1), "medium")
        v.line((f.b1, f.z0), (f.b1, f.z1), "medium")
        for z in samples(f.z0 + 60, f.z1 - 60, 110):
            v.line((f.b0, z), (f.b1, z), "thin")
    elif k == "hearth":
        v.hatch(rounded_rect(f.b0, f.z0, f.b1, f.z1, 1), "hatch45wide")
        v.rect(f.b0, f.z0, f.b1, f.z1, "thin")
    elif k == "stove":
        b0, b1 = f.b0, f.b1
        v.rect(b0 + 20, f.z0, b0 + 60, f.z0 + 60, "thin", WHITE)
        v.rect(b1 - 60, f.z0, b1 - 20, f.z0 + 60, "thin", WHITE)
        v.rect(b0, f.z0 + 60, b1, f.z1 - 25, "medium", WHITE)
        v.rect(b0 - 15, f.z1 - 25, b1 + 15, f.z1, "medium", WHITE)
        v.poly(
            rounded_rect(b0 + 50, f.z0 + 130, b1 - 50, f.z1 - 90, 30),
            "thin",
            closed=True,
        )
        cb, _ = D.CHIMNEY
        r = f.opt("flue_d") / 2
        ceiling = mm(L.ceiling_z, D.CHIMNEY[1])
        v.rect(cb - r, f.z1, cb + r, ceiling, "thin", WHITE)
    elif k == "heat_shield":
        v.hatch(rounded_rect(f.b0, f.z0, f.b1, f.z1, 1), "hatch45")
        v.rect(f.b0, f.z0, f.b1, f.z1, "thin")
    elif k == "dinette":
        seat = f.opt("seat_z")
        (a0, a1), (c0, c1) = f.opt("benches")
        for s0, s1, back_at_start in ((a0, a1, True), (c0, c1, False)):
            v.rect(s0, f.z0, s1, seat, "thin", WHITE)
            v.poly(
                rounded_rect(s0, seat, s1, seat + 80, 30),
                "thin",
                closed=True,
                fill=WHITE,
            )
            bb = (s0, s0 + 110) if back_at_start else (s1 - 110, s1)
            v.poly(
                rounded_rect(bb[0], seat, bb[1], f.z1, 40),
                "thin",
                closed=True,
                fill=WHITE,
            )
        t0, t1 = f.opt("table")
        v.rect(t0, D.FLOOR_Z + 690, t1, D.FLOOR_Z + 720, "medium", WHITE)
        tc = (t0 + t1) / 2
        v.rect(tc - 30, f.z0, tc + 30, D.FLOOR_Z + 690, "thin", WHITE)
        v.rect(tc - 200, f.z0, tc + 200, f.z0 + 20, "thin", WHITE)
    elif k in ("sofa", "armchair"):
        seat = f.opt("seat_z")
        arm = 150
        v.rect(f.b0, f.z0 + 80, f.b1, seat - 80, "thin", WHITE)
        v.poly(
            rounded_rect(f.b0 + arm, seat - 80, f.b1 - arm, seat, 30),
            "thin",
            closed=True,
            fill=WHITE,
        )
        n = 1 if k == "armchair" else 2
        for kx in range(n):
            c0 = f.b0 + arm + kx * (f.b1 - f.b0 - 2 * arm) / n
            c1 = f.b0 + arm + (kx + 1) * (f.b1 - f.b0 - 2 * arm) / n
            v.poly(
                rounded_rect(c0 + 10, seat, c1 - 10, f.z1 - 20, 60),
                "thin",
                closed=True,
                fill=WHITE,
            )
        for a in (f.b0, f.b1 - arm):
            v.poly(
                rounded_rect(a, f.z0 + 80, a + arm, seat + 180, 50),
                "thin",
                closed=True,
                fill=WHITE,
            )
        for a in (f.b0 + 40, f.b1 - 70):
            v.rect(a, f.z0, a + 30, f.z0 + 80, "thin", WHITE)
    elif k == "shelves":
        v.rect(f.b0, f.z0, f.b1, f.z1, "thin", WHITE)
        for z in (f.z0 + 100, f.z0 + 420, f.z0 + 740):
            v.line((f.b0, z), (f.b1, z), "thin")
        tv = f.opt("tv")
        if tv:
            bc = (f.b0 + f.b1) / 2
            v.rect(bc - tv / 2, f.z0 + 760, bc + tv / 2, f.z1 - 30, "thin")


def opening_ids() -> dict[str, str]:
    """Schedule tags: windows W1.., portholes P1.., port side first, aft to forward."""
    out, counts = {}, {"window": 0, "porthole": 0}
    for op in sorted(D.OPENINGS, key=lambda o: (-o.side, o.b)):
        counts[op.kind] += 1
        out[op.name] = f"{'W' if op.kind == 'window' else 'P'}{counts[op.kind]}"
    return out


OPENING_IDS = opening_ids()


def opening_elevation(v: S.View, op: D.Opening) -> None:
    if op.kind == "porthole":
        v.circle((op.b, op.zc), op.w / 2, "medium", WHITE)
        v.circle((op.b, op.zc), op.w / 2 + D.RIM_W, "thin")
        v.text((op.b, op.zc), OPENING_IDS[op.name], 1.8, anchor="middle", middle=True)
        return
    b0, b1 = op.b - op.w / 2, op.b + op.w / 2
    z0, z1 = op.zc - op.h / 2, op.zc + op.h / 2
    fl = D.FRAME_FLANGE
    v.poly(
        rounded_rect(b0 - fl, z0 - fl, b1 + fl, z1 + fl, op.r + fl),
        "thin",
        closed=True,
        fill=WHITE,
    )
    v.poly(rounded_rect(b0, z0, b1, z1, op.r), "medium", closed=True)
    v.line((op.b, z0), (op.b, z1), "thin")  # hopper windows: two lights
    tag = OPENING_IDS[op.name]
    v.text((op.b - op.w / 4, op.zc), tag, 1.8, anchor="middle", middle=True)


def side_hatch_elevation(v: S.View) -> None:
    h = D.SIDE_HATCH
    b0, b1 = h.span
    z0, z1 = h.z0, h.z0 + h.leaf_h
    v.rect(b0 - 40, z0 - 40, b1 + 40, z1 + 40, "thin", WHITE)
    v.rect(b0, z0, b1, z1, "medium")
    v.line((h.b, z0), (h.b, z1), "thin")
    for b in (h.b - 60, h.b + 60):
        v.line((b, (z0 + z1) / 2 - 50), (b, (z0 + z1) / 2 + 50), "medium")


def chimney_elevation(v: S.View) -> None:
    b, y = D.CHIMNEY
    z0 = mm(L.roof_z, y)
    c, d = D.COLLAR_D / 2, D.CHIMNEY_D / 2
    v.rect(b - c, z0, b + c, z0 + 35, "thin", WHITE)
    v.rect(b - d, z0 + 35, b + d, z0 + D.CHIMNEY_H - 30, "thin", WHITE)
    v.rect(
        b - d - 10, z0 + D.CHIMNEY_H - 30, b + d + 10, z0 + D.CHIMNEY_H, "thin", WHITE
    )
    for zb in (z0 + 160, z0 + 340):
        v.line((b - d, zb), (b + d, zb), "thin")


def pole_and_plank_side(v: S.View) -> None:
    """Pole and plank on their blocks, seen from the side."""
    top = L.block_top() * 1000
    for b in D.PLANK_BLOCKS:
        v.rect(b - 40, D.Z_ROOF_CROWN - 25, b + 40, top, "thin", WHITE)
    v.rect(D.PLANK_B[0], top, D.PLANK_B[1], top + D.PLANK_T, "thin", WHITE)
    v.rect(D.POLE_B, top, D.POLE_B + D.POLE_LEN, top + D.POLE_D, "thin", WHITE)


def vent_profile(b: float, z: float):
    r, hgt = D.VENT_D / 2, D.VENT_H
    return [
        (b - r, z),
        (b - r, z + 30),
        (b - 0.7 * r, z + 0.75 * hgt),
        (b, z + hgt),
        (b + 0.7 * r, z + 0.75 * hgt),
        (b + r, z + 30),
        (b + r, z),
    ]


def machinery_elevation(v: S.View, side: int) -> None:
    """Engine, gearbox and shaft, propeller, rudder and tiller, tanks and the weed hatch."""
    b0, b1, _, _, z0, z1 = D.ENGINE
    v.poly(rounded_rect(b0, z0, b1, z1, 40), "medium", closed=True, fill=WHITE)
    g = D.GEARBOX
    v.rect(g[0], g[4], g[1], g[5], "medium", WHITE)
    v.line((g[0], 330), (D.PROP_B + 70, D.PROP_Z), "medium")
    r = D.PROP_D / 2
    v.rect(D.PROP_B - 70, D.PROP_Z - 50, D.PROP_B + 70, D.PROP_Z + 50, "medium", WHITE)
    for sgn in (1, -1):
        v.poly(
            ellipse(D.PROP_B, D.PROP_Z + sgn * (r / 2 + 25), 55, r / 2 - 25),
            "thin",
            closed=True,
            fill=WHITE,
        )
    rb0, rb1, rz0, rz1 = D.RUDDER
    v.poly(rounded_rect(rb0, rz0, rb1, rz1, 60), "visible", closed=True, fill=WHITE)
    stock = D.RUDDER_STOCK_B
    v.rect(
        stock - D.STOCK_D / 2,
        rz1,
        stock + D.STOCK_D / 2,
        D.DECK_Z + D.STOCK_TOP,
        "thin",
        WHITE,
    )
    neck = [(b * 1000, z * 1000) for b, z in L.swan_neck()]
    v.poly(neck, "medium")
    (tb0, tz0), (tb1, tz1) = L.tiller_bar()
    v.poly([(tb0 * 1000, tz0 * 1000), (tb1 * 1000, tz1 * 1000)], "visible")
    t = D.FUEL_TANK
    v.rect(t[0], t[2], t[1], t[3], "thin")
    w0, w1, _ = D.WEED_HATCH
    v.poly([(w0, D.Z_COUNTER), (w0, D.DECK_Z)], "thin")
    v.poly([(w1, D.Z_COUNTER), (w1, D.DECK_Z)], "thin")
    v.rect(w0 - 20, D.DECK_Z, w1 + 20, D.DECK_Z + 15, "medium", WHITE)
    t = D.WATER_TANK
    v.rect(t[0], t[2], t[1], t[3], "thin")
    if side < 0:
        bt = D.BATTERIES
        v.rect(bt[0], bt[4], bt[1], bt[5], "thin", WHITE)


def deck_fittings_elevation(v: S.View, side: int) -> None:
    """Shelter plate and seat locker on this side, T-studs, fenders and gas cylinders."""
    v.rect(D.SHELTER_AFT, D.DECK_Z, 0, D.Z_SIDE_TOP, "thin")
    v.rect(D.SHELTER_AFT, D.Z_SIDE_TOP, 0, D.Z_SIDE_TOP + 30, "thin", WHITE)
    s0, s1, _, sh = D.SEAT_LOCKERS
    v.rect(s0, D.DECK_Z, s1, D.DECK_Z + sh, "thin", WHITE)
    v.rect(s0 - 15, D.DECK_Z + sh - 25, s1 + 15, D.DECK_Z + sh, "thin", WHITE)
    for b, y in D.T_STUDS:
        if y * side >= 0:
            z = mm(L.gunwale_z, b) if b > 0 else D.DECK_Z
            v.rect(b - 30, z, b + 30, z + 120, "thin", WHITE)
            v.rect(b - 80, z + 120, b + 80, z + 145, "thin", WHITE)
    rx, rz = 110, D.FENDER_H / 2
    v.poly(
        ellipse(D.STEM + D.FENDER_FWD - rx, D.FENDER_FWD_Z, rx, rz),
        "thin",
        closed=True,
        fill=WHITE,
    )
    rx = 100
    v.poly(
        ellipse(D.STERN_TIP - D.FENDER_AFT + rx, D.FENDER_AFT_Z, rx, rz - 20),
        "thin",
        closed=True,
        fill=WHITE,
    )
    d, h = D.GAS_CYLINDER
    sole = L.locker_sole_z() * 1000
    for b, y in L.cylinder_positions():
        if y * side > 0:
            b *= 1000
            v.poly(
                rounded_rect(b - d / 2, sole + 10, b + d / 2, sole + 10 + h - 60, 50),
                "thin",
                closed=True,
                fill=WHITE,
            )
            v.rect(b - 60, sole + h - 50, b + 60, sole + h + 10, "thin", WHITE)
    drain_b = (D.WELL_DECK_END + D.LOCKER_END) / 2 + 150
    v.circle((drain_b, sole + 15), D.LOCKER_DRAIN_ID / 2 + 6, "thin")


def waterline(v: S.View) -> None:
    pts = [
        (D.STERN_TIP - 300, mm(L.wl_z, D.STERN_TIP - 300)),
        (D.STEM + 300, mm(L.wl_z, D.STEM + 300)),
    ]
    v.poly(pts, "centre")
    x, y = v.P(D.STEM + 300, mm(L.wl_z, D.STEM + 300))
    up = -1 if v.v_sign > 0 else 1
    v.sheet.polyline(
        [(x - 1.2, y + up * 1.8), (x + 1.2, y + up * 1.8), (x, y)], "thin", closed=True
    )
    v.sheet.text((x + 1.6, y + (up * 0.6 if up < 0 else 2.4)), "DWL", 1.8)


def elevation(sheet: S.Sheet, side: int) -> tuple[S.View, list]:
    """Section A–A (side +1, looking to port) or B–B (side -1, looking to starboard)."""
    base = A_BASE_Y if side > 0 else B_BASE_Y
    name = "section_A" if side > 0 else "section_B"
    v = S.View(sheet, name, B0_X, base, SCALE, 1, 1 if side > 0 else -1)
    dims = []
    with sheet.group(name):
        # Lining features on this side first, so furniture can mask them.
        v.line((D.LINING_AFT, D.Z_GUNWALE), (D.LINING_FWD, D.Z_GUNWALE), "thin")
        edge = mm(L.ceiling_z, L.lining_in(1.98))
        v.line((D.LINING_AFT, edge), (D.LINING_FWD, edge), "thin")
        sheer = [(b, mm(L.gunwale_z, b)) for b in samples(D.CABIN_LEN, D.STEM, 100)]
        v.poly(sheer, "thin")
        for op in D.OPENINGS:
            if op.side == side:
                opening_elevation(v, op)
        if D.SIDE_HATCH.side == side:
            side_hatch_elevation(v)
        v.line(
            (D.HANDRAIL_B[0], L.handrail_top() * 1000),
            (D.HANDRAIL_B[1], L.handrail_top() * 1000),
            "thin",
        )
        if D.CHIMNEY[1] * side > 0:
            chimney_elevation(v)
        if D.PLANK_Y * side > 0:
            pole_and_plank_side(v)
        for b in D.VENT_BS:
            v.poly(vent_profile(b, D.Z_ROOF_CROWN), "visible", closed=True, fill=WHITE)
        deck_fittings_elevation(v, side)
        machinery_elevation(v, side)
        # Beyond the cut first, then appliances in their runs, then pieces the cut passes
        # through (steps, the bed), which stand in front of everything else.
        shown = [f for f in D.FIXTURES if on_side(f, side)]
        hosted = ("sink", "fridge", "cooker")
        for f in shown:
            if not crosses_centre(f) and f.kind not in hosted:
                elevation_fixture(v, f, side)
        for f in shown:
            if f.kind in hosted:
                elevation_fixture(v, f, side)
        for f in shown:
            if crosses_centre(f):
                elevation_fixture(v, f, side)
        cabin_cut(v)
        hull_cut(v)
        waterline(v)
        elevation_labels(v)
        elevation_notes(v, side)
    with sheet.group(f"{name}_dims"):
        dims += elevation_dims(v, side)
    return v, dims


def elevation_labels(v: S.View) -> None:
    v.text((-480, D.DECK_Z + 520), "SHELTER", 1.8, anchor="middle")
    v.text(
        ((D.ENGINE[0] + D.ENGINE[1]) / 2, (D.ENGINE[4] + D.ENGINE[5]) / 2),
        "ENGINE",
        1.8,
        anchor="middle",
        middle=True,
    )
    v.text(
        (
            (D.WATER_TANK[0] + D.WATER_TANK[1]) / 2,
            (D.WATER_TANK[2] + D.WATER_TANK[3]) / 2,
        ),
        "WATER",
        1.8,
        anchor="middle",
        middle=True,
    )


def elevation_notes(v: S.View, side: int) -> None:
    s = v.sheet
    sgn = 1 if side > 0 else -1

    def note(b, z, dx, dz, text):
        tx, ty = v.P(b, z)
        s.line((tx, ty), (tx + dx, ty - sgn * dz), "thin")
        s.circle((tx, ty), 0.35, None, fill=S.INK)
        left = dx > 0
        s.text(
            (tx + dx + (0.6 if left else -0.6), ty - sgn * dz + 0.6),
            text,
            1.8,
            anchor="start" if left else "end",
            knockout=True,
        )

    if side > 0:
        note(D.FUEL_TANK[0] + 150, D.FUEL_TANK[2] + 80, -9, -9, "DIESEL ~240 L")
        note(D.RUDDER_STOCK_B - 150, 300, -4, -14, "BALANCED RUDDER")
        note(D.PROP_B, D.PROP_Z - 120, 7, -8, "PROP 457 DIA")
        note(D.WEED_HATCH[1], D.DECK_Z + 10, 6, 6, "WEED HATCH")
        note(D.CHIMNEY[0] + 65, D.Z_ROOF_CROWN + 520, 7, 3, "REMOVABLE CHIMNEY")
        note(8600, D.FLOOR_Z + 300, 8, 6, "4.5 kW STOVE ON HEARTH")
        note(
            D.WELL_DECK_END + 200,
            L.locker_sole_z() * 1000 + 300,
            7,
            12,
            "GAS LOCKER 2 x 13 kg",
        )
        note(13300, D.WELL_SOLE_Z, -4, 12, "SELF-DRAINING WELL DECK")
    else:
        note(
            (D.BATTERIES[0] + D.BATTERIES[1]) / 2,
            D.BATTERIES[4] + 100,
            5,
            9,
            "BATTERIES (VENTED BOX)",
        )
        drain_b = (D.WELL_DECK_END + D.LOCKER_END) / 2 + 150
        note(
            drain_b,
            L.locker_sole_z() * 1000 + 15,
            6,
            -9,
            "LOCKER DRAIN 19 ID, ABOVE DWL",
        )
        note(
            D.SIDE_HATCH.b,
            D.SIDE_HATCH.z0 + D.SIDE_HATCH.leaf_h - 60,
            9,
            -3,
            "SIDE HATCH 700 CLEAR",
        )


def elevation_dims(v: S.View, side: int) -> list:
    s = v.sheet
    dims = []
    up = 1 if side > 0 else -1
    # Running dimensions from the datum (aft face of the rear bulkhead) to the openings.
    ops = sorted(op.b for op in D.OPENINGS if op.side == side)
    if D.SIDE_HATCH.side == side:
        ops = sorted([*ops, D.SIDE_HATCH.b])
    roof_y = v.y(D.Z_ROOF_CROWN + (D.CHIMNEY_H + 100 if side > 0 else 150))
    y_line = roof_y - up * 4.5
    froms = [D.Z_ROOF_CROWN] + [D.Z_ROOF_CROWN] * len(ops)
    for b in ops:
        z = D.WINDOW_ZC
        s.line(v.P(b, z - 300), v.P(b, z + 300), "centre")
    dims += S.running_h(v, 0, ops, y_line, from_v=froms, above=side > 0)
    # Heights at the stern: keel, counter, deck, cabin side top and roof crown.
    x_line = 30.0
    levels = [D.Z_COUNTER, D.DECK_Z, D.Z_SIDE_TOP, D.Z_ROOF_CROWN]
    froms = [D.SKEG_END, D.STERN_TIP, D.STERN_TIP, D.SHELTER_AFT, 0]
    dims += S.running_v(v, 0, levels, x_line, from_b=froms)
    return dims


# =========================================================================== plan


def plan_structure(v: S.View) -> None:
    zc = PLAN_CUT_Z / 1000
    hull = [(b, mm(L.plan_half, b)) for b in samples(D.STERN_TIP, D.STEM, 40)]
    v.poly(hull + [(b, -y) for b, y in reversed(hull)], "visible", closed=True)
    side_y = L.cabin_side_y(zc) * 1000
    lining = L.lining_in(zc) * 1000
    base = D.CABIN_BASE_HALF
    for sgn in (1, -1):
        v.line((D.SHELTER_AFT, sgn * base), (D.CABIN_LEN, sgn * base), "thin")
        # Cabin side, cut, broken at the openings on that side.
        cuts = sorted(
            [(op.b - op.w / 2, op.b + op.w / 2) for op in D.OPENINGS if op.side == sgn]
        )
        if D.SIDE_HATCH.side == sgn:
            cuts = sorted([*cuts, D.SIDE_HATCH.span])
        cur = D.SHELTER_AFT
        for a, c in cuts:
            v.poly([(cur, sgn * side_y), (a, sgn * side_y)], "cut")
            v.line((a, sgn * side_y), (a, sgn * lining), "thin")
            v.line((c, sgn * side_y), (c, sgn * lining), "thin")
            v.line((a, sgn * (side_y - 15)), (c, sgn * (side_y - 15)), "thin")
            v.line((a, sgn * (side_y - 35)), (c, sgn * (side_y - 35)), "thin")
            cur = c
        v.poly([(cur, sgn * side_y), (D.CABIN_LEN, sgn * side_y)], "cut")
        v.line((D.LINING_AFT, sgn * lining), (D.LINING_FWD, sgn * lining), "thin")
    # Bulkheads, cut, with doorways and their doors.
    for door in D.DOORS:
        out = -1 if door.end == "Stern" else 1
        b0, b1 = sorted((door.b, door.b - out * D.T_BULKHEAD))
        for sgn in (1, -1):
            v.rect(b0, sgn * side_y, b1, sgn * door.half, "cut")
            hinge = (door.b + out * 30, sgn * door.half)
            v.rect(door.b, sgn * door.half, door.b + out * 30, sgn * 5, "thin", WHITE)
            pts = [
                (
                    hinge[0] + out * (door.half - 5) * math.sin(math.radians(a)),
                    hinge[1] - sgn * (door.half - 5) * math.cos(math.radians(a)),
                )
                for a in range(0, 91, 10)
            ]
            v.poly(pts, "fine")
        lin = D.LINING_AFT if door.end == "Stern" else D.LINING_FWD
        for sgn in (1, -1):
            v.line((lin, sgn * lining), (lin, sgn * door.half), "thin")
    # Partitions with door leaves swung open.
    t = D.PARTITION_T / 2
    for p in D.PARTITIONS:
        y0, y1 = p.door_y
        for a, c in ((-lining, y0), (y1, lining)):
            if c - a > 1:
                v.rect(p.b - t, a, p.b + t, c, "visible", WHITE)
        hinge_y = y1 if p.hinge > 0 else y0
        free_y = y0 if p.hinge > 0 else y1
        width = abs(free_y - hinge_y)
        leaf_b = p.b + p.opens * width
        v.rect(
            min(p.b, leaf_b),
            hinge_y - p.hinge * 30,
            max(p.b, leaf_b),
            hinge_y,
            "thin",
            WHITE,
        )
        pts = [
            (
                p.b + p.opens * width * math.sin(math.radians(a)),
                hinge_y - p.hinge * width * math.cos(math.radians(a)),
            )
            for a in range(0, 91, 10)
        ]
        v.poly(pts, "fine")
    # Exterior: semi-trad shelter, counter deck, well deck, foredeck.
    b0, b1 = D.ENGINE_BOARDS
    seat = D.SEAT_LOCKERS
    inner = D.CABIN_BASE_HALF - D.T_CABIN - seat[2]
    for sgn in (1, -1):
        v.rect(
            seat[0],
            sgn * (D.CABIN_BASE_HALF - D.T_CABIN),
            seat[1],
            sgn * inner,
            "medium",
            WHITE,
        )
    for b in samples(b0, b1, 290):
        v.line((b, -inner), (b, inner), "thin")
    w0, w1, wh = D.WEED_HATCH
    v.rect(w0, -wh, w1, wh, "medium")
    lid = [
        (b, mm(L.plan_half, b) - 60) for b in samples(D.WELL_DECK_END, D.LOCKER_END, 50)
    ]
    v.poly(lid + [(b, -y) for b, y in reversed(lid)], "medium", closed=True)
    well = [
        (b, mm(L.plan_half, b) - D.T_HULL)
        for b in samples(D.CABIN_LEN, D.WELL_DECK_END, 100)
    ]
    v.poly(well, "thin")
    v.poly([(b, -y) for b, y in well], "thin")


def plan_fixture(v: S.View, f: D.Fixture) -> None:
    b0, b1, y0, y1 = f.footprint()
    k = f.kind
    cut = f.z1 > PLAN_CUT_Z and f.z0 < PLAN_CUT_Z
    style = "visible" if cut else "thin"
    if k == "steps":
        v.rect(b0, y0, b1, y1, "thin", WHITE)
        n = f.opt("risers")
        going = (b1 - b0) / (n - 1)
        for kx in range(1, n - 1):
            v.line((b0 + kx * going, y0), (b0 + kx * going, y1), "thin")
        toward = f.opt("rise_to")
        a, c = (b1 - 80, b0 + 60) if toward < 0 else (b0 + 80, b1 - 60)
        v.line((a, 0), (c, 0), "thin")
        S.arrowhead(v.sheet, v.P(c, 0), v.P(a, 0))
        v.text(((b0 + b1) / 2, y1 - 40), f"UP {n}R", 1.8, anchor="middle")
    elif k in ("cupboard", "wardrobe", "tall_unit"):
        v.rect(b0, y0, b1, y1, style, WHITE)
        if cut:
            v.line((b0, y0), (b1, y1), "fine")
            v.line((b0, y1), (b1, y0), "fine")
    elif k == "base_run":
        v.rect(b0 - 15, y0, b1, y1, "thin", WHITE)
    elif k == "fridge":
        v.rect(b0, y0, b1, y1, "hidden")
    elif k == "sink":
        bb0, bb1 = f.opt("bowl")
        v.poly(rounded_rect(bb0, y0 + 40, bb1, y1 - 40, 60), "thin", closed=True)
        for b in samples(bb1 + 80, b1 - 60, 70):
            v.line((b, y0 + 70), (b, y1 - 70), "fine")
        tap_y = y0 + 60 if f.side < 0 else y1 - 60
        v.circle(((bb0 + bb1) / 2, tap_y), 20, "thin")
    elif k == "cooker":
        v.rect(b0, y0, b1, y1, "thin", WHITE)
        for db in (0.28, 0.72):
            for dy in (0.3, 0.7):
                v.circle(
                    (b0 + db * (b1 - b0), y0 + dy * (y1 - y0)),
                    75 if db < 0.5 else 55,
                    "thin",
                )
    elif k == "wall_units":
        v.rect(b0, y0, b1, y1, "hidden")
    elif k == "bed":
        v.rect(b0, y0, b1, y1, "thin", WHITE)
        head = f.opt("head")
        hb = b1 - 60 if head > 0 else b0 + 60
        for yc in ((y0 + (y1 - y0) * 0.27), (y0 + (y1 - y0) * 0.73)):
            v.poly(
                rounded_rect(
                    hb - 380 if head > 0 else hb,
                    yc - 260,
                    hb if head > 0 else hb + 380,
                    yc + 260,
                    70,
                ),
                "thin",
                closed=True,
            )
        fold = hb - 900 if head > 0 else hb + 900
        v.line((fold, y0), (fold, y1), "thin")
        v.line((fold, y0), (fold - 250 * head, y0 + 250), "fine")
    elif k == "shower":
        v.rect(b0, y0, b1, y1, "thin", WHITE)
        corner = (b0, y0) if f.side < 0 else (b0, y1)
        r = b1 - b0
        sgn = 1 if f.side < 0 else -1
        pts = [
            (
                corner[0] + r * math.cos(math.radians(a)),
                corner[1] + sgn * r * math.sin(math.radians(a)),
            )
            for a in range(0, 91, 6)
        ]
        v.poly(pts, "medium")
        c = ((b0 + b1) / 2, (y0 + y1) / 2)
        for p in ((b0, y0), (b1, y0), (b1, y1), (b0, y1)):
            v.line(p, c, "fine")
        v.circle(c, 40, "thin")
    elif k == "vanity":
        v.rect(b0, y0, b1, y1, "thin", WHITE)
        v.poly(ellipse((b0 + b1) / 2, (y0 + y1) / 2, 170, 130), "thin", closed=True)
    elif k == "toilet":
        back = y0 if f.side < 0 else y1
        v.rect(b0, back, b1, back + f.side * -180, "thin", WHITE)
        v.poly(
            ellipse((b0 + b1) / 2, back - f.side * 360, 170, 210),
            "thin",
            closed=True,
            fill=WHITE,
        )
        v.poly(
            ellipse((b0 + b1) / 2, back - f.side * 360, 120, 160), "thin", closed=True
        )
    elif k == "towel_rail":
        v.rect(b0, y0, b1, y1, "thin", WHITE)
    elif k == "hearth":
        v.hatch(rounded_rect(b0, y0, b1, y1, 1), "cross")
        v.rect(b0, y0, b1, y1, "thin")
    elif k == "stove":
        v.rect(b0, y0, b1, y1, "medium", WHITE)
        v.rect(b0 + 30, y0, b1 - 30, y0 + 40, "thin")
        v.circle(D.CHIMNEY, f.opt("flue_d") / 2, "visible")
    elif k == "heat_shield":
        v.hatch(rounded_rect(b0, y0, b1, y1, 1), "hatch45")
        v.rect(b0, y0, b1, y1, "thin")
    elif k == "dinette":
        (a0, a1), (c0, c1) = f.opt("benches")
        for s0, s1, back_start in ((a0, a1, True), (c0, c1, False)):
            v.rect(s0, y0, s1, y1, "thin", WHITE)
            bb = s0 + 110 if back_start else s1 - 110
            v.line((bb, y0), (bb, y1), "thin")
        t0, t1 = f.opt("table")
        inner = y1 - 40 if f.side < 0 else y0 + 40
        v.rect(
            t0 + 20,
            y0 if f.side < 0 else inner,
            t1 - 20,
            inner if f.side < 0 else y1,
            "thin",
            WHITE,
        )
    elif k in ("sofa", "armchair"):
        v.poly(rounded_rect(b0, y0, b1, y1, 60), "thin", closed=True, fill=WHITE)
        back = y1 - 180 if f.side > 0 else y0 + 180
        v.line((b0 + 150, back), (b1 - 150, back), "thin")
        v.line((b0 + 150, y0), (b0 + 150, y1), "thin")
        v.line((b1 - 150, y0), (b1 - 150, y1), "thin")
    elif k == "shelves":
        v.rect(b0, y0, b1, y1, "thin", WHITE)


# Where each room's name goes in the plan (b, y), clear of furniture and door swings.
ZONE_LABELS = {
    "Bedroom": (2150, -80),
    "Bathroom": (4420, 600),
    "Galley": (6725, -80),
    "Saloon": (11045, -80),
}

PLAN_LABELS = {
    "CalorifierCupboard": ("CALORIFIER", 0),
    "ElectricsCupboard": ("ELECTRICS", 0),
    "Wardrobe": ("WARDROBE", 0),
    "Bed": ("DOUBLE BED 1905 x 1220", 0),
    "Shower": ("SHOWER", 0),
    "Vanity": ("BASIN", 0),
    "Toilet": ("WC", 0),
    "Larder": ("LARDER", 0),
    "Cooker": ("COOKER", 0),
    "Fridge": ("FRIDGE", 0),
    "Sink": ("SINK", 0),
    "Dinette": ("DINETTE", 0),
    "Sofa": ("SOFA", 0),
    "Armchair": ("CHAIR", 0),
    "Shelves": ("BOOKCASE", 0),
    "CoatLocker": ("COATS", 0),
    "LogLocker": ("LOGS", 0),
}


def plan_view(sheet: S.Sheet) -> tuple[S.View, list]:
    v = S.View(sheet, "plan", B0_X, PLAN_Y, SCALE, 1, 1)
    dims = []
    with sheet.group("plan"):
        lining = L.lining_in(PLAN_CUT_Z / 1000) * 1000
        finish = {"Oak": "boards", "Carpet": "carpet", "Vinyl": "vinyl"}
        for z in D.ZONES:
            v.hatch(
                [(z.b0, -lining), (z.b1, -lining), (z.b1, lining), (z.b0, lining)],
                finish[z.floor],
            )
        plan_structure(v)
        for f in D.FIXTURES:
            if not f.host:
                plan_fixture(v, f)
        for f in D.FIXTURES:
            if f.host:
                plan_fixture(v, f)
        hidden_plan(v)
        for f in D.FIXTURES:
            if f.name in PLAN_LABELS:
                text, _ = PLAN_LABELS[f.name]
                b0, b1, y0, y1 = f.footprint()
                h = 1.8
                if S.text_width(text, h) * SCALE > (b1 - b0) - 40:
                    v.text(
                        ((b0 + b1) / 2, (y0 + y1) / 2),
                        text,
                        h,
                        anchor="middle",
                        middle=True,
                        rot=90,
                    )
                else:
                    v.text(
                        ((b0 + b1) / 2, (y0 + y1) / 2),
                        text,
                        h,
                        anchor="middle",
                        middle=True,
                    )
        for z in D.ZONES:
            if z.name in ZONE_LABELS:
                v.text(
                    ZONE_LABELS[z.name],
                    z.name.upper(),
                    2.5,
                    anchor="middle",
                    middle=True,
                    bold=True,
                )
        v.text((-450, 0), "ENGINE BAY", 1.8, anchor="middle", middle=True)
        v.text(
            ((D.CABIN_LEN + D.WELL_DECK_END) / 2, 0),
            "WELL DECK",
            1.8,
            anchor="middle",
            middle=True,
        )
        v.text(
            ((D.WELL_DECK_END + D.LOCKER_END) / 2, 0),
            "GAS",
            1.8,
            anchor="middle",
            middle=True,
        )
        for sym, b, y in D.SAFETY:
            safety_symbol(v, sym, b, y)
        cutting_planes(v)
    with sheet.group("plan_dims"):
        dims += plan_dims(v)
    return v, dims


def safety_symbol(v: S.View, sym: str, b: float, y: float) -> None:
    x, yy = v.P(b, y)
    if sym in ("CO", "SA"):
        v.sheet.circle((x, yy), 1.9, "thin", WHITE)
    else:
        v.sheet.rect(x - 1.9, yy - 1.9, x + 1.9, yy + 1.9, "thin", WHITE)
    v.sheet.text((x, yy), sym, 1.8, anchor="middle", middle=True)


def hidden_plan(v: S.View) -> None:
    for box in (D.ENGINE, D.GEARBOX, D.BATTERIES):
        v.rect(box[0], box[2], box[1], box[3], "hidden")
    for tank in (D.FUEL_TANK, D.WATER_TANK):
        b0, b1 = tank[0], tank[1]
        pts = [(b, mm(L.plan_half, b) - 30) for b in samples(b0, b1, 50)]
        v.poly(pts + [(b, -y) for b, y in reversed(pts)], "hidden", closed=True)
    rb0, rb1, _, _ = D.RUDDER
    v.rect(rb0, -25, rb1, 25, "hidden")
    d, _ = D.GAS_CYLINDER
    for b, y in L.cylinder_positions():
        v.circle((b * 1000, y * 1000), d / 2, "hidden")
    neck = L.swan_neck()
    _, (tb1, _) = L.tiller_bar()
    v.line((neck[0][0] * 1000, 0), (tb1 * 1000, 0), "medium")
    v.circle((D.RUDDER_STOCK_B, 0), 40, "thin", WHITE)
    for b, y in D.T_STUDS:
        v.circle((b, y), 40, "thin", WHITE)
    rx = 110
    v.poly(
        ellipse(D.STEM + D.FENDER_FWD - rx, 0, rx, 150), "thin", closed=True, fill=WHITE
    )
    v.poly(
        ellipse(D.STERN_TIP - D.FENDER_AFT + 100, 0, 100, 250),
        "thin",
        closed=True,
        fill=WHITE,
    )


def cutting_planes(v: S.View) -> None:
    s = v.sheet
    xa, xb = v.x(D.STERN_TIP - 650), v.x(D.STEM + 500)
    yc = v.y(0)
    s.line((xa, yc), (xb, yc), "centre")
    for x in (xa, xb):
        s.line((x - 1.5, yc), (x + 1.5, yc), "visible")
        for sgn, letter in ((-1, "A"), (1, "B")):
            tip = (x, yc + sgn * 5.5)
            s.line((x, yc), tip, "thin")
            S.arrowhead(s, tip, (x, yc))
            s.text(
                (x + 1.2, yc + sgn * 5.0 + (0.6 if sgn > 0 else 0.4)),
                letter,
                3.5,
                bold=True,
            )
    # Transverse section C–C through the galley, looking forward (drawn on sheet 2).
    bc = SECTION_C_B
    x = v.x(bc)
    ya, yb = v.y(1150), v.y(-1150)
    for y, end in ((ya, -1), (yb, 1)):
        s.line((x, y), (x, y + end * -2.5), "visible")
        s.line((x, y), (x + 5, y), "thin")
        S.arrowhead(s, (x + 5, y), (x, y))
        s.text((x + 6, y + 1.2), "C", 3.5, bold=True)
    s.line((x, ya), (x, yb), "centre")


def plan_dims(v: S.View) -> list:
    dims = []
    top = v.y(D.HALF_BEAM) - 5.0
    stations = [0, *[z.b0 for z in D.ZONES], D.LINING_FWD, D.CABIN_LEN]
    dims += S.chain_h(v, stations, top, from_v=[None] * len(stations), h=1.8)
    # Clear widths inside: galley aisle and bedroom corridor.
    port = D.FIXTURE["GalleyPort"].footprint()
    stbd = D.FIXTURE["GalleyStarboard"].footprint()
    dims.append(S.dim_v(v, stbd[3], port[2], v.x(7700), h=1.8, text_left=False))
    bed = D.FIXTURE["Bed"].footprint()
    dims.append(S.dim_v(v, bed[3], D.LINING_IN, v.x(2900), h=1.8, text_left=False))
    # Beam over the rubbing strakes, at the stern.
    half = D.HALF_BEAM + D.STRAKE_PROUD
    dims.append(
        S.dim_v(v, -half, half, 28.5, from_b=(D.SHELTER_AFT, D.SHELTER_AFT), h=1.8)
    )
    for p in D.PARTITIONS:
        y0, y1 = p.door_y
        v.text(
            (p.b + 120, (y0 + y1) / 2),
            f"{y1 - y0} CLR",
            1.8,
            anchor="start",
            middle=True,
            rot=90,
        )
    return dims


# =========================================================================== sheet 1


def overall_dims(v: S.View, y_base: float) -> list:
    """Overall length over fenders, hull length and the stern/cabin/bow chain, above A–A."""
    dims = []
    stern_f, stem_f = D.STERN_TIP - D.FENDER_AFT, D.STEM + D.FENDER_FWD
    dims.append(
        S.dim_h(
            v,
            stern_f,
            stem_f,
            y_base,
            from_v=(D.FENDER_AFT_Z, D.FENDER_FWD_Z),
            text=f"{D.LENGTH_OVER_FENDERS} OVER FENDERS (57'-0\")",
        )
    )
    dims.append(
        S.dim_h(
            v,
            D.STERN_TIP,
            D.STEM,
            y_base + 6,
            from_v=(D.DECK_Z, D.SHEER[-1][1]),
            text=f"{D.HULL_LENGTH} HULL",
        )
    )
    stations = [D.STERN_TIP, D.SHELTER_AFT, 0, D.CABIN_LEN, D.WELL_DECK_END, D.STEM]
    dims += S.chain_h(
        v,
        stations,
        y_base + 12,
        from_v=[None, D.Z_SIDE_TOP, D.Z_ROOF_CROWN, D.Z_ROOF_CROWN, None, None],
    )
    return dims


SHEET1_NOTES = (
    (
        "Dimensions in millimetres. Lengths from the aft face of the rear cabin bulkhead"
        " (datum, origin circle); heights from the underside of the baseplate."
    ),
    (
        "Sections A-A and B-B are on the centreline and drawn in first-angle projection about"
        " the plan, so B-B is inverted with its baseplate towards the plan and all lengths"
        " align. The plan is cut 1300 above the floor; items above the cut are dashed."
    ),
    (
        "Finishes: walls ash-faced 9 ply; ceilings painted T&G; floors engineered oak (lobby,"
        " galley, saloon), carpet (bedroom), vinyl (bathroom); solid oak worktops; bathroom"
        " walls painted T&G with PVC shower panels."
    ),
    (
        "Stove: 4.5 kW multi-fuel on a raised non-combustible hearth projecting 225 in front and"
        " 150 each side; 25 calcium silicate shields on 10 air gaps; 125 flue; removable"
        " chimney (BS 8511, BSS)."
    ),
    (
        "LPG: 2 x 13 kg propane in the foredeck locker, opening to the outside only, drained"
        " overboard through 19 ID from its lowest point above the waterline; cooker with"
        " flame-failure devices and a shut-off valve."
    ),
    (
        "Safety: CO alarms to BS EN 50291 in the saloon and bedroom, smoke alarm in the galley,"
        " 5A/34B extinguishers at both exits and the galley, fire blanket by the cooker. Exits:"
        " stern doors, bow doors and side hatch."
    ),
    (
        "Concept general arrangement, not for construction. Scantlings, stability (ISO 12217-1,"
        " RCR category D) and systems to be confirmed by the builder."
    ),
)


def legend(sheet: S.Sheet, x: float, y: float) -> None:
    sheet.text((x, y), "LEGEND", 2.5, bold=True)
    rows = [
        ("FE", "rect", "Fire extinguisher 5A/34B"),
        ("FB", "rect", "Fire blanket"),
        ("CO", "circle", "CO alarm"),
        ("SA", "circle", "Smoke alarm"),
    ]
    yy = y + 4.5
    for sym, shape, text in rows:
        if shape == "rect":
            sheet.rect(x, yy - 1.9, x + 3.8, yy + 1.9, "thin", WHITE)
        else:
            sheet.circle((x + 1.9, yy), 1.9, "thin", WHITE)
        sheet.text((x + 1.9, yy), sym, 1.8, anchor="middle", middle=True)
        sheet.text((x + 5.5, yy + 0.6), text, 1.8)
        yy += 4.6
    for pattern, text in (
        ("boards", "Engineered oak"),
        ("carpet", "Carpet"),
        ("vinyl", "Vinyl"),
        ("cross", "Tiled hearth"),
    ):
        sheet.hatch(
            [(x, yy - 1.8), (x + 8, yy - 1.8), (x + 8, yy + 1.8), (x, yy + 1.8)],
            pattern,
        )
        sheet.rect(x, yy - 1.8, x + 8, yy + 1.8, "thin")
        sheet.text((x + 9.5, yy + 0.6), text, 1.8)
        yy += 4.6
    sheet.line((x, yy), (x + 8, yy), "hidden")
    sheet.text((x + 9.5, yy + 0.6), "Hidden / above the cut", 1.8)
    yy += 4.0
    sheet.line((x, yy), (x + 8, yy), "centre")
    sheet.text((x + 9.5, yy + 0.6), "Centre line, cutting plane, DWL", 1.8)


def sheet1() -> tuple[S.Sheet, dict]:
    sheet = S.Sheet("NB57-GA-001 General arrangement and furniture layout")
    S.frame(sheet)
    a, dims_a = elevation(sheet, 1)
    with sheet.group("overall_dims"):
        dims_o = overall_dims(a, 16.0)
    p, dims_p = plan_view(sheet)
    b, dims_b = elevation(sheet, -1)
    with sheet.group("labels"):
        view_label(sheet, (343.0, A_BASE_Y + 6.5), "A-A", ["LOOKING TO PORT   1:50"])
        view_label(
            sheet, (343.0, PLAN_Y + 25.5), "PLAN", ["CUT 1300 ABOVE FLOOR   1:50"]
        )
        view_label(
            sheet,
            (343.0, b.y(D.Z_ROOF_CROWN) + 9.5),
            "B-B",
            ["LOOKING TO STARBOARD   1:50", "FIRST-ANGLE POSITION, INVERTED"],
        )
    with sheet.group("notes"):
        S.notes(sheet, (22.0, 232.0), "NOTES", SHEET1_NOTES, 150.0)
        legend(sheet, 182.0, 232.0)
    with sheet.group("scale"):
        S.scale_bar(sheet, (232.0, 234.0), SCALE)
        fwd_arrow(sheet, (360.0, 233.0))
    S.title_block(
        sheet,
        S.TitleBlock(
            OWNER,
            "NB57-GA-001",
            REVISIONS[-1][0],
            DATE,
            "1/2",
            f"{D.TITLE} '{D.NAME.upper()}'",
            "GENERAL ARRANGEMENT AND FURNITURE LAYOUT: SECTIONS A-A, B-B AND PLAN",
            "General arrangement",
            STATUS,
            CREATOR,
            "-",
            "1:50",
        ),
        REVISIONS,
    )
    views = {"plan": p, "port": a, "starboard": b}
    return sheet, {
        "views": views,
        "dims": {
            "overall": dims_o,
            "port": dims_a,
            "plan": dims_p,
            "starboard": dims_b,
        },
    }


def view_label(sheet: S.Sheet, at, name: str, lines) -> None:
    x, y = at
    sheet.text((x, y), name, 5.0, bold=True)
    tx = x + S.text_width(name, 5.0) + 2.5
    for k, line in enumerate(lines):
        sheet.text(
            (tx, y - 2.6 + 3.2 * k if len(lines) > 1 else y),
            line,
            2.5 if k == 0 else 1.8,
        )


def fwd_arrow(sheet: S.Sheet, at) -> None:
    x, y = at
    sheet.line((x, y), (x + 14, y), "medium")
    S.arrowhead(sheet, (x + 16.5, y), (x, y))
    sheet.text((x + 7, y - 1.2), "FORWARD", 1.8, anchor="middle")


# =========================================================================== sheet 2

PROFILE_BASE_Y, ROOF_Y = 74.0, 104.0
SEC_SCALE, SEC_X, SEC_BASE_Y = 20.0, 100.0, 262.0


def profile_view(sheet: S.Sheet) -> tuple[S.View, list]:
    """Outboard profile from starboard: hull, cabin, openings, roof gear, hidden machinery."""
    v = S.View(sheet, "profile", B0_X, PROFILE_BASE_Y, SCALE, 1, 1)
    dims = []
    with sheet.group("profile"):
        # Hidden first: machinery, tanks, bulkheads, floor and well-deck sole, gas locker.
        for box in (D.ENGINE, D.GEARBOX, D.BATTERIES):
            v.rect(box[0], box[4], box[1], box[5], "hidden")
        for t in (D.FUEL_TANK, D.WATER_TANK):
            v.rect(t[0], t[2], t[1], t[3], "hidden")
        for b in (0, D.CABIN_LEN):
            v.line((b, 10), (b, D.Z_GUNWALE), "hidden")
        v.line(
            (D.ZONE["Bedroom"].b0, D.FLOOR_Z),
            (D.FIXTURE["BowSteps"].b0, D.FLOOR_Z),
            "hidden",
        )
        v.line((D.CABIN_LEN, D.WELL_SOLE_Z), (D.WELL_DECK_END, D.WELL_SOLE_Z), "hidden")
        sole = L.locker_sole_z() * 1000
        v.poly(
            [
                (D.WELL_DECK_END, D.WELL_SOLE_Z),
                (D.WELL_DECK_END, sole),
                (D.LOCKER_END, sole),
                (D.LOCKER_END, mm(L.gunwale_z, D.LOCKER_END)),
            ],
            "hidden",
        )
        d, h = D.GAS_CYLINDER
        for b, y in L.cylinder_positions():
            if y < 0:
                v.poly(
                    rounded_rect(
                        b * 1000 - d / 2, sole + 10, b * 1000 + d / 2, sole + h - 50, 50
                    ),
                    "hidden",
                    closed=True,
                )
        v.line((D.GEARBOX[0], 330), (D.PROP_B + 70, D.PROP_Z), "hidden")
        # Hull: keel, stem, counter, sheer, strakes and the swim's flare line.
        keel = [(b, mm(L.bottom_z, b)) for b in samples(D.STERN_POST, D.STEM, 25)]
        keel.append((D.STEM, mm(L.gunwale_z, D.STEM)))
        v.poly(keel, "visible")
        v.poly(
            [
                (D.STERN_POST, 0),
                (D.STERN_POST, D.Z_COUNTER),
                (D.STERN_TIP, D.Z_COUNTER),
                (D.STERN_TIP, D.DECK_Z),
            ],
            "visible",
        )
        sheer = [(b, mm(L.gunwale_z, b)) for b in samples(D.STERN_TIP, D.STEM, 100)]
        v.poly(sheer, "visible")
        for z in D.Z_STRAKES:
            for dz in (-D.STRAKE_H / 2, D.STRAKE_H / 2):
                v.line((D.STERN_TIP + 60, z + dz), (D.STEM - 350, z + dz), "thin")
        swim = [
            (b, mm(L.swim_full_z, b)) for b in samples(D.STERN_POST, D.SWIM_START, 50)
        ]
        v.poly(swim, "thin")
        v.rect(D.SKEG_END, 0, D.STERN_POST, 50, "visible", WHITE)
        # Cabin and shelter.
        v.poly(
            [
                (0, D.Z_GUNWALE),
                (0, D.Z_ROOF_CROWN),
                (D.CABIN_LEN, D.Z_ROOF_CROWN),
                (D.CABIN_LEN, D.Z_GUNWALE),
            ],
            "visible",
        )
        v.line((0, D.Z_SIDE_TOP), (D.CABIN_LEN, D.Z_SIDE_TOP), "thin")
        v.line((0, D.Z_ROOF_EDGE), (D.CABIN_LEN, D.Z_ROOF_EDGE), "thin")
        v.rect(D.SHELTER_AFT, D.DECK_Z, 0, D.Z_SIDE_TOP, "visible")
        v.rect(D.SHELTER_AFT, D.Z_SIDE_TOP, 0, D.Z_SIDE_TOP + 30, "thin", WHITE)
        p0, p1 = D.PANEL_B
        z0, z1 = D.PANEL_Z
        v.poly(rounded_rect(p0, z0, p1, z1, 90), "thin", closed=True)
        v.poly(
            rounded_rect(p0 + 40, z0 + 40, p1 - 40, z1 - 40, 50), "thin", closed=True
        )
        for op in D.OPENINGS:
            if op.side < 0:
                opening_elevation(v, op)
        if D.SIDE_HATCH.side < 0:
            side_hatch_elevation(v)
        # Roof gear.
        top = L.handrail_top() * 1000
        v.rect(D.HANDRAIL_B[0], top - D.HANDRAIL_D, D.HANDRAIL_B[1], top, "thin", WHITE)
        for b in samples(D.HANDRAIL_B[0], D.HANDRAIL_B[1], 1400):
            v.rect(b - 15, D.Z_ROOF_EDGE, b + 15, top - D.HANDRAIL_D, "thin", WHITE)
        chimney_elevation(v)
        pole_and_plank_side(v)
        for b in D.VENT_BS:
            v.poly(vent_profile(b, D.Z_ROOF_CROWN), "thin", closed=True, fill=WHITE)
        b0, b1, _ = D.REAR_HATCH
        v.rect(b0, D.Z_ROOF_CROWN, b1, D.Z_ROOF_CROWN + 60, "thin", WHITE)
        # Deck gear: locker lid above the sheer, studs, tiller, rudder, propeller, fenders.
        lid = L.locker_lid_z() * 1000
        v.poly(
            [
                (D.WELL_DECK_END - 20, mm(L.gunwale_z, D.WELL_DECK_END)),
                (D.WELL_DECK_END - 20, lid + 12),
                (D.LOCKER_END + 20, lid + 12),
                (D.LOCKER_END + 20, mm(L.gunwale_z, D.LOCKER_END + 20)),
            ],
            "thin",
        )
        for b, y in D.T_STUDS:
            z = mm(L.gunwale_z, b) if b > 0 else D.DECK_Z
            v.rect(b - 30, z, b + 30, z + 120, "thin", WHITE)
            v.rect(b - 80, z + 120, b + 80, z + 145, "thin", WHITE)
        r = D.PROP_D / 2
        v.rect(
            D.PROP_B - 70, D.PROP_Z - 50, D.PROP_B + 70, D.PROP_Z + 50, "medium", WHITE
        )
        for sgn in (1, -1):
            v.poly(
                ellipse(D.PROP_B, D.PROP_Z + sgn * (r / 2 + 25), 55, r / 2 - 25),
                "thin",
                closed=True,
                fill=WHITE,
            )
        rb0, rb1, rz0, rz1 = D.RUDDER
        v.poly(rounded_rect(rb0, rz0, rb1, rz1, 60), "visible", closed=True, fill=WHITE)
        stock = D.RUDDER_STOCK_B
        v.rect(
            stock - D.STOCK_D / 2,
            D.DECK_Z,
            stock + D.STOCK_D / 2,
            D.DECK_Z + D.STOCK_TOP,
            "thin",
            WHITE,
        )
        v.poly([(b * 1000, z * 1000) for b, z in L.swan_neck()], "medium")
        (tb0, tz0), (tb1, tz1) = L.tiller_bar()
        v.poly([(tb0 * 1000, tz0 * 1000), (tb1 * 1000, tz1 * 1000)], "visible")
        rx, rz = 110, D.FENDER_H / 2
        v.poly(
            ellipse(D.STEM + D.FENDER_FWD - rx, D.FENDER_FWD_Z, rx, rz),
            "thin",
            closed=True,
            fill=WHITE,
        )
        v.poly(
            ellipse(D.STERN_TIP - D.FENDER_AFT + 100, D.FENDER_AFT_Z, 100, rz - 20),
            "thin",
            closed=True,
            fill=WHITE,
        )
        waterline(v)
        profile_notes(v)
    with sheet.group("profile_dims"):
        stern_f, stem_f = D.STERN_TIP - D.FENDER_AFT, D.STEM + D.FENDER_FWD
        dims.append(
            S.dim_h(
                v,
                stern_f,
                stem_f,
                15.0,
                from_v=(D.FENDER_AFT_Z, D.FENDER_FWD_Z),
                text=f"{D.LENGTH_OVER_FENDERS} OVER FENDERS",
            )
        )
        aft_wl = mm(L.wl_z, D.STERN_POST)
        dims.append(
            S.dim_v(v, 0, aft_wl, 33.0, from_b=(D.SKEG_END, D.STERN_TIP), h=1.8)
        )
        dims.append(
            S.dim_v(v, aft_wl, D.DECK_Z, 33.0, from_b=(None, D.STERN_TIP), h=1.8)
        )
        fwd_wl = mm(L.wl_z, D.STEM)
        dims.append(
            S.dim_v(
                v,
                0,
                fwd_wl,
                398.0,
                from_b=(FOREFOOT_START_MM, D.STEM),
                h=1.8,
                text_left=False,
            )
        )
        b_air, z_air = L.air_draught_point()
        b_air *= 1000
        wl_air = mm(L.wl_z, b_air)
        dims.append(
            S.dim_v(
                v,
                wl_air,
                z_air * 1000,
                v.x(b_air) + 2.5,
                from_b=(None, b_air),
                h=1.8,
                text=f"{round(z_air * 1000 - wl_air)} AIR DRAUGHT",
                text_left=False,
            )
        )
    return v, dims


def profile_notes(v: S.View) -> None:
    s = v.sheet

    def note(b, z, dx, dz, text):
        tx, ty = v.P(b, z)
        s.line((tx, ty), (tx + dx, ty - dz), "thin")
        s.circle((tx, ty), 0.35, None, fill=S.INK)
        s.text(
            (tx + dx + (0.6 if dx > 0 else -0.6), ty - dz + 0.6),
            text,
            1.8,
            anchor="start" if dx > 0 else "end",
            knockout=True,
        )

    note(1500, D.PANEL_Z[1] - 20, 4, 6, "COACHLINE PANEL")
    note(-450, D.Z_SIDE_TOP - 100, -6, 9, "SEMI-TRAD SHELTER")
    note(-300, mm(L.swim_full_z, -300), 3, -9, "SWIM")
    note(5000, D.Z_STRAKES[1] + 25, 3, -10, "RUBBING STRAKES 50 x 13 (3)")
    note(
        (D.WELL_DECK_END + D.LOCKER_END) / 2,
        L.locker_lid_z() * 1000 + 12,
        4,
        6,
        "GAS LOCKER LID",
    )
    note((D.ENGINE[0] + D.ENGINE[1]) / 2, D.ENGINE[5] - 80, 7, 8, "ENGINE (HIDDEN)")


def roof_plan(sheet: S.Sheet) -> S.View:
    """Roof and deck plan: everything seen from above."""
    v = S.View(sheet, "roof", B0_X, ROOF_Y, SCALE, 1, 1)
    with sheet.group("roof"):
        hull = [(b, mm(L.plan_half, b)) for b in samples(D.STERN_TIP, D.STEM, 40)]
        v.poly(hull + [(b, -y) for b, y in reversed(hull)], "visible", closed=True)
        for sgn in (1, -1):
            v.line(
                (D.SHELTER_AFT, sgn * D.CABIN_BASE_HALF),
                (D.CABIN_LEN, sgn * D.CABIN_BASE_HALF),
                "thin",
            )
            v.line(
                (D.SHELTER_AFT, sgn * (D.CABIN_BASE_HALF - 3)),
                (0, sgn * (D.CABIN_BASE_HALF - 3)),
                "visible",
            )
            v.line(
                (0, sgn * D.CABIN_TOP_HALF),
                (D.CABIN_LEN, sgn * D.CABIN_TOP_HALF),
                "visible",
            )
            well = [
                (b, sgn * (mm(L.plan_half, b) - D.GUNWALE_W))
                for b in samples(D.CABIN_LEN, D.WELL_DECK_END, 100)
            ]
            v.poly(well, "thin")
            y = sgn * L.handrail_y() * 1000
            r = D.HANDRAIL_D / 2
            v.rect(D.HANDRAIL_B[0], y - r, D.HANDRAIL_B[1], y + r, "thin", WHITE)
        v.line((0, -D.CABIN_TOP_HALF), (0, D.CABIN_TOP_HALF), "visible")
        v.line(
            (D.CABIN_LEN, -D.CABIN_TOP_HALF), (D.CABIN_LEN, D.CABIN_TOP_HALF), "visible"
        )
        v.line(
            (D.CABIN_LEN, -D.CABIN_BASE_HALF), (D.CABIN_LEN, D.CABIN_BASE_HALF), "thin"
        )
        for b in D.VENT_BS:
            v.circle((b, 0), D.VENT_D / 2, "thin", WHITE)
            v.circle((b, 0), 30, "thin")
        cb, cy = D.CHIMNEY
        v.circle((cb, cy), D.COLLAR_D / 2, "thin", WHITE)
        v.circle((cb, cy), D.CHIMNEY_D / 2, "medium", WHITE)
        b0, b1, half = D.REAR_HATCH
        v.rect(b0, -half, b1, half, "thin", WHITE)
        for sgn in (1, -1):
            v.line((b0, sgn * (half + 25)), (b1 + 650, sgn * (half + 25)), "thin")
        for b in D.PLANK_BLOCKS:
            v.rect(b - 40, D.BLOCK_Y[0], b + 40, D.BLOCK_Y[1], "thin", WHITE)
        pw = D.PLANK_W / 2
        v.rect(
            D.PLANK_B[0], D.PLANK_Y - pw, D.PLANK_B[1], D.PLANK_Y + pw, "thin", WHITE
        )
        v.rect(
            D.POLE_B,
            D.POLE_Y - D.POLE_D / 2,
            D.POLE_B + D.POLE_LEN,
            D.POLE_Y + D.POLE_D / 2,
            "thin",
            WHITE,
        )
        # Semi-trad deck: seat lockers, engine boards, counter, weed hatch, studs, tiller.
        seat = D.SEAT_LOCKERS
        inner = D.CABIN_BASE_HALF - D.T_CABIN - seat[2]
        for sgn in (1, -1):
            v.rect(
                seat[0],
                sgn * (D.CABIN_BASE_HALF - D.T_CABIN),
                seat[1],
                sgn * inner,
                "thin",
                WHITE,
            )
        b0, b1 = D.ENGINE_BOARDS
        v.rect(b0, -inner, b1, inner, "thin")
        for b in samples(b0, b1, 290)[1:-1]:
            v.line((b, -inner), (b, inner), "thin")
        w0, w1, wh = D.WEED_HATCH
        v.rect(w0, -wh, w1, wh, "medium", WHITE)
        for b, y in D.T_STUDS:
            v.circle((b, y), 40, "thin", WHITE)
            v.rect(b - 12, y - 80, b + 12, y + 80, "thin")
        for (b, y), letter in ((D.DIESEL_FILLER, "D"), (D.WATER_FILLER, "W")):
            v.circle((b, y), 45, "thin", WHITE)
            v.text((b, y), letter, 1.8, anchor="middle", middle=True)
        neck = L.swan_neck()
        _, (tb1, _) = L.tiller_bar()
        v.rect(neck[0][0] * 1000, -20, tb1 * 1000, 20, "thin", WHITE)
        v.circle((D.RUDDER_STOCK_B, 0), D.STOCK_D / 2 + 10, "thin", WHITE)
        # Bow: doors, locker lid, fender.
        for door in D.DOORS:
            out = -1 if door.end == "Stern" else 1
            for sgn in (1, -1):
                v.rect(
                    door.b, sgn * 5, door.b + out * 30, sgn * door.half, "thin", WHITE
                )
        lid = [
            (b, mm(L.plan_half, b) - 60)
            for b in samples(D.WELL_DECK_END, D.LOCKER_END, 50)
        ]
        v.poly(
            lid + [(b, -y) for b, y in reversed(lid)], "medium", closed=True, fill=WHITE
        )
        v.line(
            (D.WELL_DECK_END + 40, -mm(L.plan_half, D.WELL_DECK_END) + 100),
            (D.WELL_DECK_END + 40, mm(L.plan_half, D.WELL_DECK_END) - 100),
            "thin",
        )
        d, _ = D.GAS_CYLINDER
        for b, y in L.cylinder_positions():
            v.circle((b * 1000, y * 1000), d / 2, "hidden")
        v.poly(
            ellipse(D.STEM + D.FENDER_FWD - 110, 0, 110, 150),
            "thin",
            closed=True,
            fill=WHITE,
        )
        v.poly(
            ellipse(D.STERN_TIP - D.FENDER_AFT + 100, 0, 100, 250),
            "thin",
            closed=True,
            fill=WHITE,
        )
        s = v.sheet
        xa, xb = v.x(D.STERN_TIP - 650), v.x(D.STEM + 500)
        s.line((xa, v.y(0)), (xb, v.y(0)), "centre")
        x = v.x(SECTION_C_B)
        ya, yb = v.y(1150), v.y(-1150)
        for y, end in ((ya, -1), (yb, 1)):
            s.line((x, y), (x, y + end * -2.5), "visible")
            s.line((x, y), (x + 5, y), "thin")
            S.arrowhead(s, (x + 5, y), (x, y))
            s.text((x + 6, y + 1.2), "C", 3.5, bold=True)
        roof_notes(v)
    return v


def roof_notes(v: S.View) -> None:
    """Labels for the roof and deck plan, set in the clear parts of the roof and decks."""
    s = v.sheet

    def note(target, at, text, anchor="start"):
        if target is not None:
            tx, ty = v.P(*target)
            ax, ay = v.P(*at)
            s.line((tx, ty), (ax, ay - 0.6), "thin")
            s.circle((tx, ty), 0.35, None, fill=S.INK)
        s.text(v.P(*at), text, 1.8, anchor=anchor, middle=True, knockout=True)

    hy = L.handrail_y() * 1000
    seat = D.SEAT_LOCKERS
    note((370, 200), (1150, 470), "SLIDING HATCH")
    note((D.VENT_BS[1] + 60, 50), (4300, 330), f"MUSHROOM VENT ({len(D.VENT_BS)})")
    note((D.CHIMNEY[0] + 70, D.CHIMNEY[1] - 30), (9050, 400), "CHIMNEY (REMOVABLE)")
    note(None, (9800, D.PLANK_Y), "POLE AND PLANK", "middle")
    note((5600, hy - 10), (5900, 560), "HANDRAIL")
    for sgn in (1, -1):
        y = sgn * (D.CABIN_BASE_HALF - D.T_CABIN - seat[2] / 2)
        note(None, ((seat[0] + seat[1]) / 2, y), "SEAT LOCKER", "middle")
    note(None, (-450, 260), "ENGINE BOARDS", "middle")
    note(
        ((D.WEED_HATCH[0] + D.WEED_HATCH[1]) / 2, 120),
        (-2350, 850),
        "WEED HATCH",
        "end",
    )
    note(None, (D.DIESEL_FILLER[0] + 70, D.DIESEL_FILLER[1]), "DIESEL FILLER")
    note(None, (D.WATER_FILLER[0] - 70, D.WATER_FILLER[1]), "WATER FILLER", "end")
    note(None, (13300, 300), "WELL DECK", "middle")
    note((14560, -280), (15000, -980), "GAS LOCKER LID")


def section_c(sheet: S.Sheet) -> tuple[S.View, list]:
    """Midship section C-C at 1:20 through the galley, looking forward (port on the left)."""
    v = S.View(sheet, "section_C", SEC_X, SEC_BASE_Y, SEC_SCALE, -1, 1)
    bc = SECTION_C_B / 1000
    dims = []
    with sheet.group("section_C"):
        half = [(y * 1000, z * 1000) for y, z in L.hull_half_section(bc)]
        inner = offset_polyline(
            half, [D.T_BASE, D.T_BASE, D.T_HULL, D.T_HULL, D.T_HULL, D.T_HULL]
        )
        for sgn in (1, -1):
            ring = [(sgn * y, z) for y, z in half] + [
                (sgn * y, z) for y, z in reversed(inner)
            ]
            v.fill(ring)
            # Rubbing strakes: half-round 50 x 13.
            for zs in D.Z_STRAKES:
                pts = [(sgn * D.HALF_BEAM, zs - 25)]
                for k in range(9):
                    a = math.pi * (k / 8 - 0.5)
                    pts.append(
                        (
                            sgn * (D.HALF_BEAM + D.STRAKE_PROUD * math.cos(a)),
                            zs + 25 * math.sin(a),
                        )
                    )
                pts.append((sgn * D.HALF_BEAM, zs + 25))
                v.fill(pts)
            # Insulation and linings, hull side then cabin side.
            plate_in = D.HALF_BEAM - D.T_HULL
            lin_out = D.LINING_IN + D.PLY
            zg = D.Z_GUNWALE - D.T_HULL
            v.hatch(
                [
                    (sgn * plate_in, D.FLOOR_Z),
                    (sgn * lin_out, D.FLOOR_Z),
                    (sgn * lin_out, zg),
                    (sgn * plate_in, zg),
                ],
                "insulation",
            )
            v.rect(sgn * D.LINING_IN, D.FLOOR_Z, sgn * lin_out, zg, "thin", WHITE)
            # Gunwale capping on the ledge.
            cab_in_g = L.lining_in(D.Z_GUNWALE / 1000) * 1000
            v.rect(sgn * cab_in_g, zg, sgn * (D.LINING_IN + 5), zg + 18, "thin", WHITE)
            # Cabin side plate, broken by the window, with insulation and lining behind it.
            sill = D.WINDOW_SILL
            head = sill + D.WINDOW_H
            for z0, z1 in ((D.Z_GUNWALE, sill), (head, D.Z_SIDE_TOP)):
                outer = [(sgn * mm(L.cabin_side_y, z / 1000), z) for z in (z0, z1)]
                inner_c = [
                    (sgn * (mm(L.cabin_side_y, z / 1000) - D.T_CABIN), z)
                    for z in (z1, z0)
                ]
                v.fill(outer + inner_c)
                lo = [
                    (sgn * (mm(L.cabin_side_y, z / 1000) - D.T_CABIN), z)
                    for z in (z0, z1)
                ]
                li = [(sgn * (mm(L.lining_in, z / 1000) + D.PLY), z) for z in (z1, z0)]
                v.hatch(lo + li, "insulation")
                v.poly([(sgn * mm(L.lining_in, z / 1000), z) for z in (z0, z1)], "thin")
                v.poly(li, "thin")
            # Window: frame flanges, glass and the ply reveal.
            y_out = mm(L.cabin_side_y, sill / 1000)
            for z in (sill, head):
                v.rect(
                    sgn * (y_out + 6),
                    z - (D.FRAME_FLANGE if z == sill else 0),
                    sgn * (y_out - 4),
                    z + (0 if z == sill else D.FRAME_FLANGE),
                    "thin",
                    S.INK,
                )
                v.rect(
                    sgn * (y_out - 4),
                    z - 12 if z == head else z,
                    sgn * mm(L.lining_in, z / 1000),
                    z + 12 if z == sill else z,
                    "thin",
                    WHITE,
                )
            v.line(
                (sgn * (y_out - 2), sill),
                (sgn * (mm(L.cabin_side_y, head / 1000) - 2), head),
                "medium",
            )
            # Handrail on the roof.
            hy, hz = L.handrail_y() * 1000, L.handrail_top() * 1000
            v.circle(
                (sgn * hy, hz - D.HANDRAIL_D / 2), D.HANDRAIL_D / 2, "medium", WHITE
            )
        # Roof plate, insulation and ceiling.
        roof = [(y * 1000, z * 1000) for y, z in L.roof_outer()]
        roof_in = offset_polyline(roof, [D.T_ROOF] * (len(roof) - 1))
        v.fill(roof + roof_in[::-1])
        ceil_out = offset_polyline(
            roof, [D.T_ROOF + D.INSULATION_ROOF] * (len(roof) - 1)
        )
        ceil_in = offset_polyline(
            roof, [D.T_ROOF + D.INSULATION_ROOF + D.PLY] * (len(roof) - 1)
        )
        lim = L.lining_in(1.95) * 1000
        clip = [(y, z) for y, z in ceil_out if abs(y) <= lim]
        clip_in = [(y, z) for y, z in ceil_in if abs(y) <= lim]
        v.hatch(
            [p for p in roof_in if abs(p[0]) <= lim + 60] + clip[::-1], "insulation"
        )
        v.poly(clip, "thin")
        v.poly(clip_in, "thin")
        # Floor build-up across the hull.
        w = D.HALF_BEAM - D.T_HULL - 15
        for z0, z1, layer in D.FLOOR_BUILDUP:
            if layer.startswith("concrete"):
                v.hatch([(-w, z0), (w, z0), (w, z1), (-w, z1)], "concrete")
                v.rect(-w, z0, w, z1, "thin")
            elif layer == "packers" or layer.startswith("50"):
                for yb in (-850, -450, 0, 450, 850):
                    v.rect(yb - 25, z0, yb + 25, z1, "thin", WHITE)
                    if layer.startswith("50"):
                        v.line((yb - 25, z0), (yb + 25, z1), "fine")
                        v.line((yb - 25, z1), (yb + 25, z0), "fine")
            else:
                v.rect(-D.LINING_IN, z0, D.LINING_IN, z1, "thin", WHITE)
        # Galley beyond the cut: base runs, worktops and the forward port wall unit.
        for name in ("GalleyPort", "GalleyStarboard"):
            f = D.FIXTURE[name]
            _, _, y0, y1 = f.footprint()
            carcass = D.WORKTOP_Z - 38
            v.rect(y0, f.z0, y1, carcass, "thin", WHITE)
            v.line((y0, f.z0 + 100), (y1, f.z0 + 100), "thin")
            v.line((y0, f.z0), (y1, carcass), "fine")
            v.line((y0, carcass), (y1, f.z0), "fine")
            # The worktop overhangs the carcass front by 15 towards the aisle.
            v.rect(
                y0 - (15 if y0 > 0 else 0),
                carcass,
                y1 + (15 if y1 < 0 else 0),
                D.WORKTOP_Z,
                "medium",
                WHITE,
            )
        f = D.FIXTURE["WallUnitsPortFwd"]
        _, _, y0, y1 = f.footprint()
        back = [(mm(L.lining_in, z / 1000), z) for z in (f.z0, f.z1)]
        v.poly([(y0, f.z0), *back, (y0, f.z1)], "thin", closed=True, fill=WHITE)
        # Centre line and waterline.
        v.line((0, -150), (0, D.Z_ROOF_CROWN + 250), "centre")
        wl = mm(L.wl_z, SECTION_C_B)
        v.line((-1150, wl), (1150, wl), "centre")
        x, y = v.P(0, wl)
        sheet.polyline(
            [(x - 1.2, y - 1.8), (x + 1.2, y - 1.8), (x, y)], "thin", closed=True
        )
        sheet.text((x - 1.8, y - 0.7), "DWL", 1.8, anchor="end", knockout=True)
        section_notes(v)
    with sheet.group("section_C_dims"):
        dims += section_dims(v)
    return v, dims


def section_notes(v: S.View) -> None:
    """Scantling and build-up notes down the starboard (right-hand) side, top to bottom so
    no two leaders cross."""
    s = v.sheet
    x_text = SEC_X + (D.HALF_BEAM + D.STRAKE_PROUD) / SEC_SCALE + 6
    hy = L.handrail_y() * 1000
    rows = [
        ((-hy, L.handrail_top() * 1000 - 13), "HANDRAIL 26 DIA"),
        ((-700, mm(L.roof_z, 0.7) - 2), "ROOF 4, CAMBER 75"),
        ((-700, mm(L.ceiling_z, 0.7) + 25), "T&G CEILING, 45 FOAM"),
        ((-(mm(L.cabin_side_y, 1.85) - 2), 1850), "CABIN SIDE 5, TUMBLEHOME 50"),
        ((-(mm(L.cabin_side_y, 1.45) - 2), 1450), "HOPPER WINDOW 533"),
        ((-(mm(L.lining_in, 1.3) + 4), 1300), "9 ASH PLY ON BATTENS"),
        ((-(D.HALF_BEAM - 50), D.Z_GUNWALE - 3), "GUNWALE 6, 100 WIDE"),
        ((-(D.HALF_BEAM - 25), 920), "38 SPRAY FOAM"),
        ((-(D.HALF_BEAM + 7), D.Z_STRAKES[1]), "STRAKES 50 x 13 (3)"),
        ((-(D.HALF_BEAM - 3), 420), "HULL SIDE 6"),
        ((-900, 131), "18 PLY, 50 x 50 BEARERS"),
        ((-930, 35), "BALLAST SLABS"),
        ((-980, 5), "BASEPLATE 10"),
    ]
    y_text = v.y(D.Z_ROOF_CROWN + 120)
    step = (v.y(-40) - y_text) / (len(rows) - 1)
    for k, ((y, z), text) in enumerate(rows):
        tx, ty = v.P(y, z)
        ly = y_text + k * step
        s.line((tx, ty), (x_text - 1.0, ly), "thin")
        s.circle((tx, ty), 0.35, None, fill=S.INK)
        s.text((x_text, ly + 0.6), text, 1.8)


def section_dims(v: S.View) -> list:
    dims = []
    half = D.HALF_BEAM + D.STRAKE_PROUD
    y0 = v.y(0) + 4.5
    dims.append(
        S.dim_h(
            v,
            half,
            -half,
            y0 + 5.5,
            from_v=(D.Z_STRAKES[0], D.Z_STRAKES[0]),
            h=1.8,
            text=f"{D.BEAM_OVER_STRAKES} OVER STRAKES",
        )
    )
    dims.append(
        S.dim_h(
            v,
            D.HALF_BEAM,
            -D.HALF_BEAM,
            y0,
            from_v=(80, 80),
            h=1.8,
            text=f"{2 * D.HALF_BEAM} OVER PLATING",
        )
    )
    top = v.y(D.Z_ROOF_CROWN) - 9
    dims.append(
        S.dim_h(
            v,
            D.CABIN_TOP_HALF,
            -D.CABIN_TOP_HALF,
            top,
            from_v=(D.Z_SIDE_TOP, D.Z_SIDE_TOP),
            h=1.8,
        )
    )
    gy = v.y(D.Z_GUNWALE + 140)
    dims.append(
        S.dim_h(
            v,
            -D.CABIN_BASE_HALF,
            -D.HALF_BEAM,
            gy,
            from_v=(D.Z_GUNWALE, D.Z_GUNWALE),
            h=1.8,
        )
    )
    # Heights up the port side.
    x_line = v.x(half) - 9
    levels = [
        D.FLOOR_Z,
        round(mm(L.wl_z, SECTION_C_B)),
        D.Z_GUNWALE,
        D.WINDOW_SILL,
        D.WINDOW_SILL + D.WINDOW_H,
        D.Z_SIDE_TOP,
        D.Z_ROOF_CROWN,
    ]
    dims += S.running_v(
        v,
        0,
        levels,
        x_line,
        from_b=[
            D.HALF_BEAM - 200,
            D.LINING_IN,
            D.HALF_BEAM + 20,
            D.HALF_BEAM,
            D.CABIN_BASE_HALF,
            D.CABIN_BASE_HALF,
            D.CABIN_TOP_HALF,
            0,
        ],
    )
    # Inside: headroom, worktop height, clear widths.
    dims.append(
        S.dim_v(
            v,
            D.FLOOR_Z,
            D.CEILING_CROWN,
            v.x(-120),
            h=1.8,
            text=f"{D.HEADROOM} HEADROOM",
            text_left=False,
        )
    )
    port = D.FIXTURE["GalleyPort"].footprint()
    dims.append(
        S.dim_v(v, D.FLOOR_Z, D.WORKTOP_Z, v.x(port[2] - 70), h=1.8, text_left=False)
    )
    stbd = D.FIXTURE["GalleyStarboard"].footprint()
    dims.append(S.dim_h(v, port[2], stbd[3], v.y(D.WORKTOP_Z + 260), h=1.8))
    dims.append(S.dim_h(v, D.LINING_IN, -D.LINING_IN, v.y(D.FLOOR_Z + 180), h=1.8))
    return dims


def tables(sheet: S.Sheet) -> dict:
    """Particulars, tankage, weights, ventilation and the openings schedule."""
    weights = L.weight_estimate_kg()
    water, diesel = L.tank_volume_l(D.WATER_TANK), L.tank_volume_l(D.FUEL_TANK)
    disp = L.displacement_kg()
    out = {}
    with sheet.group("tables"):
        x1, x2, x3 = 190.0, 265.0, 338.0
        y = 137.0
        particulars = [
            ("Length over fenders", f"{D.LENGTH_OVER_FENDERS:,} (57'-0\")"),
            ("Length of hull", f"{D.HULL_LENGTH:,}"),
            ("Beam over rubbing strakes", f"{D.BEAM_OVER_STRAKES:,} (6'-10\")"),
            ("Beam over plating", f"{2 * D.HALF_BEAM:,}"),
            ("Cabin length", f"{D.CABIN_LEN:,}"),
            ("Draught aft / forward", f"{D.DRAUGHT_AFT} / {D.DRAUGHT_FWD}"),
            ("Air draught (no chimney, vents)", f"{round(L.air_draught() * 1000):,}"),
            ("Headroom on centreline", f"{D.HEADROOM:,} (6'-3\")"),
            ("Displacement, design", f"{disp / 1000:.1f} t"),
            ("Stern", "semi-traditional"),
            ("Engine", "approx. 43 hp diesel"),
            ("Propeller", f"{D.PROP_D} dia, {D.PROP_BLADES} blades"),
            (
                "Plate base/hull/cabin/roof",
                f"{D.T_BASE}/{D.T_HULL}/{D.T_CABIN}/{D.T_ROOF}",
            ),
            ("Berths", "2 + 2 occasional"),
            ("Design category", "RCR D (inland)"),
        ]
        out["particulars"] = S.table(
            sheet,
            (x1, y),
            "PRINCIPAL PARTICULARS",
            particulars,
            (44, 28),
            align=("start", "end"),
        )
        tank_rows = [
            ("Fresh water, integral, bow", f"{round(water):,} L"),
            ("Diesel, integral, counter", f"{round(diesel):,} L"),
            ("LPG, foredeck locker", f"{D.GAS_CYLINDERS} x 13 kg"),
            ("WC", "cassette"),
        ]
        out["tankage"] = S.table(
            sheet,
            (x1, out["particulars"][3] + 2.5),
            "TANKAGE",
            tank_rows,
            (44, 28),
            align=("start", "end"),
        )
        machinery = ("Engine and gearbox", "Stove", "Cooker, fridge, calorifier")
        batteries = next(k for k in weights if k.startswith("Batteries"))
        weight_rows = [
            ("Steelwork (plate + framing)", weights["Steelwork"]),
            ("Fit-out", weights["Fit-out"]),
            ("Machinery and appliances", sum(weights[k] for k in machinery)),
            (batteries, weights[batteries]),
            ("LPG, 2 x 13 kg full", weights["LPG, 2 x 13 kg full"]),
            ("Crew (4) and stores", weights["Crew (4)"] + weights["Stores and gear"]),
            ("Fresh water, full", weights["Fresh water"]),
            ("Diesel, full", weights["Diesel"]),
            ("Ballast to the DWL", weights["Ballast"]),
            ("Displacement at DWL", disp),
        ]
        out["weights"] = S.table(
            sheet,
            (x2, y),
            "WEIGHT ESTIMATE (kg)",
            [(name, f"{round(kg):,}") for name, kg in weight_rows],
            (50, 20),
            align=("start", "end"),
        )
        S.notes(
            sheet, (x2, out["weights"][3] + 5.0), "NOTES", SHEET2_NOTES, 70.0, gap=2.6
        )
        req = D.ventilation_required()
        high = [v for v in D.VENTS if v.level == "high"]
        low = [v for v in D.VENTS if v.level == "low"]
        vent_rows = [
            ("V = 2200U + 650P + 440F + 550H", "mm2"),
            ("U unflued (LPG cooker)", f"{D.UNFLUED_KW} kW"),
            ("P persons", f"{D.PERSONS}"),
            ("F flued, not room-sealed", f"{D.FLUED_KW:g} kW"),
            ("H solid fuel stove", f"{D.STOVE_KW} kW"),
            ("V required", f"{round(req):,}"),
            ("Required high / low, each", f"{math.ceil(req / 2):,}"),
            (
                "High: " + ", ".join(f"{v.count} x {v.free_area:,}" for v in high),
                f"{D.ventilation_provided('high'):,}",
            ),
            (
                "Low: " + ", ".join(f"{v.count} x {v.free_area:,}" for v in low),
                f"{D.ventilation_provided('low'):,}",
            ),
        ]
        out["ventilation"] = S.table(
            sheet,
            (x3, y),
            "VENTILATION (BSS)",
            vent_rows,
            (52, 20),
            align=("start", "end"),
        )
        rows = []
        for op in sorted(D.OPENINGS, key=lambda o: (-o.side, o.b)):
            size = f"{op.w} dia" if op.kind == "porthole" else f"{op.w} x {op.h}"
            rows.append(
                (
                    OPENING_IDS[op.name],
                    "P" if op.side > 0 else "S",
                    op.kind,
                    size,
                    f"{op.b:,}",
                    f"{op.zc - op.h // 2:,}",
                )
            )
        hatch = D.SIDE_HATCH
        rows.append(
            (
                "H1",
                "P" if hatch.side > 0 else "S",
                "side hatch",
                f"{hatch.clear} clear",
                f"{hatch.b:,}",
                f"{hatch.z0:,}",
            )
        )
        out["openings"] = S.table(
            sheet,
            (x3, out["ventilation"][3] + 2.5),
            "OPENINGS SCHEDULE",
            rows,
            (8, 6, 15, 17, 13, 13),
            head=("ID", "SIDE", "TYPE", "SIZE", "CENTRE b", "SILL z"),
            align=("start", "start", "start", "start", "end", "end"),
        )
    return out


SHEET2_NOTES = (
    (
        "Waterline trimmed by the stern: 620 at the stern post, 480 at the stem. Draughts"
        " and the air draught are measured from it."
    ),
    (
        "Weights are estimates for trim and ballasting; the final ballast is set at launch"
        " to bring the boat to the DWL."
    ),
    (
        "CRT gauge: 17.37 m over fenders reaches the whole connected narrow network; 2.08 m"
        " beam; air draught under 1.75 m; Standedge cabin profile checked."
    ),
    (
        "BSS: fixed ventilation to the BSS formula; LPG locker drained overboard above the"
        " waterline; CO alarms fitted; stove 4.5 kW to BS 8511."
    ),
)


def sheet2() -> tuple[S.Sheet, dict]:
    sheet = S.Sheet("NB57-EX-002 Exterior, construction and particulars")
    S.frame(sheet)
    prof, dims_p = profile_view(sheet)
    roof = roof_plan(sheet)
    sec, dims_c = section_c(sheet)
    tabs = tables(sheet)
    with sheet.group("labels"):
        view_label(
            sheet, (343.0, PROFILE_BASE_Y + 7.0), "PROFILE", ["FROM STARBOARD   1:50"]
        )
        view_label(sheet, (343.0, ROOF_Y + 27.0), "ROOF", ["ROOF AND DECK PLAN   1:50"])
        view_label(
            sheet,
            (24.0, SEC_BASE_Y + 21.0),
            "C-C",
            ["MIDSHIP SECTION AT b 7100, LOOKING FORWARD   1:20"],
        )
    S.title_block(
        sheet,
        S.TitleBlock(
            OWNER,
            "NB57-EX-002",
            REVISIONS[-1][0],
            DATE,
            "2/2",
            f"{D.TITLE} '{D.NAME.upper()}'",
            "PROFILE, ROOF AND DECK PLAN, MIDSHIP SECTION AND PARTICULARS",
            "General arrangement",
            STATUS,
            CREATOR,
            "-",
            "1:50, 1:20",
        ),
        REVISIONS,
    )
    views = {"profile": prof, "roof": roof}
    return sheet, {
        "views": views,
        "dims": {"profile": dims_p, "section": dims_c},
        "tables": tabs,
        "section": sec,
    }


# =========================================================================== output


def design_sha256() -> str:
    """Hash of the inputs the drawing depends on (design data and line functions)."""
    h = hashlib.sha256()
    for name in ("narrowboat_design.py", "narrowboat_lines.py"):
        h.update((Path(__file__).parent / name).read_bytes())
    return h.hexdigest()


# The 1:50 views a model render can be laid over, and the sheet group each is drawn in.
OVERLAY_GROUPS = {
    "plan": "plan",
    "port": "section_A",
    "starboard": "section_B",
    "profile": "profile",
    "roof": "roof",
}


def calibration(number: str, sheet: S.Sheet, views: dict) -> dict:
    """How each 1:50 view maps model millimetres onto pixels of the sheet PNG (and of its
    lines-only overlay, which shares the page geometry): pixel = origin + sign * mm /
    mm_per_px, with b along x and the view's vertical axis (z, or y in plan) up the page
    when v_sign is +1. box_px is the view's extent for cropping."""
    px = DPI / 25.4  # pixels per sheet millimetre
    out = {}
    for name, v in views.items():
        x0, y0, x1, y1 = sheet.boxes[OVERLAY_GROUPS[name]]
        out[name] = {
            "png": f"{number}.png",
            "lines_png": f"{number}_{name}_lines.png",
            "mm_per_px": SCALE / px,
            "b0_px": round(v.ox * px, 3),
            "axis_px": round(v.oy * px, 3),
            "b_sign": v.b_sign,
            "v_sign": v.v_sign,
            "box_px": [
                math.floor(x0 * px),
                math.floor(y0 * px),
                math.ceil(x1 * px),
                math.ceil(y1 * px),
            ],
        }
    return out


def _png(svg: Path, png: Path) -> None:
    subprocess.run(
        ["rsvg-convert", "-d", str(DPI), "-p", str(DPI), str(svg), "-o", str(png)],
        check=True,
    )


def build_sheets() -> list[tuple[str, S.Sheet, dict]]:
    s1, meta1 = sheet1()
    s2, meta2 = sheet2()
    return [("NB57-GA-001", s1, meta1), ("NB57-EX-002", s2, meta2)]


def write(out: Path) -> dict:
    """Write the SVGs, PNGs, lines-only overlays, the two-page PDF and the manifest."""
    out.mkdir(parents=True, exist_ok=True)
    sheets = build_sheets()
    files, svgs, cal = {}, [], {}
    for number, sheet, meta in sheets:
        svg = out / f"{number}.svg"
        svg.write_text(sheet.svg())
        svgs.append(svg)
        png = out / f"{number}.png"
        _png(svg, png)
        files[svg.name], files[png.name] = svg, png
        cal[number] = calibration(number, sheet, meta["views"])
        for name, entry in cal[number].items():
            lines_svg = out / f"{number}_{name}_lines.svg"
            lines_svg.write_text(sheet.svg(only={OVERLAY_GROUPS[name]}, text=False))
            lines_png = out / entry["lines_png"]
            _png(lines_svg, lines_png)
            lines_svg.unlink()
            files[lines_png.name] = lines_png
    pdf = out / "NB57_drawings.pdf"
    subprocess.run(
        ["rsvg-convert", "-f", "pdf", *map(str, svgs), "-o", str(pdf)], check=True
    )
    files[pdf.name] = pdf
    manifest = {
        "design": f"{D.DESIGN_ID} {D.NAME}",
        "design_sha256": design_sha256(),
        "dpi": DPI,
        "scale": f"1:{SCALE}",
        "sheets": [number for number, _, _ in sheets],
        "calibration": cal,
        "files": {
            name: hashlib.sha256(p.read_bytes()).hexdigest()
            for name, p in sorted(files.items())
        },
    }
    (out / "drawings_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build(target: Path = OUTPUT_DIR) -> dict:
    from procgen_bpy import fresh_staging, publish

    staging = fresh_staging(target)
    manifest = write(staging)
    publish(staging, target)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR)
    ap.add_argument(
        "--scratch", type=Path, help="write here directly, without publishing"
    )
    args = ap.parse_args()
    if args.scratch:
        if args.scratch.exists():
            shutil.rmtree(args.scratch)
        manifest = write(args.scratch)
    else:
        manifest = build(args.out)
    print(json.dumps({k: manifest[k] for k in ("design", "sheets")}))


if __name__ == "__main__":
    main()
