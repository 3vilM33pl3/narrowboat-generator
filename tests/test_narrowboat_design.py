# SPDX-License-Identifier: MIT
"""Tests for the NB57 design data: gauge, hull lines, layout, clearances, safety and
hydrostatics. These are the checks a builder would make on the general arrangement."""

from __future__ import annotations

import math
import sys
from itertools import combinations, pairwise
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SCRIPTS))

import narrowboat_design as D
import narrowboat_lines as L

EPS = 1e-6


def overlap(a0, a1, b0, b1) -> float:
    return min(a1, b1) - max(a0, b0)


# --------------------------------------------------------------------------- gauge


def test_length_and_beam_fit_the_whole_network():
    assert D.LENGTH_OVER_FENDERS == 17370 <= D.CRT_MAX_LENGTH
    assert D.HULL_LENGTH == D.STEM - D.STERN_TIP == 17150
    assert -D.STERN_TIP + D.CABIN_LEN + (D.STEM - D.CABIN_LEN) == D.HULL_LENGTH
    assert D.BEAM_OVER_STRAKES == 2080 <= D.CRT_MAX_BEAM


def test_air_draught_and_standedge_profile():
    assert L.air_draught() <= D.AIR_DRAUGHT_MAX / 1000
    for height, margin in L.standedge_margins():
        assert margin > 0, f"cabin too wide {height} mm above water"


# --------------------------------------------------------------------------- hull lines


@pytest.mark.parametrize(
    ("f", "at"),
    [
        (L.plan_half, L.BOW_START),
        (L.plan_half, L.COUNTER_START),
        (L.gunwale_z, L.SHEER_START),
        (L.bottom_z, L.FOREFOOT_START),
        (L.swim_half, L.SWIM_START),
        (L.swim_full_z, L.SWIM_START),
    ],
)
def test_hull_lines_join_smoothly(f, at):
    assert f(at - 1e-4) == pytest.approx(f(at + 1e-4), abs=1e-4)


def test_hull_lines_close_where_they_should():
    assert L.plan_half(L.STEM) < 0.06  # the plan closes on the stem
    assert L.plan_half(L.STERN_TIP) < 0.01  # the counter closes at the tip
    assert L.gunwale_z(L.STEM) == pytest.approx(D.SHEER[-1][1] / 1000)
    assert (
        L.bottom_z(L.STEM) > L.gunwale_z(L.STEM) - 0.15
    )  # the stem rises to the sheer
    assert L.swim_half(L.STERN_POST) < 0.06  # the swim closes on the stern post
    bs = [L.BOW_START + 0.01 * k for k in range(round((L.STEM - L.BOW_START) / 0.01))]
    assert all(L.plan_half(a) >= L.plan_half(b) for a, b in pairwise(bs))
    assert all(L.gunwale_z(a) <= L.gunwale_z(b) + EPS for a, b in pairwise(bs))


def test_sections_are_monotone_up_the_side():
    for b in [L.STERN_POST + 0.01 * k for k in range(0, 1630, 7)]:
        sec = L.hull_half_section(b)[:6]
        assert all(q[1] >= p[1] - EPS for p, q in pairwise(sec[1:])), b
        assert all(q[0] >= p[0] - EPS for p, q in pairwise(sec)), b


# --------------------------------------------------------------------------- layout


def test_zones_tile_the_cabin_in_traditional_order():
    names = [z.name for z in D.ZONES]
    assert names == ["Lobby", "Bedroom", "Bathroom", "Galley", "Saloon"]
    assert D.ZONES[0].b0 == D.LINING_AFT and D.ZONES[-1].b1 == D.LINING_FWD
    for a, b in zip(D.ZONES, D.ZONES[1:]):
        assert a.b1 == b.b0
    bounds = {z.b0 for z in D.ZONES} | {z.b1 for z in D.ZONES}
    assert all(p.b in bounds for p in D.PARTITIONS)


def _partition_faces(zone):
    lo, hi = zone.b0, zone.b1
    for p in D.PARTITIONS:
        if p.b == zone.b0:
            lo = p.b + D.PARTITION_T / 2
        if p.b == zone.b1:
            hi = p.b - D.PARTITION_T / 2
    return lo, hi


@pytest.mark.parametrize("f", D.FIXTURES, ids=lambda f: f.name)
def test_fixture_sits_inside_its_zone(f):
    zone = D.ZONE[f.zone]
    lo, hi = _partition_faces(zone)
    b0, b1, y0, y1 = f.footprint()
    assert lo <= b0 < b1 <= hi
    assert -D.LINING_IN <= y0 < y1 <= D.LINING_IN
    if f.kind not in (
        "steps",
        "towel_rail",
        "heat_shield",
        "wall_units",
        "sink",
        "stove",
    ):
        assert f.z0 >= zone.floor_z
    y_out = max(abs(y0), abs(y1)) / 1000
    assert f.z1 / 1000 <= L.ceiling_z(min(y_out, L.lining_in(f.z1 / 1000))) + EPS


def test_major_footprints_do_not_overlap():
    major = [f for f in D.FIXTURES if f.kind in D.CARCASS_KINDS]
    for a, b in combinations(major, 2):
        fa, fb = a.footprint(), b.footprint()
        if (
            overlap(fa[0], fa[1], fb[0], fb[1]) > 0
            and overlap(fa[2], fa[3], fb[2], fb[3]) > 0
        ):
            pytest.fail(f"{a.name} overlaps {b.name}")


def test_appliances_sit_inside_their_runs():
    for f in D.FIXTURES:
        if f.host:
            hb0, hb1, hy0, hy1 = D.FIXTURE[f.host].footprint()
            b0, b1, y0, y1 = f.footprint()
            assert hb0 <= b0 < b1 <= hb1 and hy0 <= y0 < y1 <= hy1, f.name


def _blocked(b):
    """y intervals blocked at b by furniture; steps and the low hearth are walked over."""
    out = []
    for f in D.FIXTURES:
        if f.kind in D.CARCASS_KINDS | {"stove"} and f.kind not in ("hearth", "steps"):
            b0, b1, y0, y1 = f.footprint()
            if b0 <= b <= b1:
                out.append((y0, y1))
    return sorted(out)


def _widest_gap(b):
    for p in D.PARTITIONS:
        if abs(b - p.b) <= D.PARTITION_T / 2:
            return p.door_y[1] - p.door_y[0]
    cur, best = -D.LINING_IN, 0
    for y0, y1 in _blocked(b):
        best = max(best, y0 - cur)
        cur = max(cur, y1)
    return max(best, D.LINING_IN - cur)


def test_a_walkway_runs_the_length_of_the_cabin():
    end = D.FIXTURE["BowSteps"].b0
    for b in range(D.ZONE["Bedroom"].b0, end, 10):
        assert _widest_gap(b) >= 550, f"walkway pinched to {_widest_gap(b)} at b {b}"


# Pieces whose backs step inboard with the cabin side above the gunwale; everything else
# against a lining has to stay under the gunwale soffit, where the hull lining is.
FOLLOWS_TUMBLEHOME = {
    "cupboard",
    "wardrobe",
    "tall_unit",
    "wall_units",
    "shower",
    "heat_shield",
}


def test_furniture_against_the_hull_stays_under_the_gunwale():
    soffit = D.Z_GUNWALE - D.PLY
    for f in D.FIXTURES:
        if f.side and f.kind not in FOLLOWS_TUMBLEHOME:
            assert f.z1 <= soffit, f.name


def test_galley_aisle_corridor_and_doorways():
    port = D.FIXTURE["GalleyPort"].footprint()
    for name in ("GalleyStarboard", "Larder"):
        assert port[2] - D.FIXTURE[name].footprint()[3] >= 700
    bed = D.FIXTURE["Bed"].footprint()
    assert D.LINING_IN - bed[3] >= 550  # bedroom corridor
    for p in D.PARTITIONS:
        assert p.door_y[1] - p.door_y[0] >= 550
        assert -D.LINING_IN <= p.door_y[0] < p.door_y[1] <= D.LINING_IN


def test_berth_and_headroom():
    bed = D.FIXTURE["Bed"]
    length, width = bed.opt("mattress")
    assert (length, width) == (1905, 1220)
    assert bed.b1 - bed.b0 >= length and bed.depth >= width
    assert D.HEADROOM >= 1900
    assert L.ceiling_z(0.5) - D.FLOOR_Z / 1000 >= 1.85


def test_swim_clears_every_floor_and_lining():
    lining_outer = (D.LINING_IN + D.PLY) / 1000
    for zone in D.ZONES:
        for b in range(zone.b0, zone.b1 + 1, 50):
            for zk in range(zone.floor_z, D.Z_GUNWALE, 20):
                gap = L.inner_half_at(b / 1000, zk / 1000) - lining_outer
                assert gap >= 0.005, f"{zone.name} b {b} z {zk}: {gap:.3f}"


# --------------------------------------------------------------------------- openings


def test_openings_lie_on_the_cabin_side_inside_the_panel():
    inner = D.PANEL_Z[0] + 48, D.PANEL_Z[1] - 48  # inside the red coachline
    for op in D.OPENINGS:
        b0, b1 = op.span
        assert D.PANEL_B[0] + 48 <= b0 and b1 <= D.PANEL_B[1] - 48, op.name
        rim = D.RIM_W if op.kind == "porthole" else D.FRAME_FLANGE
        assert inner[0] <= op.zc - op.h / 2 - rim and op.zc + op.h / 2 + rim <= inner[1]
        assert op.zc - op.h / 2 - rim > D.Z_GUNWALE


def test_openings_are_spaced_and_clear_the_partitions():
    for side in (1, -1):
        spans = sorted(op.span for op in D.OPENINGS if op.side == side)
        if D.SIDE_HATCH.side == side:
            spans = sorted([*spans, D.SIDE_HATCH.span])
        for (_, a1), (b0, _) in pairwise(spans):
            assert b0 - a1 >= 100
        for p in D.PARTITIONS:
            for b0, b1 in spans:
                assert b1 < p.b - 100 or b0 > p.b + 100


def test_no_window_is_hidden_behind_tall_furniture():
    for op in D.OPENINGS:
        rim = D.RIM_W if op.kind == "porthole" else D.FRAME_FLANGE
        bottom = op.zc - op.h / 2 - rim
        for f in D.FIXTURES:
            b0, b1, y0, y1 = f.footprint()
            against = (y1 if op.side > 0 else -y0) >= D.LINING_IN - 100
            if against and overlap(b0, b1, *op.span) > 0:
                assert f.z1 < bottom, f"{f.name} covers {op.name}"


def test_sprays_have_room_between_the_openings():
    for _, side, hint in D.SPRAYS:
        spans = sorted(op.span for op in D.OPENINGS if op.side == side)
        if D.SIDE_HATCH.side == side:
            spans = sorted([*spans, D.SIDE_HATCH.span])
        assert not any(b0 <= hint <= b1 for b0, b1 in spans)
        before = max([b1 for b0, b1 in spans if b1 < hint], default=D.PANEL_B[0])
        after = min([b0 for b0, b1 in spans if b0 > hint], default=D.PANEL_B[1])
        assert after - before >= 500


def test_side_hatch_is_over_the_dinette():
    dinette = D.FIXTURE["Dinette"]
    t0, t1 = dinette.opt("table")
    h0, h1 = D.SIDE_HATCH.span
    assert dinette.side == D.SIDE_HATCH.side and t0 <= h0 < h1 <= t1


# --------------------------------------------------------------------------- stove


def test_stove_hearth_and_heat_shields():
    stove, hearth = D.FIXTURE["Stove"].footprint(), D.FIXTURE["Hearth"].footprint()
    assert stove[2] - hearth[2] >= 225  # in front (towards the centreline)
    assert stove[0] - hearth[0] >= 150 and hearth[1] - stove[1] >= 150
    wall = D.FIXTURE["HeatShieldWall"]
    assert wall.footprint()[2] - stove[3] >= 45
    assert wall.z1 >= D.FIXTURE["Stove"].z1 + 200
    assert D.STOVE_KW <= 4.5  # BSS guidance


def test_combustibles_are_far_from_the_stove_or_shielded():
    stove = D.FIXTURE["Stove"]
    s0, s1, sy0, sy1 = stove.footprint()
    shields = [f.footprint() for f in D.FIXTURES if f.kind == "heat_shield"]
    for f in D.FIXTURES:
        if f.kind in ("stove", "hearth", "heat_shield") or f.host:
            continue
        b0, b1, y0, y1 = f.footprint()
        gap = math.hypot(
            max(b0 - s1, s0 - b1, 0),
            max(y0 - sy1, sy0 - y1, 0),
            max(f.z0 - stove.z1, stove.z0 - f.z1, 0),
        )
        if gap < 600:
            touching = any(
                overlap(b0, b1, h[0], h[1]) >= 0 and overlap(y0, y1, h[2], h[3]) >= 0
                for h in shields
            )
            assert touching, f"{f.name} is {gap:.0f} mm from the stove, unshielded"


def test_chimney_stands_over_the_stove_clear_of_the_roof_gear():
    b, y = D.CHIMNEY
    s0, s1, sy0, sy1 = D.FIXTURE["Stove"].footprint()
    assert s0 < b < s1 and sy0 < y < sy1
    collar = D.COLLAR_D / 2
    assert (y + collar) / 1000 < L.handrail_y() - D.HANDRAIL_D / 2000
    assert all(abs(b - v) >= 500 for v in D.VENT_BS)
    plank = D.PLANK_Y - 130, D.PLANK_Y + 130
    assert y - collar > plank[1]


# --------------------------------------------------------------------------- machinery


def _inside_hull(box, clearance=0.01):
    b0, b1, y0, y1, z0, z1 = (v / 1000 for v in box)
    for b in (b0, b1):
        for z in (z0, z1):
            if max(abs(y0), abs(y1)) > L.inner_half_at(b, z) - clearance:
                return False
    return z1 < D.DECK_Z / 1000


@pytest.mark.parametrize("name", ["ENGINE", "GEARBOX", "BATTERIES"])
def test_machinery_fits_inside_the_hull(name):
    assert _inside_hull(getattr(D, name))


def test_propeller_rudder_and_skeg():
    r = D.PROP_D / 2
    assert D.PROP_Z - r >= 50 + 15  # above the skeg
    assert D.PROP_Z + r <= D.Z_COUNTER - 50  # tip clearance under the counter
    assert D.PROP_Z + r < L.wl_z(D.PROP_B / 1000) * 1000  # immersed
    assert D.RUDDER[1] < D.PROP_B - r / 2 < D.PROP_B < D.STERN_POST
    rb0, rb1, _, rz1 = D.RUDDER
    assert rb0 >= D.STERN_TIP and rz1 < D.Z_COUNTER
    assert 0.15 <= (rb1 - D.RUDDER_STOCK_B) / (rb1 - rb0) <= 0.35  # balanced blade
    assert D.SKEG_END < D.RUDDER_STOCK_B
    w0, w1, _ = D.WEED_HATCH
    assert w0 <= D.PROP_B <= w1
    assert D.FUEL_TANK[1] <= w0


def test_tanks():
    water, diesel = L.tank_volume_l(D.WATER_TANK), L.tank_volume_l(D.FUEL_TANK)
    assert 500 <= water <= 900
    assert 180 <= diesel <= 260
    assert D.WATER_TANK[3] < D.WELL_SOLE_Z
    assert D.Z_COUNTER < D.FUEL_TANK[2] < D.FUEL_TANK[3] < D.DECK_Z


def test_gas_locker():
    d, h = D.GAS_CYLINDER
    assert D.GAS_LOCKER[0] >= D.CABIN_LEN  # outside the accommodation
    assert D.LOCKER_DEPTH >= h + 60
    assert D.LOCKER_DRAIN_ID >= 19
    assert L.locker_drain_margin() >= 0.075  # outlet above the waterline
    sole, top = L.locker_sole_z(), L.locker_lid_z()
    for b, y in L.cylinder_positions():
        for bb in (b - d / 2000, b + d / 2000):
            assert m_ok(bb, abs(y) + d / 2000, sole, top)
        assert (
            D.GAS_LOCKER[0] / 1000 < b - d / 2000
            and b + d / 2000 < D.GAS_LOCKER[1] / 1000
        )
    assert len(L.cylinder_positions()) == D.GAS_CYLINDERS


def m_ok(b: float, y: float, z0: float, z1: float) -> bool:
    """A cylinder edge at (b, y) clears the plating from the sole up to the sheer (the lid's
    coaming stands above it)."""
    z1 = min(z1, L.gunwale_z(b) - 0.01)
    return all(L.inner_half_at(b, z) - 0.01 >= y for z in (z0, (z0 + z1) / 2, z1))


# --------------------------------------------------------------------------- systems


def test_displacement_gives_a_sensible_ballast():
    weights = L.weight_estimate_kg()
    assert 1500 <= weights["Ballast"] <= 5000
    assert 14000 <= L.displacement_kg() <= 20000


def test_ventilation_meets_the_bss_formula():
    required = D.ventilation_required()
    assert required == pytest.approx(2200 * 7.0 + 650 * 4 + 550 * 4.5)
    assert D.ventilation_provided("high") >= required / 2
    assert D.ventilation_provided("low") >= required / 2


def test_safety_equipment():
    symbols = [s for s, _, _ in D.SAFETY]
    assert symbols.count("CO") >= 1 and symbols.count("FE") >= 3
    assert "FB" in symbols and "SA" in symbols
    galley = D.ZONE["Galley"]
    assert any(s == "FB" and galley.b0 <= b <= galley.b1 for s, b, _ in D.SAFETY)
    for _, b, y in D.SAFETY:
        assert D.LINING_AFT <= b <= D.LINING_FWD and abs(y) <= D.LINING_IN


def test_deck_fittings_stand_on_deck():
    for b, y in D.T_STUDS:
        assert abs(y) / 1000 < L.plan_half(b / 1000) - 0.08
    s0, s1, depth, _ = D.SEAT_LOCKERS
    assert D.SHELTER_AFT <= s0 < s1 <= 0
    assert 2 * (D.CABIN_BASE_HALF - D.T_CABIN - depth) >= 1000  # room to stand between
