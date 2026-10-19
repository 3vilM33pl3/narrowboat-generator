# SPDX-License-Identifier: MIT
"""Tests for the NB57 model: it matches its design, every part is a closed solid, and the
fit-out keeps its clearances, headroom and walkway; plus the canal-art decals."""

from __future__ import annotations

import hashlib
import sys
from itertools import combinations, pairwise
from pathlib import Path

import pytest
from meshcheck import assert_closed_outward

SCRIPTS = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import narrowboat_build as B
import narrowboat_common as N
import narrowboat_design as D
import narrowboat_interior as I
import narrowboat_lines as L


@pytest.fixture(scope="module")
def exterior():
    return N.exterior_parts(N.SEED)


@pytest.fixture(scope="module")
def interior():
    return I.interior_parts(N.SEED)


def test_dimensions_follow_the_design(exterior):
    hull = [p for p in exterior if p.name in ("Hull", "Counter", "RubbingStrakes")]
    lo = [min(p.mesh.bounds()[0][k] for p in hull) for k in range(3)]
    hi = [max(p.mesh.bounds()[1][k] for p in hull) for k in range(3)]
    assert N.LOA == pytest.approx(D.HULL_LENGTH / 1000)
    assert hi[0] - lo[0] == pytest.approx(N.LOA, abs=0.05)
    assert hi[1] - lo[1] == pytest.approx(D.BEAM_OVER_STRAKES / 1000, abs=1e-6)
    assert lo[2] == pytest.approx(0.0, abs=1e-9)
    assert N.wx(0) == pytest.approx(-N.X_MID)
    fenders = next(p for p in exterior if p.name == "Fender").mesh.bounds()
    assert fenders[1][0] - fenders[0][0] == pytest.approx(
        D.LENGTH_OVER_FENDERS / 1000, abs=0.01
    )


def test_rooms_follow_the_design():
    for zone, d in zip(I.ZONES, D.ZONES, strict=True):
        assert zone.name == d.name
        assert (zone.b0, zone.b1, zone.floor_z) == pytest.approx(
            (d.b0 / 1000, d.b1 / 1000, d.floor_z / 1000)
        )
    # The lobby floor stops either side of the stern steps, which drop to the cabin floor.
    lobby = [md for md in I.floors(1).values()]
    steps = D.FIXTURE["SternSteps"].footprint()
    for md in lobby:
        for face in md.faces:
            ys = [md.verts[i][1] for i in face]
            xs = [N.bx(md.verts[i][0]) for i in face]
            zs = [md.verts[i][2] for i in face]
            if (
                max(zs) > I.ZONE["Lobby"].floor_z - 0.03
                and max(xs) <= I.ZONE["Lobby"].b1
            ):
                assert (
                    min(ys) >= steps[3] / 1000 - 1e-6
                    or max(ys) <= steps[2] / 1000 + 1e-6
                )


def test_openings_match_the_design():
    assert len(N.OPENINGS) == len(D.OPENINGS)
    for op, d in zip(N.OPENINGS, D.OPENINGS):
        assert (op.name, op.side, op.centre_mm, op.kind) == (
            d.name,
            d.side,
            d.b,
            d.kind,
        )
        assert (op.width, op.height, op.zc) == pytest.approx(
            (d.w / 1000, d.h / 1000, d.zc / 1000)
        )
    cutters = N.cutters()
    for side, key in ((1, "port"), (-1, "starboard")):
        centres = sorted(
            N.bx(sum(md.bounds()[k][0] for k in (0, 1)) / 2) * 1000
            for md in cutters[key]
        )
        assert centres == pytest.approx(
            sorted(op.b for op in D.OPENINGS if op.side == side), abs=0.5
        )


def test_openings_are_inside_the_cabin_and_spaced():
    for side in (1, -1):
        spans = N.opening_spans(side)
        assert spans[0][0] > 0.25 and spans[-1][1] < N.CABIN_LEN - 0.25
        assert all(a[1] + 0.1 <= b[0] for a, b in pairwise(spans))


def test_every_part_is_a_closed_outward_solid(exterior, interior):
    for part in exterior + interior[0]:
        assert_closed_outward(part, slots=B.MATERIALS)


def test_cutters_cover_every_opening():
    cutters = N.cutters()
    assert len(cutters["port"]) == sum(op.side > 0 for op in N.OPENINGS)
    assert len(cutters["starboard"]) == sum(op.side < 0 for op in N.OPENINGS)
    for md in cutters["port"]:
        lo, hi = md.bounds()
        # Through the steel and the lining.
        assert lo[1] < I.lining_y(1.5) - 0.01 and hi[1] > N.cabin_side_y(1.5)


def test_headroom_and_cabin_section():
    assert I.ceiling_z(0.0) - N.Z_FLOOR >= 1.9
    assert I.ceiling_z(0.5) - N.Z_FLOOR >= 1.85
    for z in (1.1, 1.5, 1.9):
        # Above the gunwale the lining stays inside the cabin side.
        assert I.lining_y(z) + I.LINING_T < N.cabin_side_y(z) - N.CABIN_PLATE


def test_linings_stay_inside_the_hull_plating():
    """Below the gunwale, the lining's back is inside the plating wherever the swim and bow
    close in, for every room at its own floor height."""
    for zone in I.ZONES:
        for k in range(41):
            b = zone.b0 + (zone.b1 - zone.b0) * k / 40
            for z in (
                zone.floor_z,
                (zone.floor_z + N.Z_GUNWALE) / 2,
                N.Z_GUNWALE - 0.02,
            ):
                assert I.HULL_LINING + I.LINING_T < L.inner_half_at(b, z), (
                    zone.name,
                    b,
                    z,
                )


def test_furniture_stays_inside_the_lining(interior):
    _, items, _ = interior
    floor = min(z.floor_z for z in I.ZONES)
    for name, (lo, hi) in items:
        assert lo[2] >= floor - 0.021, name
        assert hi[2] <= I.ceiling_z(0.0), name
        assert (
            N.wx(1000 * I.B_IN0) - 1e-6 <= lo[0]
            and hi[0] <= N.wx(1000 * I.B_IN1) + 1e-6
        ), name
        for z in (lo[2], hi[2]):
            for y in (lo[1], hi[1]):
                assert abs(y) <= I.lining_y(z) + 0.003, (name, y, z)


def test_major_furniture_does_not_overlap(interior):
    _, items, _ = interior
    major = {f.name for f in D.FIXTURES if f.kind in D.CARCASS_KINDS}
    boxes = [
        (n, b)
        for n, b in items
        if n.split(".")[1] in major or "tread" in n or "bench" in n
    ]
    assert {n.split(".")[1] for n, _ in boxes} >= major - {
        "SternSteps",
        "BowSteps",
        "Dinette",
    }
    for (na, (alo, ahi)), (nb, (blo, bhi)) in combinations(boxes, 2):
        overlap = [min(ahi[k], bhi[k]) - max(alo[k], blo[k]) for k in range(3)]
        assert min(overlap) <= 0.002, (na, nb, overlap)


def test_there_is_a_walkway_the_whole_length(interior):
    """At body height a free strip at least 0.5 m wide runs from the stern steps to the bow
    steps, through both bathroom doorways."""
    _, items, _ = interior
    start, end = I.ZONE["Bedroom"].b0 + 0.02, D.FIXTURE["BowSteps"].b0 / 1000 - 0.02
    for i in range(300):
        b = start + (end - start) * i / 299
        x = N.wx(1000 * b)
        half = I.lining_y(1.0)
        blocked = [
            (lo[1], hi[1])
            for _, (lo, hi) in items
            if lo[0] <= x <= hi[0] and lo[2] < 1.4 and hi[2] > 0.35
        ]
        for p in D.PARTITIONS:
            if abs(b - p.b / 1000) <= D.PARTITION_T / 2000:
                blocked += [(-half, p.door_y[0] / 1000), (p.door_y[1] / 1000, half)]
        cursor, best = -half, 0.0
        for a, c in sorted(blocked):
            best = max(best, a - cursor)
            cursor = max(cursor, c)
        best = max(best, half - cursor)
        assert best >= 0.5, (round(b, 2), best)


def test_lights_sit_inside_the_cabin_clear_of_the_openings(interior):
    _, _, lights = interior
    assert len(lights) == len(I.SCONCES) + len(I.downlights())
    for _, (x, y, z), watts in lights:
        assert N.wx(0) < x < N.wx(1000 * N.CABIN_LEN)
        assert abs(y) < I.lining_y(z)
        assert N.Z_FLOOR < z < I.ceiling_z(y)
        assert watts > 0
    for side, b in I.SCONCES:
        assert all(not (a - 0.1 <= b <= c + 0.1) for a, c in N.opening_spans(side)), (
            side,
            b,
        )


def test_lake_stops_at_the_hull():
    """The lakeside water is a ring from the hull's waterline outline outwards: no face over
    the cabin, every face facing up, and open water beside the boat."""
    verts, faces = B.lake_mesh()

    def inside(p, poly):
        hit = False
        for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
            if (y0 > p[1]) != (y1 > p[1]) and p[0] < x0 + (p[1] - y0) * (x1 - x0) / (
                y1 - y0
            ):
                hit = not hit
        return hit

    quads = [[verts[i][:2] for i in f] for f in faces]
    for q in quads:
        area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(q, q[1:] + q[:1])) / 2
        assert area > 0
    for b, y in ((0.0, 0.0), (3.0, 0.8), (-0.8, -0.9), (14.6, 0.3)):
        assert not any(inside((N.wx(1000 * b), y), q) for q in quads), (b, y)
    assert any(inside((0.0, 1.5), q) for q in quads)
    assert B.TRIM < 0  # trimmed by the stern: the bow rises in the lake


def test_build_is_deterministic():
    def digest(parts):
        h = hashlib.sha256()
        for p in parts:
            h.update(p.name.encode())
            for v in p.mesh.verts:
                h.update(f"{v[0]:.6f},{v[1]:.6f},{v[2]:.6f};".encode())
        return h.hexdigest()

    assert digest(N.exterior_parts(1)) == digest(N.exterior_parts(1))
    assert digest(I.interior_parts(1)[0]) == digest(I.interior_parts(1)[0])


# --------------------------------------------------------------------------- decal panels


def _col(rot, c):
    return tuple(rot[r][c] for r in range(3))


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def test_each_cabin_side_carries_three_sprays_with_the_reference_mid_side():
    for _, side_name in N.SIDES:
        cabin = {
            p.design: N.bx(p.origin[0])
            for p in N.decal_panels()
            if p.name.endswith(side_name) and p.design.startswith("Cabin")
        }
        assert set(cabin) == {"CabinAft", "CabinCentre", "CabinFore"}
        assert cabin["CabinAft"] < cabin["CabinCentre"] < cabin["CabinFore"]
    assert N.ROSE_DESIGNS["CabinCentre"] == N.C.PRESETS["reference"]


def test_cabin_roses_sit_inside_the_panel_clear_of_every_opening():
    clear = N.DECAL_CLEAR - 1e-9
    for p in N.decal_panels():
        if not p.design.startswith("Cabin"):
            continue
        side = 1 if p.name.endswith("Port") else -1
        b = N.bx(p.origin[0])
        b0, b1 = b - p.width / 2, b + p.width / 2
        assert b0 - clear >= N.PANEL_B[0] + N.PANEL_INNER, p.name
        assert b1 + clear <= N.PANEL_B[1] - N.PANEL_INNER, p.name
        assert p.height / 2 + clear <= N.PANEL_H / 2 - N.PANEL_INNER, p.name
        for a, z in N.opening_spans(side):
            assert b1 + clear <= a or z <= b0 - clear, (p.name, a, z)
        # Same frame and height as the coachlines, so the slab lies on the painted side.
        assert p.origin[2] == pytest.approx(N.PANEL_ZC)
        assert abs(p.origin[1]) == pytest.approx(N.cabin_side_y(N.PANEL_ZC))


def test_door_and_hatch_decals_sit_inside_their_leaf_borders():
    border = N.LEAF_LINE_INSET + N.LEAF_LINE_W
    leaves = {d.name: d for d in N.door_leaves()}
    hatch = {name: (u0, u1) for name, u0, u1 in N.hatch_leaves()}
    on_leaf: dict[str, list] = {}
    for p in N.decal_panels():
        leaf = next(
            (name for name in [*leaves, *hatch] if p.name.startswith(name)), None
        )
        if leaf is None:
            continue
        on_leaf.setdefault(leaf, []).append(p)
        cu, cv = p.centre
        if leaf in leaves:
            d = leaves[leaf]
            half_w, half_h = (d.y1 - d.y0) / 2, (d.z1 - d.z0) / 2
            assert p.origin == pytest.approx(
                (d.face_x, (d.y0 + d.y1) / 2, (d.z0 + d.z1) / 2)
            )
            u0, u1, v0, v1 = -half_w, half_w, -half_h, half_h
        else:
            u0, u1 = hatch[leaf]
            v0, v1 = 0.0, N.HATCH_LEAF_H
            assert p.face == -N.HATCH_PROUD
        assert u0 + border < cu - p.width / 2 and cu + p.width / 2 < u1 - border, p.name
        assert v0 + border < cv - p.height / 2 and cv + p.height / 2 < v1 - border, (
            p.name
        )
    assert set(on_leaf) == set(leaves) | set(hatch)
    # Every door and hatch leaf: a castle above, roses below, not overlapping.
    for name in [*leaves, *hatch]:
        kinds = {N.design_kind(p.design): p for p in on_leaf[name]}
        assert set(kinds) == {"castle", "roses"}, name
        castle, roses = kinds["castle"], kinds["roses"]
        gap = (castle.centre[1] - castle.height / 2) - (
            roses.centre[1] + roses.height / 2
        )
        assert gap >= N.DOOR_GAP - 1e-9, name
    # Each pair of leaves is painted as mirror images.
    for pair in (("SternDoorPort", "SternDoorStarboard"), ("HatchAft", "HatchFore")):
        mirrored = {
            p.name: p.mirror
            for p in N.decal_panels()
            if any(p.name.startswith(leaf) for leaf in pair)
        }
        assert set(mirrored.values()) == {False, True}


def test_castles_stand_toward_the_outer_edges_of_their_pair():
    castles = [p for p in N.decal_panels() if N.design_kind(p.design) == "castle"]
    pairs: dict[str, list] = {}
    for p in castles:
        pairs.setdefault(p.design, []).append(p)
    assert set(pairs) == set(N.CASTLE_DESIGNS)
    for design, pair in pairs.items():
        assert len(pair) == 2 and {p.mirror for p in pair} == {False, True}, design
        mid = [sum(c) / 2 for c in zip(*(N.panel_centre(p) for p in pair))]
        for p in pair:
            outward = _sub(N.panel_centre(p), mid)
            # Seen from outside, the texture's right is -along, or +along when mirrored.
            right = tuple(-c if not p.mirror else c for c in _col(p.rot, 0))
            towards = N.CASTLE_DESIGNS[design].side
            assert _dot(right, outward) * towards > 0, p.name


@pytest.mark.parametrize("panel", N.decal_panels(), ids=lambda p: p.name)
def test_decal_slab_is_proud_of_its_face_and_reads_from_outside(panel):
    md = N.decal_slab(panel)
    along, up, into = (_col(panel.rot, c) for c in range(3))
    centre = N.panel_centre(panel)
    depth = [_dot(_sub(v, centre), into) for v in md.verts]
    # Off the paint toward the viewer (against `into`), never sunk into it.
    assert min(depth) == pytest.approx(-N.DECAL_PROUD[1])
    assert max(depth) == pytest.approx(-N.DECAL_PROUD[0])
    assert panel.height / panel.width == pytest.approx(N.design_aspect(panel.design))
    front = [(f, uv) for f, uv in zip(md.faces, md.uvs) if uv[0] != N.DECAL_OUTSIDE_UV]
    assert len(front) == 1 and len(md.faces) == 6
    face, uv = front[0]
    assert {round(c, 6) for u in uv for c in u} == {0.0, 1.0}
    # Seen from outside, right is -along; U must grow that way unless the leaf is mirrored.
    right = [_dot(md.verts[i], along) * -1 for i in face]
    us = [u for u, _ in uv]
    rising = max(zip(right, us))[1] > min(zip(right, us))[1]
    assert rising != panel.mirror
    height = [_dot(md.verts[i], up) for i in face]
    vs = [v for _, v in uv]
    assert max(zip(height, vs))[1] == pytest.approx(1.0)


def test_decal_parts_have_materials_and_collections(exterior):
    names = {p.name: p for p in exterior}
    for design in N.DECAL_DESIGNS:
        assert f"Decal{design}" in B.MATERIALS
        assert any(p.slots == (f"Decal{design}",) for p in exterior), design
    # Port-side parts hide with the port side in the cutaway; the hatch is to starboard.
    for name in (
        "DecalCabinCentrePort",
        "SeatLockerPort",
        "ShelterRailPort",
        "CabinPort",
    ):
        assert B.group_of(name) == "CabinPort", name
    hatch = N.HATCH_NAME
    assert B.group_of(f"DecalHatchCastle{hatch}") == (
        "CabinPort" if hatch == "Port" else "Cabin"
    )
    assert f"HatchLinesCream{hatch}" in names
    for name in (
        "DecalSternDoorCastle",
        "DecalSternDoorRoses",
        "DecalBowDoorCastle",
        "DecalBowDoorRoses",
        "DoorLinesCream",
        "DecalCabinAftStarboard",
    ):
        assert B.group_of(name) == "Cabin", name
    assert names["DoorLinesCream"].slots == ("CoachCream",)


def test_decal_designs_fit_their_textures():
    for design, spec in N.ROSE_DESIGNS.items():
        x0, y0, x1, y1 = N.C.extent(N.C.compose(spec))
        vx, vy, vw, vh = N.C.view_box(spec, fit=True)
        assert vx < x0 and x1 < vx + vw and vy < y0 and y1 < vy + vh, design
    for design, spec in N.CASTLE_DESIGNS.items():
        assert N.K.validate(spec) == []
        assert N.design_view_box(design) == (0, 0, spec.width, spec.height)
        assert N.texture_name(design) == f"castle_{design}.png"


# --------------------------------------------------------------------------- weathered build


def test_clean_materials_never_touch_the_weathered_builders(exterior, monkeypatch):
    made = []
    monkeypatch.setattr(
        B,
        "MATERIALS",
        {s: (lambda bpy, n, s=s: made.append(s) or n) for s in B.MATERIALS},
    )

    def refuse(*args):
        raise AssertionError("a weathered builder ran in a clean build")

    monkeypatch.setattr(B, "WEATHERED", {s: refuse for s in B.WEATHERED})
    monkeypatch.setattr(B.WX, "false_colour", refuse)
    out = B.build_materials(None, exterior)
    assert set(out) == {s for p in exterior for s in p.slots}
    assert out["HullBlack"] == "M_NB_HullBlack"


def test_weathered_slots_cover_the_exterior_paint_and_metal(exterior):
    slots = {s for p in exterior for s in p.slots}
    for slot in ("HullBlack", "CabinGreen", "CoachCream", "CoachRed", "RoofOxide"):
        assert slot in B.WEATHERED
    assert all(f"Decal{d}" in B.WEATHERED for d in N.DECAL_DESIGNS)
    assert set(B.FALSE_BASE) <= set(B.WEATHERED)
    assert set(B.WX.USES) == set(B.FALSE_BASE)
    assert {s for s in slots if not s.startswith("Decal")} - set(B.WEATHERED) <= {
        "Glass",
        "Stainless",
    }


def test_shots_parse_names_and_groups():
    assert B.parse_shots(None, "clean") == B.DEFAULT_SHOTS
    assert set(B.WEAR_SHOTS) <= set(B.parse_shots(None, "weathered"))
    shots = B.parse_shots("stage1,hero,wear_weld", "weathered")
    assert shots[0] == "hero" and shots.count("hero") == 1 and "wear_weld" in shots
    with pytest.raises(ValueError):
        B.parse_shots("hero,nonsense", "clean")


def test_output_guards(tmp_path):
    B.check_out(tmp_path / "scratch", "weathered", True, False)
    B.check_out(N.OUTPUT_DIR, "clean", False, False)
    with pytest.raises(RuntimeError):
        B.check_out(N.OUTPUT_DIR, "weathered", False, True)
    with pytest.raises(RuntimeError):
        B.check_out(B.default_out("weathered"), "weathered", False, False)
    B.check_out(B.default_out("weathered"), "weathered", False, True)
    with pytest.raises(RuntimeError):
        B.check_out(N.OUTPUT_DIR, "clean", True, False)  # a partial render over the clean build
    assert B.blend_name("weathered") == "Narrowboat_NB57_Weathered.blend"
    assert B.blend_name("clean") == N.BLEND_NAME


def test_handrail_legs_are_where_the_rail_posts_stand():
    rails, _, _ = N.roof_gear(N.SEED)
    xs = {round(v[0], 4) for v in rails.verts}
    for b in N.handrail_legs():
        x = N.wx(1000 * b)
        assert any(abs(x - 0.015 - v) < 1e-4 for v in xs)
        assert any(abs(x + 0.015 - v) < 1e-4 for v in xs)


def test_wear_cameras_look_at_the_boat():
    aims = B.WM.camera_aims(B.WM.WEARS["weathered"], N.SEED)
    specs = B.WX.wear_camera_specs(aims)
    assert specs["wear_weld"]["target"][0] == N.wx(aims["weld"])
    assert specs["wear_porthole"]["target"][0] == N.wx(aims["porthole"])
    assert specs["wear_counter"] == B.WX.wear_camera_specs()["wear_counter"]
    for name, spec in specs.items():
        (lx, ly, lz), (tx, ty, tz) = spec["location"], spec["target"]
        assert abs(ty) <= 1.1 and 0.5 < tz < 2.3, name
        assert 0.5 < ((lx - tx) ** 2 + (ly - ty) ** 2 + (lz - tz) ** 2) ** 0.5 < 4.0, (
            name
        )
