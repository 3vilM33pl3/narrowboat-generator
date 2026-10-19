# SPDX-License-Identifier: MIT
"""Tests for the NB57 drawing set: sheet format, ISO line and lettering sets, dimension
values against the design, layout clearances, determinism and the overlay calibration."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from itertools import combinations
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SCRIPTS))

import drawing_svg as S
import narrowboat_blueprint as B
import narrowboat_design as D
import narrowboat_lines as L

SVG_NS = "{http://www.w3.org/2000/svg}"


@pytest.fixture(scope="module")
def sheets():
    return {number: (sheet, meta) for number, sheet, meta in B.build_sheets()}


def values(dims) -> list[int]:
    return [d.value for d in dims]


def overlaps(a, b, tol=0.15) -> bool:
    return (
        min(a[2], b[2]) - max(a[0], b[0]) > tol
        and min(a[3], b[3]) - max(a[1], b[1]) > tol
    )


def test_sheets_are_a3_svgs(sheets):
    assert list(sheets) == ["NB57-GA-001", "NB57-EX-002"]
    for sheet, _ in sheets.values():
        root = ET.fromstring(sheet.svg().split("\n", 1)[1])
        assert root.tag == f"{SVG_NS}svg"
        assert (root.get("width"), root.get("height")) == ("420mm", "297mm")
        assert root.get("viewBox") == "0 0 420 297"
        ids = {g.get("id") for g in root.iter(f"{SVG_NS}g")}
        assert {"frame", "titleblock"} <= ids


def test_lines_and_lettering_come_from_the_iso_sets(sheets):
    for sheet, _ in sheets.values():
        assert sheet.widths <= set(S.WIDTHS.values())
        assert sheet.heights <= set(S.TEXT_HEIGHTS)


def test_overall_dimensions_follow_the_design(sheets):
    dims = sheets["NB57-GA-001"][1]["dims"]["overall"]
    assert values(dims[:2]) == [D.LENGTH_OVER_FENDERS, D.HULL_LENGTH]
    chain = dims[2:]
    assert sum(values(chain)) == D.HULL_LENGTH
    assert values(chain) == [
        950,
        900,
        D.CABIN_LEN,
        D.WELL_DECK_END - D.CABIN_LEN,
        D.STEM - D.WELL_DECK_END,
    ]


def test_zone_chain_runs_bulkhead_to_bulkhead(sheets):
    plan = sheets["NB57-GA-001"][1]["dims"]["plan"]
    chain = [d for d in plan if d.axis == "b"]
    assert sum(values(chain)) == D.CABIN_LEN
    zones = [z.b1 - z.b0 for z in D.ZONES]
    assert values(chain)[1:-1] == zones


def test_running_dimensions_are_the_opening_centres(sheets):
    dims = sheets["NB57-GA-001"][1]["dims"]
    port = sorted(op.b for op in D.OPENINGS if op.side > 0)
    starboard = sorted([*(op.b for op in D.OPENINGS if op.side < 0), D.SIDE_HATCH.b])
    assert [d.value for d in dims["port"] if d.axis == "b"] == port
    assert [d.value for d in dims["starboard"] if d.axis == "b"] == starboard


def test_plan_gives_the_clear_widths(sheets):
    plan = values(sheets["NB57-GA-001"][1]["dims"]["plan"])
    assert 734 in plan  # galley aisle
    assert D.LINING_IN - D.FIXTURE["Bed"].footprint()[3] in plan  # bedroom corridor
    assert D.BEAM_OVER_STRAKES in plan


def test_section_and_profile_dimensions(sheets):
    meta = sheets["NB57-EX-002"][1]
    section = values(meta["dims"]["section"])
    for v in (D.BEAM_OVER_STRAKES, 2 * D.HALF_BEAM, 2 * D.CABIN_TOP_HALF, D.GUNWALE_W):
        assert v in section
    assert D.HEADROOM in section and 2 * D.LINING_IN in section and 900 in section
    profile = values(meta["dims"]["profile"])
    assert D.LENGTH_OVER_FENDERS in profile
    assert round(L.air_draught() * 1000) in profile
    assert D.DRAUGHT_AFT in profile and D.DRAUGHT_FWD in profile


def test_schedule_lists_every_opening_once(sheets):
    ids = B.OPENING_IDS
    assert len(set(ids.values())) == len(D.OPENINGS)
    box = sheets["NB57-EX-002"][1]["tables"]["openings"]
    assert box[3] - box[1] == pytest.approx(3.6 * (len(D.OPENINGS) + 3) + 0.6)


MAIN = {
    "NB57-GA-001": ("section_A", "plan", "section_B", "notes", "scale", "titleblock"),
    "NB57-EX-002": ("profile", "roof", "section_C", "tables", "titleblock"),
}


@pytest.mark.parametrize("number", list(MAIN))
def test_layout_stays_inside_the_frame_without_collisions(sheets, number):
    sheet, _ = sheets[number]
    x0, y0, x1, y1 = S.FRAME
    for name, box in sheet.boxes.items():
        if name != "frame":
            assert x0 <= box[0] and y0 <= box[1] and box[2] <= x1 and box[3] <= y1, name
    for a, b in combinations(MAIN[number], 2):
        assert not overlaps(sheet.boxes[a], sheet.boxes[b], 0.0), f"{a} overlaps {b}"


@pytest.mark.parametrize("number", list(MAIN))
def test_no_lettering_collides(sheets, number):
    texts = sheets[number][0].texts
    for a, b in combinations(texts, 2):
        assert not overlaps(a[2:], b[2:]), (
            f"{a[1]!r} ({a[0]}) collides with {b[1]!r} ({b[0]})"
        )


def test_drawing_is_deterministic(sheets):
    again = {number: sheet.svg() for number, sheet, _ in B.build_sheets()}
    assert again == {number: sheet.svg() for number, (sheet, _) in sheets.items()}


def test_calibration_maps_model_millimetres_to_pixels(sheets):
    sheet, meta = sheets["NB57-GA-001"]
    cal = B.calibration("NB57-GA-001", sheet, meta["views"])
    px = B.DPI / 25.4
    for name, view in meta["views"].items():
        c = cal[name]
        assert c["mm_per_px"] == pytest.approx(5.0)
        for b, v in ((0, 0), (D.CABIN_LEN, 1000), (D.STERN_TIP, -500)):
            x, y = view.P(b, v)
            assert x * px == pytest.approx(
                c["b0_px"] + c["b_sign"] * b / c["mm_per_px"]
            )
            assert y * px == pytest.approx(
                c["axis_px"] - c["v_sign"] * v / c["mm_per_px"]
            )
    assert cal["starboard"]["v_sign"] == -1  # B-B is drawn inverted


TOOLS = all(shutil.which(t) for t in ("rsvg-convert", "pdfinfo"))


@pytest.mark.skipif(not TOOLS, reason="needs rsvg-convert and pdfinfo")
def test_written_set(tmp_path, sheets):
    manifest = B.write(tmp_path)
    on_disk = json.loads((tmp_path / "drawings_manifest.json").read_text())
    assert on_disk == manifest
    assert manifest["design_sha256"] == B.design_sha256()
    for number, (sheet, meta) in sheets.items():
        assert manifest["calibration"][number] == B.calibration(
            number, sheet, meta["views"]
        )
        for entry in manifest["calibration"][number].values():
            assert (tmp_path / entry["lines_png"]).is_file()
    for name in manifest["files"]:
        assert (tmp_path / name).is_file()
    info = subprocess.run(
        ["pdfinfo", str(tmp_path / "NB57_drawings.pdf")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "Pages:           2" in info and "(A3)" in info
    with (tmp_path / "NB57-GA-001.png").open("rb") as fh:
        head = fh.read(24)
    assert (
        int.from_bytes(head[16:20], "big") == 4200
        and int.from_bytes(head[20:24], "big") == 2970
    )
