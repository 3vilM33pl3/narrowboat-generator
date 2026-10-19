# SPDX-License-Identifier: MIT
"""Tests for the canal-rose SVG generator: brush geometry, layouts, SVG output and publishing."""

from __future__ import annotations

import json
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SCRIPTS))

import canal_roses as C
from procgen_bpy import sha256

SVG_NS = "{http://www.w3.org/2000/svg}"


def all_specs():
    return [*C.PRESETS.values(), *(C.random_spec(s) for s in range(40))]


def test_smooth_path_is_closed_cubic_curve():
    d = C.smooth_path([(0, 0), (10, 0), (10, 10), (0, 10)])
    assert d.startswith("M0 0") and d.endswith("Z")
    assert d.count("C") == 4
    with pytest.raises(ValueError):
        C.smooth_path([(0, 0), (1, 1)])


def test_ribbon_offsets_by_half_width():
    center = [(x, 0.0) for x in range(5)]
    pts = C.ribbon(center, [2.0] * 5)
    assert {round(abs(y), 6) for _, y in pts} == {2.0}
    assert len(pts) == 10


def test_strokes_taper_to_points():
    pts = C.leaf_stroke((0, 0), (100, 0), 10)
    ys = [abs(y) for _, y in pts]
    assert max(ys) == pytest.approx(5, abs=0.2)
    # Both ends close to the centreline: a pointed leaf, not a blunt bar.
    assert min(abs(y) for x, y in pts if x < 1) < 0.5
    assert min(abs(y) for x, y in pts if x > 99) < 0.5


def test_comma_has_round_head_and_pointed_tail():
    pts = C.comma((0, 0), (100, 0), 20, head=0.25)
    widest = max(pts, key=lambda p: abs(p[1]))
    assert 15 < widest[0] < 40
    assert abs(widest[1]) == pytest.approx(10, abs=0.3)


def test_scallop_stays_within_radius():
    import random

    pts = C.scallop((0, 0), 100, 8, random.Random(1))
    radii = [math.hypot(*p) for p in pts]
    assert max(radii) <= 100 * 1.02 + 1e-6
    assert min(radii) >= 100 * (1 - 0.13) * 0.98


@pytest.mark.parametrize("tone", sorted(C.ROSE_TONES))
def test_every_rose_tone_paints_a_full_rose(tone):
    import random

    shapes = C.rose(C.ROSE_TONES[tone], random.Random(3))
    roles = {s.role for s in shapes}
    assert {"rim", "body", "cup", "spiral", "eye", "front", "sweep"} <= roles
    assert all(s.fill.startswith("#") for s in shapes)


def test_reference_layout_matches_the_panel():
    L = C.reference_layout()
    roses = [f for f in L.flowers if f.kind == "rose"]
    assert sorted(f.tone for f in roses) == [
        "pink",
        "pink",
        "red",
        "red",
        "yellow",
        "yellow",
    ]
    assert sum(f.kind == "daisy" for f in L.flowers) == 6
    # The two red roses flank the pink/yellow block, as on the photo.
    reds = sorted(f.x for f in roses if f.tone == "red")
    others = [f.x for f in roses if f.tone != "red"]
    assert reds[0] < min(others) and reds[1] > max(others)


@pytest.mark.parametrize("spec", all_specs(), ids=lambda s: s.name)
def test_specs_validate_and_fit_their_canvas(spec):
    assert C.validate(spec) == []
    L = C.compose(spec)
    assert L.flowers
    for f in L.flowers:
        assert 0 <= f.x <= L.width and 0 <= f.y <= L.height, (spec.name, f)
    assert sum(f.kind == "rose" for f in L.flowers) == (
        6 if spec.layout == "reference" else spec.roses
    )


@pytest.mark.parametrize("spec", all_specs(), ids=lambda s: s.name)
def test_svg_is_well_formed_and_has_no_text(spec):
    root = ET.fromstring(C.render_svg(spec))
    assert root.tag == f"{SVG_NS}svg"
    L = C.compose(spec)
    assert root.get("viewBox") == f"0 0 {C.fmt(L.width)} {C.fmt(L.height)}"
    # The reference's corner number must never come back: no text elements at all.
    assert not root.findall(f".//{SVG_NS}text")
    ids = {g.get("id") for g in root.iter(f"{SVG_NS}g")}
    assert {"leaf-strokes", "leaves", "flowers"} <= ids
    ground = root.find(f"{SVG_NS}rect")
    assert ground is not None and ground.get("fill") == C.PALETTES[spec.palette].ground


def test_transparent_svg_has_no_ground():
    root = ET.fromstring(C.render_svg(C.PRESETS["reference"], background=False))
    assert root.find(f"{SVG_NS}rect") is None


def test_rendering_is_deterministic_and_seeded():
    spec = C.PRESETS["spray_cream"]
    assert C.render_svg(spec) == C.render_svg(spec)
    other = C.Spec(**{**spec.__dict__, "seed": spec.seed + 1})
    assert C.render_svg(other) != C.render_svg(spec)
    assert C.random_spec(5) == C.random_spec(5)


def test_presets_are_distinct():
    svgs = {C.render_svg(s) for s in C.PRESETS.values()}
    assert len(svgs) == len(C.PRESETS)
    assert len({(s.layout, s.palette) for s in C.PRESETS.values()}) == len(C.PRESETS)


@pytest.mark.parametrize(
    "bad, message",
    [
        (C.Spec("x", layout="star"), "unknown layout"),
        (C.Spec("x", palette="mauve"), "unknown palette"),
        (C.Spec("x", tones=()), "at least one rose tone"),
        (C.Spec("x", tones=("teal",)), "unknown rose tone"),
        (C.Spec("x", layout="posy", roses=2), "posy needs"),
        (C.Spec("x", roses=0), "roses must be"),
        (C.Spec("x", width=50), "canvas"),
        (C.Spec("x", layout="single", roses=2), "exactly 1 rose"),
    ],
)
def test_validate_rejects_bad_specs(bad, message):
    assert any(message in e for e in C.validate(bad))
    with pytest.raises(ValueError):
        C.compose(bad)


def test_build_publishes_svgs_and_manifest(tmp_path):
    out = tmp_path / "CanalRoses" / "v001"
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"\x89PNG fake")
    specs = [C.PRESETS["reference"], C.PRESETS["single_rose"]]
    manifest = C.build(specs, out, ref, render=False)
    assert not out.with_name("v001.staging").exists()
    on_disk = json.loads((out / "canal_roses_manifest.json").read_text())
    assert on_disk == manifest
    assert [v["name"] for v in on_disk["variants"]] == ["reference", "single_rose"]
    for v in on_disk["variants"]:
        for key in ("svg", "svg_transparent"):
            path = out / v["files"][key]["path"]
            ET.parse(path)
            assert sha256(path) == v["files"][key]["sha256"]
    assert (out / on_disk["reference"]["path"]).read_bytes() == ref.read_bytes()

    # Republishing replaces the folder wholesale; nothing stale survives.
    C.build([C.PRESETS["posy_round"]], out, None, render=False)
    names = sorted(p.name for p in (out / "SVG").iterdir())
    assert names == ["posy_round.svg", "posy_round_transparent.svg"]
    assert not (out / "Reference").exists()


@pytest.mark.skipif(
    not (shutil.which("rsvg-convert") and shutil.which("magick")),
    reason="needs rsvg-convert and ImageMagick",
)
def test_build_renders_previews(tmp_path):
    out = tmp_path / "v001"
    manifest = C.build([C.PRESETS["single_rose"]], out, None, render=True)
    preview = out / manifest["variants"][0]["files"]["preview"]["path"]
    assert preview.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (out / manifest["contact_sheet"]).is_file()


def test_column_layout_is_a_spray_on_end():
    spec = C.PRESETS["door_column"]
    L = C.compose(spec)
    x0, y0, x1, y1 = C.extent(L)
    assert y1 - y0 > 2 * (x1 - x0)
    flat = C.spray_layout(
        C.Spec(**{**spec.__dict__, "width": spec.height, "height": spec.width}),
        __import__("random").Random(spec.seed),
        0.23,
    )
    assert [(f.y, f.x) for f in flat.flowers] == [(f.x, f.y) for f in L.flowers]


def _points(svg: str):
    """Every path control point in canvas coordinates, through each group's transform."""
    import re

    root = ET.fromstring(svg)
    number = re.compile(r"-?\d+(?:\.\d+)?")

    def walk(node, xf):
        t = node.get("transform")
        if t:
            tx, ty, rot, sc = map(float, number.findall(t))
            c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
            outer = xf

            def xf(p, outer=outer, tx=tx, ty=ty, c=c, s=s, sc=sc):
                x, y = p[0] * sc, p[1] * sc
                return outer((tx + x * c - y * s, ty + x * s + y * c))

        if node.tag == f"{SVG_NS}path":
            vals = [float(v) for v in number.findall(node.get("d"))]
            yield from (xf(p) for p in zip(vals[::2], vals[1::2]))
        for child in node:
            yield from walk(child, xf)

    return root, list(walk(root, lambda p: p))


@pytest.mark.parametrize(
    "spec",
    [
        C.PRESETS["door_column"],
        C.PRESETS["reference"],
        C.Spec("h", layout="posy", roses=5, daisies=4, width=600, height=600, seed=41),
        C.Spec("a", layout="spray", roses=3, daisies=2, width=840, height=400, seed=21),
    ],
    ids=lambda s: s.name,
)
def test_fitted_svg_contains_every_mark(spec):
    # Béziers stay inside their control points' hull, so this bounds every curve drawn.
    root, pts = _points(C.render_svg(spec, background=False, fit=True))
    vx, vy, vw, vh = map(float, root.get("viewBox").split())
    assert (vx, vy, vw, vh) == pytest.approx(C.view_box(spec, fit=True), abs=0.06)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    assert vx <= min(xs) and max(xs) <= vx + vw
    assert vy <= min(ys) and max(ys) <= vy + vh
