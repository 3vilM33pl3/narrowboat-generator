# SPDX-License-Identifier: MIT
"""Tests for the canal-castle SVG scenes: validation, structure, framing and determinism."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SCRIPTS))

import canal_castles as K

SVG = "{http://www.w3.org/2000/svg}"
XLINK = "{http://www.w3.org/1999/xlink}"


def specs():
    out = list(K.PRESETS.values())
    for frame in K.FRAMES:
        for side in (-1, 1):
            for towers in (2, 3, 4):
                out.append(
                    K.Spec(
                        f"{frame}_{side}_{towers}",
                        frame=frame,
                        side=side,
                        towers=towers,
                        seed=towers * 7 + side,
                    )
                )
    return out


@pytest.mark.parametrize("spec", specs(), ids=lambda s: s.name)
def test_scene_is_well_formed_framed_and_textless(spec):
    assert K.validate(spec) == []
    root = ET.fromstring(K.render_svg(spec))
    assert root.get("viewBox") == f"0 0 {K.fmt(spec.width)} {K.fmt(spec.height)}"
    assert not root.findall(f".//{SVG}text")
    scene = root.find(f"{SVG}g[@id='scene']")
    assert scene is not None and scene.get("clip-path") == "url(#frame)"
    assert root.find(f".//{SVG}clipPath[@id='frame']") is not None
    # The lake reflects the castle hill through a <use> of the same group.
    hill = root.find(f".//{SVG}g[@id='castle-hill']")
    assert hill is not None and len(hill) > 20
    uses = root.findall(f".//{SVG}use")
    assert [u.get(f"{XLINK}href") for u in uses] == ["#castle-hill"]
    # Cream and red frame lines sit outside the clipped scene.
    borders = [p for p in root if p.tag == f"{SVG}path" and p.get("fill") == "none"]
    assert [b.get("stroke") for b in borders] == [K.FRAME_CREAM, K.FRAME_RED]


def _roof_xs(svg: str) -> list[float]:
    """x of every roof apex, from the cone paths' final L command."""
    return [float(m) for m in re.findall(rf'fill="{K.ROOF}" d="[^"]*L(-?[\d.]+) ', svg)]


def test_side_puts_the_castle_on_that_side():
    for side in (-1, 1):
        spec = K.Spec("s", side=side)
        xs = _roof_xs(K.render_svg(spec))
        assert xs and all((x - spec.width / 2) * side > 0 for x in xs)


def test_bridge_arches_follow_the_spec():
    def arches(spec):
        # Openings are opaque; each one's reflection on the water is a faded copy.
        root = ET.fromstring(K.render_svg(spec))
        paths = root.iter(f"{SVG}path")
        return sum(p.get("fill") == K.ARCH and p.get("opacity") is None for p in paths)

    assert arches(K.Spec("a", arches=0)) == 0
    assert arches(K.Spec("a", arches=2)) == 2
    assert arches(K.Spec("a", arches=4)) == 4


def test_rendering_is_deterministic_and_seeded():
    spec = K.PRESETS["castle_day"]
    assert K.render_svg(spec) == K.render_svg(spec)
    assert K.render_svg(K.Spec("castle_day", seed=2)) != K.render_svg(spec)


@pytest.mark.parametrize(
    "bad, message",
    [
        (K.Spec("x", frame="star"), "unknown frame"),
        (K.Spec("x", sky="storm"), "unknown sky"),
        (K.Spec("x", side=0), "side must be"),
        (K.Spec("x", towers=5), "towers must be"),
        (K.Spec("x", arches=9), "arches must be"),
        (K.Spec("x", width=20), "canvas"),
    ],
)
def test_validate_rejects_bad_specs(bad, message):
    assert any(message in e for e in K.validate(bad))
    with pytest.raises(ValueError):
        K.render_svg(bad)


@pytest.mark.skipif(not shutil.which("rsvg-convert"), reason="needs rsvg-convert")
def test_rsvg_renders_every_preset(tmp_path):
    for name, spec in K.PRESETS.items():
        svg = tmp_path / f"{name}.svg"
        svg.write_text(K.render_svg(spec))
        png = tmp_path / f"{name}.png"
        subprocess.run(["rsvg-convert", str(svg), "-o", str(png)], check=True)
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
