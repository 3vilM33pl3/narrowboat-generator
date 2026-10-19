#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build narrowboat NB57 'Wren' (exterior and full interior) as a render-ready Blender scene.

blender --background --factory-startup --python-exit-code 1 --python src/narrowboat_build.py -- \
    [--seed N] [--out DIR] [--samples N] [--no-render] [--hdri FILE] [--wear LEVEL]

Geometry is pure Python, built from the original design in narrowboat_design: the hull,
cabin, semi-trad stern and deck gear in narrowboat_common, the fit-out in
narrowboat_interior. Windows and portholes are live Boolean modifiers (exact solver) driven
by cutter objects in the hidden `Cutters` collections, so openings stay editable. Collections
separate the hull, cabin, port side, roof, ceiling, interior and lights; the cutaway renders
hide the roof, ceiling and port side. Materials are procedural Cycles graphs: traditional
livery outside (black hull, deep green cabin, cream and red coachlines, red oxide roof) and
the design's finishes inside (ash-faced ply, painted tongue and groove, engineered oak,
carpet and vinyl, oak worktops). Canal art is painted on as alpha-textured decal slabs:
roses from src/canal_roses.py on the cabin-side panels, and on each end-door and hatch
leaf a castle scene from src/canal_castles.py over a bunch of roses. The decals' SVGs and
PNGs (rsvg-convert) are written to Textures/ and the .blend links them relatively.

The previews include the plan, both elevations and the outboard profile rendered at the
drawing set's scale and laid over its linework (narrowboat_blueprint, build/Drawings), so
the model can be checked against its own drawings; those stay in the grey studio, level.
The beauty shots and interiors are taken on the lake: lit and backed by an HDRI when one is given
(turned so its sun stands behind each camera), with the boat afloat in a water ring that
stops at its waterline and pitched to its trimmed design waterline. The build refuses to
run against drawings made from a different design.

Outputs (default build/NB57, written to a staging folder
and swapped in): Narrowboat_NB57.blend, Reference/ (the drawing set), Textures/, Previews/
and narrowboat_manifest.json.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import narrowboat_common as N
import narrowboat_dents as DN
import narrowboat_design as D
import narrowboat_interior as I
import narrowboat_wear as WM
import narrowboat_weathered as WX
from procgen_bpy import (
    Graph,
    fresh_scene,
    keep_previews,
    lookdev,
    make_cameras,
    make_object,
    new_collection,
    over_grey,
    pixels,
    publish,
    receipts,
    render,
    save_png,
    sha256,
    write_json,
)

PROPS = {"generator": "src/narrowboat_build.py", "build_version": N.BUILD_VERSION}
WARM = (1.0, 0.72, 0.45)

# --------------------------------------------------------------------------- materials


def paint(bpy, name: str, colour, rough: float = 0.28):
    """Clean enamel: faint large-scale variation, orange peel, a little crevice grime."""
    g = Graph(bpy, name, colour)
    obj = g.coords.outputs["Object"]
    big = g.noise(obj, 0.8, detail=2.0)
    tint = g.node(
        "ShaderNodeHueSaturation",
        {"Value": g.map_range(big, 0.3, 0.7, 0.95, 1.05), "Color": colour},
    ).outputs["Color"]
    grime = g.map_range(g.occlusion(0.06), 0.6, 1.0, 0.35, 0.0)
    col = g.mix(grime, tint, (0.01, 0.01, 0.008))
    peel = g.noise(obj, 260.0, detail=3.0)
    return g.finish(
        col,
        g.math("ADD", rough, g.math("MULTIPLY", grime, 0.3)),
        g.bump(peel, 0.03, 0.001),
    )


def hull_black(bpy, name: str):
    """Bitumen-black steel: satin, uneven sheen, dull and faintly green below the waterline."""
    g = Graph(bpy, name, (0.012, 0.012, 0.012))
    obj = g.coords.outputs["Object"]
    z = g.object_axis("Z")
    water = g.map_range(z, 0.62, 0.5)
    col = g.mix(
        g.math("MULTIPLY", water, 0.6), (0.014, 0.014, 0.013), (0.02, 0.026, 0.016)
    )
    rough = g.math(
        "ADD",
        g.map_range(g.noise(obj, 3.0, detail=4.0), 0.3, 0.7, 0.32, 0.52),
        g.math("MULTIPLY", water, 0.3),
    )
    return g.finish(col, rough, g.bump(g.noise(obj, 40.0, detail=6.0), 0.05, 0.002))


def non_slip(bpy, name: str, colour):
    g = Graph(bpy, name, colour)
    obj = g.coords.outputs["Object"]
    grit = g.noise(obj, 700.0, detail=2.0)
    col = g.mix(
        g.math("MULTIPLY", g.map_range(grit, 0.4, 0.7), 0.25),
        colour,
        tuple(c * 1.5 for c in colour),
    )
    return g.finish(col, 0.82, g.bump(grit, 0.35, 0.0015))


def metal(bpy, name: str, colour, rough: float, tarnish: float = 0.0):
    g = Graph(bpy, name, colour)
    obj = g.coords.outputs["Object"]
    col = colour
    if tarnish:
        stain = g.map_range(g.noise(obj, 12.0, detail=5.0), 0.45, 0.75)
        col = g.mix(
            g.math("MULTIPLY", stain, tarnish), colour, tuple(c * 0.55 for c in colour)
        )
    return g.finish(
        col,
        g.math("ADD", rough, g.math("MULTIPLY", g.noise(obj, 30.0), 0.06)),
        g.geometry.outputs["Normal"],
        metallic=1.0,
        specular=0.5,
    )


def glass(bpy, name: str, roughness: float = 0.0, tint=(0.92, 0.96, 0.94)):
    """Window glass that lets daylight through: shadow rays see a transparent surface."""
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*tint, 0.3)
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*tint, 1)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Transmission Weight"].default_value = 1.0
    bsdf.inputs["IOR"].default_value = 1.5
    clear = nt.nodes.new("ShaderNodeBsdfTransparent")
    path = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(path.outputs["Is Shadow Ray"], mix.inputs[0])
    nt.links.new(bsdf.outputs["BSDF"], mix.inputs[1])
    nt.links.new(clear.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat


def oak(bpy, name: str, base, rough: float = 0.45):
    g = Graph(bpy, name, base)
    grain = g.grain(26.0, 10.0, detail=6.0)
    big = g.noise(g.coords.outputs["Object"], 0.9, detail=2.0)
    plank = g.math("SUBTRACT", g.attribute("plank_rand"), 0.5)
    value = g.math(
        "ADD", g.map_range(big, 0.3, 0.7, 0.9, 1.08), g.math("MULTIPLY", plank, 0.25)
    )
    tinted = g.node("ShaderNodeHueSaturation", {"Value": value, "Color": base}).outputs[
        "Color"
    ]
    col = g.mix(g.math("MULTIPLY", grain, 0.55), tinted, tuple(c * 0.55 for c in base))
    grime = g.map_range(g.occlusion(0.05), 0.6, 1.0, 0.4, 0.0)
    col = g.mix(grime, col, tuple(c * 0.3 for c in base))
    return g.finish(col, rough, g.bump(grain, 0.06, 0.001))


def tongue_groove(bpy, name: str, colour, pitch: float = 0.1):
    """White painted T&G boards: a V-groove every `pitch` metres along U."""
    g = Graph(bpy, name, colour)
    bands = g.node(
        "ShaderNodeTexWave",
        # Saw bands repeat every 2*pi / 20 units of the scaled vector.
        {
            "Vector": g.coords.outputs["UV"],
            "Scale": math.pi / (10 * pitch),
            "Distortion": 0.0,
        },
        wave_type="BANDS",
        bands_direction="X",
        wave_profile="SAW",
    ).outputs["Factor"]
    groove = g.map_range(bands, 0.0, 0.03, 1.0, 0.0)
    col = g.mix(
        g.math("MULTIPLY", groove, 0.6), colour, tuple(c * 0.55 for c in colour)
    )
    grain = g.grain(30.0, 10.0)
    col = g.mix(g.math("MULTIPLY", grain, 0.04), col, tuple(c * 0.9 for c in colour))
    return g.finish(col, 0.55, g.bump(g.math("SUBTRACT", 1.0, groove), 0.5, 0.002))


def plain(bpy, name: str, colour, rough: float, specular: float = 0.3):
    g = Graph(bpy, name, colour)
    return g.finish(colour, rough, g.geometry.outputs["Normal"], specular=specular)


def granite(bpy, name: str):
    """Steel-grey granite worktop: fine light and dark speckles, polished."""
    base = (0.07, 0.07, 0.075)
    g = Graph(bpy, name, base)
    obj = g.coords.outputs["Object"]
    light = g.map_range(g.noise(obj, 180.0, detail=3.0), 0.62, 0.66)
    dark = g.map_range(
        g.noise(g.mapping(obj, (1.3, 1.3, 1.3)), 140.0, detail=3.0), 0.36, 0.33
    )
    cloud = g.map_range(g.noise(obj, 6.0, detail=4.0), 0.3, 0.7, 0.85, 1.2)
    col = g.node("ShaderNodeHueSaturation", {"Value": cloud, "Color": base}).outputs[
        "Color"
    ]
    col = g.mix(light, col, (0.42, 0.42, 0.43))
    col = g.mix(dark, col, (0.01, 0.01, 0.012))
    return g.finish(col, 0.12, g.geometry.outputs["Normal"], specular=0.5)


def tiles(bpy, name: str, colour, size: float, grout=(0.2, 0.2, 0.2)):
    """Square tiles on the part's metre UVs with grout lines and a little per-tile shade."""
    g = Graph(bpy, name, colour)
    brick = g.node(
        "ShaderNodeTexBrick",
        {
            "Vector": g.coords.outputs["UV"],
            "Color1": colour,
            "Color2": tuple(c * 0.85 for c in colour),
            "Mortar": grout,
            "Scale": 1.0,
            "Mortar Size": 0.003,
            "Mortar Smooth": 0.2,
            "Brick Width": size,
            "Row Height": size,
        },
        offset=0.0,
        offset_frequency=1,
        squash=1.0,
        squash_frequency=1,
    )
    joint = brick.outputs["Factor"]
    rough = g.map_range(joint, 0.0, 1.0, 0.22, 0.85)
    return g.finish(
        brick.outputs["Color"],
        rough,
        g.bump(g.math("SUBTRACT", 1.0, joint), 0.4, 0.002),
        specular=0.45,
    )


def vinyl(bpy, name: str, colour):
    """Seamless sheet vinyl: a soft mottle and fine flecks under a satin wear layer."""
    g = Graph(bpy, name, colour)
    obj = g.coords.outputs["Object"]
    mottle = g.map_range(g.noise(obj, 3.0, detail=4.0), 0.35, 0.65, 0.9, 1.08)
    col = g.node("ShaderNodeHueSaturation", {"Value": mottle, "Color": colour}).outputs[
        "Color"
    ]
    fleck = g.map_range(g.noise(obj, 320.0, detail=1.0), 0.63, 0.67)
    col = g.mix(
        g.math("MULTIPLY", fleck, 0.6), col, tuple(min(1.0, c * 1.7) for c in colour)
    )
    return g.finish(
        col, 0.38, g.bump(g.noise(obj, 45.0, detail=2.0), 0.04, 0.0005), specular=0.4
    )


def water(bpy, name: str):
    """Canal water: dark, mirror-like at grazing angles, with fine wind ripples running
    across it (the plane's object coordinates are metres)."""
    base = (0.012, 0.028, 0.026)
    g = Graph(bpy, name, base)
    obj = g.coords.outputs["Object"]
    ripple = g.noise(g.mapping(obj, (0.7, 1.8, 1.0)), 2.2, detail=6.0, roughness=0.55)
    swell = g.noise(obj, 0.15, detail=2.0)
    height = g.math("ADD", ripple, g.math("MULTIPLY", swell, 0.6))
    return g.finish(base, 0.04, g.bump(height, 0.18, 0.03), specular=0.5)


def carpet(bpy, name: str, colour):
    g = Graph(bpy, name, colour)
    obj = g.coords.outputs["Object"]
    fibre = g.noise(obj, 900.0, detail=2.0)
    patch = g.map_range(g.noise(obj, 2.0, detail=3.0), 0.3, 0.7, 0.9, 1.06)
    col = g.node("ShaderNodeHueSaturation", {"Value": patch, "Color": colour}).outputs[
        "Color"
    ]
    col = g.mix(g.math("MULTIPLY", fibre, 0.3), col, tuple(c * 0.6 for c in colour))
    return g.finish(col, 1.0, g.bump(fibre, 0.4, 0.002), specular=0.1)


def fabric(bpy, name: str, colour):
    g = Graph(bpy, name, colour)
    uv = g.coords.outputs["UV"]
    warp = g.node(
        "ShaderNodeTexWave",
        {"Vector": uv, "Scale": 60.0},
        wave_type="BANDS",
        bands_direction="X",
    ).outputs["Factor"]
    weft = g.node(
        "ShaderNodeTexWave",
        {"Vector": uv, "Scale": 60.0},
        wave_type="BANDS",
        bands_direction="Y",
    ).outputs["Factor"]
    weave = g.math("MULTIPLY", warp, weft)
    col = g.mix(
        g.math("MULTIPLY", weave, 0.25),
        colour,
        tuple(min(1.0, c * 1.4) for c in colour),
    )
    grime = g.map_range(g.occlusion(0.06), 0.5, 1.0, 0.4, 0.0)
    col = g.mix(grime, col, tuple(c * 0.4 for c in colour))
    return g.finish(col, 0.92, g.bump(weave, 0.15, 0.001), specular=0.15)


def emissive(bpy, name: str, colour, strength: float):
    g = Graph(bpy, name, colour)
    g.set(g.bsdf.inputs["Emission Color"], colour)
    g.set(g.bsdf.inputs["Emission Strength"], strength)
    return g.finish(colour, 0.5, g.geometry.outputs["Normal"])


def rope(bpy, name: str):
    base = (0.28, 0.2, 0.11)
    g = Graph(bpy, name, base)
    twist = g.node(
        "ShaderNodeTexWave",
        {
            "Vector": g.coords.outputs["Object"],
            "Scale": 14.0,
            "Distortion": 1.0,
            "Detail": 3.0,
        },
        wave_type="BANDS",
        bands_direction="DIAGONAL",
    ).outputs["Factor"]
    col = g.mix(g.math("MULTIPLY", twist, 0.6), (0.09, 0.06, 0.03), base)
    return g.finish(col, 0.95, g.bump(twist, 0.6, 0.004), specular=0.1)


MATERIALS = {
    "HullBlack": lambda bpy, n: hull_black(bpy, n),
    "DeckNonSlip": lambda bpy, n: non_slip(bpy, n, (0.02, 0.02, 0.02)),
    "CabinGreen": lambda bpy, n: paint(bpy, n, (0.008, 0.06, 0.03)),
    "CoachCream": lambda bpy, n: paint(bpy, n, (0.62, 0.5, 0.3), 0.3),
    "CoachRed": lambda bpy, n: paint(bpy, n, (0.32, 0.015, 0.012), 0.3),
    "RoofOxide": lambda bpy, n: non_slip(bpy, n, (0.16, 0.035, 0.022)),
    "Brass": lambda bpy, n: metal(bpy, n, (0.95, 0.72, 0.36), 0.22, tarnish=0.35),
    "Stainless": lambda bpy, n: metal(bpy, n, (0.78, 0.78, 0.8), 0.16),
    "FrameGold": lambda bpy, n: metal(bpy, n, (0.82, 0.62, 0.32), 0.3),
    "Glass": lambda bpy, n: glass(bpy, n),
    "FrostedGlass": lambda bpy, n: glass(bpy, n, 0.35),
    "CastIron": lambda bpy, n: plain(bpy, n, (0.02, 0.02, 0.02), 0.65),
    "TillerWood": lambda bpy, n: oak(bpy, n, (0.32, 0.15, 0.06), 0.25),
    "Rope": lambda bpy, n: rope(bpy, n),
    "Oak": lambda bpy, n: oak(bpy, n, (0.38, 0.22, 0.1)),
    "OakLight": lambda bpy, n: oak(bpy, n, (0.55, 0.37, 0.19)),
    "WhiteTG": lambda bpy, n: tongue_groove(bpy, n, (0.78, 0.77, 0.72)),
    "CreamTG": lambda bpy, n: tongue_groove(bpy, n, (0.74, 0.68, 0.55)),
    "SageTG": lambda bpy, n: tongue_groove(bpy, n, (0.36, 0.42, 0.32)),
    "Ash": lambda bpy, n: oak(bpy, n, (0.6, 0.48, 0.32), 0.5),
    "OakWorktop": lambda bpy, n: oak(bpy, n, (0.42, 0.25, 0.11), 0.35),
    "Vinyl": lambda bpy, n: vinyl(bpy, n, (0.15, 0.16, 0.17)),
    "WhitePanel": lambda bpy, n: plain(bpy, n, (0.8, 0.79, 0.76), 0.7, 0.2),
    "Granite": lambda bpy, n: granite(bpy, n),
    "TileFloor": lambda bpy, n: tiles(bpy, n, (0.045, 0.045, 0.05), 0.3),
    "TileWall": lambda bpy, n: tiles(
        bpy, n, (0.06, 0.06, 0.065), 0.15, (0.35, 0.35, 0.34)
    ),
    "Carpet": lambda bpy, n: carpet(bpy, n, (0.2, 0.18, 0.15)),
    "Upholstery": lambda bpy, n: fabric(bpy, n, (0.04, 0.07, 0.1)),
    "Bedding": lambda bpy, n: fabric(bpy, n, (0.7, 0.68, 0.62)),
    "Duvet": lambda bpy, n: fabric(bpy, n, (0.2, 0.26, 0.24)),
    "Ceramic": lambda bpy, n: plain(bpy, n, (0.85, 0.85, 0.83), 0.06, 0.5),
    "BlackGlass": lambda bpy, n: plain(bpy, n, (0.01, 0.01, 0.01), 0.08, 0.5),
    "Screen": lambda bpy, n: plain(bpy, n, (0.004, 0.006, 0.01), 0.03, 0.5),
    "LampShade": lambda bpy, n: emissive(bpy, n, (1.0, 0.82, 0.6), 3.0),
}
MATERIALS.update(
    {
        f"Decal{design}": (lambda bpy, n, design=design: decal(bpy, n, design))
        for design in N.DECAL_DESIGNS
    }
)


DECAL_TEXTURE_PX = 2048  # longer side of each decal texture


def write_decal_textures(folder: Path) -> dict:
    """Each decal design as a transparent-ground SVG (roses cropped to their paint, castles
    to their frame), rasterised by rsvg-convert with its longer side at DECAL_TEXTURE_PX."""
    if not shutil.which("rsvg-convert"):
        raise RuntimeError("rsvg-convert is needed to rasterise the decals")
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for design in N.DECAL_DESIGNS:
        png = folder / N.texture_name(design)
        svg = png.with_suffix(".svg")
        svg.write_text(N.design_svg(design))
        _, _, w, h = N.design_view_box(design)
        size = ["-w" if w >= h else "-h", str(DECAL_TEXTURE_PX)]
        subprocess.run(["rsvg-convert", *size, str(svg), "-o", str(png)], check=True)
        spec = (N.CASTLE_DESIGNS | N.ROSE_DESIGNS)[design]
        out[design] = {
            "kind": N.design_kind(design),
            "spec": spec.name,
            "svg": {"path": f"Textures/{svg.name}", "sha256": sha256(svg)},
            "png": {"path": f"Textures/{png.name}", "sha256": sha256(png)},
        }
    return out


def load_decal_textures(bpy, folder: Path) -> None:
    for design in N.DECAL_DESIGNS:
        image = bpy.data.images.load(str(folder / N.texture_name(design)))
        image.name = N.texture_name(design)
        image.colorspace_settings.name = "sRGB"
        image.alpha_mode = "STRAIGHT"


def decal(bpy, name: str, design: str):
    """Canal art painted on the enamel: the design's texture over full transparency, so the
    panel's own paint shows round it, with the paint's gloss and orange peel."""
    g = Graph(bpy, name, (0.6, 0.25, 0.15))
    tex = g.node(
        "ShaderNodeTexImage",
        {"Vector": g.coords.outputs["UV"]},
        image=bpy.data.images[N.texture_name(design)],
        extension="CLIP",  # the slab's edges sample outside 0-1: transparent
        interpolation="Linear",
    )
    g.set(g.bsdf.inputs["Alpha"], tex.outputs["Alpha"])
    peel = g.noise(g.coords.outputs["Object"], 260.0, detail=3.0)
    return g.finish(tex.outputs["Color"], 0.3, g.bump(peel, 0.03, 0.001))


# The weathered look, slot by slot (stages 1-3: base ageing, then the damage on the hull
# and cabin, then the roof, decks and fittings). Slots missing here keep their clean
# material.
WEATHERED = {
    "HullBlack": lambda bpy, n, ctx: WX.aged_hull(bpy, n, ctx),
    "DeckNonSlip": lambda bpy, n, ctx: WX.aged_non_slip(
        bpy, n, (0.032, 0.031, 0.029), ctx
    ),
    "CabinGreen": lambda bpy, n, ctx: WX.aged_paint(bpy, n, (0.008, 0.06, 0.03), ctx),
    # The coachlines are painted over the green, which their chips show first.
    "CoachCream": lambda bpy, n, ctx: WX.aged_paint(
        bpy,
        n,
        (0.56, 0.42, 0.205),
        ctx,
        rough=0.5,
        chalk=0.0,
        under=(0.012, 0.07, 0.036),
        touch=False,
        stain_k=0.55,
    ),
    "CoachRed": lambda bpy, n, ctx: WX.aged_paint(
        bpy,
        n,
        (0.30, 0.05, 0.04),
        ctx,
        rough=0.5,
        chalk=0.6,
        under=(0.012, 0.07, 0.036),
        touch=False,
    ),
    "RoofOxide": lambda bpy, n, ctx: WX.aged_roof(
        bpy, n, (0.16, 0.035, 0.022), (0.30, 0.17, 0.12), 0.22, 0.4, ctx
    ),
    # One tarnish for all the brass, the window frames included.
    "Brass": lambda bpy, n, ctx: WX.aged_metal(
        bpy,
        n,
        (0.30, 0.20, 0.085),
        0.5,
        (0.17, 0.13, 0.06),
        (0.09, 0.065, 0.03),
        hands=True,
        ctx=ctx,
    ),
    "FrameGold": lambda bpy, n, ctx: WX.aged_metal(
        bpy,
        n,
        (0.38, 0.27, 0.12),
        0.42,
        (0.24, 0.18, 0.08),
        (0.12, 0.09, 0.04),
        0.9,
        frame=True,
    ),
    "CastIron": lambda bpy, n, ctx: WX.aged_iron(bpy, n),
    "TillerWood": lambda bpy, n, ctx: WX.aged_timber(bpy, n, (0.30, 0.17, 0.09), 0.35),
    "Glass": lambda bpy, n, ctx: WX.aged_glass(bpy, n),
    "Rope": lambda bpy, n, ctx: WX.aged_rope(bpy, n),
}
WEATHERED.update(
    {
        f"Decal{design}": (lambda bpy, n, ctx, d=design: WX.aged_decal(bpy, n, d, ctx))
        for design in N.DECAL_DESIGNS
    }
)
# Clean colours of the exterior slots, under the false-colour view of the masks.
FALSE_BASE = {
    "HullBlack": (0.012, 0.012, 0.012),
    "DeckNonSlip": (0.02, 0.02, 0.02),
    "CabinGreen": (0.008, 0.06, 0.03),
    "CoachCream": (0.62, 0.5, 0.3),
    "CoachRed": (0.32, 0.015, 0.012),
    "RoofOxide": (0.16, 0.035, 0.022),
    "Brass": (0.95, 0.72, 0.36),
    "FrameGold": (0.82, 0.62, 0.32),
    "CastIron": (0.02, 0.02, 0.02),
    "TillerWood": (0.32, 0.15, 0.06),
    "Rope": (0.28, 0.2, 0.11),
}


def build_materials(bpy, parts, wear="clean", ctx=None, false_colour=False) -> dict:
    used = []
    for part in parts:
        for slot in part.slots:
            if slot not in used:
                used.append(slot)
    if wear == "clean":
        return {slot: MATERIALS[slot](bpy, f"M_NB_{slot}") for slot in used}
    out = {}
    for slot in used:
        name = f"M_NB_{slot}"
        if false_colour and slot.startswith("Decal"):
            design = slot.removeprefix("Decal")
            out[slot] = WX.false_colour(
                bpy, name, (0.6, 0.45, 0.7), ctx, WX.DECAL_USES, design=design
            )
        elif false_colour and slot in FALSE_BASE:
            out[slot] = WX.false_colour(
                bpy,
                name,
                FALSE_BASE[slot],
                ctx,
                WX.USES[slot],
                hull=slot == "HullBlack",
            )
        elif slot in WEATHERED:
            out[slot] = WEATHERED[slot](bpy, name, ctx)
        else:
            out[slot] = MATERIALS[slot](bpy, name)
    return out


# --------------------------------------------------------------------------- objects

HULL_PARTS = (
    "Hull",
    "Counter",
    "RubbingStrakes",
    "RearDeck",
    "EngineBoards",
    "WellDeck",
    "SternGear",
    "Tiller",
    "DeckBrass",
    "Fender",
    "GasCylinders",
    "Machinery",
)
ROOF_PARTS = ("Roof", "RearHatch", "Handrails", "RoofBrass", "Chimney", "PoleAndPlank")
GROUPS = (
    ("CabinPort", lambda n: n.endswith("Port") or n.startswith("LiningPort")),
    ("Roof", lambda n: n in ROOF_PARTS),
    ("Ceiling", lambda n: n.startswith("Ceiling")),
    ("Hull", lambda n: n in HULL_PARTS),
    (
        "Cabin",
        lambda n: (
            n.endswith("Starboard")
            # Decal* here are the end-door decals; the port-side ones matched CabinPort.
            or n.startswith(("LiningStarboard", "Decal"))
            or n in ("Bulkheads", "EndDoors", "DoorLinesCream")
        ),
    ),
    ("Interior", lambda n: True),
)


def group_of(name: str) -> str:
    return next(group for group, test in GROUPS if test(name))


def add_openings(obj, collection) -> None:
    mod = obj.modifiers.new("Openings", "BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.operand_type = "COLLECTION"
    mod.collection = collection
    mod.solver = "EXACT"
    obj.modifiers.move(len(obj.modifiers) - 1, 0)  # cut before bevelling the edges


def flue_ceiling() -> str:
    """The ceiling part the stove flue passes through: that of the room it stands in."""
    zone = next(z for z in I.ZONES if z.b0 <= N.CHIMNEY_B <= z.b1)
    return f"Ceiling{zone.ceiling}"


# Parts pushed in by the dents and the rippled plating of a worn build: the plates, and
# whatever is painted on or welded to them, so it follows the plate.
DENTED = {
    "Hull": DN.HULL,
    "Counter": DN.HULL,
    "RubbingStrakes": DN.HULL,
    "CabinPort": DN.CABIN,
    "CabinStarboard": DN.CABIN,
}
DENT_EDGE = 0.06  # m: longest edge left on a dented plate, so a dish has the vertices


def dent_surface(name: str) -> str | None:
    if name in DENTED:
        return DENTED[name]
    if name.startswith(("CoachlineCream", "CoachlineRed", "DecalCabin")):
        return DN.CABIN
    return None


def dent_object(
    obj, surface: str, items: list, wear, max_edge: float = DENT_EDGE
) -> int:
    """Densify the plate where dents can reach, then push its vertices in by the dent
    field. Inner and outer faces of a plate move together (by position, scaled by how
    far the surface faces sideways), so it keeps its thickness. Returns the vertex count."""
    import bmesh  # Blender's, only there when running inside it
    import numpy as np

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    z_lo = 0.12 if surface == DN.HULL else WM.GUNWALE / 1000 - 0.01
    z_hi = WM.GUNWALE / 1000 if surface == DN.HULL else WM.SIDE_TOP / 1000 + 0.01

    def reachable(e):
        a, b = e.verts[0].co, e.verts[1].co
        z = (a.z + b.z) / 2
        return z_lo <= z <= z_hi and abs(a.y + b.y) / 2 > 0.25

    for _ in range(16):
        long = [e for e in bm.edges if e.calc_length() > max_edge and reachable(e)]
        if not long:
            break
        bmesh.ops.subdivide_edges(bm, edges=long, cuts=1, use_grid_fill=True)
    bm.normal_update()
    bm.verts.ensure_lookup_table()
    co = np.array([v.co[:] for v in bm.verts])
    ny = np.abs(np.array([v.normal.y for v in bm.verts]))
    u = WM.u_wall(1000 * (co[:, 0] + N.X_MID))
    z = 1000 * co[:, 2]
    side = np.where(co[:, 1] >= 0, 1, -1)
    push = np.zeros(len(co))
    for s in (1, -1):
        sel = side == s
        if sel.any():
            push[sel] = DN.depth(wear, items, s, surface, u[sel], z[sel])
    co[:, 1] -= side * push / 1000 * np.clip((ny - 0.2) / 0.5, 0, 1)
    for v, c in zip(bm.verts, co):
        v.co = c
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return len(co)


def use_painted(bpy, objects, prep: Path, textures: Path) -> list:
    """Show the Painter-textured low poly in place of the procedural exterior: hide every
    part the bake prep put in a texture set (the overlays too, they are in the textures
    now), append its <Set>_low objects and give each the set's exported maps."""
    import narrowboat_bakeprep as BP

    for obj in objects:
        if BP.set_of(obj.name) is not None:
            obj.hide_render = obj.hide_viewport = True
    with bpy.data.libraries.load(str(prep / "NB57_bakeprep.blend")) as (src, dst):
        dst.objects = [n for n in src.objects if n.endswith("_low")]
    root = bpy.data.objects["Boat"]
    collection = bpy.data.collections["Hull"]
    lows = []
    for obj in dst.objects:
        collection.objects.link(obj)
        obj.parent = root
        s = obj.name.removesuffix("_low")
        g = Graph(bpy, f"M_NB57_{s}", (0.5, 0.5, 0.5))
        uv = g.coords.outputs["UV"]

        def image(kind, colour=False, s=s, g=g, uv=uv):
            img = bpy.data.images.load(str(textures / f"NB57_{s}_{kind}.png"))
            img.colorspace_settings.name = "sRGB" if colour else "Non-Color"
            return g.node("ShaderNodeTexImage", {"Vector": uv}, image=img).outputs[
                "Color"
            ]

        normal = g.node(
            "ShaderNodeNormalMap", {"Color": image("Normal_OpenGL")}
        ).outputs["Normal"]
        rough = g.separate(image("Roughness"))[0]
        metal = g.separate(image("Metallic"))[0]
        g.finish(image("BaseColor", True), rough, normal, metallic=metal, specular=0.5)
        obj.data.materials.clear()
        obj.data.materials.append(g.mat)
        lows.append(obj)
    return lows


def build_boat(bpy, scene, seed: int, wear="clean", ctx=None, false_colour=False):
    exterior = N.exterior_parts(seed)
    interior, items, lights = I.interior_parts(seed)
    parts = exterior + interior
    materials = build_materials(bpy, parts, wear, ctx, false_colour)
    props = (
        PROPS
        if wear == "clean"
        else PROPS | {"wear": wear, "wear_version": WM.WEAR_VERSION}
    )
    collections = {name: new_collection(bpy, scene, name) for name, _ in GROUPS}
    objects = []
    for part in parts:
        obj = make_object(
            bpy,
            part,
            materials,
            collections[group_of(part.name)],
            part.name,
            props=props,
        )
        if part.name.startswith("Decal"):
            # Paint has no thickness: the decal slab must not bevel, and must not occlude.
            # The paint shader's crevice AO counts shadow-visible geometry even where the
            # decal is transparent, which would grime a dark box round every spray.
            obj.modifiers.remove(obj.modifiers["Bevel"])
            obj.visible_shadow = False
        objects.append(obj)
    if wear != "clean":
        level = WM.WEARS[wear]
        dents = DN.dents(level, seed)
        for obj in objects:
            surface = dent_surface(obj.name)
            if surface is not None:
                dent_object(obj, surface, dents, level)
    cutters = new_collection(bpy, scene, "Cutters")
    groups = {}
    for key, mds in N.cutters().items():
        sub = bpy.data.collections.new(f"Cutters{key.title()}")
        cutters.children.link(sub)
        groups[key] = sub
        for k, md in enumerate(mds):
            mesh = bpy.data.meshes.new(f"Cutter{key.title()}{k}")
            mesh.from_pydata(md.verts, [], md.faces)
            obj = bpy.data.objects.new(mesh.name, mesh)
            obj.display_type = "WIRE"
            obj.hide_render = True
            sub.objects.link(obj)
    by_name = {o.name: o for o in objects}
    for name, obj in by_name.items():
        if name == "CabinPort" or name.startswith("LiningPort"):
            add_openings(obj, groups["port"])
        elif name == "CabinStarboard" or name.startswith("LiningStarboard"):
            add_openings(obj, groups["starboard"])
        elif name in ("Roof", flue_ceiling()):
            add_openings(obj, groups["roof"])
    lamps = new_collection(bpy, scene, "Lights")
    for k, (kind, pos, watts) in enumerate(lights):
        data = bpy.data.lights.new(f"{kind}{k}", "POINT")
        data.energy = watts
        data.color = WARM
        data.shadow_soft_size = 0.03
        light = bpy.data.objects.new(data.name, data)
        light.location = pos
        lamps.objects.link(light)
    # Everything that belongs to the boat hangs off one empty, so the lakeside shots can
    # trim it by the stern while the drawing-matched views keep it level.
    root = bpy.data.objects.new("Boat", None)
    scene.collection.objects.link(root)
    for obj in [*objects, *cutters.all_objects, *lamps.objects]:
        obj.parent = root
    return parts, objects, materials, items, lights


# The sun of the lakeside HDRI the build was tuned with, from the centroid of its brightest pixels: azimuth and elevation
# (degrees) in Blender's world frame before any turn.
HDRI_SUN = (-36.1, 31.7)
# Lakeside shots light the boat from behind the camera, the sun this far round to the side.
SUN_FROM_CAMERA = 40.0
# Interiors take the sun through the starboard windows.
INTERIOR_SUN_AZIMUTH = -90.0
# The design waterline is trimmed by the stern; in the lake the boat is pitched to it.
TRIM = math.atan2(D.DRAUGHT_FWD - D.DRAUGHT_AFT, D.STEM - D.STERN_POST)


def hdri_world(bpy, scene, strength: float = 1.0) -> bool:
    """The lakeside HDRI as the world, behind a Mapping node that turns it about Z."""
    if N.HDRI is None or not N.HDRI.is_file():
        return False
    nt = scene.world.node_tree
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(N.HDRI), check_existing=True)
    coords = nt.nodes.new("ShaderNodeTexCoord")
    turn = nt.nodes.new("ShaderNodeMapping")
    turn.name = "HDRI_Turn"
    nt.links.new(coords.outputs["Generated"], turn.inputs["Vector"])
    nt.links.new(turn.outputs["Vector"], env.inputs["Vector"])
    background = nt.nodes.get("Background")
    nt.links.new(env.outputs["Color"], background.inputs["Color"])
    background.inputs["Strength"].default_value = strength
    return True


def find_sun(bpy, path: Path, share: float = 0.0005) -> tuple[float, float]:
    """Azimuth and elevation (degrees, Blender's world frame) of an HDRI's sun: the
    direction-averaged centroid of its brightest pixels, as for HDRI_SUN."""
    import numpy as np

    image = bpy.data.images.load(str(path), check_existing=True)
    w, h = image.size
    px = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(px)
    lum = px.reshape(h, w, 4)[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    count = max(1, int(lum.size * share))
    rows, cols = np.unravel_index(np.argpartition(lum.ravel(), -count)[-count:], lum.shape)
    # Blender's equirectangular lookup: u = 0.5 - atan2(y, x) / 2pi, v = 0.5 + elevation / pi.
    az = (0.5 - (cols + 0.5) / w) * 2 * np.pi
    el = ((rows + 0.5) / h - 0.5) * np.pi
    weight = lum[rows, cols]
    d = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
    x, y, z = (d * weight[:, None]).sum(axis=0)
    return (
        round(math.degrees(math.atan2(y, x)), 1),
        round(math.degrees(math.atan2(z, math.hypot(x, y))), 1),
    )


def set_sun_azimuth(scene, azimuth_deg: float) -> float:
    """Turn the HDRI so its sun stands at this azimuth; returns the turn in radians. The
    Mapping node rotates the lookup direction, so the sky turns the other way."""
    node = scene.world.node_tree.nodes.get("HDRI_Turn")
    turn = math.radians(HDRI_SUN[0] - azimuth_deg)
    if node is not None:
        node.inputs["Rotation"].default_value = (0.0, 0.0, turn)
    return turn


def sun_behind(spec: dict) -> float:
    """Sun azimuth (degrees) for a camera: behind it, SUN_FROM_CAMERA to one side."""
    (lx, ly, _), (tx, ty, _) = spec["location"], spec["target"]
    if "sun" in spec:
        return spec["sun"]
    return math.degrees(math.atan2(ly - ty, lx - tx)) + SUN_FROM_CAMERA


def lake_mesh(radius: float = 400.0, samples: int = 160):
    """(verts, faces) of the lake: a ring of quads from the hull's waterline outline, just
    inside the plating, out to a far circle, so the water stops at the hull and the cabin
    stays dry. The plan is convex, so every quad runs straight out along its own bearing.
    The 0.5 degree trim moves the outline by millimetres, so the hole needs no updating."""
    bs = [
        N.STERN_TIP + 0.02 + (N.STEM - 0.03 - N.STERN_TIP) * k / (samples - 1)
        for k in range(samples)
    ]
    port = [(N.wx(1000 * b), max(N.plan_half(b) - 0.004, 0.002)) for b in bs]
    # Counter-clockwise seen from above.
    inner = [(x, -y) for x, y in port] + [(x, y) for x, y in reversed(port)]
    outer = [
        (radius * x / math.hypot(x, y), radius * y / math.hypot(x, y)) for x, y in inner
    ]
    n = len(inner)
    verts = [(x, y, 0.0) for x, y in inner] + [(x, y, 0.0) for x, y in outer]
    faces = [(i, n + i, n + (i + 1) % n, (i + 1) % n) for i in range(n)]
    return verts, faces


def lake(bpy, collection):
    """The water at the design waterline amidships (the boat is pitched to it)."""
    verts, faces = lake_mesh()
    mesh = bpy.data.meshes.new("Lake_Water")
    mesh.from_pydata(verts, [], faces)
    mesh.materials.append(water(bpy, "M_NB_LakeWater"))
    obj = bpy.data.objects.new("Lake_Water", mesh)
    obj.location = (
        0.0,
        0.0,
        N.m(D.DRAUGHT_AFT) + math.tan(TRIM) * (N.X_MID - N.STERN_POST),
    )
    obj.hide_render = True
    collection.objects.link(obj)
    return obj


# --------------------------------------------------------------------------- cameras


def ortho_camera(view: str, cal: dict) -> dict:
    """An orthographic camera framing one view of the drawing set at its scale, from the
    view's calibration in the drawings manifest. The plan is looked down on; the elevations
    are seen from outside the boat: starboard and the profile directly, port from its own
    side (the overlay mirrors it back), and B-B, drawn inverted, is flipped to match."""
    x0, y0, x1, y1 = cal["box_px"]
    mpp = cal["mm_per_px"] / 1000
    b = ((x0 + x1) / 2 - cal["b0_px"]) * cal["b_sign"] * mpp
    v = (cal["axis_px"] - (y0 + y1) / 2) * cal["v_sign"] * mpp
    spec = {"ortho": (x1 - x0) * mpp, "resolution": (x1 - x0, y1 - y0)}
    x = W(b)
    if view == "plan":
        spec.update(location=(x, v, 40.0), rotation=(0, 0, 0))
    elif view == "port":
        spec.update(
            location=(x, 40.0, v), rotation=(math.radians(90), 0, math.radians(180))
        )
    else:  # starboard (B-B) and the outboard profile, both from starboard
        spec.update(location=(x, -40.0, v), rotation=(math.radians(90), 0, 0))
    return spec


def W(b: float) -> float:
    return N.wx(1000 * b)


def zone_span(name: str) -> tuple[float, float]:
    z = I.ZONE[name]
    return z.b0, z.b1


def camera_specs(calibration: dict) -> dict:
    """Ortho cameras from the drawing calibration; the rest placed from the design."""
    specs = {
        "plan": ortho_camera("plan", calibration["NB57-GA-001"]["plan"]),
        "elev_port": ortho_camera("port", calibration["NB57-GA-001"]["port"]),
        "elev_starboard": ortho_camera(
            "starboard", calibration["NB57-GA-001"]["starboard"]
        ),
        "profile": ortho_camera("profile", calibration["NB57-EX-002"]["profile"]),
    }
    hatch = N.HATCH.b / 1000
    spray = next(b for d, s, b in D.SPRAYS if d == "CabinCentre" and s > 0) / 1000
    g0, g1 = zone_span("Galley")
    s0, s1 = zone_span("Saloon")
    w0, w1 = zone_span("Bathroom")
    e0, e1 = zone_span("Bedroom")
    hs = N.HATCH_SIDE
    specs.update(
        {
            "hero": {
                "lens": 32,
                "location": (W(N.STEM) + 4.2, -6.8, 3.1),
                "target": (W(9.0), 0.0, 1.05),
                "resolution": (1920, 1080),
            },
            "stern": {
                "lens": 30,
                "location": (W(N.STERN_TIP) - 4.0, 5.6, 3.0),
                "target": (W(3.0), 0.0, 1.15),
                "resolution": (1920, 1080),
            },
            "panels_stern": {
                "lens": 26,
                "location": (W(N.STERN_TIP) + 0.05, -0.3, 2.05),
                "target": (W(0.0), 0.0, 1.5),
                "resolution": (1600, 1000),
            },
            "panels_bow": {
                "lens": 30,
                "location": (W(N.STEM) - 0.4, 1.4, 2.25),
                "target": (W(N.CABIN_LEN), 0.05, 1.45),
                "resolution": (1600, 1000),
            },
            "panels_hatch": {
                "lens": 30,
                "location": (W(hatch - 1.9), hs * 3.3, 1.95),
                "target": (W(hatch), hs * 0.9, 1.5),
                "resolution": (1600, 1000),
            },
            "panels_port": {
                "lens": 30,
                "location": (W(spray), 3.1, 1.75),
                "target": (W(spray), 0.9, 1.56),
                "resolution": (1600, 1000),
            },
            "cutaway": {
                "lens": 28,
                "location": (W(6.4), 8.2, 7.6),
                "target": (W(6.2), -0.2, 0.7),
                "resolution": (2400, 1100),
            },
            "galley": {
                "lens": 16,
                "location": (W(g0 + 0.12), 0.0, 1.62),
                "target": (W(g1), -0.1, 1.15),
                "resolution": (1600, 1000),
            },
            "saloon": {
                "lens": 16,
                "location": (W(s1 - 0.75), 0.1, 1.6),
                "target": (W(s0 + 0.5), -0.1, 1.0),
                "resolution": (1600, 1000),
            },
            "bathroom": {
                "lens": 16,
                # From inside the forward doorway, starboard of the open leaf, looking aft.
                "location": (W(w1 - 0.15), -0.05, 1.62),
                "target": (W(w0 + 0.3), -0.55, 0.9),
                "resolution": (1600, 1000),
            },
            "bedroom": {
                "lens": 16,
                "location": (W(e1 - 0.1), 0.62, 1.65),
                "target": (W(e0 + 0.6), -0.3, 0.85),
                "resolution": (1600, 1000),
            },
        }
    )
    return specs


# --------------------------------------------------------------------------- previews

# Drawing-matched views stay in the grey studio, level, so they overlay the drawing set.
ORTHO = ("elev_starboard", "elev_port", "profile")
# Beauty shots and interiors are taken on the lake: HDRI lit and seen, the boat afloat.
BEAUTY = ("hero", "stern", "panels_stern", "panels_bow", "panels_hatch", "panels_port")
INTERIOR = ("galley", "saloon", "bathroom", "bedroom")
STUDIO_LIGHTS = ("Lookdev_Key", "Lookdev_Fill", "Lookdev_Floor")
# Weathered builds add a top view with the roof on and close-ups of the wear.
WEAR_SHOTS = tuple(WX.wear_camera_specs())
DEFAULT_SHOTS = (*ORTHO, "plan", "cutaway", *BEAUTY, *INTERIOR)
ALL_SHOTS = (*DEFAULT_SHOTS, "roof", *WEAR_SHOTS)
SHOT_GROUPS = {
    "ortho": ORTHO,
    "beauty": BEAUTY,
    "interior": INTERIOR,
    "wear": WEAR_SHOTS,
    "stage1": ("hero", "stern", "panels_port", *ORTHO, "roof"),
    "stage2": (
        "hero",
        "stern",
        "panels_hatch",
        "wear_weld",
        "wear_porthole",
        "wear_counter",
        "wear_bow",
        "wear_doors",
    ),
    "stage3": ("hero", "stern", "panels_stern", "wear_roof", "wear_bow"),
}


def parse_shots(text: str | None, wear: str) -> tuple[str, ...]:
    """Shot names from a comma-separated list of names and groups; by default today's set,
    plus the roof view and the close-ups for a weathered build."""
    if not text:
        return DEFAULT_SHOTS if wear == "clean" else ALL_SHOTS
    out: list[str] = []
    for item in (i.strip() for i in text.split(",")):
        for name in SHOT_GROUPS.get(item, (item,)):
            if name not in ALL_SHOTS:
                raise ValueError(f"unknown shot or group: {name}")
            if name not in out:
                out.append(name)
    return tuple(out)


def default_out(wear: str) -> Path:
    return N.OUTPUT_DIR if wear == "clean" else N.ASSET_ROOT / f"NB57-{wear}"


def blend_name(wear: str) -> str:
    if wear == "clean":
        return N.BLEND_NAME
    return N.BLEND_NAME.removesuffix(".blend") + f"_{wear.title()}.blend"


def check_out(out: Path, wear: str, partial: bool, publish: bool) -> None:
    """Refuse outputs that would overwrite what should stay: the clean build with a
    weathered one, and anything under the build folder from a candidate (weathered, partial
    or false-colour) build without --publish."""
    out = out.resolve()
    if wear != "clean" and out == N.OUTPUT_DIR.resolve():
        raise RuntimeError("a weathered build never replaces the clean build")
    root = N.ASSET_ROOT.resolve()
    inside = out == root or root in out.parents
    if inside and (wear != "clean" or partial) and not publish:
        raise RuntimeError(
            f"{out}: candidate builds go to a scratch folder; pass --publish to write here"
        )


def lakeside(bpy, scene, on: bool, sun_azimuth: float | None = None) -> None:
    """Switch between the grey studio and the lake: studio suns and shadow catcher off,
    the water on, the boat trimmed to its waterline, the HDRI turned for its sun."""
    for name in STUDIO_LIGHTS:
        bpy.data.objects[name].hide_render = on
    bpy.data.objects["Lake_Water"].hide_render = not on
    bpy.data.objects["Boat"].rotation_euler = (0.0, TRIM if on else 0.0, 0.0)
    if on and sun_azimuth is not None:
        set_sun_azimuth(scene, sun_azimuth)


def set_hidden(bpy, names, hidden: bool) -> None:
    for name in names:
        bpy.data.collections[name].hide_render = hidden


# Each overlay: the render, the drawing view it is compared with, and how the render is
# flipped to read like the drawing (port is seen from its own side; B-B is drawn inverted).
OVERLAYS = {
    "nb_plan_overlay.png": (("plan", "NB57-GA-001", "plan", None),),
    "nb_elevations_compare.png": (
        ("elev_port", "NB57-GA-001", "port", "mirror"),
        ("elev_starboard", "NB57-GA-001", "starboard", "flip"),
    ),
    "nb_profile_overlay.png": (("profile", "NB57-EX-002", "profile", None),),
}


def drawing_lines(bpy, reference: Path, cal: dict):
    """The view's lines-only drawing, cropped to the view's box."""
    x0, y0, x1, y1 = cal["box_px"]
    return pixels(bpy, reference / cal["lines_png"])[y0:y1, x0:x1, :3]


def render_previews(
    bpy,
    scene,
    cams: dict,
    folder: Path,
    samples: int,
    reference: Path,
    calibration: dict,
    specs: dict,
    wanted: tuple[str, ...] = DEFAULT_SHOTS,
) -> list[str]:
    import numpy as np

    folder.mkdir(parents=True, exist_ok=True)
    shots = {}

    def shoot(name, transparent, render_samples, hidden=()):
        set_hidden(bpy, hidden, True)
        scene.render.film_transparent = transparent
        cam, resolution = cams[name]
        path = folder / f"nb_{name}.rgba.png"
        render(bpy, scene, cam, resolution, path, render_samples)
        rgba = pixels(bpy, path)
        path.unlink()
        set_hidden(bpy, hidden, False)
        rgb = over_grey(rgba) if transparent else rgba[..., :3]
        save_png(bpy, rgb, folder / f"nb_{name}.png")
        shots[name] = rgb
        return rgb

    lakeside(bpy, scene, False)
    for name in ORTHO:
        if name in wanted:
            shoot(name, True, samples)
    if "plan" in wanted:
        shoot("plan", True, samples, ("Roof", "Ceiling"))
    if "roof" in wanted:
        shoot("roof", True, samples)
    if "cutaway" in wanted:
        shoot("cutaway", True, samples, ("Roof", "Ceiling", "CabinPort"))
    for name in (*BEAUTY, *WEAR_SHOTS):
        if name in wanted:
            lakeside(bpy, scene, True, sun_behind(specs[name]))
            shoot(name, False, samples)
    lakeside(bpy, scene, True, INTERIOR_SUN_AZIMUTH)
    for name in INTERIOR:
        if name in wanted:
            shoot(name, False, samples * 2)
    lakeside(bpy, scene, False)

    def stack(rows):
        """Rows one above the other, padded with white to the widest."""
        width = max(r.shape[1] for r in rows)
        padded = [
            np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=1.0)
            for r in rows
        ]
        return np.concatenate(padded, axis=0)

    for name, rows in OVERLAYS.items():
        if not all(shot in wanted for shot, _, _, _ in rows):
            continue
        out = []
        for shot, sheet, view, flip in rows:
            lines = drawing_lines(bpy, reference, calibration[sheet][view])
            rgb = shots[shot]
            if flip == "mirror":
                rgb = rgb[:, ::-1]
            elif flip == "flip":
                rgb = rgb[::-1]
            out += [lines, rgb, red_ink(np, rgb, lines)]
        save_png(bpy, stack(out), folder / name)
    return sorted(p.name for p in folder.iterdir())


def red_ink(np, render_rgb, lines):
    """The render with the drawing's linework laid over it in red."""
    ink = 1 - np.clip(lines.mean(axis=2, keepdims=True), 0, 1)
    ink = np.clip(ink * 1.6, 0, 1)
    red = np.array([0.9, 0.08, 0.05], dtype=np.float32)
    return render_rgb * (1 - ink) + red * ink


# --------------------------------------------------------------------------- main


def opening_table() -> list[dict]:
    return [
        {
            "name": op.name,
            "side": "port" if op.side > 0 else "starboard",
            "centre_mm_from_rear_bulkhead": round(op.centre_mm, 1),
            "centre_z_m": round(op.zc, 4),
            "width_m": round(op.width, 4),
            "height_m": round(op.height, 4),
            "kind": op.kind,
        }
        for op in N.OPENINGS
    ]


def decal_table() -> list[dict]:
    return [
        {
            "name": p.name,
            "design": p.design,
            "kind": N.design_kind(p.design),
            "part": p.part,
            "width_m": round(p.width, 4),
            "height_m": round(p.height, 4),
            "centre_m": [round(c, 4) for c in N.panel_centre(p)],
            "mirrored": p.mirror,
        }
        for p in N.decal_panels()
    ]


def load_drawings(folder: Path) -> dict:
    """The drawing set's manifest, refusing drawings made from a different design."""
    import narrowboat_blueprint as BP

    path = folder / "drawings_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"drawing set missing: {path} (run narrowboat_blueprint.py)"
        )
    drawings = json.loads(path.read_text())
    if drawings["design_sha256"] != BP.design_sha256():
        raise RuntimeError(
            "the drawing set is out of date with narrowboat_design; rebuild it"
        )
    return drawings


def build(bpy, args) -> dict:
    wear = getattr(args, "wear", "clean")
    shots = getattr(args, "shots", DEFAULT_SHOTS)
    false_colour = getattr(args, "false_colour", False)
    out = Path(args.out) if args.out else default_out(wear)
    check_out(
        out,
        wear,
        shots != DEFAULT_SHOTS or false_colour,
        getattr(args, "publish", False),
    )
    drawings = load_drawings(N.DRAWINGS)
    staging = out.with_name(out.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    reference = staging / "Reference"
    reference.mkdir(parents=True)
    for name in [*drawings["files"], "drawings_manifest.json"]:
        shutil.copy2(N.DRAWINGS / name, reference / name)
    # Textures go straight into staging: the .blend saved beside them links them relatively.
    textures = write_decal_textures(staging / "Textures")
    wear_record = None
    if wear != "clean":
        level = WM.WEARS[wear]
        layers = WM.layers(level, args.seed)
        wear_record = WM.write_maps(
            staging / "Textures" / "WearMaps", level, layers, WM.wear_maps(layers)
        )
        del layers
    scene = fresh_scene(bpy)
    load_decal_textures(bpy, staging / "Textures")
    ctx = None
    if wear != "clean":
        ctx = WX.load_maps(
            bpy, staging / "Textures" / "WearMaps", wear_record, WM.WEARS[wear]
        )
    parts, objects, materials, items, lights = build_boat(
        bpy, scene, args.seed, wear, ctx, false_colour
    )
    if getattr(args, "painted", None):
        prep, tex = (Path(p) for p in args.painted)
        use_painted(bpy, objects, prep, tex)
    render_info = lookdev(
        bpy, scene, new_collection(bpy, scene, "Lookdev"), args.samples
    )
    if getattr(args, "device", "auto") == "cpu":
        # For when another program holds the GPU's memory.
        scene.cycles.device = "CPU"
        render_info["device"] = "CPU"
    render_info["hdri"] = str(N.HDRI) if hdri_world(bpy, scene) else None
    render_info["lakeside"] = {
        "shots": [*BEAUTY, *INTERIOR],
        "hdri_sun_deg": HDRI_SUN,
        "trim_deg": round(math.degrees(TRIM), 3),
    }
    lake(bpy, bpy.data.collections["Lookdev"])
    calibration = drawings["calibration"]
    specs = camera_specs(calibration)
    if wear != "clean" or set(shots) & {"roof", *WEAR_SHOTS}:
        # The roof view and the close-ups, for a worn build or to compare against one.
        specs["roof"] = dict(specs["plan"])
        # A clean build aims them where the weathered one finds its wear, to compare.
        level = WM.WEARS.get(wear, WM.WEARS["weathered"])
        specs.update(WX.wear_camera_specs(WM.camera_aims(level, args.seed)))
    cams = make_cameras(bpy, bpy.data.collections["Lookdev"], specs)
    # Perspective cameras ride with the boat, so its trim on the lake keeps their framing.
    for name, (cam, _) in cams.items():
        if "ortho" not in specs[name]:
            cam.parent = bpy.data.objects["Boat"]
    hero, hero_res = cams["hero"]
    scene.camera = hero
    scene.render.resolution_x, scene.render.resolution_y = hero_res
    parts_receipts = receipts(bpy, parts, objects, materials)
    evaluated = {r["name"]: r["triangles_rendered"] for r in parts_receipts}

    blend = staging / blend_name(wear)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend), compress=True, relative_remap=True)
    if args.no_render:
        previews = keep_previews(out, staging)
    else:
        previews = render_previews(
            bpy,
            scene,
            cams,
            staging / "Previews",
            args.samples,
            reference,
            calibration,
            specs,
            shots,
        )
    scripts = Path(__file__).resolve().parent
    manifest = {
        "asset": f"Narrowboat/{D.DESIGN_ID}",
        "design": f"{D.DESIGN_ID} {D.NAME}",
        "build_version": N.BUILD_VERSION,
        "seed": args.seed,
        "blender_version": bpy.app.version_string,
        "units": "metres; +X to the bow, +Y to port, Z up; z = 0 under the baseplate; "
        f"origin at mid-hull (rear cabin bulkhead at x = {N.wx(0):.3f})",
        "drawings": {
            "path": "Reference/",
            "sheets": drawings["sheets"],
            "design_sha256": drawings["design_sha256"],
            "files": drawings["files"],
        },
        "generator": {
            name: sha256(scripts / name)
            for name in (
                "narrowboat_build.py",
                "narrowboat_common.py",
                "narrowboat_interior.py",
                "narrowboat_design.py",
                "narrowboat_lines.py",
                "canal_roses.py",
                "canal_castles.py",
                "procgen_mesh.py",
                "procgen_bpy.py",
            )
        },
        "blend": {"path": blend.name, "sha256": sha256(blend)},
        "dimensions_m": {
            "length_over_fenders": D.LENGTH_OVER_FENDERS / 1000,
            "hull_length": round(N.LOA, 3),
            "beam_over_strakes": D.BEAM_OVER_STRAKES / 1000,
            "cabin": N.CABIN_LEN,
            "roof_crown": N.Z_ROOF_CROWN,
            "headroom_centreline": D.HEADROOM / 1000,
        },
        "openings": opening_table(),
        "decal_panels": decal_table(),
        "decal_textures": textures,
        "interior_items": [
            {
                "name": n,
                "bounds_min_m": [round(c, 3) for c in b[0]],
                "bounds_max_m": [round(c, 3) for c in b[1]],
            }
            for n, b in items
        ],
        "lights": [
            {"kind": k, "location_m": [round(c, 3) for c in p], "watts": w}
            for k, p, w in lights
        ],
        "render": render_info,
        "parts": parts_receipts,
        "previews": previews,
    }
    if wear != "clean":
        manifest["wear"] = {
            **wear_record,
            "false_colour": false_colour,
            "shots": list(shots),
            "weathered_slots": sorted(s for s in materials if s in WEATHERED),
            "clean_slots": sorted(s for s in materials if s not in WEATHERED),
            "dents": DN.summary(DN.dents(WM.WEARS[wear], args.seed)),
        }
        for name in (
            "narrowboat_dents.py",
            "narrowboat_wear.py",
            "narrowboat_weathered.py",
            "procgen_masks.py",
        ):
            manifest["generator"][name] = sha256(scripts / name)
    write_json(staging / "narrowboat_manifest.json", manifest)
    publish(staging, out)
    return {"manifest": manifest, "triangles": evaluated}


def main() -> None:
    import bpy

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=N.SEED)
    parser.add_argument(
        "--out", default=None, help="default: build/NB57, or build/NB57-<wear> for a worn build"
    )
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument(
        "--hdri",
        default=None,
        help="equirectangular .hdr/.exr for the lakeside shots, or blender:NAME for one of "
        "the CC0 worlds Blender ships (forest, sunset, courtyard, ...); default: studio",
    )
    parser.add_argument(
        "--hdri-sun",
        type=float,
        default=None,
        metavar="AZIMUTH",
        help="azimuth in degrees of the HDRI's sun (default: found from its brightest pixels)",
    )
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--wear", choices=WM.LEVELS, default="clean")
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument(
        "--painted",
        nargs=2,
        metavar=("PREP", "TEXTURES"),
        help="render the Painter-textured low poly (narrowboat_bakeprep output and the "
        "exported maps) in place of the procedural exterior",
    )
    parser.add_argument(
        "--shots", default=None, help="comma-separated shots or groups to render"
    )
    parser.add_argument(
        "--false-colour",
        action="store_true",
        help="show the wear masks in flat false colour (placement review)",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="allow a worn, partial or false-colour build under the asset folder",
    )
    args = parser.parse_args(
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    )
    if args.false_colour and args.wear == "clean":
        parser.error("--false-colour needs --wear")
    args.shots = parse_shots(args.shots, args.wear)
    if args.hdri is not None:
        hdri = Path(args.hdri)
        if args.hdri.startswith("blender:"):
            name = args.hdri.removeprefix("blender:")
            # system_resource only resolves folders, so look the file up inside one.
            worlds = bpy.utils.system_resource("DATAFILES", path="studiolights/world")
            hdri = Path(worlds) / f"{name}.exr"
        if not hdri.is_file():
            parser.error(f"--hdri: no such file {hdri}")
        N.HDRI = hdri.resolve()
    global HDRI_SUN
    if args.hdri_sun is not None:
        HDRI_SUN = (args.hdri_sun, HDRI_SUN[1])
    elif N.HDRI is not None:
        HDRI_SUN = find_sun(bpy, N.HDRI)
        print(f"NB_HDRI_SUN {HDRI_SUN}")
    if not bpy.app.background:
        raise RuntimeError("Run in a background Blender process")
    if bpy.data.filepath:
        raise RuntimeError(
            "Refusing to run with an opened .blend; use --factory-startup"
        )
    result = build(bpy, args)
    print("NB_PARTS " + json.dumps(result["triangles"]))
    print("NB_TRIANGLES " + str(sum(result["triangles"].values())))


if __name__ == "__main__":
    main()
