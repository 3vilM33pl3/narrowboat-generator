#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Texture the weathered NB57 'Wren' in Substance Painter, by its remote scripting API.

python3 src/narrowboat_painter.py all --prep DIR --out DIR [--size 4096]
python3 src/narrowboat_painter.py {create,bake,masks,layers,export,save} ...

Painter must be running with --enable-remote-scripting (port 60041). The script never
closes a project it did not make: an open project elsewhere, or with unsaved changes, stops
it. DIR --prep is a narrowboat_bakeprep.py output.

1. create: a project from Mesh/NB57_low.fbx, one texture set per material, OpenGL normals.
2. bake: normal, AO, curvature, world normal, position, thickness and height from
   Mesh/NB57_high.fbx, matched by name (<Set>_low / <Set>_high); exported to bakes/.
3. masks: the final greyscale masks per set (masks/), from the bake prep's wear
   potentials thresholded as the Blender shaders do, plus edge wear from the curvature
   bake and grime from the AO bake.
4. layers: one stack per set: the livery, then grime, stains, verdigris, chips through to
   primer, rust, walked polish, algae, moss, soot and standing water, each a fill with
   colour, roughness, metallic and height under its mask.
5. export: BaseColor, Normal (OpenGL and DirectX), Height, Roughness, Metallic and AO per
   set into textures/.
6. save: Painter/NB57_Weathered.spp.
"""

from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

URL = "http://127.0.0.1:60041/run.json"
SETS = ("Hull", "Cabin", "Ends", "Roof", "Decks", "Fittings")
PROJECT = "NB57_Weathered.spp"
BAKERS = (
    "Normal",
    "AO",
    "Curvature",
    "WorldSpaceNormal",
    "Position",
    "Thickness",
    "Height",
)
EPS = 0.025  # half width of a threshold's ramp, as in narrowboat_weathered
# Rust shows only this far above the shaders' threshold, and stains are this strong:
# the user asked twice for about 30% less rust than the first Painter pass (2026-10-18). Where rust
# drops out the paint is still chipped, so the wear stays; it shows grey primer instead
# (a red-oxide primer read as more rust).
RUST_LIFT = 0.15
STAIN = 0.5


def _srgb(linear):
    """Linear RGB (the Blender shaders' constants) to the sRGB Painter's colours take."""
    return tuple(
        round(12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055, 4)
        for c in linear
    )


# Each wear layer: (name, mask, base colour (linear, as in narrowboat_weathered) or
# None, roughness, metallic, height). Bottom to top; a None channel is left alone.
LAYERS = (
    ("02_Chalk", "chalk", (0.26, 0.27, 0.25), 0.62, None, None),
    ("03_RoofFade", "fade", (0.30, 0.17, 0.12), 0.9, None, None),
    ("04_Grime", "dirt", (0.075, 0.08, 0.07), 0.75, None, None),
    ("05_Cavity", "cavity", (0.012, 0.014, 0.010), 0.8, None, None),
    ("06_WaterlineScum", "scum", (0.075, 0.07, 0.055), 0.75, 0.0, None),
    ("07_WaterlineAlgae", "band", (0.035, 0.045, 0.02), 0.55, 0.0, 0.05),
    ("08_SlimeLip", "slime", (0.022, 0.03, 0.012), 0.25, 0.0, 0.1),
    ("09_RustStain", "stain", (0.16, 0.07, 0.025), 0.6, None, None),
    ("10_Verdigris", "verd", (0.02, 0.12, 0.10), 0.7, 0.0, None),
    ("11_EdgeWear", "edge", (0.12, 0.12, 0.11), 0.55, 0.0, -0.04),
    ("12_Chips", "chip", (0.12, 0.12, 0.11), 0.6, 0.0, -0.12),
    ("13_Rust", "rust", (0.13, 0.05, 0.016), 0.9, 0.0, 0.12),
    ("14_RustPits", "pits", (0.028, 0.015, 0.008), 0.95, 0.0, -0.08),
    ("15_Polish", "polish", (0.03, 0.03, 0.028), 0.35, None, None),
    ("16_Algae", "algae", (0.035, 0.05, 0.02), 0.4, 0.0, None),
    ("17_Moss", "moss", (0.045, 0.085, 0.018), 0.95, 0.0, 0.6),
    ("18_Soot", "soot", (0.012, 0.011, 0.010), 0.85, 0.0, None),
    ("19_StandingWater", "pool", (0.09, 0.045, 0.02), 0.3, 0.0, None),
)


# --------------------------------------------------------------------------- remote


def remote(code: str):
    body = json.dumps(
        {"python": base64.b64encode(f"eval({code!r}, globals())".encode()).decode()}
    ).encode()
    request = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    result = json.loads(urllib.request.urlopen(request, timeout=900).read())
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(result["error"])
    return result


def execute(code: str):
    return remote(f"exec({code!r}, globals())")


def wait_idle(timeout: float = 1800) -> None:
    start = time.monotonic()
    while remote("project.is_busy()"):
        if time.monotonic() - start > timeout:
            raise TimeoutError("Painter stayed busy")
        time.sleep(2)


def boot() -> None:
    execute(
        "import substance_painter as sp\n"
        "import substance_painter.project as project\n"
        "from pathlib import Path"
    )


# --------------------------------------------------------------------------- steps


def create(prep: Path, out: Path, size: int) -> None:
    if remote("project.is_open()"):
        path = remote("project.file_path()")
        if not path or Path(path).resolve() != (out / "Painter" / PROJECT).resolve():
            raise RuntimeError("Another project is open in Painter; leaving it alone")
        if remote("project.needs_saving()"):
            raise RuntimeError("The open project has unsaved changes; save it first")
        execute("project.close()")
    if (out / "Painter" / PROJECT).exists():
        raise FileExistsError(f"{out / 'Painter' / PROJECT} exists; use a new --out")
    low = prep / "Mesh" / "NB57_low.fbx"
    log2 = int(size).bit_length() - 1
    execute(
        f"project.create(mesh_file_path={str(low)!r}, settings=project.Settings("
        "normal_map_format=project.NormalMapFormat.OpenGL, "
        f"default_texture_resolution={size}, import_cameras=False))"
    )
    wait_idle()
    names = remote("[t.name() for t in sp.textureset.all_texture_sets()]")
    missing = [s for s in SETS if f"NB57_{s}" not in names]
    if missing:
        raise RuntimeError(f"texture sets missing: {missing} (have {names})")
    (out / "Painter").mkdir(parents=True, exist_ok=True)
    execute(f"project.save_as({str(out / 'Painter' / PROJECT)!r})")
    wait_idle()
    print(json.dumps({"log2": log2, "texture_sets": names}))


def bake(prep: Path, out: Path, size: int, distance: float, sets=SETS) -> dict:
    high = (prep / "Mesh" / "NB57_high.fbx").as_uri()
    log2 = int(size).bit_length() - 1
    execute(
        "import substance_painter.event as events\n"
        "NB_BAKE = {}\n"
        "def nb_done(event):\n"
        "    NB_BAKE['status'] = event.status.name\n"
        "events.DISPATCHER.connect_strong(events.BakingProcessEnded, nb_done)\n"
    )
    report = {}
    for s in sets:
        execute(
            f"ts = sp.textureset.TextureSet.from_name('NB57_{s}')\n"
            "bp = sp.baking.BakingParameters.from_texture_set(ts)\n"
            "cp = bp.common()\n"
            "sp.baking.BakingParameters.set({\n"
            f"    cp['HipolyMesh']: {high!r},\n"
            "    cp['FilterMethod']: cp['FilterMethod'].enum_value('By mesh name'),\n"
            "    cp['LowpolySuffix']: '_low', cp['HipolySuffix']: '_high',\n"
            f"    cp['OutputSize']: ({log2}, {log2}),\n"
            "    cp['SubSampling']: cp['SubSampling'].enum_value('Subsampling 2x2'),\n"
            "    cp['RelativeScale']: False,\n"
            f"    cp['MaxHeight']: {distance}, cp['MaxDepth']: {distance},\n"
            "})\n"
            "bp.set_enabled_bakers([getattr(sp.textureset.MeshMapUsage, u) for u in "
            f"{list(BAKERS)!r}])\n"
            "NB_BAKE['status'] = 'running'\n"
            "NB_STOP = sp.baking.bake_async(ts)\n"
        )
        start = time.monotonic()
        while remote("NB_BAKE['status']") == "running":
            if time.monotonic() - start > 3600:
                raise TimeoutError(f"bake of {s} did not finish")
            time.sleep(3)
        status = remote("NB_BAKE['status']")
        if status != "Success":
            raise RuntimeError(f"bake of {s}: {status}")
        wait_idle()
        report[s] = status
    execute("sp.ui.switch_to_mode(sp.ui.UIMode.Edition)")
    for s in sets:
        export_maps(out / "bakes", f"NB57_{s}", bakes=True, size=size)
    execute("project.save()")
    wait_idle()
    (out / "bakes" / "bake_report.json").write_text(json.dumps(report, indent=1))
    return report


def _spec(name, source, kind, gray=False):
    channels = ["L"] if gray else list("RGB")
    return {
        "fileName": "$textureSet_" + name,
        "channels": [
            {
                "destChannel": c,
                "srcChannel": c,
                "srcMapType": kind,
                "srcMapName": source,
            }
            for c in channels
        ],
    }


def export_maps(folder: Path, ts: str, bakes: bool, size: int) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    if bakes:
        maps = [
            _spec(n, s, "meshMap", g)
            for n, s, g in (
                ("BakeNormal", "normal_base", False),
                ("BakeAO", "ambient_occlusion", True),
                ("BakeCurvature", "curvature", True),
                ("BakeWorldNormal", "world_space_normals", False),
                ("BakePosition", "position", False),
                ("BakeThickness", "thickness", True),
                ("BakeHeight", "height", True),
            )
        ]
    else:
        maps = [
            _spec("BaseColor", "baseColor", "documentMap"),
            _spec("Normal_OpenGL", "Normal_OpenGL", "virtualMap"),
            _spec("Normal_DirectX", "Normal_DirectX", "virtualMap"),
            _spec("Height", "height", "documentMap", True),
            _spec("Roughness", "roughness", "documentMap", True),
            _spec("Metallic", "metallic", "documentMap", True),
            _spec("AO", "AO_Mixed", "virtualMap", True),
        ]
    log2 = int(size).bit_length() - 1
    config = {
        "exportPath": str(folder),
        "exportShaderParams": False,
        "defaultExportPreset": "NB57",
        "exportPresets": [{"name": "NB57", "maps": maps}],
        "exportList": [{"rootPath": ts}],
        "exportParameters": [
            {
                "parameters": {
                    "fileFormat": "png",
                    "bitDepth": "16" if bakes else "8",
                    "sizeLog2": log2,
                    "paddingAlgorithm": "infinite",
                    "dithering": False,
                }
            },
            {
                "filter": {"outputMaps": ["$textureSet_Height"]},
                "parameters": {"bitDepth": "16"},
            },
        ],
    }
    execute(f"NB_EXPORT = sp.export.export_project_textures({config!r})")
    result = remote(
        "{'status': NB_EXPORT.status.name, 'message': NB_EXPORT.message, "
        "'textures': {str(k): v for k, v in NB_EXPORT.textures.items()}}"
    )
    if result["status"] != "Success":
        raise RuntimeError(result)
    return result


# --------------------------------------------------------------------------- masks


def _read(path: Path) -> np.ndarray:
    a = np.asarray(Image.open(path))
    scale = 65535.0 if a.dtype == np.uint16 or a.max() > 255 else 255.0
    return a.astype(np.float32) / scale


def _smooth(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def _noise(shape, cell, seed):
    """Smooth value noise (0-1) with features about `cell` pixels across."""
    rng = np.random.default_rng(seed)
    gh, gw = shape[0] // cell + 2, shape[1] // cell + 2
    grid = rng.random((gh, gw)).astype(np.float32)
    small = Image.fromarray(grid)
    return np.asarray(small.resize((gw * cell, gh * cell), Image.BICUBIC), np.float32)[
        : shape[0], : shape[1]
    ]


def masks(prep: Path, out: Path, size: int) -> dict:
    """The final masks per set, white = the layer shows."""
    manifest = json.loads((prep / "bakeprep_manifest.json").read_text())
    t = manifest["threshold"]
    th = manifest["thresholds"]
    folder = out / "masks"
    folder.mkdir(parents=True, exist_ok=True)
    coverage = {}
    for k, s in enumerate(SETS):
        a = _read(prep / "Textures" / f"{s}_MaskA.png")
        b = _read(prep / "Textures" / f"{s}_MaskB.png")
        c = _read(prep / "Textures" / f"{s}_MaskC.png")
        d = _read(prep / "Textures" / f"{s}_MaskD.png")
        surf = _read(prep / "Textures" / f"{s}_Surface.png")
        shape = a.shape[:2]
        if shape != (size, size):
            raise ValueError(f"{s}: masks are {shape}, project is {size}")
        curv = _read(out / "bakes" / f"NB57_{s}_BakeCurvature.png")
        ao = _read(out / "bakes" / f"NB57_{s}_BakeAO.png")
        curv = curv if curv.ndim == 2 else curv[..., 0]
        ao = ao if ao.ndim == 2 else ao[..., 0]
        # The shaders' ragged threshold edges, at about 15 mm and 3 mm.
        jit = 0.4 * (_noise(shape, 24, k) - 0.5) + 0.1 * (
            _noise(shape, 5, 50 + k) - 0.5
        )
        painted = surf[..., 0] < 0.5  # not metal
        rust = _smooth(a[..., 0] + jit, t + RUST_LIFT - EPS, t + RUST_LIFT + EPS)
        chip = _smooth(a[..., 1] + jit, t - EPS, t + EPS) * (1 - rust)
        out_masks = {
            "rust": rust,
            "pits": rust
            * _smooth(_noise(shape, 14, 100 + k), 0.55, 0.7)
            * _smooth(a[..., 0] - t, 0.08, 0.25),
            "chip": chip,
            "stain": np.clip(a[..., 2] * 1.1 * STAIN, 0, 0.9 * STAIN),
            "dirt": np.clip(b[..., 0] * 0.85, 0, 1),
            "moss": _smooth(b[..., 1] + 0.5 * jit, th["moss"] - EPS, th["moss"] + EPS),
            "algae": 0.7
            * _smooth(b[..., 1] + 0.5 * jit, th["algae"] - EPS, th["algae"] + EPS),
            "soot": np.clip(b[..., 2] * 1.3, 0, 0.85),
            "pool": 0.55 * _smooth(c[..., 0] + 0.3 * jit, 0.25, 0.6),
            "polish": 0.6 * _smooth(c[..., 1], 0.05, 0.35),
            "verd": np.clip(c[..., 2] * 0.7, 0, 0.75),
            # Paint worn off sharp convex edges, broken up so it is not a ruled line.
            "edge": painted
            * _smooth(curv, 0.58, 0.7)
            * _smooth(_noise(shape, 10, 200 + k) + 0.3 * jit, 0.45, 0.6),
            # Dirt settled in the corners and joints.
            "cavity": 0.45 * _smooth(1 - ao, 0.45, 0.85),
            "chalk": painted * np.clip(d[..., 1], 0, 1) * (s not in ("Hull", "Roof")),
            "fade": np.clip(d[..., 2], 0, 1) * (s == "Roof"),
        }
        # The waterline, from the height above it (-0.15..0.35 m as 0..1), on the hull:
        # a scummy band whose top wanders, green-black algae low in it, a wet slime lip
        # at the water.
        zero = np.zeros(shape, np.float32)
        out_masks.update(scum=zero, band=zero, slime=zero)
        if s == "Hull":
            h = d[..., 0] * 0.5 - 0.15
            top = (
                0.12
                + 0.05 * (_noise(shape, 120, 300) - 0.5)
                + 0.015 * (_noise(shape, 12, 301) - 0.5)
            )
            band = _smooth(h, -0.004, 0.004) * (1 - _smooth(h - top, -0.014, 0.004))
            lip = _smooth(h, -0.065, -0.05) * (1 - _smooth(h, 0.015, 0.03))
            out_masks.update(
                scum=0.8 * band * _smooth(h, 0.0, 0.07),
                band=0.75 * band * (1 - _smooth(h, 0.0, 0.07)),
                slime=lip * _smooth(_noise(shape, 30, 302), 0.38, 0.52),
            )
        # Rope, timber and brass do not flake or rust like painted steel: only surfaces
        # that are neither metal nor very rough (rope) take the paint damage.
        steel = painted & (surf[..., 1] < 0.85)
        for k_ in ("rust", "pits", "chip", "edge"):
            out_masks[k_] = out_masks[k_] * (steel if s == "Fittings" else painted)
        # Nothing grows on the brass, the rims or the chimney.
        for k_ in ("moss", "algae"):
            out_masks[k_] = out_masks[k_] * painted
        coverage[s] = {}
        for name, m in out_masks.items():
            m = np.clip(m, 0, 1)
            Image.fromarray(np.round(m * 255).astype(np.uint8), "L").save(
                folder / f"{s}_{name}.png"
            )
            coverage[s][name] = round(float((m > 0.5).mean()), 4)
    (folder / "coverage.json").write_text(json.dumps(coverage, indent=1))
    return coverage


# --------------------------------------------------------------------------- layers

REMOTE_LAYERS = r"""
CT = sp.textureset.ChannelType
Color = sp.colormanagement.Color
NB_RES = {}
def nb_resource(path):
    if path not in NB_RES:
        NB_RES[path] = sp.resource.import_project_resource(
            path, sp.resource.Usage.TEXTURE).identifier()
    return NB_RES[path]
def nb_source(value, colour):
    if colour:
        return Color(*value, sp.colormanagement.GenericColorSpace.sRGB)
    return Color(value, value, value, sp.colormanagement.GenericColorSpace.Raw)
def nb_fill(stack, name, color=None, rough=None, metal=None, height=None,
            mask=None, bitmap=None, rough_map=None, metal_map=None):
    layer = sp.layerstack.insert_fill(sp.layerstack.InsertPosition.from_textureset_stack(stack))
    layer.set_name(name)
    chans = {CT.BaseColor: color if color is not None else bitmap,
             CT.Roughness: rough if rough is not None else rough_map,
             CT.Metallic: metal if metal is not None else metal_map,
             CT.Height: height}
    layer.active_channels = {c for c, v in chans.items() if v is not None}
    if bitmap is not None:
        layer.set_source(CT.BaseColor, nb_resource(bitmap))
    elif color is not None:
        layer.set_source(CT.BaseColor, nb_source(color, True))
    for ch, value, map_ in ((CT.Roughness, rough, rough_map), (CT.Metallic, metal, metal_map)):
        if map_ is not None:
            layer.set_source(ch, nb_resource(map_))
        elif value is not None:
            layer.set_source(ch, nb_source(value, False))
    if height is not None:
        layer.set_source(CT.Height, nb_source(height, False))
    if mask is not None:
        layer.add_mask(sp.layerstack.MaskBackground.Black)
        fill = sp.layerstack.insert_fill(
            sp.layerstack.InsertPosition.inside_node(layer, sp.layerstack.NodeStack.Mask))
        fill.set_name('Mask')
        fill.set_source(None, nb_resource(mask))
    return layer
"""


def layers(prep: Path, out: Path) -> None:
    execute(REMOTE_LAYERS)
    split = out / "masks" / "surface"
    split.mkdir(parents=True, exist_ok=True)
    for s in SETS:
        surf = _read(prep / "Textures" / f"{s}_Surface.png")
        for i, name in ((0, "metal"), (1, "rough")):
            Image.fromarray(np.round(surf[..., i] * 255).astype(np.uint8), "L").save(
                split / f"{s}_{name}.png"
            )
        livery = str(prep / "Textures" / f"{s}_Livery.png")
        code = [
            f"stack = sp.textureset.TextureSet.from_name('NB57_{s}').get_stack()",
            "for node in sp.layerstack.get_root_layer_nodes(stack):",
            "    sp.layerstack.delete_node(node)",
            (
                "for ch, fmt in ((CT.Height, sp.textureset.ChannelFormat.L16),"
                " (CT.Roughness, sp.textureset.ChannelFormat.L8),"
                " (CT.Metallic, sp.textureset.ChannelFormat.L8)):"
            ),
            "    if not stack.has_channel(ch):",
            "        stack.add_channel(ch, fmt)",
            (
                f"nb_fill(stack, '01_Livery', bitmap={livery!r}, height=0.0,"
                f" rough_map={str(split / f'{s}_rough.png')!r},"
                f" metal_map={str(split / f'{s}_metal.png')!r})"
            ),
        ]
        for name, mask, colour, rough, metal, height in LAYERS:
            path = out / "masks" / f"{s}_{mask}.png"
            colour = _srgb(colour) if colour is not None else None
            code.append(
                f"nb_fill(stack, {name!r}, color={colour!r}, rough={rough!r},"
                f" metal={metal!r}, height={height!r}, mask={str(path)!r})"
            )
        execute("\n".join(code))
        wait_idle()
    execute("project.save()")
    wait_idle()


def export(out: Path, size: int) -> dict:
    results = {}
    for s in SETS:
        results[s] = export_maps(out / "textures", f"NB57_{s}", bakes=False, size=size)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step", choices=("all", "create", "bake", "masks", "layers", "export")
    )
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--distance", type=float, default=3.0)  # cm, Painter's unit
    parser.add_argument("--sets", default=",".join(SETS))
    args = parser.parse_args()
    steps = (
        ("create", "bake", "masks", "layers", "export")
        if args.step == "all"
        else (args.step,)
    )
    if any(s != "masks" for s in steps):
        boot()
    for step in steps:
        print(f"== {step}", flush=True)
        if step == "create":
            create(args.prep, args.out, args.size)
        elif step == "bake":
            sets = tuple(args.sets.split(","))
            print(json.dumps(bake(args.prep, args.out, args.size, args.distance, sets)))
        elif step == "masks":
            print(json.dumps(masks(args.prep, args.out, args.size)))
        elif step == "layers":
            layers(args.prep, args.out)
        elif step == "export":
            print(json.dumps(list(export(args.out, args.size))))


if __name__ == "__main__":
    main()
