# SPDX-License-Identifier: MIT
"""Low and high poly of the weathered NB57 'Wren' for Substance Painter, with the maps
Painter starts from.

blender --background --factory-startup --python-exit-code 1 \\
    --python src/narrowboat_bakeprep.py -- --out DIR [--size 4096] [--seed 1]

For each texture set (Hull, Cabin, Ends, Roof, Decks, Fittings) it writes:
- `Mesh/NB57_low.fbx`: the clean exterior, one object `<Set>_low` per set with its
  own material, cut into strips along the boat and unwrapped without overlaps into 0-1.
  The paint overlays (coachlines, door lines, canal art) are not in it: they are colour.
- `Mesh/NB57_high.fbx`: the same parts plus the overlays, dented and densified
  (narrowboat_dents), one object `<Set>_high` per set, for Painter's bakers.
- `Textures/<Set>_Livery.png`: the clean paint, coachlines and canal art, baked from the
  high poly onto the low's UVs (Painter's base colour starts from it).
- `Textures/<Set>_Surface.png`: (metallic, roughness) of the clean materials, from the
  high poly, so Painter knows brass from paint without an ID map.
- `Textures/<Set>_MaskA/B/C.png`: the wear potentials in the low's UVs, packed
  A = (rust, chip, stain), B = (dirt, moss, soot), C = (pool, polish, verd); Painter
  thresholds them at `threshold` from the manifest, so the damage still follows the welds,
  openings and dents.
- `NB57_bakeprep.blend` and `bakeprep_manifest.json`.

Glazing, the engine and the interior keep their own materials and are not textured.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import narrowboat_build as B
import narrowboat_common as N
import narrowboat_dents as DN
import narrowboat_wear as WM
import narrowboat_weathered as WX
from procgen_bpy import Graph, fresh_scene, sha256

SETS = ("Hull", "Cabin", "Ends", "Roof", "Decks", "Fittings")
# Paint with no thickness of its own: colour on the plate, so in the high poly only.
OVERLAYS = (
    "CoachlineCream",
    "CoachlineRed",
    "DoorLinesCream",
    "HatchLinesCream",
    "Decal",
)
STRIP = 2.0  # m: long parts are cut into strips this long, so they pack into a square
HIGH_EDGE = 0.02  # m: longest edge on a dented high-poly plate
GREEN = (0.008, 0.06, 0.03)  # the paint under the canal art


def set_of(name: str) -> str | None:
    """Which texture set a part belongs to (None: not textured)."""
    if name.startswith(("Glazing", "Machinery")):
        return None
    if name in ("Hull", "Counter", "RubbingStrakes", "SternGear"):
        return "Hull"
    if name.startswith(
        ("CabinPort", "CabinStarboard", "CoachlineCream", "CoachlineRed")
    ):
        return "Cabin"
    if name.startswith("DecalCabin"):
        return "Cabin"
    if name.startswith(
        (
            "Bulkheads",
            "EndDoors",
            "RearHatch",
            "SideHatch",
            "DoorLines",
            "HatchLines",
            "Decal",
            "SeatLocker",
            "ShelterRail",
        )
    ):
        return "Ends"
    if name in ("Roof", "Handrails"):
        return "Roof"
    if name.startswith(("RearDeck", "WellDeck", "EngineBoards", "SeatLid")):
        return "Decks"
    if name.startswith(
        (
            "RoofBrass",
            "Chimney",
            "PoleAndPlank",
            "Tiller",
            "DeckBrass",
            "Fender",
            "GasCylinders",
            "WindowFrames",
            "PortholeRims",
        )
    ):
        return "Fittings"
    return None


def is_overlay(name: str) -> bool:
    return name.startswith(OVERLAYS)


def evaluated_copy(bpy, obj, name: str):
    """A new object holding `obj`'s mesh with its modifiers applied."""
    dg = bpy.context.evaluated_depsgraph_get()
    mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(dg), depsgraph=dg)
    copy = bpy.data.objects.new(name, mesh)
    copy.matrix_world = obj.matrix_world
    bpy.context.scene.collection.objects.link(copy)
    return copy


def join(bpy, objs, name: str):
    for o in bpy.context.selected_objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = obj.data.name = name
    return obj


def one_material(bpy, obj, name: str):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    for p in obj.data.polygons:
        p.material_index = 0
    return mat


def project_uvs(obj) -> int:
    """UVs by projection, true to scale in metres, one island per (part, facing, side of
    the boat, strip along it):
    - faces that look sideways unroll along the hull's girth (u = girth, v = z), so the
      bow's curve is not stretched;
    - faces that look up or down are mapped in plan (x, y);
    - hidden faces (undersides, the plates' inner faces) get a fifth of the density;
    - faces that look fore or aft are mapped in (y, z).
    Long islands are cut every STRIP metres so the set packs into a square. Returns the
    number of islands."""
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    # Cut across the boat every STRIP metres: a long plate is often a single n-gon.
    xs = [v.co.x for v in bm.verts]
    for k in range(math.floor(min(xs) / STRIP) + 1, math.ceil(max(xs) / STRIP)):
        geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
        bmesh.ops.bisect_plane(
            bm, geom=geom, plane_co=(k * STRIP, 0, 0), plane_no=(1, 0, 0)
        )
    bm.faces.ensure_lookup_table()
    bm.faces.index_update()
    uv = bm.loops.layers.uv.verify()
    part = {}
    for k, island in enumerate(_components(bm)):
        for f in island:
            part[f.index] = k
    keys: dict = {}
    for f in bm.faces:
        n, c = f.normal, f.calc_center_median()
        ax, ay, az = abs(n.x), abs(n.y), abs(n.z)
        strip = math.floor(c.x / STRIP)
        if ay >= ax and ay >= az:
            mode, sign = "side", 1 if n.y > 0 else -1
        elif az >= ax:
            mode, sign = "plan", 1 if n.z > 0 else -1
        else:
            mode, sign = "end", 1 if n.x > 0 else -1
        key = (part[f.index], mode, sign, c.y > 0, strip if mode != "end" else 0)
        island = keys.setdefault(key, len(keys))
        # Hidden faces get a fifth of the density: undersides, and the plates' inner
        # faces (they look towards the centreline, behind the linings).
        hidden = (mode == "plan" and sign < 0) or (
            mode == "side" and sign * c.y < 0 and abs(c.y) > 0.3
        )
        scale = 0.2 if hidden else 1.0
        for loop in f.loops:
            x, y, z = loop.vert.co
            if mode == "side":
                u = float(WM.u_wall(1000 * (x + N.X_MID))) / 1000
                u, v = u * sign, z
            elif mode == "plan":
                u, v = x * sign, y
            else:
                u, v = y * -sign, z
            # Islands start far apart so none touch before packing.
            loop[uv].uv = (scale * u + 100.0 * island, scale * v)
    bm.to_mesh(obj.data)
    bm.free()
    return len(keys)


def _components(bm):
    """Face sets connected through shared edges."""
    seen, out = set(), []
    for f in bm.faces:
        if f.index in seen:
            continue
        stack, comp = [f], []
        seen.add(f.index)
        while stack:
            g = stack.pop()
            comp.append(g)
            for e in g.edges:
                for h in e.link_faces:
                    if h.index not in seen:
                        seen.add(h.index)
                        stack.append(h)
        out.append(comp)
    return out


def unwrap(bpy, obj, margin: float) -> dict:
    """Project, then pack the islands into 0-1 at one uniform scale."""
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name="UVMap")
    islands = project_uvs(obj)
    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.select_all(action="SELECT")
    bpy.ops.uv.pack_islands(
        rotate=True, scale=True, margin=margin, shape_method="CONCAVE"
    )
    bpy.ops.object.mode_set(mode="OBJECT")
    return {"islands": islands, **texel_density(obj)}


def texel_density(obj) -> dict:
    """Pixels per metre at 1 px = 1/4096 of the UV square: median and spread."""
    import numpy as np

    me = obj.data
    uv = me.uv_layers[0].data
    ratios = []
    for p in me.polygons:
        if p.area < 1e-6:
            continue
        loops = list(p.loop_indices)
        pts = [uv[i].uv for i in loops]
        a = 0.0
        for i in range(len(pts)):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % len(pts)]
            a += x0 * y1 - x1 * y0
        ratios.append(math.sqrt(abs(a) / 2 / p.area) * 4096)
    r = np.array(ratios)
    return {
        "px_per_m_at_4k": round(float(np.median(r)), 1),
        "p10": round(float(np.percentile(r, 10)), 1),
        "p90": round(float(np.percentile(r, 90)), 1),
    }


def livery_material(bpy, obj) -> None:
    """The decals' art over the green they are painted on, opaque, so a bake that
    projects from the high poly picks up the art and not the decal slab's empty corners."""
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None or not mat.name.startswith("M_NB_Decal"):
            continue
        design = mat.name.removeprefix("M_NB_Decal")
        g = Graph(bpy, f"Livery_{design}", GREEN)
        tex = g.node(
            "ShaderNodeTexImage",
            {"Vector": g.coords.outputs["UV"]},
            image=bpy.data.images[N.texture_name(design)],
            extension="CLIP",
        )
        col = g.mix(tex.outputs["Alpha"], GREEN, tex.outputs["Color"])
        g.finish(col, 0.3, g.geometry.outputs["Normal"])
        slot.material = g.mat


def albedo_material(bpy, obj) -> None:
    """Make each of `obj`'s materials emit its own base colour, so an emission bake reads
    the albedo of metal too (a diffuse bake sees none: metal has no diffuse)."""
    done = set()
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None or mat.name in done or not mat.use_nodes:
            continue
        done.add(mat.name)
        nt = mat.node_tree
        bsdf = next(
            (n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None
        )
        out = next(
            (n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"), None
        )
        if bsdf is None or out is None:
            continue
        emit = nt.nodes.new("ShaderNodeEmission")
        base = bsdf.inputs["Base Color"]
        if base.is_linked:
            nt.links.new(base.links[0].from_socket, emit.inputs["Color"])
        else:
            emit.inputs["Color"].default_value = base.default_value
        nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])


def surface_material(bpy, obj) -> None:
    """Replace `obj`'s materials with flat emission of (metallic, roughness, 0) read from
    each clean material, so one bake records which parts are metal and how glossy."""
    made = {}
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None:
            continue
        if mat.name not in made:
            bsdf = next(
                (
                    n
                    for n in mat.node_tree.nodes
                    if n.bl_idname == "ShaderNodeBsdfPrincipled"
                ),
                None,
            )
            metal = rough = None
            if bsdf is not None:
                if not bsdf.inputs["Metallic"].is_linked:
                    metal = bsdf.inputs["Metallic"].default_value
                if not bsdf.inputs["Roughness"].is_linked:
                    rough = bsdf.inputs["Roughness"].default_value
            name = mat.name
            if metal is None:
                metal = (
                    1.0
                    if any(k in name for k in ("Brass", "Frame", "Stainless"))
                    else 0.0
                )
            if rough is None:
                rough = 0.3 if metal > 0.5 else 0.6
            g = Graph(bpy, f"Surface_{name}", (0, 0, 0))
            g.set(g.bsdf.inputs["Emission Color"], (metal, rough, 0.0, 1.0))
            g.set(g.bsdf.inputs["Emission Strength"], 1.0)
            g.finish((0.0, 0.0, 0.0), 1.0, g.geometry.outputs["Normal"], specular=0.0)
            made[mat.name] = g.mat
        slot.material = made[mat.name]


PACKS = {
    "A": ("rust", "chip", "stain"),
    "B": ("dirt", "moss", "soot"),
    "C": ("pool", "polish", "verd"),
    # Height above the waterline (-0.15..0.35 m as 0..1), paint chalking, roof fading.
    "D": ("waterline", "chalk", "fade"),
}


def _masks(g, ctx, set_name: str) -> dict:
    """The wear channels of one set, sampled as its shaders sample them."""
    wearing = WX.Wearing(g, ctx)
    if set_name == "Hull":
        m = dict(wearing.masks(gunwale_only=True, planes=("side", "plan")))
    else:
        m = dict(wearing.masks())
    if set_name in ("Ends", "Roof"):
        # Paint above the side top (handrails, the rear hatch): rust at the handrail
        # feet and knocks along the rails; the plan map's channels stay as they are.
        _, _, z = g.separate(g.coords.outputs["Object"])
        rf = WX.roof_fittings(g, wearing, wearing.masks(planes=("side", "ends")), z)
        for ch in ("rust", "chip"):
            m[ch] = g.math("MAXIMUM", m[ch], rf[ch])
    if set_name == "Fittings":
        m["polish"] = g.math("MAXIMUM", m["polish"], WX.handled(g))
    obj = g.coords.outputs["Object"]
    x, y, z = g.separate(obj)
    m["waterline"] = g.map_range(WX.waterline_height(g, x, z), -0.15, 0.35)
    m["chalk"] = WX.chalk_amount(g, ctx.wear, z, obj, 1.0, dirt=m["dirt"])
    _, _, nz = g.separate(g.coords.outputs["Normal"])
    edge = g.map_range(g.math("ABSOLUTE", y), 0.62, 0.86)
    fade = g.madd(edge, 0.4 - 0.22, 0.22)
    m["fade"] = g.math("MULTIPLY", fade, g.madd(g.math("MAXIMUM", nz, 0.0), 0.35, 0.65))
    return m


def mask_material(bpy, name: str, ctx, set_name: str):
    """Emission = one pack of wear channels per material, as PACKS lists them."""
    packs = {}
    for pack, channels in PACKS.items():
        g = Graph(bpy, f"{name}_{pack}", (0, 0, 0))
        m = _masks(g, ctx, set_name)
        rgb = g.node(
            "ShaderNodeCombineColor",
            {"Red": m[channels[0]], "Green": m[channels[1]], "Blue": m[channels[2]]},
        ).outputs["Color"]
        g.set(g.bsdf.inputs["Emission Color"], rgb)
        g.set(g.bsdf.inputs["Emission Strength"], 1.0)
        g.finish((0.0, 0.0, 0.0), 1.0, g.geometry.outputs["Normal"], specular=0.0)
        packs[pack] = g.mat
    return packs


def bake_masks(bpy, out: Path, lows: dict, ctx, size: int) -> dict:
    files = {}
    for s, low in lows.items():
        files[s] = {}
        for pack, mat in mask_material(bpy, f"Mask_{s}", ctx, s).items():
            image = new_image(bpy, f"{s}_Mask{pack}", size)
            bake_image(bpy, low, image, mat, "EMIT")
            path = out / "Textures" / f"{s}_Mask{pack}.png"
            save(image, path, colour=False)
            files[s][f"Mask{pack}"] = {
                "file": f"Textures/{path.name}",
                "sha256": sha256(path),
            }
        one_material(bpy, low, f"NB57_{s}")
    return files


def bake_image(bpy, obj, image, mat, bake_type, selected=None, **kw):
    """Bake into `image` through an active image node in `mat` on `obj`."""
    nt = mat.node_tree
    node = nt.nodes.new("ShaderNodeTexImage")
    node.image = image
    nt.nodes.active = node
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    if selected is not None:
        selected.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.bake(
        type=bake_type,
        use_selected_to_active=selected is not None,
        margin=16,
        **kw,
    )
    nt.nodes.remove(node)


def new_image(bpy, name: str, size: int):
    """A float image to bake into: its pixels come back linear and unclamped."""
    image = bpy.data.images.new(name, size, size, alpha=False, float_buffer=True)
    image.colorspace_settings.name = "Non-Color"
    return image


def save(image, path: Path, colour: bool) -> None:
    """Write the baked pixels: colour as 8-bit sRGB, data as 16-bit linear. Done here
    rather than by Blender, whose save applies the scene's view transform."""
    import numpy as np
    import procgen_masks as M

    w, h = image.size
    px = np.empty(w * h * 4, np.float32)
    image.pixels.foreach_get(px)
    rgb = np.clip(px.reshape(h, w, 4)[::-1, :, :3], 0.0, 1.0)  # Blender rows: bottom up
    if colour:
        srgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
        M.write_png(path, np.round(srgb * 255).astype(np.uint8))
    else:
        M.write_png(path, np.round(rgb * 65535).astype(np.uint16))


def remask(bpy, out: Path, args) -> None:
    """Re-bake only the wear masks, into the UVs of the saved low poly (the meshes Painter
    already holds stay exactly as they are); updates the manifest's mask entries."""
    bpy.ops.wm.open_mainfile(filepath=str(out / "NB57_bakeprep.blend"))
    manifest = json.loads((out / "bakeprep_manifest.json").read_text())
    level = WM.WEARS[manifest["wear"]]
    layers = WM.layers(level, manifest["seed"])
    record = WM.write_maps(
        out / "Textures" / "WearMaps", level, layers, WM.wear_maps(layers)
    )
    del layers
    ctx = WX.load_maps(bpy, out / "Textures" / "WearMaps", record, level)
    lows = {
        s: bpy.data.objects[f"{s}_low"] for s in SETS if f"{s}_low" in bpy.data.objects
    }
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 4
    scene.cycles.device = "GPU"
    for s, masks in bake_masks(bpy, out, lows, ctx, manifest["size"]).items():
        manifest["files"][s].update(masks)
    manifest["packs"] = {k: list(v) for k, v in PACKS.items()}
    manifest["thresholds"] = record["thresholds"]
    (out / "bakeprep_manifest.json").write_text(json.dumps(manifest, indent=1))


def main() -> None:
    import bpy

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=N.SEED)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--wear", default="weathered")
    parser.add_argument(
        "--masks-only",
        action="store_true",
        help="re-bake the wear masks into the saved NB57_bakeprep.blend's UVs",
    )
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    out = Path(args.out)
    if args.masks_only:
        remask(bpy, out, args)
        return
    (out / "Mesh").mkdir(parents=True, exist_ok=True)
    (out / "Textures").mkdir(parents=True, exist_ok=True)
    level = WM.WEARS[args.wear]

    scene = fresh_scene(bpy)
    textures = B.write_decal_textures(out / "Textures" / "Decals")
    B.load_decal_textures(bpy, out / "Textures" / "Decals")
    layers = WM.layers(level, args.seed)
    record = WM.write_maps(
        out / "Textures" / "WearMaps", level, layers, WM.wear_maps(layers)
    )
    del layers
    ctx = WX.load_maps(bpy, out / "Textures" / "WearMaps", record, level)
    _, objects, _, _, _ = B.build_boat(bpy, scene, args.seed)
    dents = DN.dents(level, args.seed)

    groups: dict[str, dict[str, list]] = {s: {"low": [], "high": []} for s in SETS}
    for obj in objects:
        s = set_of(obj.name)
        if s is None:
            continue
        high = evaluated_copy(bpy, obj, f"{obj.name}_h")
        surface = B.dent_surface(obj.name)
        if surface is not None:
            B.dent_object(high, surface, dents, level, max_edge=HIGH_EDGE)
        livery_material(bpy, high)
        groups[s]["high"].append(high)
        if not is_overlay(obj.name):
            groups[s]["low"].append(evaluated_copy(bpy, obj, f"{obj.name}_l"))
    # The build's own objects (with their booleans and cutters) are no longer needed.
    for obj in list(bpy.data.objects):
        if obj.name.endswith(("_h", "_l")):
            continue
        bpy.data.objects.remove(obj)

    margin = 8 / args.size
    lows, highs, info = {}, {}, {}
    for s in SETS:
        if not groups[s]["low"]:
            continue
        low = join(bpy, groups[s]["low"], f"{s}_low")
        one_material(bpy, low, f"NB57_{s}")
        info[s] = {"texel": unwrap(bpy, low, margin)}
        info[s]["low_tris"] = sum(len(p.vertices) - 2 for p in low.data.polygons)
        high = join(bpy, groups[s]["high"], f"{s}_high")
        info[s]["high_tris"] = sum(len(p.vertices) - 2 for p in high.data.polygons)
        lows[s], highs[s] = low, high

    scene.render.engine = "CYCLES"
    scene.cycles.samples = 4
    scene.cycles.device = "GPU"
    bake = scene.render.bake
    bake.cage_extrusion = 0.03
    bake.max_ray_distance = 0.08
    files = {}
    for s, low in lows.items():
        files[s] = {}
        # Livery: the high poly's clean paint and art, onto the low's UVs.
        image = new_image(bpy, f"{s}_Livery", args.size)
        mat = bpy.data.materials.new(f"Bake_{s}_Livery")
        mat.use_nodes = True
        albedo_material(bpy, highs[s])
        bake_image(bpy, low, image, mat, "EMIT", selected=highs[s])
        path = out / "Textures" / f"{s}_Livery.png"
        save(image, path, colour=True)
        files[s]["Livery"] = {"file": f"Textures/{path.name}", "sha256": sha256(path)}
        # Which parts are metal and how glossy, from the clean materials of the high.
        surface_material(bpy, highs[s])
        image = new_image(bpy, f"{s}_Surface", args.size)
        mat = bpy.data.materials.new(f"Bake_{s}_Surface")
        mat.use_nodes = True
        bake_image(bpy, low, image, mat, "EMIT", selected=highs[s])
        path = out / "Textures" / f"{s}_Surface.png"
        save(image, path, colour=False)
        files[s]["Surface"] = {"file": f"Textures/{path.name}", "sha256": sha256(path)}
        one_material(bpy, low, f"NB57_{s}")

    for s, masks in bake_masks(bpy, out, lows, ctx, args.size).items():
        files[s].update(masks)
    for which, objs in (("low", lows), ("high", highs)):
        for o in bpy.context.selected_objects:
            o.select_set(False)
        for o in objs.values():
            o.select_set(True)
        path = out / "Mesh" / f"NB57_{which}.fbx"
        bpy.ops.export_scene.fbx(
            filepath=str(path),
            use_selection=True,
            apply_unit_scale=True,
            axis_forward="-Z",
            axis_up="Y",
            mesh_smooth_type="FACE",
            use_mesh_modifiers=False,
            bake_space_transform=False,
        )
        files[f"mesh_{which}"] = {"file": f"Mesh/{path.name}", "sha256": sha256(path)}

    bpy.ops.wm.save_as_mainfile(filepath=str(out / "NB57_bakeprep.blend"))
    manifest = {
        "asset": "Narrowboat/NB57",
        "stage": "bakeprep",
        "wear": args.wear,
        "seed": args.seed,
        "size": args.size,
        "threshold": level.threshold,
        "thresholds": record["thresholds"],
        "packs": {k: list(v) for k, v in PACKS.items()},
        "sets": info,
        "files": files,
        "dents": DN.summary(dents),
        "decals": textures,
    }
    (out / "bakeprep_manifest.json").write_text(json.dumps(manifest, indent=1))
    print("BAKEPREP " + json.dumps(info))


if __name__ == "__main__":
    main()
