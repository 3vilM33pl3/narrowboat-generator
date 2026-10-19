# SPDX-License-Identifier: MIT
"""Blender-side helpers shared by the procedural generators (bpy is passed in, never imported).

Shader graphs (`Graph`), object creation from `procgen_mesh.MeshData` parts with live Bevel
modifiers and custom attributes, the grey-studio lookdev, cameras, render-to-numpy helpers,
receipts and atomic folder publishing.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

BACKGROUND = 157 / 255  # grey backdrop of the reference sheets, sRGB


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- materials


class Graph:
    """Small helper for building shader node trees; numbers and tuples become defaults."""

    def __init__(self, bpy, name: str, viewport):
        self.mat = bpy.data.materials.new(name)
        self.mat.diffuse_color = (*viewport, 1.0)
        self.nt = self.mat.node_tree
        self.nt.nodes.clear()
        out = self.nt.nodes.new("ShaderNodeOutputMaterial")
        self.bsdf = self.nt.nodes.new("ShaderNodeBsdfPrincipled")
        self.nt.links.new(self.bsdf.outputs["BSDF"], out.inputs["Surface"])
        self._coords = None
        self._geometry = None

    def set(self, socket, value) -> None:
        if hasattr(value, "is_output"):
            self.nt.links.new(value, socket)
        elif isinstance(value, tuple) and len(value) == 3 and socket.type == "RGBA":
            socket.default_value = (*value, 1.0)
        else:
            socket.default_value = value

    def node(self, kind: str, inputs=None, **props):
        n = self.nt.nodes.new(kind)
        for key, value in props.items():
            setattr(n, key, value)
        for key, value in (inputs or {}).items():
            socket = n.inputs[key] if isinstance(key, int) else _by_id(n.inputs, key)
            self.set(socket, value)
        return n

    @property
    def coords(self):
        if self._coords is None:
            self._coords = self.node("ShaderNodeTexCoord")
        return self._coords

    @property
    def geometry(self):
        if self._geometry is None:
            self._geometry = self.node("ShaderNodeNewGeometry")
        return self._geometry

    def math(self, op: str, a, b=0.0, clamp=False):
        return self.node(
            "ShaderNodeMath", {0: a, 1: b}, operation=op, use_clamp=clamp
        ).outputs[0]

    def map_range(self, v, f0, f1, t0=0.0, t1=1.0, clamp=True):
        return self.node(
            "ShaderNodeMapRange", {0: v, 1: f0, 2: f1, 3: t0, 4: t1}, clamp=clamp
        ).outputs[0]

    def mix(self, fac, a, b, blend="MIX"):
        n = self.node(
            "ShaderNodeMix",
            data_type="RGBA",
            blend_type=blend,
            clamp_factor=True,
        )
        self.set(_by_id(n.inputs, "Factor_Float"), fac)
        self.set(_by_id(n.inputs, "A_Color"), a)
        self.set(_by_id(n.inputs, "B_Color"), b)
        return _by_id(n.outputs, "Result_Color")

    def mapping(self, vec, scale):
        return self.node("ShaderNodeMapping", {"Vector": vec, "Scale": scale}).outputs[
            0
        ]

    def noise(self, vec, scale, detail=2.0, roughness=0.5, distortion=0.0):
        return self.node(
            "ShaderNodeTexNoise",
            {
                "Vector": vec,
                "Scale": scale,
                "Detail": detail,
                "Roughness": roughness,
                "Distortion": distortion,
            },
        ).outputs["Factor"]

    def grain(self, scale, distortion, detail=4.0):
        """Wood grain along U: meandering growth rings plus streaky fibres (noise stretched
        along U), so boards read as timber rather than regular stripes."""
        uv = self.coords.outputs["UV"]
        rings = self.node(
            "ShaderNodeTexWave",
            {
                "Vector": self.mapping(uv, (0.05, 1.0, 1.0)),
                "Scale": scale,
                "Distortion": distortion,
                "Detail": detail,
                "Detail Scale": 0.6,
                "Detail Roughness": 0.6,
            },
            wave_type="BANDS",
            bands_direction="Y",
        ).outputs["Factor"]
        fibres = self.noise(
            self.mapping(uv, (1.0, 40.0, 1.0)), scale / 8, detail=8.0, roughness=0.65
        )
        return self.math(
            "ADD",
            self.math("MULTIPLY", rings, 0.45),
            self.math("MULTIPLY", self.map_range(fibres, 0.3, 0.7), 0.55),
        )

    def attribute(self, name: str):
        return self.node("ShaderNodeAttribute", attribute_name=name).outputs["Factor"]

    def combine(self, x, y, z=0.0):
        return self.node("ShaderNodeCombineXYZ", {"X": x, "Y": y, "Z": z}).outputs[0]

    def separate(self, vec):
        """(x, y, z) sockets of a vector."""
        n = self.node("ShaderNodeSeparateXYZ", {0: vec})
        return n.outputs["X"], n.outputs["Y"], n.outputs["Z"]

    def madd(self, a, b, c):
        """a * b + c."""
        return self.node(
            "ShaderNodeMath", {0: a, 1: b, 2: c}, operation="MULTIPLY_ADD"
        ).outputs[0]

    def image_rgb(self, image, vector, interpolation="Linear", extension="CLIP"):
        """(r, g, b) sockets of a data image sampled at `vector`."""
        tex = self.node(
            "ShaderNodeTexImage",
            {"Vector": vector},
            image=image,
            interpolation=interpolation,
            extension=extension,
        )
        rgb = self.node("ShaderNodeSeparateColor", {0: tex.outputs["Color"]})
        return rgb.outputs["Red"], rgb.outputs["Green"], rgb.outputs["Blue"]

    def object_axis(self, axis: str):
        return self.node(
            "ShaderNodeSeparateXYZ", {0: self.coords.outputs["Object"]}
        ).outputs[axis]

    def edge(self, radius: float):
        """1 on convex/concave edges, from how far the Bevel normal leans off the true normal."""
        bevel = self.node("ShaderNodeBevel", {"Radius": radius}, samples=8)
        dot = self.node(
            "ShaderNodeVectorMath",
            operation="DOT_PRODUCT",
        )
        self.nt.links.new(bevel.outputs["Normal"], dot.inputs[0])
        self.nt.links.new(self.geometry.outputs["Normal"], dot.inputs[1])
        return self.map_range(dot.outputs["Value"], 0.998, 0.93)

    def occlusion(self, distance: float):
        return self.node(
            "ShaderNodeAmbientOcclusion", {"Distance": distance}, samples=12
        ).outputs["AO"]

    def bump(self, height, strength, distance, normal=None):
        inputs = {"Height": height, "Strength": strength, "Distance": distance}
        if normal is not None:
            inputs["Normal"] = normal
        return self.node("ShaderNodeBump", inputs).outputs["Normal"]

    def finish(self, color, roughness, normal, metallic=0.0, specular=0.3):
        # Weathered paint and timber are matte: a low specular level keeps the lookdev key
        # from washing top surfaces out in the overhead view.
        self.set(self.bsdf.inputs["Specular IOR Level"], specular)
        self.set(self.bsdf.inputs["Base Color"], color)
        self.set(self.bsdf.inputs["Roughness"], roughness)
        self.set(self.bsdf.inputs["Metallic"], metallic)
        self.set(self.bsdf.inputs["Normal"], normal)
        return self.mat


def _by_id(sockets, identifier):
    for s in sockets:
        if s.identifier == identifier or s.name == identifier:
            return s
    raise KeyError(identifier)


def make_object(
    bpy, part, materials: dict, collection, name: str, offset=(0, 0, 0), props=None
):
    md = part.mesh
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(md.verts, [], md.faces)
    if mesh.validate(verbose=True):
        raise RuntimeError(f"{name}: mesh.validate() changed generated geometry")
    uv = mesh.uv_layers.new(name="UVMap")
    uv.uv.foreach_set(
        "vector", [c for face in md.uvs for corner in face for c in corner]
    )
    mesh.polygons.foreach_set("material_index", md.mats)
    mesh.attributes.new("plank_rand", "FLOAT", "FACE").data.foreach_set(
        "value", md.plank
    )
    mesh.attributes.new("hand_wear", "FLOAT", "POINT").data.foreach_set(
        "value", md.wear
    )
    for slot in part.slots:
        mesh.materials.append(materials[slot])
    mesh.shade_smooth()
    mesh.set_sharp_from_angle(angle=math.radians(35))
    obj = bpy.data.objects.new(name, mesh)
    obj.location = offset
    collection.objects.link(obj)
    bevel = obj.modifiers.new("Bevel", "BEVEL")
    bevel.width = 0.003
    bevel.segments = 2
    bevel.limit_method = "ANGLE"
    bevel.angle_limit = math.radians(35)
    bevel.harden_normals = True
    bevel.use_clamp_overlap = True
    for key, value in (props or {}).items():
        obj[key] = value
    return obj


def new_collection(bpy, scene, name: str):
    collection = bpy.data.collections.new(name)
    scene.collection.children.link(collection)
    return collection


def look_at(cam, target) -> None:
    from mathutils import Vector

    cam.rotation_euler = (
        (Vector(target) - cam.location).to_track_quat("-Z", "Y").to_euler()
    )


def lookdev(bpy, scene, collection, samples: int) -> dict:
    """Grey studio: shadow-catcher floor, soft key and fill suns, grey world, Cycles."""
    floor = bpy.data.meshes.new("Lookdev_Floor")
    floor.from_pydata(
        [(-50, -50, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0)], [], [(0, 1, 2, 3)]
    )
    floor_obj = bpy.data.objects.new("Lookdev_Floor", floor)
    floor_obj.is_shadow_catcher = True
    # The catcher still bounces light: give it the reference backdrop's grey, not white.
    backdrop = bpy.data.materials.new("Lookdev_Backdrop")
    backdrop.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (
        0.3,
        0.3,
        0.3,
        1,
    )
    backdrop.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9
    floor.materials.append(backdrop)
    collection.objects.link(floor_obj)
    for name, energy, angle, rot in (
        ("Lookdev_Key", 3.0, 40, (14, 0, 25)),
        ("Lookdev_Fill", 1.0, 60, (60, 0, -25)),
    ):
        light = bpy.data.objects.new(name, bpy.data.lights.new(name, "SUN"))
        light.data.energy = energy
        light.data.angle = math.radians(angle)
        light.data.color = (1.0, 0.97, 0.92)
        light.rotation_euler = tuple(math.radians(a) for a in rot)
        collection.objects.link(light)
    world = bpy.data.worlds.new("Lookdev_World")
    scene.world = world
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.4, 0.4, 0.4, 1)
    background.inputs["Strength"].default_value = 0.7

    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA", "HIP", "NONE"):
        try:
            prefs.compute_device_type = kind
        except TypeError:
            continue
        prefs.get_devices()
        devices = [d for d in prefs.devices if d.type == kind]
        if devices or kind == "NONE":
            for d in prefs.devices:
                d.use = d.type == kind
            break
    scene.cycles.device = "GPU" if kind != "NONE" else "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    # Standard keeps the saturated paint of the reference; AgX washes the orange to salmon.
    scene.view_settings.view_transform = "Standard"
    return {
        "engine": "CYCLES",
        "device": kind,
        "samples": samples,
        "view_transform": "Standard",
    }


def make_cameras(bpy, collection, specs: dict) -> dict:
    out = {}
    for name, spec in specs.items():
        data = bpy.data.cameras.new(f"Cam_{name}")
        cam = bpy.data.objects.new(f"Cam_{name}", data)
        collection.objects.link(cam)
        cam.location = spec["location"]
        if "ortho" in spec:
            data.type = "ORTHO"
            data.ortho_scale = spec["ortho"]
            cam.rotation_euler = spec["rotation"]
        else:
            data.lens = spec["lens"]
            if "target" in spec:
                look_at(cam, spec["target"])
            else:
                cam.rotation_euler = spec["rotation"]
        data.clip_end = spec.get("clip_end", 200)
        # Optional: a clip plane that cuts sections, and the lens shift and sensor fit of
        # a camera matched to a drawing.
        if "clip_start" in spec:
            data.clip_start = spec["clip_start"]
        if "shift" in spec:
            data.shift_x, data.shift_y = spec["shift"]
        if "sensor_fit" in spec:
            data.sensor_fit = spec["sensor_fit"]
        if "sensor_width" in spec:
            data.sensor_width = spec["sensor_width"]
        out[name] = (cam, spec["resolution"])
    return out


def pixels(bpy, path: Path):
    import numpy as np

    image = bpy.data.images.load(str(path))
    w, h = image.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(buf)
    bpy.data.images.remove(image)
    return np.flipud(buf.reshape(h, w, 4))


def save_png(bpy, rgb, path: Path, alpha=None) -> None:
    """Write top-row-first RGB pixels as PNG; `alpha` (h x w) adds a straight alpha."""
    import numpy as np

    h, w = rgb.shape[:2]
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[..., :3] = rgb[..., :3]
    if alpha is not None:
        rgba[..., 3] = alpha
    image = bpy.data.images.new(path.stem, w, h, alpha=alpha is not None)
    image.pixels.foreach_set(np.flipud(rgba).ravel())
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)


def over_grey(rgba):
    a = rgba[..., 3:4]
    return rgba[..., :3] * a + BACKGROUND * (1 - a)


def render(bpy, scene, cam, resolution, path: Path, samples: int) -> None:
    scene.camera = cam
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.cycles.samples = samples
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def publish(staging: Path, target: Path) -> None:
    """Swap a fully written staging folder into place, keeping the old one until it succeeds."""
    old = target.with_name(target.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if target.exists():
        target.rename(old)
    staging.rename(target)
    if old.exists():
        shutil.rmtree(old)


def fresh_staging(target: Path) -> Path:
    staging = target.with_name(target.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging


def keep_previews(target: Path, staging: Path) -> list[str]:
    """With --no-render, carry the last successful previews over instead of dropping them."""
    old = target / "Previews"
    if not old.is_dir():
        return []
    shutil.copytree(old, staging / "Previews")
    return sorted(p.name for p in old.iterdir())


def receipts(bpy, parts, objects, materials) -> list[dict]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    out = []
    for part, obj in zip(parts, objects):
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        mesh.calc_loop_triangles()
        lo, hi = part.mesh.bounds()
        out.append(
            {
                "name": obj.name,
                "material_slots": [materials[s].name for s in part.slots],
                "vertices": len(part.mesh.verts),
                "faces": len(part.mesh.faces),
                "triangles_rendered": len(mesh.loop_triangles),
                "volume_m3": round(part.mesh.volume(), 6),
                "bounds_min_m": [round(c + o, 4) for c, o in zip(lo, obj.location)],
                "bounds_max_m": [round(c + o, 4) for c, o in zip(hi, obj.location)],
            }
        )
        evaluated.to_mesh_clear()
    return out


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def fresh_scene(bpy):
    """Factory settings with an empty scene: nothing of a user's session is ever touched."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    return scene


# --------------------------------------------------------------------------- buildings


def hdri_sky(bpy, scene, samples: int, hdri: Path, turn: float, exposure, haze) -> dict:
    """Cycles set up as for the lookdev, but lit by an HDRI alone, turned `turn` radians
    about Z: its sun casts the shadows, so the studio suns and the catcher floor go. Camera
    rays below the horizon see `haze` instead of the map's ground."""
    info = lookdev(bpy, scene, new_collection(bpy, scene, "Lookdev"), samples)
    for name in ("Lookdev_Key", "Lookdev_Fill", "Lookdev_Floor"):
        bpy.data.objects.remove(bpy.data.objects[name])
    bpy.data.meshes.remove(bpy.data.meshes["Lookdev_Floor"])
    bpy.data.materials.remove(bpy.data.materials["Lookdev_Backdrop"])
    nt = scene.world.node_tree
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(hdri), check_existing=True)
    coords = nt.nodes.new("ShaderNodeTexCoord")
    rotate = nt.nodes.new("ShaderNodeMapping")
    rotate.inputs["Rotation"].default_value = (0.0, 0.0, turn)
    nt.links.new(coords.outputs["Generated"], rotate.inputs["Vector"])
    nt.links.new(rotate.outputs["Vector"], env.inputs["Vector"])
    background = nt.nodes.get("Background")
    below = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(coords.outputs["Generated"], below.inputs[0])
    fade = nt.nodes.new("ShaderNodeMapRange")
    nt.links.new(below.outputs["Z"], fade.inputs[0])
    fade.inputs[1].default_value = 0.1
    fade.inputs[2].default_value = 0.0
    camera = nt.nodes.new("ShaderNodeLightPath")
    mask = nt.nodes.new("ShaderNodeMath")
    mask.operation = "MULTIPLY"
    nt.links.new(fade.outputs[0], mask.inputs[0])
    nt.links.new(camera.outputs["Is Camera Ray"], mask.inputs[1])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    nt.links.new(mask.outputs[0], _by_id(mix.inputs, "Factor_Float"))
    nt.links.new(env.outputs["Color"], _by_id(mix.inputs, "A_Color"))
    _by_id(mix.inputs, "B_Color").default_value = haze
    nt.links.new(_by_id(mix.outputs, "Result_Color"), background.inputs["Color"])
    background.inputs["Strength"].default_value = 1.0
    scene.render.film_transparent = False
    scene.view_settings.exposure = exposure
    scene.cycles.max_bounces = 8
    info.update(
        hdri=str(hdri), hdri_turn_deg=round(math.degrees(turn), 1), exposure=exposure
    )
    return info


def text_object(
    bpy,
    collection,
    name,
    body,
    size,
    location,
    rotation,
    material,
    font,
    props=None,
    extrude=0.003,
    bevel=0.0006,
):
    """Centred text curve placed by a 3x3 rotation (rows) and a location."""
    from mathutils import Matrix

    curve = bpy.data.curves.new(name, "FONT")
    curve.body = body
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    curve.size = size
    curve.extrude = extrude
    curve.bevel_depth = bevel
    if font is not None:
        curve.font = font
    curve.materials.append(material)
    obj = bpy.data.objects.new(name, curve)
    obj.matrix_world = Matrix.Translation(location) @ Matrix(rotation).to_4x4()
    collection.objects.link(obj)
    for key, value in (props or {}).items():
        obj[key] = value
    return obj


def reference_image(bpy, path: Path):
    image = bpy.data.images.load(str(path), check_existing=True)
    image.pack()
    return image


def drawing_empty(
    bpy, collection, image, name, m_per_px, axis_px, base_px, location=(0.0, 4.0, 0.0)
):
    """A drawing as an image empty standing in the xz plane at its true scale, its column
    `axis_px` on x = 0 and its row `base_px` on z = 0; hidden from renders."""
    w, h = image.size
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = "IMAGE"
    empty.data = image
    empty.empty_display_size = max(w, h) * m_per_px
    empty.empty_image_offset = (-axis_px / w, -(h - base_px) / h)
    empty.location = location
    empty.rotation_euler = (math.pi / 2, 0.0, 0.0)
    empty.hide_render = True
    collection.objects.link(empty)
    return empty


def red_ink(np, render_rgb, lines, threshold: float = 0.12, gain: float = 2.2):
    """The render with a drawing's linework laid over it in red."""
    ink = 1 - np.clip(lines.mean(axis=2, keepdims=True), 0, 1)
    ink = np.clip((ink - threshold) * gain, 0, 1)
    red = np.array([0.9, 0.08, 0.05], dtype=np.float32)
    return render_rgb * (1 - ink) + red * ink


def edge_ink(np, render_rgb, picture, threshold: float = 0.06, gain: float = 5.0):
    """The render with a picture's edges (its luminance gradient) laid over it in red: for
    coloured drawings, where `red_ink` would ink every dark field as line work."""
    lum = picture[..., :3].mean(axis=2)
    gy, gx = np.gradient(lum)
    ink = np.clip((np.sqrt(gx * gx + gy * gy) - threshold) * gain, 0, 1)[..., None]
    red = np.array([0.9, 0.08, 0.05], dtype=np.float32)
    return render_rgb * (1 - ink) + red * ink


def upscale(np, rgb, k: int):
    return np.repeat(np.repeat(rgb, k, axis=0), k, axis=1)


def daylight_glass(bpy, name: str, tint=(0.92, 0.96, 0.94), roughness: float = 0.0):
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
