# SPDX-License-Identifier: MIT
"""Pure-Python mesh building blocks shared by the procedural generators (no bpy).

`MeshData` is a polygon soup with per-corner UVs, per-face material slot and plank value
and per-vertex hand wear; the builders (`loft`, `band`, `box`, `sweep`, `ring`, `lathe`)
return closed, outward-wound solids so tests can check every part by divergence and edge
pairing. Coordinates are metres; profiles are 2D (y, z) polygons lofted along X.
"""

from __future__ import annotations

import math
import random
from bisect import bisect_right
from dataclasses import dataclass, field
from itertools import pairwise


class Pchip:
    """Monotone cubic (Fritsch-Carlson) through (x, y) knots, evaluated at abs(x).

    The slope at x = 0 is forced to zero so the mirrored curve is smooth at midships.
    """

    def __init__(self, knots):
        self.x = [k[0] for k in knots]
        self.y = [k[1] for k in knots]
        n = len(self.x)
        h = [self.x[i + 1] - self.x[i] for i in range(n - 1)]
        delta = [(self.y[i + 1] - self.y[i]) / h[i] for i in range(n - 1)]
        d = [0.0] * n
        for i in range(1, n - 1):
            if delta[i - 1] * delta[i] > 0:
                w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
                d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
        d[-1] = delta[-1]
        self.h, self.d = h, d

    def __call__(self, x: float) -> float:
        x = min(max(abs(x), self.x[0]), self.x[-1])
        i = min(bisect_right(self.x, x) - 1, len(self.x) - 2)
        h = self.h[i]
        t = (x - self.x[i]) / h
        t2, t3 = t * t, t * t * t
        return (
            (2 * t3 - 3 * t2 + 1) * self.y[i]
            + (t3 - 2 * t2 + t) * h * self.d[i]
            + (-2 * t3 + 3 * t2) * self.y[i + 1]
            + (t3 - t2) * h * self.d[i + 1]
        )


def _normals(points):
    out = []
    for (y0, z0), (y1, z1) in pairwise(points):
        length = math.hypot(y1 - y0, z1 - z0)
        out.append((-(z1 - z0) / length, (y1 - y0) / length))
    return out


def offset_polyline(points, dists):
    """Offset an open 2D polyline to its left (inboard for a section) by per-segment distances.

    Interior vertices are mitred: the intersection of the two neighbouring offset lines.
    """
    normals = _normals(points)
    out = [
        (
            points[0][0] + normals[0][0] * dists[0],
            points[0][1] + normals[0][1] * dists[0],
        )
    ]
    for j in range(1, len(points) - 1):
        (ay, az), (by, bz) = normals[j - 1], normals[j]
        d0, d1 = dists[j - 1], dists[j]
        det = ay * bz - az * by
        if abs(det) < 1e-9:
            dy, dz = ay * d0, az * d0
        else:
            dy = (d0 * bz - az * d1) / det
            dz = (ay * d1 - d0 * by) / det
        out.append((points[j][0] + dy, points[j][1] + dz))
    ny, nz = normals[-1]
    out.append((points[-1][0] + ny * dists[-1], points[-1][1] + nz * dists[-1]))
    return out


def y_at_z(poly, z: float) -> float:
    """Y of a section polyline (centreline first) where it reaches height z on its side."""
    for (y0, z0), (y1, z1) in pairwise(poly[1:]):
        if z0 <= z <= z1 or (z1 < z0 and z1 <= z <= z0):
            return y0 if z1 == z0 else y0 + (y1 - y0) * (z - z0) / (z1 - z0)
    if z < poly[1][1]:
        return poly[1][0]
    (y0, z0), (y1, z1) = poly[-2], poly[-1]
    return y0 + (y1 - y0) * (z - z0) / (z1 - z0)


def point_along(p, q, t: float):
    return (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)


def linspace(a: float, b: float, n: int) -> list[float]:
    return [a + (b - a) * i / (n - 1) for i in range(n)]


@dataclass
class MeshData:
    """Polygon soup with per-corner UVs, per-face slot and plank value, per-vertex hand wear."""

    verts: list = field(default_factory=list)
    faces: list = field(default_factory=list)
    uvs: list = field(default_factory=list)
    mats: list = field(default_factory=list)
    plank: list = field(default_factory=list)
    wear: list = field(default_factory=list)

    def add_vert(self, v, wear: float = 0.0) -> int:
        self.verts.append(tuple(v))
        self.wear.append(wear)
        return len(self.verts) - 1

    def add_face(self, idx, uv, mat: int, plank: float) -> None:
        self.faces.append(tuple(idx))
        self.uvs.append(tuple(uv))
        self.mats.append(mat)
        self.plank.append(plank)

    def extend(self, other: MeshData) -> None:
        base = len(self.verts)
        self.verts += other.verts
        self.wear += other.wear
        self.faces += [tuple(i + base for i in f) for f in other.faces]
        self.uvs += other.uvs
        self.mats += other.mats
        self.plank += other.plank

    def flip(self) -> None:
        self.faces = [tuple(reversed(f)) for f in self.faces]
        self.uvs = [tuple(reversed(u)) for u in self.uvs]

    def volume(self) -> float:
        vol = 0.0
        for f in self.faces:
            a = self.verts[f[0]]
            for i in range(1, len(f) - 1):
                b, c = self.verts[f[i]], self.verts[f[i + 1]]
                vol += (
                    a[0] * (b[1] * c[2] - b[2] * c[1])
                    - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0])
                )
        return vol / 6

    def orient_outward(self) -> MeshData:
        if self.volume() < 0:
            self.flip()
        return self

    def transformed(self, rot, offset) -> MeshData:
        """Apply a proper rotation (3x3 rows) and translation."""
        out = MeshData(
            [
                tuple(
                    sum(rot[r][c] * v[c] for c in range(3)) + offset[r]
                    for r in range(3)
                )
                for v in self.verts
            ],
            list(self.faces),
            list(self.uvs),
            list(self.mats),
            list(self.plank),
            list(self.wear),
        )
        return out

    def mirrored_y(self) -> MeshData:
        out = MeshData(
            [(x, -y, z) for x, y, z in self.verts],
            list(self.faces),
            list(self.uvs),
            list(self.mats),
            list(self.plank),
            list(self.wear),
        )
        out.flip()
        return out

    def bounds(self):
        xs, ys, zs = zip(*self.verts)
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def loft(
    xs,
    profile,
    *,
    edge_mats=None,
    cap_mats=(0, 0),
    grain: str = "x",
    plank: float = 0.0,
    wear=None,
) -> MeshData:
    """Closed tube along X through `profile(x)` polygons (constant point count), capped.

    UVs are metres: U along the grain (X for grain "x", around the profile for "across"),
    V the other way; caps use (y, z).
    """
    md = MeshData()
    rings = [profile(x) for x in xs]
    n = len(rings[0])
    edge_mats = edge_mats or [0] * n
    ids = [
        [md.add_vert((x, y, z), wear(x, y, z) if wear else 0.0) for y, z in ring]
        for x, ring in zip(xs, rings)
    ]
    perim = []
    for ring in rings:
        s = [0.0]
        for j in range(n):
            (y0, z0), (y1, z1) = ring[j], ring[(j + 1) % n]
            s.append(s[-1] + math.hypot(y1 - y0, z1 - z0))
        perim.append(s)
    for i in range(len(xs) - 1):
        for j in range(n):
            k = (j + 1) % n
            corners = [(i, j, j), (i, k, j + 1), (i + 1, k, j + 1), (i + 1, j, j)]
            uv = []
            for ii, jj, sj in corners:
                u, v = xs[ii], perim[ii][sj]
                uv.append((u, v) if grain == "x" else (v, u))
            md.add_face([ids[ii][jj] for ii, jj, _ in corners], uv, edge_mats[j], plank)
    for ring_i, mat, order in ((0, cap_mats[0], -1), (-1, cap_mats[1], 1)):
        idx = list(range(n))[::order]
        md.add_face(
            [ids[ring_i][j] for j in idx],
            [rings[ring_i][j] for j in idx],
            mat,
            plank,
        )
    return md.orient_outward()


def band(outer, inner, x0: float, x1: float, *, mat: int = 0, plank: float = 0.0):
    """Closed solid between two matching 2D polylines, extruded from x0 to x1 (frames)."""
    md = MeshData()
    n = len(outer)
    o0 = [md.add_vert((x0, y, z)) for y, z in outer]
    o1 = [md.add_vert((x1, y, z)) for y, z in outer]
    i0 = [md.add_vert((x0, y, z)) for y, z in inner]
    i1 = [md.add_vert((x1, y, z)) for y, z in inner]
    s = [0.0]
    for (y0, z0), (y1, z1) in pairwise(outer):
        s.append(s[-1] + math.hypot(y1 - y0, z1 - z0))
    for j in range(n - 1):
        k = j + 1
        md.add_face(
            (o0[j], o0[k], i0[k], i0[j]),
            ((s[j], 0), (s[k], 0), (s[k], 0.05), (s[j], 0.05)),
            mat,
            plank,
        )
        md.add_face(
            (o1[j], i1[j], i1[k], o1[k]),
            ((s[j], 0), (s[j], 0.05), (s[k], 0.05), (s[k], 0)),
            mat,
            plank,
        )
        md.add_face(
            (o0[j], o1[j], o1[k], o0[k]),
            ((s[j], 0), (s[j], x1 - x0), (s[k], x1 - x0), (s[k], 0)),
            mat,
            plank,
        )
        md.add_face(
            (i0[j], i0[k], i1[k], i1[j]),
            ((s[j], 0), (s[k], 0), (s[k], x1 - x0), (s[j], x1 - x0)),
            mat,
            plank,
        )
    for j, flip in ((0, False), (n - 1, True)):
        f = [o0[j], i0[j], i1[j], o1[j]]
        md.add_face(
            f[::-1] if flip else f,
            ((0, 0), (0.05, 0), (0.05, 0.05), (0, 0.05)),
            mat,
            plank,
        )
    return md.orient_outward()


def box(x0, x1, y0, y1, z0, z1, *, mat=0, grain="x", plank=0.0) -> MeshData:
    return loft(
        [x0, x1],
        lambda x: [(y0, z0), (y1, z0), (y1, z1), (y0, z1)],
        edge_mats=[mat] * 4,
        cap_mats=(mat, mat),
        grain=grain,
        plank=plank,
    )


def mirror_full(half):
    """Full section polyline from +Y top, through the centreline, to -Y top."""
    return list(reversed(half)) + [(-y, z) for y, z in half[1:]]


def ring_profile(radius_y: float, radius_z: float, n: int = 12):
    return [
        (
            radius_y * math.cos(2 * math.pi * k / n),
            radius_z * math.sin(2 * math.pi * k / n),
        )
        for k in range(n)
    ]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _unit(v):
    length = math.sqrt(sum(c * c for c in v))
    return tuple(c / length for c in v)


# Proper rotations mapping the loft axis (local X) elsewhere.
ROT_X_TO_Y = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
ROT_X_TO_Z = ((0, 0, -1), (0, 1, 0), (1, 0, 0))


def rot_z(angle: float):
    c, s = math.cos(angle), math.sin(angle)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def rot_y(angle: float):
    c, s = math.cos(angle), math.sin(angle)
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))


def matmul(a, b):
    return tuple(
        tuple(sum(a[r][k] * b[k][c] for k in range(3)) for c in range(3))
        for r in range(3)
    )


def _sub(a, b):
    return tuple(p - q for p, q in zip(a, b))


def _scale(a, k: float):
    return tuple(p * k for p in a)


def _dot(a, b) -> float:
    return sum(p * q for p, q in zip(a, b))


def path_frames(path, up=None):
    """Tangent and a-axis at every point of a 3D polyline: (tangents, axes). The axes are
    rotation-minimising (double reflection) starting from `up`, so what is swept along
    the path does not twist."""
    n = len(path)
    tangents = [
        _unit(_sub(path[min(i + 1, n - 1)], path[max(i - 1, 0)])) for i in range(n)
    ]
    r = (
        up
        if up is not None
        else ((0, 0, 1) if abs(tangents[0][2]) < 0.9 else (1, 0, 0))
    )
    if abs(_dot(_unit(r), tangents[0])) > 0.95:
        # The hint runs along the path: any other axis gives a valid starting frame.
        r = (1, 0, 0) if abs(tangents[0][0]) < 0.9 else (0, 1, 0)
    frames = [_unit(_sub(r, _scale(tangents[0], _dot(r, tangents[0]))))]
    for i in range(n - 1):
        v1 = _sub(path[i + 1], path[i])
        c1 = _dot(v1, v1)
        rl = _sub(frames[-1], _scale(v1, 2 * _dot(v1, frames[-1]) / c1))
        tl = _sub(tangents[i], _scale(v1, 2 * _dot(v1, tangents[i]) / c1))
        v2 = _sub(tangents[i + 1], tl)
        c2 = _dot(v2, v2)
        frames.append(
            _unit(rl if c2 < 1e-12 else _sub(rl, _scale(v2, 2 * _dot(v2, rl) / c2)))
        )
    return tangents, frames


def sweep(
    path, profile, *, up=None, mat: int = 0, plank: float = 0.0, scales=None
) -> MeshData:
    """Closed tube along a 3D polyline with a 2D (a, b) profile, capped at both ends.

    Frames are rotation-minimising (double reflection) starting from `up` as the a-axis, so
    rope and bent strips do not twist. U runs along the path in metres. `scales` (one per
    path point) multiplies the profile ring by ring, for tapering branches and roots; V
    then runs around the unscaled profile.
    """
    n = len(path)
    tangents, frames = path_frames(path, up)
    md = MeshData()
    m = len(profile)
    rings = []
    for i, (p, t, a_axis) in enumerate(zip(path, tangents, frames)):
        b_axis = _cross(t, a_axis)
        s = 1.0 if scales is None else scales[i]
        rings.append(
            [
                md.add_vert(
                    tuple(
                        p[k] + a_axis[k] * a * s + b_axis[k] * b * s for k in range(3)
                    )
                )
                for a, b in profile
            ]
        )
    along = [0.0]
    for a, b in pairwise(path):
        along.append(along[-1] + math.dist(a, b))
    around = [0.0]
    for j in range(m):
        around.append(around[-1] + math.dist(profile[j], profile[(j + 1) % m]))
    for i in range(n - 1):
        for j in range(m):
            k = (j + 1) % m
            md.add_face(
                (rings[i][j], rings[i][k], rings[i + 1][k], rings[i + 1][j]),
                (
                    (along[i], around[j]),
                    (along[i], around[j + 1]),
                    (along[i + 1], around[j + 1]),
                    (along[i + 1], around[j]),
                ),
                mat,
                plank,
            )
    md.add_face(tuple(reversed(rings[0])), tuple(reversed(profile)), mat, plank)
    md.add_face(tuple(rings[-1]), tuple(profile), mat, plank)
    return md.orient_outward()


def spline(points, step: float):
    """Catmull-Rom through control points (x, y, z, radius), resampled about every `step`
    metres; radius rides along as a fourth coordinate."""
    pts = [tuple(p) for p in points]
    ext = [pts[0], *pts, pts[-1]]
    out = [pts[0]]
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        n = max(1, math.ceil(math.dist(p1[:3], p2[:3]) / step))
        for k in range(1, n + 1):
            t = k / n
            t2, t3 = t * t, t * t * t
            out.append(
                tuple(
                    0.5
                    * (
                        2 * p1[c]
                        + (p2[c] - p0[c]) * t
                        + (2 * p0[c] - 5 * p1[c] + 4 * p2[c] - p3[c]) * t2
                        + (3 * p1[c] - p0[c] - 3 * p2[c] + p3[c]) * t3
                    )
                    for c in range(4)
                )
            )
    return out


def tube(path, radii, sides: int, *, up=None):
    """Capped round tube along `path` with one radius per point (`sweep` with a scaled
    circle). Ring i's vertices are i * sides ... i * sides + sides - 1; returns
    (mesh, sides)."""
    r0 = max(radii)
    profile = [
        (r0 * math.cos(2 * math.pi * k / sides), r0 * math.sin(2 * math.pi * k / sides))
        for k in range(sides)
    ]
    md = sweep(list(path), profile, up=up, scales=[r / r0 for r in radii])
    return md, sides


def axis_frame(start, through):
    """Rotation whose local X runs from `start` through `through`, with local Y horizontal."""
    d = _unit(tuple(through[i] - start[i] for i in range(3)))
    y = _unit(_cross((0, 0, 1), d))
    z = _cross(d, y)
    rot = tuple(tuple((d, y, z)[c][r] for c in range(3)) for r in range(3))
    return rot, tuple(start), d


@dataclass
class Part:
    name: str
    slots: tuple
    mesh: MeshData


def seeded(namespace: str, seed: int, key: str) -> random.Random:
    """Deterministic RNG per (namespace, seed, key), independent of call order."""
    return random.Random(f"{namespace}/{seed}/{key}")


def rounded_rect(width: float, height: float, radius: float, segments: int = 4):
    """Counter-clockwise rounded rectangle centred on the origin (2D points)."""
    radius = max(min(radius, width / 2 - 1e-4, height / 2 - 1e-4), 0.0)
    hw, hh = width / 2 - radius, height / 2 - radius
    corners = ((hw, hh, 0.0), (-hw, hh, 0.5), (-hw, -hh, 1.0), (hw, -hh, 1.5))
    pts = []
    for cx, cy, start in corners:
        for k in range(segments + 1):
            a = math.pi * (start + 0.5 * k / segments)
            pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    return pts


def circle(radius: float, segments: int = 24):
    return [
        (
            radius * math.cos(2 * math.pi * k / segments),
            radius * math.sin(2 * math.pi * k / segments),
        )
        for k in range(segments)
    ]


def prism(
    polygon, z0: float, z1: float, *, mat: int = 0, plank: float = 0.0, grain="x"
):
    """Closed prism of a convex-or-simple 2D (x, y) polygon between heights z0 and z1."""
    local = loft(
        [z0, z1],
        lambda z: [(y, -x) for x, y in polygon],
        edge_mats=[mat] * len(polygon),
        cap_mats=(mat, mat),
        grain=grain,
        plank=plank,
    )
    return local.transformed(ROT_X_TO_Z, (0, 0, 0))


def ring(outer, inner, depth: float, *, mat: int = 0, plank: float = 0.0) -> MeshData:
    """Closed frame between two matching closed 2D (x, y) loops, extruded 0..depth along +Z
    (window frames, porthole rims)."""
    md = MeshData()
    n = len(outer)
    o0 = [md.add_vert((x, y, 0.0)) for x, y in outer]
    o1 = [md.add_vert((x, y, depth)) for x, y in outer]
    i0 = [md.add_vert((x, y, 0.0)) for x, y in inner]
    i1 = [md.add_vert((x, y, depth)) for x, y in inner]
    for j in range(n):
        k = (j + 1) % n
        for quad in (
            (o0[j], o0[k], o1[k], o1[j]),
            (i0[j], i1[j], i1[k], i0[k]),
            (o0[j], i0[j], i0[k], o0[k]),
            (o1[j], o1[k], i1[k], i1[j]),
        ):
            md.add_face(quad, [md.verts[i][:2] for i in quad], mat, plank)
    return md.orient_outward()


def lathe(profile, segments: int = 24, *, mat: int = 0, plank: float = 0.0) -> MeshData:
    """Solid of revolution about +Z from (radius, z) samples ordered bottom to top; radii must
    stay positive (end caps close the solid)."""
    radii = iter([r for r, _ in profile])
    # loft samples each station once, in order: taking radii by position rather than keying
    # them by z keeps flat moulding steps, where two samples share a height.
    local = loft(
        [z for _, z in profile],
        lambda z: circle(next(radii), segments),
        edge_mats=[mat] * segments,
        cap_mats=(mat, mat),
        plank=plank,
    )
    return local.transformed(ROT_X_TO_Z, (0, 0, 0))


def slice_segments(md: MeshData, axis: int, value: float):
    """Where the plane (coordinate `axis` = value) cuts a mesh: one 2D segment per face it
    crosses, in the other two coordinates (x, y for a level cut; y, z across x). Fans
    each face into triangles; faces lying in the plane give nothing."""
    keep = [i for i in range(3) if i != axis]
    out = []
    for f in md.faces:
        pts = [md.verts[i] for i in f]
        for k in range(1, len(pts) - 1):
            tri = (pts[0], pts[k], pts[k + 1])
            d = [p[axis] - value for p in tri]
            hits = []
            for i in range(3):
                j = (i + 1) % 3
                if (d[i] < 0) != (d[j] < 0):
                    t = d[i] / (d[i] - d[j])
                    a, b = tri[i], tri[j]
                    hits.append(tuple(a[c] + (b[c] - a[c]) * t for c in keep))
            if len(hits) == 2:
                out.append((hits[0], hits[1]))
    return out
