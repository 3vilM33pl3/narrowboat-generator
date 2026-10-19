# SPDX-License-Identifier: MIT
"""Mesh checks shared by the tests."""

from __future__ import annotations

from collections import Counter


def assert_closed_outward(part, slots):
    """Every edge used once in each direction (closed, consistently wound), positive volume
    (outward), and per-face/per-vertex attributes that line up with the faces and slots."""
    md = part.mesh
    directed = Counter()
    for f in md.faces:
        for i in range(len(f)):
            directed[(f[i], f[(i + 1) % len(f)])] += 1
    assert max(directed.values()) == 1, part.name
    assert all((b, a) in directed for a, b in directed), part.name
    assert md.volume() > 0, part.name
    assert len(md.uvs) == len(md.faces) == len(md.mats) == len(md.plank)
    assert len(md.wear) == len(md.verts)
    assert all(len(u) == len(f) for u, f in zip(md.uvs, md.faces))
    assert max(md.mats) < len(part.slots), part.name
    assert set(part.slots) <= set(slots), part.name
