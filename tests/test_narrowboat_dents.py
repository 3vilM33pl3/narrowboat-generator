# SPDX-License-Identifier: MIT
"""Dents and rippled plating of the worn NB57: where they go, how deep, and that the
frames, openings and plate edges stay true."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import narrowboat_dents as DN
import narrowboat_wear as W

WEAR = W.WEARS["weathered"]


def test_dents_are_deterministic_per_seed():
    assert DN.dents(WEAR, 1) == DN.dents(WEAR, 1)
    assert DN.dents(WEAR, 1) != DN.dents(WEAR, 2)


def test_counts_and_depths():
    items = DN.dents(WEAR, 1)
    for side in (1, -1):
        hull = [d for d in items if d.side == side and d.surface == DN.HULL]
        cabin = [d for d in items if d.side == side and d.surface == DN.CABIN]
        lo = WEAR.dents_bow[0] + WEAR.dents_stern[0] + WEAR.dents_mid[0]
        assert len(hull) >= lo
        assert WEAR.dents_cabin[0] <= len(cabin) <= WEAR.dents_cabin[1]
        assert all(d.depth <= 26 for d in hull)
        assert all(d.depth <= 11 for d in cabin)
    # The bow takes the most knocks.
    u_mid = float(W.u_wall(9000))
    bow = [d for d in items if d.surface == DN.HULL and d.u > u_mid]
    assert sum(d.depth for d in bow) > 0.5 * sum(
        d.depth for d in items if d.surface == DN.HULL
    )


def test_cabin_dents_keep_clear_of_openings_and_art():
    for d in DN.dents(WEAR, 1):
        if d.surface == DN.CABIN:
            assert DN._clear_of_openings(d.side, d.u, d.z, d.r), d
            assert W.GUNWALE < d.z - 0.5 * d.r and d.z + d.r < W.SIDE_TOP


def test_hull_dents_sit_between_waterline_and_gunwale():
    for d in DN.dents(WEAR, 1):
        if d.surface == DN.HULL:
            assert DN.wl_mm(d.u) + 40 <= d.z < W.GUNWALE, d


def test_ripple_is_zero_on_frames_and_plate_edges():
    u = np.arange(0, 12 * DN.CABIN_FRAMES, DN.CABIN_FRAMES)
    for z in (W.GUNWALE + 300, W.GUNWALE + 600):
        assert np.allclose(DN.ripple(WEAR, 1, DN.HULL, u * 0 + 5000, 150.0), 0)
        assert np.abs(DN.ripple(WEAR, 1, DN.CABIN, u, z)).max() < 1e-6
    mid = DN.ripple(WEAR, 1, DN.CABIN, u + DN.CABIN_FRAMES / 2, W.GUNWALE + 500)
    assert mid.max() <= WEAR.ripple_cabin + 1e-9
    assert np.allclose(DN.ripple(WEAR, 1, DN.CABIN, 3000.0, W.SIDE_TOP - 10), 0)


def test_plate_stays_flat_round_the_openings():
    items = DN.dents(WEAR, 1)
    for side in (1, -1):
        for op in W.openings(side):
            u0, u1, z0, z1 = W.frame_box(op)
            uu, zz = np.meshgrid(np.linspace(u0, u1, 25), np.linspace(z0, z1, 25))
            under = W.inside_frame(op, uu, zz)
            push = DN.depth(WEAR, items, side, DN.CABIN, uu[under], zz[under])
            assert np.abs(push).max() < 0.05


def test_depth_peaks_at_a_dent():
    items = DN.dents(WEAR, 1)
    d = next(i for i in items if i.surface == DN.HULL)
    centre = DN.depth(WEAR, items, d.side, d.surface, d.u, d.z)
    assert centre >= d.depth - 1e-6
    outside = DN.depth(WEAR, [d], d.side, d.surface, d.u + 1.01 * d.r, d.z)
    ripple = DN.ripple(WEAR, d.side, d.surface, d.u + 1.01 * d.r, d.z)
    assert np.isclose(outside, ripple)


def test_dents_leave_paint_damage():
    items = DN.dents(WEAR, 1)
    src = DN.dent_sources(WEAR, items, 1)
    chips = [b for b in src.blobs if b.layer == "chip"]
    assert len(chips) == len(items)
    rust = [b for b in src.blobs if b.layer == "rust"]
    assert len(src.runs) == len(rust) > 0
