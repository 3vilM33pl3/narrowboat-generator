# SPDX-License-Identifier: MIT
"""Tests for the weathered NB57's wear maps: the girth and roof coordinates, where each
kind of wear starts and is allowed to go, coverage, touch-ups, determinism and the files.
Rasters are checked at 8 mm per pixel to keep the suite quick."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SCRIPTS))

import narrowboat_common as N
import narrowboat_design as D
import narrowboat_lines as L
import narrowboat_wear as W

WEAR = W.WEARS["weathered"]
PX = 8.0
CABIN_KINDS = ("corner", "sill", "porthole", "hatch", "roofspot", "dirt")


@pytest.fixture(scope="module")
def src():
    return W.sources(WEAR, 1)


@pytest.fixture(scope="module")
def lay():
    return W.layers(WEAR, 1, px=PX)


def test_girth_is_b_along_the_parallel_body_and_unrolls_the_ends():
    for b in (-900.0, 0.0, 5000.0, 12000.0):
        assert abs(float(W.u_wall(b)) - b) < 1e-6
    bs = np.linspace(1000 * W.X_TIP, D.STEM, 400)
    assert np.all(np.diff(W.u_wall(bs)) > 0)
    # Round the counter and the bow the girth runs ahead of b.
    assert float(W.u_wall(1000 * W.X_TIP)) < 1000 * W.X_TIP - 300
    assert float(W.u_wall(D.STEM)) > D.STEM + 100
    # The end plate joins the counter wall at its corner.
    assert (
        abs(float(W.u_plate(1000 * W.W_END)) - float(W.u_wall(1000 * W.X_TIP))) < 1e-6
    )
    assert abs(float(W.u_plate(0.0)) - 1000 * W.U0) < 1e-9


def test_girth_lut_decodes_to_the_side_u():
    lut, info = W.girth_lut()
    assert np.all(np.diff(lut) > 0) and lut[0] == 0 and abs(lut[-1] - 1) < 1e-12
    for b in (-1.5, 0.0, 6.0, 14.5):
        x = N.wx(1000 * b)
        t = np.interp(x, np.linspace(info["x0"], info["x1"], info["size"]), lut)
        u = info["u0"] + t * (info["u1"] - info["u0"]) + info["w_end"]
        assert abs(1000 * u - float(W.u_wall(1000 * b))) < 2.0


def test_frames_round_trip_and_cover_the_boat():
    fr = W.frames(PX)
    for f in fr.values():
        cols = np.array([0.0, 10.5, f.w - 1])
        assert np.allclose(f.col(f.u(cols)), cols)
        assert np.allclose(f.row(f.v(cols)), cols)
        u, v = f.grid()
        assert abs(u[0, 0] - (f.u0 + 0.5 * f.px)) < 1e-3
        assert abs(v[0, 0] - (f.v1 - 0.5 * f.px)) < 1e-3
    side, plan = fr["side"], fr["plan"]
    assert side.u0 < 1000 * W.U0 and side.u1 > float(W.u_wall(D.STEM))
    assert side.v1 >= D.Z_SIDE_TOP + D.CORNER_R and side.v0 <= 0
    assert plan.u0 < D.STERN_TIP - D.FENDER_AFT and plan.u1 > D.STEM + D.FENDER_FWD
    assert plan.v1 > D.HALF_BEAM + D.STRAKE_PROUD


def test_roof_v_wraps_the_corner():
    assert abs(float(W.roof_v(400.0, 2100.0)) - 400.0) < 1e-9
    assert abs(float(W.roof_v(-400.0, 2100.0)) + 400.0) < 1e-9
    corner = float(W.roof_v(1000 * L.CABIN_TOP_HALF, D.Z_SIDE_TOP + 1e-6))
    assert abs(corner - W.V_ROOF) < 0.5
    top = float(W.roof_v(W.ROOF_YC + 1e-3, D.Z_SIDE_TOP + D.CORNER_R))
    assert abs(top - W.ROOF_YC) < 0.5


def test_window_runs_hang_from_their_frames_and_stop_at_the_gunwale(src):
    for side, where in W.SIDES:
        for run in [r for r in src.runs if r.where == where and r.kind == "corner"]:
            ops = [
                op
                for op in W.openings(side)
                if W.frame_box(op)[0] < run.u < W.frame_box(op)[1]
            ]
            assert ops, run
            edge = W.frame_edge(ops[0], run.u)
            # It leaves from the bottom of the teardrop of rust hanging off the corner.
            assert edge - 35 <= run.v <= edge, run
            assert abs(run.v - run.length - W.GUNWALE) < 1e-6
            assert any(
                bl.layer == "rust"
                and bl.kind == "corner"
                and abs(bl.u - run.u) < bl.ru + 1
                and bl.v > run.v
                for bl in src.blobs
                if bl.where == where
            )


def test_cabin_runs_never_start_in_frames_or_reach_the_hull(src):
    for side, where in W.SIDES:
        for run in src.runs:
            if run.where != where or run.kind not in CABIN_KINDS:
                continue
            assert not any(
                W.inside_frame(op, run.u, run.v, grow=-1) for op in W.openings(side)
            )
            assert run.v - run.length >= W.GUNWALE - 1e-6, run


def test_every_weeping_source_feeds_a_weld_cluster(src):
    for side, where in W.SIDES:
        clusters = src.welds[where]
        assert len(clusters) >= WEAR.weld_clusters[0]
        for run in [
            r for r in src.runs if r.where == where and r.kind in ("corner", "hatch")
        ]:
            assert any(c - l / 2 - 1 <= run.u <= c + l / 2 + 1 for c, l in clusters), (
                run
            )


def test_hull_runs_stop_above_the_waterline(src):
    for run in src.runs:
        if run.where in ("port", "starboard") and run.kind in (
            "hull",
            "knock",
            "scuff",
            "refresh",
            "shoulder",
        ):
            b = float(W.b_of_u(run.u))
            assert run.v - run.length > 1000 * L.wl_z(b / 1000) + 50, run


def test_nothing_is_drawn_inside_the_frames(lay):
    frame = lay.frames["side"]
    uu, zz = frame.grid()
    for side, where in W.SIDES:
        a = lay.side[where]
        inside = np.zeros(a["rust"].shape, bool)
        for op in W.openings(side):
            inside |= W.inside_frame(op, uu, zz, grow=-2 * PX)
        for k in ("rust", "stain", "dirt", "touch"):
            assert a[k][inside].max(initial=0) < 0.05, (where, k)


def near_frame(side, u, z, grow):
    boxes = [W.frame_box(op) for op in W.openings(side)]
    if D.SIDE_HATCH.side == side:
        boxes.append(W.hatch_box())
    return any(
        a0 - grow < u < a1 + grow and b0 - grow < z < b1 + grow
        for a0, a1, b0, b1 in boxes
    )


def test_rust_on_the_cabin_sides_only_starts_at_features(src):
    """Open cabin paint above the weld and below the roof edge only rusts at a frame or
    the hatch; everything else rusts at the foot, the roof edge or on the hull."""
    for side, where in W.SIDES:
        for item in [
            *src.of(where, W.Blob),
            *src.of(where, W.Patch),
            *src.of(where, W.Run),
        ]:
            if item.layer != "rust" or item.kind == "dent":
                continue
            u = getattr(item, "u", None)
            z = getattr(item, "v", None)
            if u is None:
                u, z = (item.u0 + item.u1) / 2, (item.v0 + item.v1) / 2
            on_cabin = W.u_wall(W.CABIN_B[0]) < u < W.u_wall(W.CABIN_B[1])
            if on_cabin and W.GUNWALE + 75 < z < W.SIDE_TOP - 30:
                assert near_frame(side, u, z, 40), item


def test_rasterised_cabin_rust_stays_by_its_features(lay):
    """Open cabin paint rusts by a frame or the hatch, in a dent, or where the paint has
    flaked off (rust only ever shows through a chip there)."""
    import narrowboat_dents as DN

    frame = lay.frames["side"]
    uu, zz = frame.grid()
    t = WEAR.threshold
    band = (zz > W.GUNWALE + 90) & (zz < W.SIDE_TOP - 45)
    dents = DN.dents(WEAR, 1)
    for side, where in W.SIDES:
        a = lay.side[where]
        rust = (a["rust"] > t) & band
        rust &= (uu > W.u_wall(W.CABIN_B[0])) & (uu < W.u_wall(W.CABIN_B[1]))
        rows, cols = np.nonzero(rust)
        for r, c in zip(rows[::7], cols[::7]):
            u, z = float(uu[0, c]), float(zz[r, 0])
            dented = any(
                d.side == side and (u - d.u) ** 2 + (z - d.z) ** 2 < d.r**2
                for d in dents
            )
            flaked = a["chip"][r, c] > 0.9 * t
            assert near_frame(side, u, z, 70) or dented or flaked, (where, r, c)


def test_coverage_is_in_range(lay):
    cov = W.coverage(WEAR, lay)
    for where in ("port", "starboard"):
        c = cov[where]
        # Hard-used: a few percent rusting through, more flaked, but most paint holds.
        assert 0.01 < c["cabin_rust"] < 0.08
        assert 0.02 < c["cabin_stain"] < 0.15
        assert 0.03 < c["cabin_chip"] < 0.15
        assert 0.0003 < c["cabin_touch"] < 0.08
    assert 0.8 * WEAR.moss_edge < cov["roof"]["moss"] < 0.1
    assert abs(cov["roof"]["algae"] - WEAR.algae) < 0.02


def test_touchups_cover_damage_but_never_decals(lay):
    frame = lay.frames["side"]
    uu, zz = frame.grid()
    for side, where in W.SIDES:
        a = lay.side[where]
        touch = a["touch"] > 0.05
        assert touch.any()
        for u0, u1, z0, z1 in W.decal_rects(side):
            inside = (uu > u0 + PX) & (uu < u1 - PX) & (zz > z0 + PX) & (zz < z1 - PX)
            assert not (touch & inside).any(), where
        _, cabin = W._cabin_mask(frame, side)
        hidden = touch & cabin
        for k in ("rust", "stain", "chip", "verd"):
            assert a[k][hidden].max(initial=0) == 0, (where, k)


def test_roof_moss_grows_where_water_and_debris_collect(lay):
    frame = lay.frames["plan"]
    bb, vv = frame.grid()
    roof = (bb >= 0) & (bb <= D.CABIN_LEN) & (np.abs(vv) <= W.V_ROOF)
    moss = (lay.plan["moss"] > lay.thresholds["moss"]) & roof
    assert moss.any()
    near = np.abs(vv) >= W.HANDRAIL_Y - 2 * PX
    for side in (1, -1):
        for b in W.handrail_feet():
            near = near | ((bb - b) ** 2 + (vv - side * W.HANDRAIL_Y) ** 2 <= 250**2)
    for b in D.VENT_BS:
        near = near | ((bb - b) ** 2 + vv**2 <= 200**2)
    for b in D.PLANK_BLOCKS:
        near = near | (
            (np.abs(bb - b) < 130) & (vv > D.BLOCK_Y[0] - 70) & (vv < D.BLOCK_Y[1] + 70)
        )
    pw = D.PLANK_W / 2 + 60
    near = near | (
        (bb > D.PLANK_B[0] - 60)
        & (bb < D.PLANK_B[1] + 60)
        & (np.abs(vv - D.PLANK_Y) < pw)
    )
    assert (moss & near).sum() >= 0.99 * moss.sum()
    edge = moss & (np.abs(vv) >= W.HANDRAIL_Y)
    assert edge.sum() >= 0.3 * moss.sum()
    # Along the edges, more on the port side than to starboard.
    assert (edge & (vv > 0)).sum() > (edge & (vv < 0)).sum()


def test_feet_polish_follows_footfall_not_whole_parts(lay):
    frame = lay.frames["plan"]
    bb, vv = frame.grid()
    polish = lay.plan["polish"]
    # The bow door threshold is walked on; the roof amidships is not.
    r, c = int(frame.row(0.0)), int(frame.col(12890.0))
    assert polish[r, c] > 0.4
    mid = (np.abs(bb - 6000) < 1500) & (np.abs(vv) < 500)
    assert polish[np.broadcast_to(mid, polish.shape)].max() < 0.05


def test_soot_drifts_aft_of_the_chimney(lay):
    frame = lay.frames["plan"]
    bb, vv = frame.grid()
    soot = lay.plan["soot"]
    mass = soot.sum()
    cb, cy = D.CHIMNEY
    assert (soot * bb).sum() / mass < cb
    near = (bb - cb) ** 2 + (vv - cy) ** 2 <= 3500**2
    assert soot[np.broadcast_to(near, soot.shape)].sum() >= 0.95 * mass


def test_wear_is_deterministic_per_seed(lay):
    again = W.layers(WEAR, 1, px=PX)
    other = W.layers(WEAR, 2, px=PX)
    for where in ("port", "starboard"):
        for k, v in lay.side[where].items():
            assert np.array_equal(v, again.side[where][k])
    assert not np.array_equal(lay.side["port"]["stain"], other.side["port"]["stain"])
    assert np.array_equal(lay.plan["moss"], again.plan["moss"])


def test_maps_and_files(lay, tmp_path):
    maps = W.wear_maps(lay)
    side = lay.frames["side"]
    assert maps["SideA"].shape == (2 * side.h, side.w, 3)
    assert maps["SideA"].dtype == np.uint8
    assert maps["PlanB"].shape == (lay.frames["plan"].h, lay.frames["plan"].w, 3)
    assert maps["Girth"].dtype == np.uint16 and maps["Girth"].shape == (1, 2048)
    record = W.write_maps(tmp_path, WEAR, lay, maps)
    for name, image in maps.items():
        info = record["maps"][name]
        back = np.asarray(Image.open(tmp_path / info["file"]))
        assert np.array_equal(back.reshape(image.shape).astype(image.dtype), image)
        assert len(info["sha256"]) == 64
    assert record["level"] == "weathered" and record["version"] == W.WEAR_VERSION
    assert set(record["thresholds"]) == {"potential", "moss", "algae"}
    assert record["girth"]["w_end"] == W.W_END


def test_porthole_picks_take_the_level_share_of_all_portholes():
    picks = W.porthole_picks(WEAR, 1)
    ports = {
        (side, float(op.b))
        for side, _ in W.SIDES
        for op in W.openings(side)
        if op.kind == "porthole"
    }
    assert len(picks["verd"]) == round(WEAR.verdigris * len(ports))
    assert len(picks["weep"]) == round(WEAR.porthole_weep * len(ports))
    assert picks["verd"] <= ports and picks["weep"] <= ports
    assert picks == W.porthole_picks(WEAR, 1)


def test_picked_portholes_get_verdigris_and_weeps(src):
    picks = W.porthole_picks(WEAR, 1)
    for side, where in W.SIDES:
        for op in W.openings(side):
            if op.kind != "porthole":
                continue
            key = (side, float(op.b))
            rim = [b for b in src.of(where, W.Blob) if b.kind == "rim" and b.u == op.b]
            tendrils = [
                r
                for r in src.of(where, W.Run)
                if r.kind == "rim" and abs(r.u - op.b) < 300
            ]
            weep = [
                b
                for b in src.of(where, W.Blob)
                if b.kind == "porthole" and abs(b.u - op.b) < 40
            ]
            assert bool(rim) == bool(tendrils) == (key in picks["verd"])
            assert bool(weep) == (key in picks["weep"])
            # Tendrils leave the rim narrow, then spread, break up and fade going down.
            for r in tendrils:
                assert r.width[1] >= 2 * r.width[0] and r.amp[1] < 0.1 * r.amp[0]
                assert r.beads >= 1.0


def test_camera_aims_find_the_seeded_wear(src):
    aims = W.camera_aims(WEAR, 1)
    assert (1, aims["porthole"]) in W.porthole_picks(WEAR, 1)["verd"]
    op = next(o for o in W.openings(-1) if float(o.b) == aims["weld"])
    assert op.kind != "porthole"
    hw, _, _ = W._frame_shape(op)
    assert any(abs(c - op.b) < hw + ln / 2 for c, ln in src.welds["starboard"])


def test_cabin_foot_rust_stays_off_the_hull_edge(src):
    """Amidships the hull's top edge shares (u, z) with the cabin foot in the side map,
    so the weld beads and their blisters stop just above it rather than ringing it."""
    frame = W.frames(PX)["side"]
    below = int(np.floor(frame.row(W.GUNWALE)))
    for _, where in W.SIDES:
        only = W.Sources(
            blobs=[b for b in src.of(where, W.Blob) if b.kind == "blister"],
            patches=[p for p in src.of(where, W.Patch) if p.kind == "weld"],
        )
        arrays = {"rust": np.zeros((frame.h, frame.w), np.float32)}
        W._draw(arrays, frame, only, where, W.seeded(W.NAMESPACE, 1, "test"))
        assert arrays["rust"].max() > WEAR.threshold
        assert arrays["rust"][below:].max() == 0


def test_weld_beads_have_ragged_tops_and_thin_out(src):
    frame = W.frames(2.0)["side"]
    patch = max(
        (p for p in src.of("starboard", W.Patch) if p.kind == "weld"),
        key=lambda p: p.u1 - p.u0,
    )
    assert patch.ragged > 0 and patch.taper > 0
    arrays = {"rust": np.zeros((frame.h, frame.w), np.float32)}
    only = W.Sources(patches=[patch])
    W._draw(arrays, frame, only, "starboard", W.seeded(W.NAMESPACE, 1, "test"))
    rust = arrays["rust"] > WEAR.threshold
    cols = np.nonzero(rust.any(axis=0))[0]
    tops = [np.nonzero(rust[:, c])[0].min() for c in cols]
    assert max(tops) - min(tops) >= 2  # rows: the top edge is not ruled
    # Thinned towards its ends: the potential at its very ends is well below the middle.
    c0, c1 = int(frame.col(patch.u0)), int(frame.col(patch.u1))
    span = arrays["rust"][:, c0:c1].max(axis=0)
    k = max(len(span) // 20, 1)
    middle = span[len(span) // 4 : 3 * len(span) // 4].mean()
    assert span[:k].mean() < 0.6 * middle and span[-k:].mean() < 0.6 * middle


def test_hull_scuffs_are_streaky_scrapes_bright_only_in_scratches(src):
    """Knocks are drawn out fore and aft; strake scuffs are long streaky scrapes that
    never reach the shaders' bright-steel level (threshold + 0.3), which only the
    scratches along them reach; nothing is keyed to the strakes' ends, where it stacked
    into a column."""
    steel = WEAR.threshold + 0.3
    aspects = [
        b.ru / b.rv
        for where in ("port", "starboard")
        for b in src.of(where, W.Blob)
        if b.layer == "chip" and b.kind == "knock"
    ]
    assert min(aspects) > 1.2 and float(np.median(aspects)) > 2.5
    for where in ("port", "starboard"):
        patches = src.of(where, W.Patch)
        scuffs = [p for p in patches if p.layer == "chip" and p.kind == "scuff"]
        assert scuffs
        for p in scuffs:
            assert p.amp * (1 + 0.6 * p.breakup) < steel and p.stretch >= 6, p
        assert not [p for p in patches if p.kind == "strake-end"]
        scratches = [r for r in src.of(where, W.Run) if r.kind == "scratch"]
        assert scratches and max(r.amp[0] for r in scratches) > steel


def test_roof_edge_runs_hang_from_rust_spots(src):
    for _, where in W.SIDES:
        runs = [r for r in src.of(where, W.Run) if r.kind == "roofspot"]
        assert runs
        spots = [b for b in src.of(where, W.Blob) if b.kind == "roofspot"]
        for r in runs:
            assert any(abs(b.u - r.u) < b.ru and b.v > r.v for b in spots), r
            assert r.v - r.length >= W.GUNWALE


def test_weld_beads_carry_soft_blotches_of_stain(src):
    """Above each stretch of weld rust, a few soft blotches of stain that creep up the
    plate (ellipses, so no ruled strip and no spikes), cut off at the cabin foot."""
    for _, where in W.SIDES:
        creep = [b for b in src.of(where, W.Blob) if b.kind == "creep"]
        for weld in [p for p in src.of(where, W.Patch) if p.kind == "weld"]:
            mine = [b for b in creep if weld.u0 <= b.u <= weld.u1]
            assert mine, weld
            for b in mine:
                assert b.layer == "stain" and b.ru > b.rv and b.floor == W.FOOT


def test_the_bow_shoulder_strakes_are_always_scuffed(src):
    shoulder = float(W.u_wall(D.STEM - 3000))
    for where in ("port", "starboard"):
        for zk, weight in W.STRAKES:
            if weight >= 0.6:
                assert any(
                    z == zk and b > shoulder for z, _, b, _ in src.scuffs[where]
                ), (where, zk)
