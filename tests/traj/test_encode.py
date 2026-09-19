"""Tests for traj.encode (step8 spec section 2.5): the only section-9 property
that's M0's job is "encoding is reversible" -- the others (hybrid vs. DP+SED on
straight lines, hybrid vs. oracle on circles, knot removal never exceeding tol,
DP vs. brute force, shift invariance) need hybrid.py/singular.py/knot_removal.py,
which belong to M1/M2."""

from __future__ import annotations

import zlib

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from traj.encode import (
    COORD_SCALE,
    TIME_SCALE,
    LineSegment,
    SplineSegment,
    decode_segments,
    encode_segments,
)

_coord = st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False, width=64)


def _line(n: int, seed: int) -> LineSegment:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 1000.0, n))
    xy = rng.uniform(-1000.0, 1000.0, (n, 2))
    return LineSegment(t=t, xy=xy)


def _spline(m: int, seed: int, t_start: float = 0.0, t_end: float = 100.0) -> SplineSegment:
    rng = np.random.default_rng(seed)
    internal = np.sort(rng.uniform(t_start + 0.1, t_end - 0.1, m)) if m > 0 else np.empty(0)
    control_xy = rng.uniform(-1000.0, 1000.0, (m + 4, 2))
    return SplineSegment(t_start=t_start, t_end=t_end, internal_knots=internal, control_xy=control_xy)


def _assert_close(a: np.ndarray, b: np.ndarray, atol_t=None, atol_xy=None):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    atol = atol_t if atol_t is not None else atol_xy
    assert a.shape == b.shape
    if a.size == 0:
        return
    assert np.max(np.abs(a - b)) <= atol + 1e-9


def test_round_trip_single_line_segment():
    seg = _line(12, seed=1)
    blob = encode_segments([seg])
    [out] = decode_segments(blob)
    assert isinstance(out, LineSegment)
    _assert_close(out.t, seg.t, atol_t=1.0 / TIME_SCALE)
    _assert_close(out.xy, seg.xy, atol_xy=1.0 / COORD_SCALE)


def test_round_trip_single_spline_segment():
    seg = _spline(5, seed=2)
    blob = encode_segments([seg])
    [out] = decode_segments(blob)
    assert isinstance(out, SplineSegment)
    assert abs(out.t_start - seg.t_start) <= 1.0 / TIME_SCALE + 1e-9
    assert abs(out.t_end - seg.t_end) <= 1.0 / TIME_SCALE + 1e-9
    _assert_close(out.internal_knots, seg.internal_knots, atol_t=1.0 / TIME_SCALE)
    _assert_close(out.control_xy, seg.control_xy, atol_xy=1.0 / COORD_SCALE)


def test_round_trip_two_line_segments_sharing_boundary():
    seg0 = _line(6, seed=3)
    boundary_t = float(seg0.t[-1])
    boundary_xy = seg0.xy[-1].copy()
    seg1 = _line(5, seed=4)
    seg1.t = np.concatenate([[boundary_t], boundary_t + 1.0 + np.sort(np.random.default_rng(5).uniform(0.0, 50.0, 4))])
    seg1.xy = np.vstack([boundary_xy, seg1.xy[1:]])

    blob = encode_segments([seg0, seg1])
    out0, out1 = decode_segments(blob)
    _assert_close(out0.t, seg0.t, atol_t=1.0 / TIME_SCALE)
    _assert_close(out0.xy, seg0.xy, atol_xy=1.0 / COORD_SCALE)
    _assert_close(out1.t, seg1.t, atol_t=1.0 / TIME_SCALE)
    _assert_close(out1.xy, seg1.xy, atol_xy=1.0 / COORD_SCALE)
    # the shared boundary vertex isn't stored twice: the RAW (pre-zlib) stream
    # for the 2-segment encoding must be exactly 2 bytes longer than a single
    # (n0 + n1 - 1)-vertex line's raw stream -- one extra type byte and one
    # extra count-varint byte for the second segment's header, and nothing
    # else (no duplicated vertex payload).
    solo = LineSegment(t=np.concatenate([seg0.t, seg1.t[1:]]), xy=np.vstack([seg0.xy, seg1.xy[1:]]))
    raw_two = zlib.decompress(encode_segments([seg0, seg1]))
    raw_solo = zlib.decompress(encode_segments([solo]))
    assert len(raw_two) == len(raw_solo) + 2


def test_round_trip_line_then_spline_sharing_boundary():
    seg0 = _line(6, seed=6)
    boundary_t = float(seg0.t[-1])
    boundary_xy = seg0.xy[-1].copy()
    seg1 = _spline(4, seed=7, t_start=boundary_t, t_end=boundary_t + 100.0)
    seg1.control_xy[0] = boundary_xy

    blob = encode_segments([seg0, seg1])
    out0, out1 = decode_segments(blob)
    _assert_close(out0.t, seg0.t, atol_t=1.0 / TIME_SCALE)
    _assert_close(out0.xy, seg0.xy, atol_xy=1.0 / COORD_SCALE)
    assert abs(out1.t_start - seg1.t_start) <= 1.0 / TIME_SCALE + 1e-9
    assert abs(out1.t_end - seg1.t_end) <= 1.0 / TIME_SCALE + 1e-9
    _assert_close(out1.control_xy, seg1.control_xy, atol_xy=1.0 / COORD_SCALE)


def test_mismatched_boundary_raises():
    seg0 = _line(4, seed=8)
    seg1 = _line(4, seed=9)  # unrelated, does NOT share seg0's last vertex
    with pytest.raises(ValueError, match="does not share its boundary"):
        encode_segments([seg0, seg1])


def test_degenerate_two_point_line_segment():
    seg = LineSegment(t=np.array([0.0, 1.0]), xy=np.array([[0.0, 0.0], [1.0, 1.0]]))
    blob = encode_segments([seg])
    [out] = decode_segments(blob)
    _assert_close(out.t, seg.t, atol_t=1.0 / TIME_SCALE)
    _assert_close(out.xy, seg.xy, atol_xy=1.0 / COORD_SCALE)


def test_degenerate_zero_internal_knot_spline_segment():
    seg = _spline(0, seed=10)
    assert seg.control_xy.shape == (4, 2)
    blob = encode_segments([seg])
    [out] = decode_segments(blob)
    assert len(out.internal_knots) == 0
    _assert_close(out.control_xy, seg.control_xy, atol_xy=1.0 / COORD_SCALE)


def test_encode_segments_rejects_empty_list():
    with pytest.raises(ValueError):
        encode_segments([])


@given(
    n=st.integers(min_value=2, max_value=15),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
def test_line_round_trip_property(n, seed):
    seg = _line(n, seed)
    [out] = decode_segments(encode_segments([seg]))
    _assert_close(out.t, seg.t, atol_t=1.0 / TIME_SCALE)
    _assert_close(out.xy, seg.xy, atol_xy=1.0 / COORD_SCALE)


@given(
    m=st.integers(min_value=0, max_value=10),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
def test_spline_round_trip_property(m, seed):
    seg = _spline(m, seed)
    [out] = decode_segments(encode_segments([seg]))
    _assert_close(out.internal_knots, seg.internal_knots, atol_t=1.0 / TIME_SCALE)
    _assert_close(out.control_xy, seg.control_xy, atol_xy=1.0 / COORD_SCALE)
