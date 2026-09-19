"""Step8 spec section 2.5: one binary encoding shared by every representation (a
"pure" DP+SED polyline, a "pure" spline, and -- from M2 on -- a hybrid made of
several segments of either kind), so byte-count comparisons between methods are
never biased by using a more generous format for one of them.

Stream layout: `varint(num_segments)`, then per segment `[type: 1 byte][count:
varint]` followed by its quantized, delta-encoded payload. Coordinates are
quantized to 1 cm, time to 10 ms (`COORD_SCALE`/`TIME_SCALE` = 100, matching
step3's own `BYTE_SCALE` -- this formalizes step3's ad hoc scheme into a
reusable container, not a new quantization choice). The whole concatenated
stream is compressed once with `zlib` level 9.

Segment type is stored as a full byte rather than a literal bit (the spec's own
wording, section 2.5): zlib compresses the redundant 7 bits away, and the
simplification is applied identically to every method, so it does not bias the
fairness comparison the spec's last sentence in 2.5 is actually protecting
(ADR-0021).

Boundary dedup: segment `i > 0`'s own first (time, point) -- a line segment's
first vertex, or a spline segment's `(t_start, first control point)` -- is not
re-stored, since by construction (spec section 2.1's shared cut points `c_i`)
it is identical to segment `i - 1`'s last (time, point) -- a line segment's
last vertex, or a spline segment's `(t_end, last control point)`. A clamped
cubic spline's first/last control point exactly equals `C(t_start)`/`C(t_end)`
(the clamping property), so both segment kinds share the same boundary-point
notion. `encode_segments` verifies this equality (to quantization precision)
and raises `ValueError` if a caller passes segments that do not actually share
their boundary -- a silent mismatch here would corrupt every later segment's
carried-forward boundary point, the same class of bug ADR-0014/ADR-0020 guard
against elsewhere in this codebase.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

COORD_SCALE = 100.0  # 1 cm
TIME_SCALE = 100.0  # 10 ms
_INT32_SAFE_BOUND = 2_000_000_000.0
_BOUNDARY_ATOL_M = 1e-6  # boundary-equality check is on FLOAT inputs, before quantization
_BOUNDARY_ATOL_S = 1e-6

_TYPE_LINE = 0
_TYPE_SPLINE = 1


@dataclass
class LineSegment:
    t: np.ndarray  # (n,) vertex times, ascending
    xy: np.ndarray  # (n, 2) vertex coords


@dataclass
class SplineSegment:
    t_start: float
    t_end: float
    internal_knots: np.ndarray  # (m,) interior knot times, ascending, strictly inside (t_start, t_end)
    control_xy: np.ndarray  # (m + 4, 2) control points; degree fixed at 3 (spline_lsq.DEGREE), not stored


# ------------------------------------------------------------------ varint --

def _write_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint encoding requires a non-negative value")
    out = bytearray()
    while True:
        b = value & 0x7F
        value >>= 7
        if value:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


# ------------------------------------------------------- quantize / delta --

def _quantize(values: np.ndarray, scale: float) -> np.ndarray:
    """float -> scaled int32, clipped the same way step3's `_stream_bytes` is
    (a candidate that blew up numerically is rejected at the honest-check
    stage anyway; the clip just keeps this path from raising on overflow)."""
    scaled = np.clip(np.asarray(values, dtype=float) * scale, -_INT32_SAFE_BOUND, _INT32_SAFE_BOUND)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=_INT32_SAFE_BOUND, neginf=-_INT32_SAFE_BOUND)
    return np.round(scaled).astype(np.int32)


def _delta_encode(ints: np.ndarray) -> bytes:
    if len(ints) == 0:
        return b""
    deltas = np.empty_like(ints)
    deltas[0] = ints[0]
    deltas[1:] = np.diff(ints)
    return deltas.tobytes()


def _delta_decode(buf: bytes, pos: int, count: int) -> tuple[np.ndarray, int]:
    if count == 0:
        return np.empty(0, dtype=np.int32), pos
    nbytes = count * 4
    deltas = np.frombuffer(buf, dtype=np.int32, count=count, offset=pos)
    ints = np.cumsum(deltas.astype(np.int64)).astype(np.int32)
    return ints, pos + nbytes


def _write_stream(values: np.ndarray, scale: float) -> bytes:
    return _delta_encode(_quantize(values, scale))


def _read_stream(buf: bytes, pos: int, count: int, scale: float) -> tuple[np.ndarray, int]:
    ints, pos = _delta_decode(buf, pos, count)
    return ints.astype(np.float64) / scale, pos


def _check_boundary(prev_t: float, prev_xy: np.ndarray, t: float, xy: np.ndarray) -> None:
    if abs(t - prev_t) > _BOUNDARY_ATOL_S or float(np.hypot(*(xy - prev_xy))) > _BOUNDARY_ATOL_M:
        raise ValueError(
            f"segment does not share its boundary with the previous one: "
            f"prev=({prev_t}, {tuple(prev_xy)}) vs. this=({t}, {tuple(xy)})"
        )


# --------------------------------------------------------------- encode --

def encode_segments(segments: list[LineSegment | SplineSegment]) -> bytes:
    if not segments:
        raise ValueError("encode_segments requires at least one segment")

    parts = [_write_varint(len(segments))]
    prev_t = prev_xy = None

    for i, seg in enumerate(segments):
        if isinstance(seg, LineSegment):
            t = np.asarray(seg.t, dtype=float)
            xy = np.asarray(seg.xy, dtype=float)
            if i > 0:
                _check_boundary(prev_t, prev_xy, t[0], xy[0])
                t, xy = t[1:], xy[1:]
            parts.append(bytes([_TYPE_LINE]))
            parts.append(_write_varint(len(t)))
            parts.append(_write_stream(t, TIME_SCALE))
            parts.append(_write_stream(xy[:, 0], COORD_SCALE))
            parts.append(_write_stream(xy[:, 1], COORD_SCALE))
            prev_t = float(seg.t[-1])
            prev_xy = np.asarray(seg.xy[-1], dtype=float)
        elif isinstance(seg, SplineSegment):
            m = len(seg.internal_knots)
            cxy = np.asarray(seg.control_xy, dtype=float)
            if len(cxy) != m + 4:
                raise ValueError(f"spline segment: expected {m + 4} control points for {m} internal knots, got {len(cxy)}")
            parts.append(bytes([_TYPE_SPLINE]))
            parts.append(_write_varint(m))
            if i == 0:
                parts.append(_write_stream(np.array([seg.t_start]), TIME_SCALE))
            else:
                _check_boundary(prev_t, prev_xy, seg.t_start, cxy[0])
            parts.append(_write_stream(np.array([seg.t_end]), TIME_SCALE))
            parts.append(_write_stream(np.asarray(seg.internal_knots, dtype=float), TIME_SCALE))
            cx, cy = (cxy[:, 0], cxy[:, 1]) if i == 0 else (cxy[1:, 0], cxy[1:, 1])
            parts.append(_write_stream(cx, COORD_SCALE))
            parts.append(_write_stream(cy, COORD_SCALE))
            prev_t = float(seg.t_end)
            prev_xy = cxy[-1]
        else:
            raise TypeError(f"unknown segment type: {type(seg)!r}")

    return zlib.compress(b"".join(parts), 9)


# --------------------------------------------------------------- decode --

def decode_segments(blob: bytes) -> list[LineSegment | SplineSegment]:
    buf = zlib.decompress(blob)
    pos = 0
    num_segments, pos = _read_varint(buf, pos)

    segments: list[LineSegment | SplineSegment] = []
    prev_t = prev_xy = None

    for i in range(num_segments):
        seg_type = buf[pos]
        pos += 1
        count, pos = _read_varint(buf, pos)

        if seg_type == _TYPE_LINE:
            t, pos = _read_stream(buf, pos, count, TIME_SCALE)
            x, pos = _read_stream(buf, pos, count, COORD_SCALE)
            y, pos = _read_stream(buf, pos, count, COORD_SCALE)
            xy = np.column_stack([x, y])
            if i > 0:
                t = np.concatenate([[prev_t], t])
                xy = np.vstack([prev_xy, xy])
            segments.append(LineSegment(t=t, xy=xy))
            prev_t = float(t[-1])
            prev_xy = xy[-1]
        elif seg_type == _TYPE_SPLINE:
            m = count
            if i == 0:
                t_start_arr, pos = _read_stream(buf, pos, 1, TIME_SCALE)
                t_start = float(t_start_arr[0])
            else:
                t_start = prev_t
            t_end_arr, pos = _read_stream(buf, pos, 1, TIME_SCALE)
            t_end = float(t_end_arr[0])
            internal_knots, pos = _read_stream(buf, pos, m, TIME_SCALE)
            cp_count = (m + 4) if i == 0 else (m + 3)
            cx, pos = _read_stream(buf, pos, cp_count, COORD_SCALE)
            cy, pos = _read_stream(buf, pos, cp_count, COORD_SCALE)
            control_xy = np.column_stack([cx, cy])
            if i > 0:
                control_xy = np.vstack([prev_xy, control_xy])
            segments.append(
                SplineSegment(t_start=t_start, t_end=t_end, internal_knots=internal_knots, control_xy=control_xy)
            )
            prev_t = t_end
            prev_xy = control_xy[-1]
        else:
            raise ValueError(f"unknown segment type byte: {seg_type}")

    return segments
