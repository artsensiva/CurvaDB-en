"""GPS track cleaning: time/speed breaks, single-point outliers, Beijing
bbox, length refilter.

Motivation (benchmarks/results/step0_diagnostics.md,
benchmarks/diagnose_fit.py): anomalous GPS jumps and large time gaps
inside a "track" are the main reason the spline can't keep tol between
timestamps. We cut a track into segments wherever the raw data physically
cannot be one continuous track.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from traj.io import DEFAULT_DATA_DIR, DEFAULT_SEED, MAX_POINTS, MIN_POINTS, Track, load_tracks

MAX_DT_S = 30.0
MAX_SPEED_MPS = 70.0
# Beijing (municipality), approximate -- GeoLife was recorded almost
# entirely within these bounds; guards against isolated geocoding outliers.
BEIJING_BBOX = (39.4, 41.6, 115.7, 117.5)  # lat_min, lat_max, lon_min, lon_max

# For load_clean_tracks: load raw tracks with no length cap -- the
# 50..2000 filter is applied AFTER splitting, to the segments, not to the
# raw track.
_RAW_MIN_POINTS = 1
_RAW_MAX_POINTS = 1_000_000


@dataclass
class CleanStats:
    n_tracks_in: int = 0
    n_points_in: int = 0
    n_split_segments: int = 0  # extra segments produced by break-splitting
    n_outliers_removed: int = 0  # single-point segments
    n_points_dropped_bbox: int = 0
    n_dropped_short_or_long: int = 0  # segments that failed the length refilter
    n_tracks_out: int = 0
    n_points_out: int = 0


def _segment_breaks(t: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Indices where a new segment starts (a dt or speed break precedes it)."""
    dt = np.diff(t)
    dist = np.hypot(*np.diff(xy, axis=0).T)
    speed = np.divide(dist, dt, out=np.full_like(dist, np.inf), where=dt > 0)
    break_mask = (dt > MAX_DT_S) | (speed > MAX_SPEED_MPS)
    return np.nonzero(break_mask)[0] + 1


def _split_track(track: Track) -> list[Track]:
    breaks = _segment_breaks(track.t, track.xy)
    bounds = [0, *breaks.tolist(), len(track.t)]
    if len(bounds) <= 2:
        return [track]
    segments = []
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        segments.append(
            Track(
                track_id=f"{track.track_id}_seg{i}",
                lat=track.lat[a:b], lon=track.lon[a:b],
                t=track.t[a:b] - track.t[a], xy=track.xy[a:b],
            )
        )
    return segments


def _drop_single_point_outliers(segments: list[Track]) -> list[Track]:
    return [s for s in segments if len(s.t) > 1]


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[a, b) index ranges of contiguous True runs in mask."""
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return []
    gaps = np.nonzero(np.diff(idx) > 1)[0]
    starts = [idx[0]] + [idx[g + 1] for g in gaps]
    ends = [idx[g] + 1 for g in gaps] + [idx[-1] + 1]
    return list(zip(starts, ends))


def _apply_bbox(segments: list[Track], bbox: tuple[float, float, float, float]) -> list[Track]:
    """Keeps only points inside bbox; splits into sub-segments at the
    points where a segment leaves the bbox, instead of just dropping
    points (otherwise a hidden gap would appear inside the "segment"
    that the dt/speed split would no longer see)."""
    lat_min, lat_max, lon_min, lon_max = bbox
    out = []
    for s in segments:
        mask = (s.lat >= lat_min) & (s.lat <= lat_max) & (s.lon >= lon_min) & (s.lon <= lon_max)
        if mask.all():
            out.append(s)
            continue
        for j, (a, b) in enumerate(_contiguous_runs(mask)):
            out.append(
                Track(
                    track_id=f"{s.track_id}_bbox{j}",
                    lat=s.lat[a:b], lon=s.lon[a:b],
                    t=s.t[a:b] - s.t[a], xy=s.xy[a:b],
                )
            )
    return out


def _refilter_length(segments: list[Track], min_points: int, max_points: int) -> list[Track]:
    return [s for s in segments if min_points <= len(s.t) <= max_points]


def clean_tracks(
    tracks: list[Track],
    bbox: tuple[float, float, float, float] = BEIJING_BBOX,
    min_points: int = MIN_POINTS,
    max_points: int = MAX_POINTS,
) -> tuple[list[Track], CleanStats]:
    """Splits tracks at dt/speed breaks, drops single-point outliers,
    clips to the Beijing bbox, then refilters by point count."""
    stats = CleanStats(n_tracks_in=len(tracks), n_points_in=sum(len(tr.t) for tr in tracks))

    segments: list[Track] = []
    for tr in tracks:
        segs = _split_track(tr)
        stats.n_split_segments += len(segs) - 1
        segments.extend(segs)

    n_before = len(segments)
    segments = _drop_single_point_outliers(segments)
    stats.n_outliers_removed = n_before - len(segments)

    points_before_bbox = sum(len(s.t) for s in segments)
    segments = _apply_bbox(segments, bbox)
    points_after_bbox = sum(len(s.t) for s in segments)
    stats.n_points_dropped_bbox = points_before_bbox - points_after_bbox

    n_before_len = len(segments)
    segments = _refilter_length(segments, min_points, max_points)
    stats.n_dropped_short_or_long = n_before_len - len(segments)

    stats.n_tracks_out = len(segments)
    stats.n_points_out = sum(len(s.t) for s in segments)
    return segments, stats


def load_clean_tracks(
    data_dir: str = DEFAULT_DATA_DIR,
    n: int = 200,
    seed: int = DEFAULT_SEED,
    bbox: tuple[float, float, float, float] = BEIJING_BBOX,
    min_points: int = MIN_POINTS,
    max_points: int = MAX_POINTS,
) -> tuple[list[Track], CleanStats]:
    """Loads `n` raw tracks (no length cap -- the 50..2000 filter is
    applied after splitting, to the segments) and cleans them."""
    raw_tracks = load_tracks(
        data_dir=data_dir, n=n, min_points=_RAW_MIN_POINTS, max_points=_RAW_MAX_POINTS, seed=seed,
    )
    return clean_tracks(raw_tracks, bbox=bbox, min_points=min_points, max_points=max_points)
