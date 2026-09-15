"""Чистка GPS-треков: разрывы по времени/скорости, одиночные выбросы,
bbox Пекина, повторный фильтр по числу точек.

Мотивация (benchmarks/results/step0_diagnostics.md,
benchmarks/diagnose_fit.py): аномальные GPS-скачки и большие разрывы по
времени внутри "трека" — главная причина, по которой сплайн не может
держать tol между метками времени. Режем трек на сегменты там, где сырые
данные физически не могут быть одним непрерывным треком.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from traj.io import DEFAULT_DATA_DIR, DEFAULT_SEED, MAX_POINTS, MIN_POINTS, Track, load_tracks

MAX_DT_S = 30.0
MAX_SPEED_MPS = 70.0
# Пекин (муниципалитет), приблизительно — GeoLife записан почти целиком
# внутри этих границ; служит защитой от единичных выбросов геокодирования.
BEIJING_BBOX = (39.4, 41.6, 115.7, 117.5)  # lat_min, lat_max, lon_min, lon_max

# Для load_clean_tracks: загружаем сырые треки без ограничения на длину —
# фильтр 50..2000 применяется ПОСЛЕ резки, к сегментам, а не к сырому треку.
_RAW_MIN_POINTS = 1
_RAW_MAX_POINTS = 1_000_000


@dataclass
class CleanStats:
    n_tracks_in: int = 0
    n_points_in: int = 0
    n_split_segments: int = 0  # доп. сегменты, порождённые резкой по разрывам
    n_outliers_removed: int = 0  # одиночные точки-сегменты
    n_points_dropped_bbox: int = 0
    n_dropped_short_or_long: int = 0  # сегментов, не прошедших refilter длины
    n_tracks_out: int = 0
    n_points_out: int = 0


def _segment_breaks(t: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Индексы начала новых сегментов (разрыв по dt или скорости перед ними)."""
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
    """[a, b) индексы непрерывных серий True в mask."""
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return []
    gaps = np.nonzero(np.diff(idx) > 1)[0]
    starts = [idx[0]] + [idx[g + 1] for g in gaps]
    ends = [idx[g] + 1 for g in gaps] + [idx[-1] + 1]
    return list(zip(starts, ends))


def _apply_bbox(segments: list[Track], bbox: tuple[float, float, float, float]) -> list[Track]:
    """Оставляет только точки внутри bbox; режет на подсегменты по местам
    выпадения из bbox, а не просто выкидывает точки (иначе внутри
    "сегмента" появился бы скрытый разрыв, который резка по dt/скорости
    уже не увидит)."""
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
    """Режет треки по разрывам dt/скорости, убирает одиночные выбросы,
    ограничивает bbox Пекина, затем заново фильтрует по числу точек."""
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
    """Загружает `n` сырых треков (без ограничения длины — фильтр 50..2000
    применяется после резки, к сегментам) и чистит их."""
    raw_tracks = load_tracks(
        data_dir=data_dir, n=n, min_points=_RAW_MIN_POINTS, max_points=_RAW_MAX_POINTS, seed=seed,
    )
    return clean_tracks(raw_tracks, bbox=bbox, min_points=min_points, max_points=max_points)
