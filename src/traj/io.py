"""Загрузка GPS-треков GeoLife и их проекция в метры."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

EARTH_RADIUS_M = 6_371_000.0
HEADER_LINES = 6
MIN_POINTS = 50
MAX_POINTS = 2000
DEFAULT_SEED = 42
DEFAULT_N_TRACKS = 200
DEFAULT_DATA_DIR = "data/geolife"


@dataclass
class Track:
    track_id: str
    lat: np.ndarray  # исходные градусы
    lon: np.ndarray  # исходные градусы
    t: np.ndarray  # секунды от начала трека
    xy: np.ndarray  # Nx2, метры, equirectangular от общей опорной точки


def _parse_plt(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Читает .plt, отбрасывая дубликаты по времени. None, если файл пуст/битый."""
    lat: list[float] = []
    lon: list[float] = []
    t: list[float] = []
    seen: set[str] = set()

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()[HEADER_LINES:]

    for line in lines:
        parts = line.strip().split(",")
        if len(parts) < 7:
            continue
        try:
            la = float(parts[0])
            lo = float(parts[1])
            date_s = parts[5]
            time_s = parts[6]
            y, mo, d = (int(x) for x in date_s.split("-"))
            hh, mm, ss = (int(x) for x in time_s.split(":"))
        except (ValueError, IndexError):
            continue

        key = f"{date_s} {time_s}"
        if key in seen:
            continue
        seen.add(key)

        dt = datetime(y, mo, d, hh, mm, ss, tzinfo=timezone.utc)
        lat.append(la)
        lon.append(lo)
        t.append(dt.timestamp())

    if not lat:
        return None
    return np.array(lat), np.array(lon), np.array(t)


def _discover_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("*/Trajectory/*.plt"))


def load_tracks(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    n: int = DEFAULT_N_TRACKS,
    min_points: int = MIN_POINTS,
    max_points: int = MAX_POINTS,
    seed: int = DEFAULT_SEED,
) -> list[Track]:
    """Загружает до `n` треков длиной [min_points, max_points] точек.

    Порядок файлов фиксируется через `seed`. Опорная точка для проекции —
    среднее по центроидам ВСЕХ загруженных треков (каждый трек с равным
    весом, независимо от числа точек в нём), а не центр каждого трека
    отдельно.
    """
    data_dir = Path(data_dir)
    files = _discover_files(data_dir)
    rng = random.Random(seed)
    rng.shuffle(files)

    raw: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for path in files:
        if len(raw) >= n:
            break
        parsed = _parse_plt(path)
        if parsed is None:
            continue
        lat, lon, t = parsed
        order = np.argsort(t, kind="stable")
        lat, lon, t = lat[order], lon[order], t[order]
        if not (min_points <= len(lat) <= max_points):
            continue
        track_id = f"{path.parent.parent.name}/{path.stem}"
        raw.append((track_id, lat, lon, t))

    if not raw:
        raise RuntimeError(f"Не найдено ни одного подходящего трека в {data_dir}")

    centroids_lat = np.array([r[1].mean() for r in raw])
    centroids_lon = np.array([r[2].mean() for r in raw])
    lat0 = float(centroids_lat.mean())
    lon0 = float(centroids_lon.mean())
    lat0_rad = np.radians(lat0)

    tracks = []
    for track_id, lat, lon, t in raw:
        x = EARTH_RADIUS_M * np.radians(lon - lon0) * np.cos(lat0_rad)
        y = EARTH_RADIUS_M * np.radians(lat - lat0)
        xy = np.column_stack([x, y])
        t_rel = t - t[0]
        tracks.append(Track(track_id=track_id, lat=lat, lon=lon, t=t_rel, xy=xy))

    return tracks
