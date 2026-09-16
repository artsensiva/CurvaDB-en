"""Loading GeoLife GPS tracks and projecting them into meters."""

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
    lat: np.ndarray  # original degrees
    lon: np.ndarray  # original degrees
    t: np.ndarray  # seconds since track start
    xy: np.ndarray  # Nx2, meters, equirectangular from a shared reference point


def _parse_plt(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Reads a .plt file, dropping time-based duplicates. None if the file is empty/corrupt."""
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
    """Loads up to `n` tracks with [min_points, max_points] points.

    File order is fixed via `seed`. The projection's reference point is the
    mean of the centroids of ALL loaded tracks (each track weighted
    equally, regardless of its point count), not the center of each track
    individually.
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
        raise RuntimeError(f"No suitable tracks found in {data_dir}")

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
