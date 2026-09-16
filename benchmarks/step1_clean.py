"""Step 1, п.1: чистка треков — сколько треков/точек отброшено.

Запуск: venv/bin/python benchmarks/step1_clean.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from traj.clean import BEIJING_BBOX, MAX_DT_S, MAX_SPEED_MPS, load_clean_tracks  # noqa: E402
from _report_utils import upsert_section  # noqa: E402

SEED = 42
N_TRACKS = 200
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_MD = os.path.join(RESULTS_DIR, "step1.md")
TITLE = "# Step 1: чистка данных, честный фиттинг сплайна, A/B\n"
SECTION_HEADER = "## 1. Чистка треков"


def _section_md(stats) -> str:
    lines = [
        f"{SECTION_HEADER}\n",
        (
            f"Параметры: разрыв по времени > {MAX_DT_S:.0f}с ИЛИ скорость > "
            f"{MAX_SPEED_MPS:.0f} м/с — резать; одиночные точки-сегменты — "
            f"выбросить; bbox Пекина {BEIJING_BBOX} (lat_min, lat_max, lon_min, "
            "lon_max); затем заново фильтр 50..2000 точек на сегмент.\n"
        ),
        f"Загружено сырых треков: {stats.n_tracks_in}, точек: {stats.n_points_in}.\n",
        "| Этап | Значение |",
        "|---|---|",
        f"| доп. сегментов от резки по разрывам | {stats.n_split_segments} |",
        f"| одиночных точек-выбросов убрано | {stats.n_outliers_removed} |",
        f"| точек отброшено вне bbox Пекина | {stats.n_points_dropped_bbox} |",
        f"| сегментов отброшено фильтром длины (не 50..2000) | {stats.n_dropped_short_or_long} |",
        f"| **итог: треков** | {stats.n_tracks_out} (из {stats.n_tracks_in} исходных) |",
        f"| **итог: точек** | {stats.n_points_out} (из {stats.n_points_in} исходных) |",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    _tracks, stats = load_clean_tracks(n=N_TRACKS, seed=SEED)
    print(
        f"in: {stats.n_tracks_in} tracks / {stats.n_points_in} points -> "
        f"out: {stats.n_tracks_out} tracks / {stats.n_points_out} points"
    )
    print(
        f"split_segments={stats.n_split_segments} outliers={stats.n_outliers_removed} "
        f"bbox_dropped_points={stats.n_points_dropped_bbox} "
        f"length_refilter_dropped={stats.n_dropped_short_or_long}"
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(OUT_MD):
        with open(OUT_MD, "w") as f:
            f.write(TITLE + "\n")
    upsert_section(OUT_MD, SECTION_HEADER, _section_md(stats))
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
