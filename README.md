# CurvaDB

Исследование: даёт ли хранение GPS-траекторий кубическими B-сплайнами
выигрыш над классической схемой "упрощение Дугласом-Пекером (DP) +
поиск по дискретной метрике Фреше"? Итог — отрицательный результат по
сжатию, устойчивый вплоть до теоретического предела представимости
сплайном. Полный разбор — [docs/findings.md](docs/findings.md).

## Шаг / вопрос / ответ

| Шаг | Вопрос | Ответ |
|---|---|---|
| step0 | Как наивный сплайн смотрится против DP по recall/сжатию/latency? | Разгромно хуже (recall 0.707 против 0.997) — но это оказалась методическая ошибка: `fit()` держал tol только в метках времени, между ними сплайн улетал на километры (`step0_diagnostics.md`). |
| step1 | Что будет, если исправить методику (честный густой контроль ошибки + чистка GPS-разрывов)? | Recall почти сравнялся (0.972 против 0.996 у DP), но по сжатию DP выигрывает на КАЖДОМ tol (в 3.7 раза при tol=10м). Сплайн надёжнее DP+PCHIP по кинематике (не даёт выбросов ускорения). |
| step2 | Есть ли зона (шум/tol), где сплайн всё же компактнее? | Узкая немонотонная зона нашлась (0.78-0.85x при tol=1-5м, sigma<=0.1м) — но применима только к RTK/лидарной точности, не к потребительскому GPS. |
| step3 | Что если убрать привязку сплайна к ломаной (честный МНК-фиттер) и проверить решающими критериями, зафиксированными заранее? | Все три критерия провалены. Даже оракул (сплайн на истинной, бесшумной кривой) не компактнее DP+SED — **вывод step2 о зоне выигрыша отменён**. |

Подробности, точные цифры и методика — в [docs/findings.md](docs/findings.md).

## Установка

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

Данные — GeoLife Trajectories 1.3 (`data/geolife/<user_id>/Trajectory/*.plt`),
в репозиторий не входят (см. `.gitignore`).

## Как воспроизвести

```bash
venv/bin/pytest tests/traj/                          # тесты

venv/bin/python benchmarks/step0.py                  # первый (сломанный) замер
venv/bin/python benchmarks/step1_clean.py             # чистка данных
venv/bin/python benchmarks/step1_spline_fit.py         # честный густой фиттинг
venv/bin/python benchmarks/step1_compression.py        # сжатие (гипотеза A)
venv/bin/python benchmarks/step1_kinematics.py         # кинематика (гипотеза B)
venv/bin/python benchmarks/step1_search.py             # recall@10
venv/bin/python benchmarks/step2_crossover.py --pilot  # поиск ниши (пилот)
venv/bin/python benchmarks/step2_crossover.py          # поиск ниши (полный прогон)
venv/bin/python benchmarks/step3_decisive.py --pilot   # решающий эксперимент (пилот)
venv/bin/python benchmarks/step3_decisive.py           # решающий эксперимент (полный прогон)
```

Каждый скрипт пишет свою секцию в `benchmarks/results/<step>.md`.

## Известные альтернативы

- **PostGIS** ([`ST_FrechetDistance`](https://postgis.net/docs/ST_FrechetDistance.html)) —
  дискретная метрика Фреше как встроенная функция SQL над `geometry`;
  готовое индустриальное решение без необходимости писать свою ДП.
- **[MobilityDB](https://mobilitydb.com/)** — расширение PostgreSQL/PostGIS
  для траекторных данных (`tgeompoint` и т.п.), с временными операциями,
  индексами и метриками схожести из коробки.
- **Map-matching** (например, [Valhalla](https://github.com/valhalla/valhalla),
  [OSRM](https://project-osrm.org/)) — привязка GPS-трека к дорожной сети;
  другой подход к сжатию/представлению траектории, снимающий часть шума
  за счёт внешней информации о графе дорог, а не только геометрии трека.
- **Сглаживание Калманом** (constant-acceleration + RTS) — в step1
  (`docs/findings.md`, `benchmarks/results/step1.md` §4) показал лучшую
  среднюю точность восстановления скорости/ускорения среди всех трёх
  сравненных методов, ценой заметно более низкого recall детекции
  резких манёвров — альтернатива для задач, где важна кинематика, а не
  геометрическое сжатие/поиск.

## Старый проект

Level 1 (Hilbert-индекс над текстовыми эмбеддингами, `src/level1`) не
развивается в этой ветке. Архив прежнего README — в
[docs/legacy.md](docs/legacy.md).
