# CurvaDB

Исследование: даёт ли хранение GPS-траекторий кубическими B-сплайнами
выигрыш над классической схемой "упрощение Дугласом-Пекером (DP) +
поиск по дискретной метрике Фреше"? Итог — отрицательный результат по
сжатию для потребительского GPS; для точных данных выигрыш в
проверенных условиях тоже не найден, но вопрос до конца не закрыт (см.
гипотезу H1 в [docs/findings.md](docs/findings.md)). Проект завершается
как исследование — следующий шаг не код, а интервью с отраслью (см.
[docs/next_steps.md](docs/next_steps.md)).

## Этапы

| Этап/шаг | Вопрос | Ответ |
|---|---|---|
| A. Исходная идея | Семантический поиск через "кривые" над эмбеддингами? | Реализовано (Level 1-2), но не доведено до сравнения с baseline. |
| B. Критика | Состоятельна ли идея математически? | Нет — 4 несвязанные задачи, порядок координат эмбеддинга произволен. Пивот на GPS-траектории. |
| C. Проверка фактов | Пуста ли ниша, честно ли сравнение? | Ниша не пуста; сравнение "80КБ/2-4КБ" нечестное; рекомендованы 10-15 интервью — не проведены. |
| D. Организация | — | Ветка `trajectory-pivot`, данные GeoLife, промпты файлами. |
| step0 | Как наивный сплайн смотрится против DP? | Разгромно хуже (recall 0.707 против 0.997) — оказалась методическая ошибка (`step0_diagnostics.md`). |
| step1 | Что покажет честная методика? | Recall почти сравнялся (0.972/0.996), но сжатие DP выигрывает на каждом tol (3.7x). |
| step2 | Есть ли ниша сжатия по шуму/tol? | Узкая зона нашлась (tol=1-5м, sigma<=0.1м) — но только для RTK/лидарной точности. |
| step3 | Выдержит ли гипотеза решающую проверку без привязки к ломаной? | Все три критерия (K1-K3) провалены; даже оракул не компактнее DP+SED. **Вывод step2 о нише отменён.** |
| Резолюция | Что дальше? | Не код — 8-10 интервью с отраслью; порог возврата к коду ≥3/10. |

Полная хронология с цифрами и коммитами — [docs/history.md](docs/history.md).

## Документы

- [docs/history.md](docs/history.md) — полная история проекта, от
  исходной идеи до резолюции.
- [docs/findings.md](docs/findings.md) — итоги исследования: вопрос,
  ключевые цифры, вывод, ограничения, открытые гипотезы H1/H2.
- [docs/next_steps.md](docs/next_steps.md) — резолюция, план интервью,
  порог возврата к коду, набросок эксперимента по H2.
- [docs/blog_draft.md](docs/blog_draft.md) — текст поста для публикации, для
  инженерной аудитории.
- [docs/legacy.md](docs/legacy.md) — архив README до пивота на
  траектории (исходная идея с эмбеддингами).
- [docs/prompts/](docs/prompts/) — промпты step3-step5 (step0-step2
  давались в чате, пересказаны в `docs/prompts/README.md`).
- [benchmarks/results/](benchmarks/results/) — сырые результаты
  каждого шага (step0.md, step0_diagnostics.md, step1.md, step2.md,
  step3.md).

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
