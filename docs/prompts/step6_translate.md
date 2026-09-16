# Step6 — English as the primary language, Russian preserved

Это репозиторий CurvaDB-en (приватная английская версия проекта CurvaDB).
Работаем в ветке english. Новых экспериментов не запускать, логику кода не
менять, агентов-исследователей не запускать. Цель: английский — основной язык
репозитория, русские тексты сохранены в docs/ru/ и README.ru.md.
data/ — символическая ссылка, не трогать и не коммитить.

## Глоссарий (использовать единообразно везде)
- ломаная Дугласа-Пекера / DP-ломаная -> DP polyline
- DP+SED -> DP+SED (time-aware Douglas-Peucker, synchronized Euclidean distance)
- допуск tol -> tolerance (tol)
- достижимо / недостижимо -> reachable / unreachable
- доля достижимых треков -> reachability
- густая проверка / густая сетка -> dense check / dense grid
- ошибка до истинной кривой -> error against the ground-truth curve
- оракул -> oracle
- привязка сплайна к ломаной -> spline tethered to the polyline
- ошибка хорды -> chord error
- чистка треков -> track cleaning
- критерии К1-К3 -> criteria K1-K3
- зафиксированы до запуска -> pre-registered
- отрицательный результат -> negative result
- резолюция -> resolution; открытые гипотезы H1/H2 -> open hypotheses H1/H2

## Пункты (коммит после каждого)

1. Код: src/traj/*.py, benchmarks/*.py, tests/**/*.py — перевести комментарии,
   docstrings, сообщения print и строки, из которых генерируются отчёты.
   Имена функций и переменных не менять. Проверка: venv/bin/pytest tests/ -q
   проходит; git diff не содержит изменений логики. src/level1 не трогать.

2. Результаты: скопировать benchmarks/results/*.md в docs/ru/results/, затем
   перевести оригиналы. ВСЕ числа, таблицы и идентификаторы треков сохранить.
   Проверка: скрипт извлекает все числа из каждой пары файлов (ru и en)
   и сравнивает множества; расхождения исправить.

3. Документы: git mv docs/history.md, findings.md, next_steps.md,
   blog_draft.md, legacy.md в docs/ru/, создать английские версии в docs/ с
   теми же именами. blog_draft.md — адаптация для англоязычной инженерной
   аудитории, не дословный перевод. Ссылки на репозиторий заменить на
   github.com/artsensiva/CurvaDB-en. Числа сверить тем же скриптом.

4. CLAUDE.md, TODO.md — перевести на месте. docs/prompts/*.md не переводить
   (исторические артефакты), только docs/prompts/README.md — перевести
   и отметить, что сами промпты на русском.

5. README: git mv README.md README.ru.md, первой строкой в нём
   "English version: README.md", ссылки поправить на docs/ru/. Создать
   README.md на английском той же структуры, первой строкой
   "Русская версия: README.ru.md", ссылки на английские docs/.

6. Финальная проверка: все относительные ссылки во всех .md ведут на
   существующие файлы; pytest проходит; в src/traj, benchmarks, tests,
   README.md, docs/*.md (кроме docs/ru и docs/prompts) нет кириллицы —
   grep -rlP '[А-Яа-яЁё]' с выводом списка, исключения только осознанные.
   Затем git push -u origin english.

Если лимит сессии заканчивается — закоммитить текущий пункт и записать в
TODO.md, с какого пункта продолжать. Уточняющих вопросов не задавать.
Сначала короткий план (5-10 строк), жди «ок».
