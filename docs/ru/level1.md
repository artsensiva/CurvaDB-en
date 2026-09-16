# 📚 CurvaDB Level 1 - Complete Documentation

## Overview

**Level 1: Minimal Hilbert Vector DB** - это proof-of-concept реализация семантического поиска, использующая space-filling Hilbert curves для индексации эмбеддингов.

**Статус**: ✅ Реализован | ❌ Не достиг целевых метрик | 🔄 Переход к Level 2

**Дата разработки**: Ноябрь 2024
**Версия**: 0.1.0

---

## Ключевая идея

Вместо хранения и поиска в полном N-мерном пространстве:
1. **Редуцируем** эмбеддинги до меньших измерений (PCA: 384D → 12D)
2. **Преобразуем** в 1-мерные Hilbert distances (12D → 1D)
3. **Индексируем** в LMDB key-value store
4. **Ищем** через range queries
5. **Реранкируем** по cosine similarity для точности

**Гипотеза**: Hilbert curve сохраняет пространственную локальность → близкие в семантическом пространстве векторы будут близки на кривой Hilbert.

**Результат**: ❌ Гипотеза НЕ подтвердилась для семантического поиска

---

## Архитектура

```
Text Input
    ↓
[TextEmbedder] → SentenceTransformer embeddings (384D)
    ↓
[DimensionalityReducer] → PCA reduction (12D) + normalization [0, 1023]
    ↓
[HilbertIndexer] → Map to 1D Hilbert distance (up to 2^120)
    ↓
[LMDBStorage] → Store metadata with Hilbert key (128-bit keys)
    ↓
[EmbeddingStore] → Separate mmap storage for embeddings
    ↓
Search: Query → Hilbert distance → Range query → Rerank → Top-K
```

### Компоненты (6 модулей, ~600 LOC)

#### 1. TextEmbedder ([src/level1/embedder.py](../../src/level1/embedder.py))
- **Назначение**: Преобразование текста в dense embeddings
- **Модель**: `all-MiniLM-L6-v2` (384 dimensions)
- **Технологии**: SentenceTransformers
- **Возможности**: Batch encoding, GPU support

#### 2. DimensionalityReducer ([src/level1/reducer.py](../../src/level1/reducer.py))
- **Назначение**: PCA reduction + normalization
- **Входные данные**: 384D embeddings
- **Выходные данные**: 12D normalized integers [0, 1023]
- **Особенности**: Fit на первом батче, сохранение модели

#### 3. HilbertIndexer ([src/level1/hilbert_index.py](../../src/level1/hilbert_index.py))
- **Назначение**: Преобразование N-D → 1-D Hilbert distance
- **Библиотека**: `hilbertcurve`
- **Конфигурация**: p=10 bits, n=12 dimensions → max distance 2^120
- **Теория**: Сохраняет пространственную локальность

#### 4. LMDBStorage ([src/level1/storage.py](../../src/level1/storage.py))
- **Назначение**: Key-value storage с range queries
- **LMDB**: Lightning Memory-Mapped Database
- **Ключи**: 128-bit Hilbert distances (16 bytes)
- **Значения**: JSON metadata (doc_id, text, emb_idx)
- **Map size**: 100MB (оптимизировано с 10GB)

#### 5. EmbeddingStore ([src/level1/embedding_store.py](../../src/level1/embedding_store.py))
- **Назначение**: Compact storage для embeddings
- **Технология**: numpy memmap (memory-mapped files)
- **Формат**: float32 array shape (capacity, 384)
- **Auto-expansion**: Динамическое увеличение capacity
- **Преимущества**: Низкое потребление RAM, быстрый доступ

#### 6. SearchEngine ([src/level1/search.py](../../src/level1/search.py))
- **Назначение**: Two-stage retrieval
- **Stage 1**: Approximate search via Hilbert range query
- **Stage 2**: Reranking by cosine similarity
- **Radius**: Adaptive (20% → 50% при недостатке кандидатов)

#### 7. MinimalCurveDB ([src/level1/minimal_db.py](../../src/level1/minimal_db.py))
- **Назначение**: Main API class
- **Методы**: `add_batch()`, `search()`, `count()`, `close()`
- **Context manager**: Поддержка `with` statement

---

## Разработка: Проблемы и решения

### Проблема 1: Overflow Hilbert distances (2^72 > 64-bit)

**Симптом**:
```
struct.error: int too large to convert
```

**Причина**:
- Начальная конфигурация: p=12, n=6 → max distance 2^72
- LMDB использовал 8-byte (64-bit) keys
- 2^72 > 2^64 → overflow

**Решение**:
- Изменил LMDB keys на 16-byte (128-bit):
  ```python
  key_bytes = key.to_bytes(16, byteorder='big', signed=False)
  ```
- Финальная конфигурация: p=10, n=12 → 2^120 (fits in 128 bits)

**Файлы**: [src/level1/storage.py](../../src/level1/storage.py) lines 50-55

---

### Проблема 2: Пустые результаты поиска

**Симптом**: `db.search()` возвращает пустые результаты даже на релевантных запросах

**Причина**: Search radius слишком мал
- Версия 1: `radius = k * 100` (для k=10 → 1000)
- Max distance для p=10, n=12: ~2^120
- 1000 / 2^120 ≈ 0.0000000001% пространства

**Решение (итерационно)**:
1. **v1**: Увеличил до 1% max distance
2. **v2**: Увеличил до 20% max distance
3. **v3**: Auto-expansion до 50% при недостатке кандидатов

**Текущая логика**:
```python
# Default: 20% of space
radius = self.hilbert.get_max_distance() // 5

# Auto-expand to 50% if too few candidates
if len(candidates) < k:
    expanded_radius = self.hilbert.get_max_distance() // 2
```

**Результат**: Все равно низкий recall (см. бенчмарки)

**Файлы**: [src/level1/minimal_db.py](../../src/level1/minimal_db.py) lines 198-240

---

### Проблема 3: Размер БД 10GB для 1K документов

**Симптом**: Database size на диске 10.24 GB для всего 1000 документов

**Причины**:
1. **LMDB map_size**: Установлен в 10GB (резервирует sparse file)
2. **Хранение embeddings в LMDB**: 384 floats × 4 bytes = 1.5KB на документ
3. **Неэффективная сериализация**: JSON с полными эмбеддингами

**Решение**:
1. **Создал EmbeddingStore**: Отдельный memmap storage
2. **LMDB только для метаданных**: doc_id, text, emb_idx (без embeddings!)
3. **Уменьшил map_size**: 10GB → 100MB
4. **Уменьшил initial_capacity**: 10000 → 1000 (auto-expand)

**Результат**:
- 1K docs: 101 MB (было 10GB) - улучшение 100x
- 10K docs: 116 MB (было бы ~100GB)

**Файлы**:
- [src/level1/embedding_store.py](../../src/level1/embedding_store.py) (новый модуль)
- [src/level1/storage.py](../../src/level1/storage.py) lines 18-20
- [src/level1/minimal_db.py](../../src/level1/minimal_db.py) lines 80-83, 147-163

---

### Проблема 4: EmbeddingStore expansion error

**Симптом**:
```
ValueError: Cannot load file containing pickled data when allow_pickle=False
```

**Причина**: После `_expand_capacity()` пытались загрузить memmap file через `np.load()` вместо `np.memmap()`

**Решение**:
```python
# БЫЛО (неправильно):
self.embeddings = np.load(self.embeddings_file, mmap_mode='r+')

# СТАЛО (правильно):
self.embeddings = np.memmap(
    self.embeddings_file,
    dtype=np.float32,
    mode='r+',
    shape=(new_capacity, self.embedding_dim)
)
```

**Файлы**: [src/level1/embedding_store.py](../../src/level1/embedding_store.py) lines 153-159

---

### Проблема 5: Recall деградирует с ростом датасета

**Симптом**:
- 1K docs: Recall@10 = 8%
- 10K docs: Recall@10 = 0.7% (в 11 раз хуже!)

**Причина**: **Фундаментальное ограничение подхода**

Hilbert curve хорошо сохраняет **геометрическую близость**, но НЕ **семантическую близость**:
- PCA теряет семантическую информацию (384D → 12D)
- Hilbert mapping дополнительно искажает отношения
- Чем больше документов, тем более разрознены релевантные документы в Hilbert space

**Вывод**: Подход PCA + Hilbert curve **не масштабируется** для семантического поиска

---

## Benchmark Results

### MS MARCO 1K Dataset

**Конфигурация**: 1000 docs, 100 queries, p=10, n=12

| Metric | Target | CurvaDB | Baseline | Status |
|--------|--------|---------|----------|--------|
| **Recall@10** | ≥99% | **8%** | 100% | ❌ |
| **MRR@10** | ≥95% | **8%** | 100% | ❌ |
| **P50 Latency** | <5ms | **11.92ms** | 11.63ms | ❌ |
| **P95 Latency** | <10ms | **12.82ms** | 13.22ms | ❌ |
| **DB Size** | <10MB | **101.5MB** | N/A | ❌ |
| **QPS** | ≥100 | **81.2** | 85.1 | ❌ |

**Speedup vs Baseline**: 0.95x (CurvaDB немного медленнее на малом датасете)

---

### MS MARCO 10K Dataset

**Конфигурация**: 10000 docs, 500 queries, p=10, n=12

| Metric | Target | CurvaDB | Baseline | Status |
|--------|--------|---------|----------|--------|
| **Recall@1** | ~95% | **0.7%** | 91.3% | ❌ |
| **Recall@5** | ~98% | **0.7%** | 98.6% | ❌ |
| **Recall@10** | ≥99% | **0.7%** | 99.3% | ❌ |
| **Recall@20** | ~99.5% | **0.7%** | 99.6% | ❌ |
| **MRR@10** | ≥95% | **0.8%** | 95.8% | ❌ |
| **P50 Latency** | <5ms | **11.97ms** | 15.03ms | ❌ |
| **P95 Latency** | <10ms | **13.12ms** | 16.28ms | ❌ |
| **DB Size** | <10MB | **116.1MB** | N/A | ❌ |
| **QPS** | ≥100 | **78.1** | 66.4 | ❌ |
| **Indexing** | - | 33.35s (299.9 docs/sec) | 27.35s | - |
| **Memory** | - | 292.1 MB | - | - |

**Speedup vs Baseline**: 1.18x-1.26x (CurvaDB быстрее на большем датасете!)

**Критическое наблюдение**:
- ✅ Скорость есть (1.2x speedup)
- ❌ Качество катастрофически низкое (0.7% recall)
- 📉 Recall деградирует с ростом датасета (8% → 0.7%)

---

## Масштабирование

### Recall vs Dataset Size

| Dataset | Recall@10 | Degradation |
|---------|-----------|-------------|
| 1K docs | 8% | baseline |
| 10K docs | **0.7%** | **-91%** |

**Вывод**: Система практически **неработоспособна** для реального поиска при масштабировании.

### Resource Usage

| Dataset | Indexing Time | DB Size | Memory | P95 Latency |
|---------|---------------|---------|--------|-------------|
| 1K | 6.33s | 101.5 MB | 211.7 MB | 12.82ms |
| 10K | 33.35s | 116.1 MB | 292.1 MB | 13.12ms |

**Наблюдения**:
- Линейный рост времени индексации (~300 docs/sec)
- Сублинейный рост размера БД (компрессия LMDB)
- Стабильная latency (~12ms независимо от размера)

---

## API Reference

### MinimalCurveDB

```python
class MinimalCurveDB:
    def __init__(
        self,
        db_path: str = "./curve_db",
        p: int = 10,  # Hilbert precision (bits per dimension)
        n: int = 12,  # Reduced dimensions
        model_name: str = "all-MiniLM-L6-v2"
    )
```

**Параметры**:
- `db_path`: Путь к директории БД
- `p`: Hilbert precision (10-16, recommend 10)
- `n`: Target dimensions after PCA (6-12, recommend 12)
- `model_name`: SentenceTransformer model

**Методы**:

```python
def add_batch(
    self,
    doc_ids: List[str],
    texts: List[str],
    show_progress: bool = False
) -> None
```
Добавляет документы. Первый батч должен иметь ≥ n документов для fit PCA.

```python
def search(
    self,
    query: str,
    k: int = 10,
    radius: Optional[int] = None,
    rerank: bool = True
) -> List[Tuple[str, str, float]]
```
Возвращает список `(doc_id, text, similarity_score)`.

```python
def count() -> int
def get_stats() -> dict
def close() -> None
```

**Context Manager**:
```python
with MinimalCurveDB('./my_db') as db:
    db.add_batch(doc_ids, texts)
    results = db.search('query')
```

---

## Ограничения

### Фундаментальные (не решаемы в Level 1)

1. ❌ **Низкий recall (0.7-8%)**
   - PCA теряет семантическую информацию
   - Hilbert curve не сохраняет семантическую близость
   - Не решается увеличением радиуса поиска

2. ❌ **Деградация с ростом датасета**
   - Recall падает с 8% → 0.7% при росте 1K → 10K
   - Чем больше документов, тем хуже качество

3. ❌ **Размер БД > целевого**
   - 116 MB для 10K docs (цель <10MB)
   - LMDB overhead + separate embedding storage

### Технические

4. ⚠️ **Append-only** (нет update/delete)
   - Workaround: rebuild или tombstones

5. ⚠️ **PCA fitting требует ≥ n документов в первом батче**

6. ⚠️ **Single-threaded indexing**

7. ⚠️ **Нет metadata filtering**

---

## Выводы

### ✅ Что работает

1. **Архитектура корректна**: 6 модулов интегрированы, end-to-end pipeline работает
2. **Тесты проходят**: 15 unit tests + E2E test
3. **Скорость достигнута**: 1.2x speedup vs baseline на 10K
4. **Компактное хранение**: EmbeddingStore с memmap эффективен
5. **Масштабируемая индексация**: ~300 docs/sec стабильно

### ❌ Что не работает

1. **Recall катастрофически низкий**: 0.7% вместо 99%
2. **Деградация при масштабировании**: -91% recall при росте 1K → 10K
3. **Все целевые метрики не достигнуты**: 0/6 targets passed

### 🔍 Корневая причина

**Гипотеза PCA + Hilbert curve НЕ подтвердилась**:
- Hilbert curve сохраняет **геометрическую**, но НЕ **семантическую** близость
- Трансформация 384D → 12D → 1D теряет критически важную информацию
- Подход фундаментально непригоден для семантического поиска

### 📊 Практическое значение Level 1

Level 1 служит **proof-of-concept** и **baseline** для Level 2:
- ✅ Доказывает работоспособность инфраструктуры
- ✅ Выявляет ограничения простого подхода
- ✅ Мотивирует переход к true curve-based representation (FPCA + splines)

---

## Следующие шаги: Level 2

**Проблемы Level 1 → Решения Level 2**:

| Проблема Level 1 | Решение Level 2 |
|------------------|-----------------|
| PCA теряет семантику | **FPCA** (Functional PCA) - сохраняет динамику |
| 1D Hilbert не сохраняет семантику | **Spline curves** - истинное curve-based представление |
| Recall 0.7% | Ожидаемый Recall >90% с FPCA |
| Деградация при росте | Масштабируемые curve-based индексы |

**Level 2 Roadmap** (см. [SPECIFY.md](../../SPECIFY.md)):
- Functional PCA для temporal/semantic patterns
- B-spline/Bezier curve representation
- Curve similarity metrics (Fréchet distance)
- Advanced indexing (curve clustering)

---

## Файлы и код

### Структура проекта

```
src/level1/
├── minimal_db.py        (главный API класс, 336 LOC)
├── embedder.py          (text → embeddings, 73 LOC)
├── reducer.py           (PCA + normalization, 95 LOC)
├── hilbert_index.py     (N-D → 1-D mapping, 45 LOC)
├── storage.py           (LMDB key-value, 115 LOC)
├── embedding_store.py   (memmap storage, 189 LOC)
└── search.py            (two-stage retrieval, 88 LOC)

Total: ~940 LOC (без комментариев и docstrings)
```

### Benchmarks

```
benchmarks/
├── level1_bench.py      (original benchmark)
├── level1_bench_v2.py   (improved methodology)
└── results/
    ├── level1_results.json
    └── level1_10k_results.json
```

### Ключевые алгоритмы

**Indexing**:
```python
embeddings = embedder.encode(texts)           # 384D
reduced = reducer.transform(embeddings)        # 12D [0, 1023]
hilbert_dists = hilbert.points_to_distances(reduced)  # 1D [0, 2^120]
storage.put(hilbert_dist, metadata)
embedding_store.add_batch(embeddings)
```

**Search**:
```python
query_emb = embedder.encode([query])[0]
query_reduced = reducer.transform(query_emb)
query_dist = hilbert.points_to_distances([query_reduced])[0]

# Range query: [query_dist - radius, query_dist + radius]
candidates = storage.range_query(start_dist, end_dist)

# Rerank by cosine similarity
scores = cosine_similarity(query_emb, candidate_embeddings)
top_k = sort_by_score(scores)[:k]
```

---

## Запуск

### Установка

```bash
git clone https://github.com/artsensiva/CurvaDB.git
cd CurvaDB
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Quick Start

```python
from src.level1.minimal_db import MinimalCurveDB

# Initialize
db = MinimalCurveDB(db_path='./my_db', p=10, n=12)

# Add documents
doc_ids = ['doc1', 'doc2', 'doc3']
texts = [
    'Machine learning is a subset of AI',
    'Deep learning uses neural networks',
    'I love cooking pasta'
]
db.add_batch(doc_ids, texts)

# Search
results = db.search('artificial intelligence', k=2)
for doc_id, text, score in results:
    print(f"{doc_id}: {text} (score: {score:.3f})")

db.close()
```

### Тесты

```bash
# Unit tests
pytest tests/ -v

# End-to-end
python test_e2e.py

# Benchmarks
python benchmarks/level1_bench_v2.py
```

---

## Заключение

**Level 1 выполнил свою роль как proof-of-concept**:
- ✅ Инфраструктура работает
- ✅ Выявлены фундаментальные ограничения простого подхода
- ✅ Готов переход к Level 2

**Ключевой урок**: Пространственные структуры (Hilbert curves) эффективны для геометрического поиска, но **недостаточны для семантического поиска**. Требуется более sophisticated подход с сохранением семантических отношений.

**Статус**: ⏸️ Завершен, переход к Level 2
**Дата**: 2024-11-18
**Автор**: Artem Galukhin (artsensiva)

---

**Next**: [Level 2 Documentation](level2.md) (в разработке)
