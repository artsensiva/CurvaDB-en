# Level 2: Curve-Based Semantic Search

## Overview

Level 2 реализует истинное curve-based представление текстов с использованием B-spline кривых, Functional PCA (FPCA) и метрик расстояния между кривыми для семантического поиска.

**Статус**: MVP реализован и протестирован
**Текущий Recall@10**: 31.17%
**Целевой Recall@10**: ≥75%

---

## Архитектура

```
Text Input
    ↓
[Token Embeddings] → Per-token vectors (SentenceTransformer)
    ↓
[PCA Reduction] → 384D → 8D (scipy splprep limitation)
    ↓
[CurveFitter] → Parametric curves (B-splines)
    ↓
[FPCA] → Low-dimensional curve features (20 components)
    ↓
[Search] → FPCA distance → Candidates → DTW reranking → Top-K
```

---

## Реализованные компоненты

### 1. Curve Fitter (`src/level2/curve_fitter.py`)

Конвертация последовательности token embeddings в параметрические B-spline кривые.

**Ключевые особенности**:
- Адаптивный выбор степени сплайна в зависимости от количества токенов
- Поддержка периодических и непериодических кривых
- Автоматическое понижение степени для коротких текстов

```python
class CurveFitter:
    def __init__(self, smoothing=0.0, degree=3, periodic=False)
    def tokens_to_curve(self, token_embeddings) -> Curve
    def fit_batch(self, embeddings_list) -> List[Curve]
```

### 2. Functional PCA (`src/level2/fpca.py`)

Анализ главных компонент в функциональном пространстве кривых.

**Ключевые особенности**:
- Использует scikit-fda для FPCA
- Преобразует многомерные кривые в 1D через flattening
- Сохранение/загрузка моделей

```python
class FunctionalPCA:
    def __init__(self, n_components=10, n_sample_points=50)
    def fit(self, curves: List[Curve])
    def transform(self, curves) -> np.ndarray
```

### 3. Distance Metrics (`src/level2/distances.py`)

Метрики расстояния между кривыми:
- **Fréchet distance**: учитывает траекторию кривой
- **DTW (Dynamic Time Warping)**: устойчив к временным искажениям
- **L2 distance**: быстрое поточечное сравнение
- **Cosine distance**: угловая метрика

### 4. CurveDB (`src/level2/curve_db.py`)

Основной класс базы данных, объединяющий все компоненты.

**Ключевые особенности**:
- Token-level embeddings вместо chunk embeddings
- PCA редукция эмбеддингов (384D → 8D)
- Двухэтапный поиск: FPCA filtering + curve distance reranking
- Персистентное хранение (pickle/numpy)

---

## Выполненные улучшения

### Проблема: 23% запросов не могли построить кривую

**Причина**: Короткие тексты при chunking давали идентичные эмбеддинги, что вызывало ошибку scipy splprep.

**Решения**:

1. **Переход на token-level embeddings**
   - Использование `output_value='token_embeddings'` в SentenceTransformer
   - Каждый токен дает уникальный embedding
   - 100% успешное построение кривых

2. **Адаптивный выбор степени сплайна**
   ```python
   if n_tokens < self.degree + 1:
       degree = max(1, n_tokens - 1)
   else:
       degree = self.degree
   ```

### Улучшение параметров для повышения recall

| Параметр | До | После | Эффект |
|----------|-----|-------|--------|
| n_fpca_components | 10 | 20 | Больше variance captured |
| distance_metric | frechet | dtw | Лучше для warped sequences |
| rerank_factor | 10 | 20 | Больше кандидатов для reranking |

---

## Результаты бенчмарков

### Датасет: MS MARCO 1K

- 1000 документов
- 100 запросов с relevance judgments

### Итоговые метрики

| Метрика | Значение |
|---------|----------|
| **Recall@10 (mean)** | 31.17% |
| **Recall@10 (median)** | 0.00% |
| **Std Deviation** | 0.458 |
| **Query Time (mean)** | 291ms |
| **Query Time (median)** | 284ms |

### Конфигурация

```python
CurveDB(
    n_fpca_components=20,
    hilbert_order=8,
    curve_degree=3,
    distance_metric='dtw',
    n_sample_points=50,
    curve_dim=8
)
```

### Сравнение с Level 1

| Система | Recall@10 | Улучшение |
|---------|-----------|-----------|
| Level 1 (PCA + Hilbert) | 8% | baseline |
| Level 2 (initial) | 10.67% | +33% |
| **Level 2 (optimized)** | **31.17%** | **+289%** |

---

## Анализ результатов

### Сильные стороны

1. **100% надежность curve fitting** - все документы и запросы успешно конвертируются в кривые
2. **Значительное улучшение recall** - почти 4x лучше Level 1
3. **Гибкая архитектура** - легко менять метрики и параметры

### Проблемы

1. **Высокая дисперсия результатов**
   - Median = 0% означает, что большинство запросов не находят релевантных документов
   - Некоторые запросы получают 100% recall, другие - 0%
   - Указывает на то, что curve representation работает для определенных паттернов, но не для всех

2. **Latency**
   - 291ms значительно выше целевых <100ms
   - Bottleneck: DTW distance computation для reranking

3. **Gap до целевого recall**
   - 31% vs 75% target
   - Требуются дополнительные улучшения

---

## Анализ дальнейших разработок

### Приоритет 1: Улучшение качества представления

#### 1.1 Многоуровневая архитектура кривых
```python
class HierarchicalCurve:
    """Кривые на разных уровнях гранулярности"""
    sentence_curve: Curve    # Sentence-level
    token_curve: Curve       # Token-level
    combined_score: float    # Weighted combination
```

**Ожидаемый эффект**: +15-20% recall за счет лучшего захвата семантики на разных уровнях.

#### 1.2 Улучшение FPCA
- Увеличить n_components до 30-50
- Использовать weighted FPCA с учетом важности компонент
- Исследовать Kernel FPCA для нелинейных паттернов

**Ожидаемый эффект**: +10-15% recall

#### 1.3 Ensemble метрик
```python
def combined_distance(curve1, curve2):
    return (
        0.4 * dtw_distance(curve1, curve2) +
        0.3 * frechet_distance(curve1, curve2) +
        0.3 * cosine_distance(curve1, curve2)
    )
```

**Ожидаемый эффект**: +5-10% recall

### Приоритет 2: Оптимизация производительности

#### 2.1 Параллельный DTW
```python
from concurrent.futures import ThreadPoolExecutor

def parallel_rerank(query_curve, candidates, n_workers=4):
    with ThreadPoolExecutor(n_workers) as executor:
        distances = executor.map(
            lambda c: dtw_distance(query_curve, c),
            candidates
        )
```

**Ожидаемый эффект**: 3-4x ускорение reranking

#### 2.2 Приближенный DTW
- FastDTW с радиусом ограничения
- LB_Keogh lower bound для pruning

**Ожидаемый эффект**: 5-10x ускорение с минимальной потерей качества

#### 2.3 Кэширование частых паттернов
```python
class CurveCache:
    """LRU cache для часто используемых curve distances"""
    def __init__(self, max_size=10000):
        self.cache = {}
```

### Приоритет 3: Улучшение индексации

#### 3.1 Cluster-based retrieval
1. Кластеризация кривых в FPCA пространстве (K-means, 50-100 clusters)
2. При поиске: найти ближайшие кластеры → искать внутри кластеров

**Ожидаемый эффект**: 10x ускорение с потерей <5% recall

#### 3.2 Hierarchical Navigable Small World (HNSW)
- Использовать HNSW для FPCA scores вместо простого range query
- faiss или hnswlib

**Ожидаемый эффект**: O(log N) вместо O(N) для первичного поиска

### Приоритет 4: Гибридный подход

#### 4.1 Level 1 + Level 2 Fusion
```python
def hybrid_search(query, k=10):
    level1_results = level1_db.search(query, k=50)
    level2_results = level2_db.search(query, k=50)

    # Reciprocal Rank Fusion
    return fuse_results(level1_results, level2_results, k=k)
```

**Ожидаемый эффект**: Комбинирует скорость Level 1 и качество Level 2

#### 4.2 Adaptive routing
- Классификатор для выбора лучшего метода для каждого запроса
- Короткие запросы → Level 1
- Длинные/сложные → Level 2

---

## План действий

### Фаза 1: Quick wins (1-2 дня)
- [ ] Увеличить n_fpca_components до 30
- [ ] Реализовать parallel reranking
- [ ] Добавить FastDTW с radius=1

### Фаза 2: Core improvements (3-5 дней)
- [ ] Hierarchical curves (sentence + token)
- [ ] Ensemble метрик
- [ ] Cluster-based retrieval

### Фаза 3: Advanced optimizations (1 неделя)
- [ ] HNSW для FPCA space
- [ ] Hybrid Level 1 + Level 2
- [ ] Query-dependent routing

### Target после всех улучшений
- Recall@10: ≥70%
- Query latency: <100ms
- QPS: ≥10

---

## Использование

### Создание базы данных
```python
from level2.curve_db import CurveDB

db = CurveDB(
    db_path="./my_curve_db",
    n_fpca_components=20,
    distance_metric='dtw',
    curve_degree=3
)
```

### Добавление документов
```python
texts = ["Document 1 text...", "Document 2 text...", ...]
db.add_documents(texts)
db.fit()
db.save()
```

### Поиск
```python
results = db.search("query text", k=10, rerank_factor=20)
for text, distance in results:
    print(f"{distance:.4f}: {text[:100]}...")
```

---

## Выводы

Level 2 MVP демонстрирует жизнеспособность curve-based подхода к семантическому поиску:

1. **Успех**: 289% улучшение над Level 1, 100% надежность curve fitting
2. **Проблема**: Высокая дисперсия результатов указывает на ограничения текущего представления
3. **Путь вперед**: Hierarchical curves + ensemble metrics + оптимизация производительности

Curve-based representation имеет теоретические преимущества для захвата последовательной семантики текста. Однако требуется дальнейшая работа над качеством представления и производительностью для достижения production-ready состояния.
