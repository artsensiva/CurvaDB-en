# Level 2 Development Roadmap

## Overview

**Level 2: Full Curve-Based DB** - Реализация истинного curve-based представления с функциональным анализом данных (FPCA) и метриками расстояния между кривыми.

**Цель**: Достичь Recall@10 ≥75% за счет лучшего сохранения семантической информации через curve representation.

**Срок**: 2.5-3 недели

---

## Ключевые улучшения vs Level 1

| Aspect | Level 1 | Level 2 |
|--------|---------|---------|
| **Representation** | PCA reduction (384D → 12D) | Spline curves from token embeddings |
| **Dimensionality Reduction** | Standard PCA | Functional PCA (FPCA) |
| **Similarity Metric** | Cosine similarity | Fréchet distance, DTW |
| **Index Structure** | Hilbert curve only | Hilbert (FPCA scores) + curve reranking |
| **Expected Recall@10** | 0.7-8% | ≥75% |

---

## Architecture

```
Text Input
    ↓
[Tokenization] → Token-level embeddings (transformer)
    ↓
[CurveFitter] → Parametric curves (B-splines)
    ↓
[FPCA] → Low-dimensional curve features
    ↓
[Hilbert Index] → 1D distance from FPCA scores
    ↓
[Storage] → RocksDB (curve params + FPCA scores)
    ↓
Search: Query curve → FPCA → Hilbert range → Candidate curves
    ↓
[Curve Distance] → Rerank by Fréchet distance → Top-K
```

---

## Phase 1: Core Infrastructure (Week 1)

### Day 1-2: Setup and Dependencies

**Tasks**:
- [ ] Update requirements.txt with Level 2 dependencies
- [ ] Install and test:
  - scipy ≥1.11.0 (spline interpolation)
  - scikit-fda ≥0.8.0 (functional data analysis)
  - similaritymeasures ≥0.5.0 (Fréchet, DTW)
  - python-rocksdb ≥0.7.0 (storage)
  - torch ≥2.0.0 (token embeddings)
- [ ] Create src/level2/ directory structure
- [ ] Write Level 2 architecture document

**Deliverables**:
- Updated requirements.txt
- Verified dependency installation
- docs/level2_architecture.md

---

### Day 3-4: Curve Fitting Module

**File**: `src/level2/curve_fitter.py`

**Core Class**: `CurveFitter`

**Methods**:
```python
class CurveFitter:
    def __init__(self, smoothing: float = 0.1, degree: int = 3):
        """Initialize curve fitter with spline parameters."""

    def tokens_to_curve(self, token_embeddings: np.ndarray) -> Curve:
        """Convert token embeddings to parametric curve."""
        # Use scipy.interpolate.splprep
        # Return Curve object with (knots, coefficients, degree)

    def fit_batch(self, texts: List[str]) -> List[Curve]:
        """Fit curves for multiple texts."""
```

**Curve Representation**:
```python
@dataclass
class Curve:
    knots: np.ndarray       # Spline knots
    coefficients: np.ndarray  # Spline coefficients
    degree: int             # Spline degree
    dimension: int          # Embedding dimension

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Evaluate curve at parameter values t."""
```

**Challenges**:
- Variable-length token sequences → need uniform parameterization
- Smoothing parameter tuning
- Handle edge cases (very short/long texts)

**Tests**:
- Test on simple token sequences
- Verify curve smoothness
- Check parameter count vs original dimension

---

### Day 5-6: Functional PCA Module

**File**: `src/level2/fpca.py`

**Core Class**: `FunctionalPCA`

**Methods**:
```python
class FunctionalPCA:
    def __init__(self, n_components: int = 10, n_basis: int = 20):
        """Initialize FPCA with scikit-fda."""

    def fit(self, curves: List[Curve]):
        """Fit FPCA on corpus of curves."""
        # Convert curves to functional data objects
        # Perform FPCA decomposition

    def transform(self, curves: List[Curve]) -> np.ndarray:
        """Project curves to FPCA space."""
        # Returns shape (n_curves, n_components)

    def inverse_transform(self, scores: np.ndarray) -> List[Curve]:
        """Reconstruct curves from FPCA scores."""
```

**Key Concepts**:
- FPCA finds principal components in function space
- Captures smooth variation patterns
- Better than standard PCA for temporal/sequential data

**Tests**:
- Verify reconstruction quality
- Check explained variance
- Test on simple synthetic curves

---

### Day 7: Curve Distance Metrics

**File**: `src/level2/distances.py`

**Functions**:
```python
def frechet_distance(curve1: Curve, curve2: Curve, n_samples: int = 100) -> float:
    """Compute discrete Fréchet distance between curves."""
    # Sample both curves uniformly
    # Use similaritymeasures.frechet_dist

def dtw_distance(curve1: Curve, curve2: Curve, n_samples: int = 100) -> float:
    """Compute Dynamic Time Warping distance."""
    # Use similaritymeasures.dtw

def curve_similarity_matrix(curves: List[Curve], metric: str = 'frechet') -> np.ndarray:
    """Compute pairwise distance matrix."""
```

**Metrics Comparison**:
- **Fréchet**: Good for smooth curves, respects trajectory
- **DTW**: Better for warped/stretched sequences
- **L2**: Fast baseline (point-wise difference)

**Tests**:
- Identical curves → distance = 0
- Opposite curves → large distance
- Triangle inequality (if applicable)

---

## Phase 2: Storage and Retrieval (Week 2)

### Day 8-9: Enhanced Storage

**File**: `src/level2/storage.py`

**Core Class**: `CurveStorage`

**Methods**:
```python
class CurveStorage:
    def __init__(self, db_path: str):
        """Initialize RocksDB storage."""

    def put_curve(self, doc_id: str, curve: Curve, fpca_scores: np.ndarray, metadata: dict):
        """Store curve with FPCA scores and metadata."""

    def get_curve(self, doc_id: str) -> Tuple[Curve, np.ndarray, dict]:
        """Retrieve curve by document ID."""

    def range_query_fpca(self, start: float, end: float, limit: int) -> List[tuple]:
        """Range query on FPCA-based Hilbert index."""
```

**Storage Format**:
```python
{
    "doc_id": str,
    "curve_params": bytes,  # pickle(Curve)
    "fpca_scores": bytes,   # numpy array
    "hilbert_dist": int,    # from FPCA scores
    "metadata": dict,
    "text": str
}
```

**Optimizations**:
- Compress curve parameters (msgpack)
- Index FPCA scores → Hilbert mapping
- Efficient serialization

---

### Day 10-11: Two-Stage Retrieval

**File**: `src/level2/retrieval.py`

**Core Class**: `CurveRetrieval`

**Methods**:
```python
class CurveRetrieval:
    def __init__(self, storage: CurveStorage, fpca: FunctionalPCA):
        """Initialize retrieval engine."""

    def search(
        self,
        query_curve: Curve,
        k: int = 10,
        radius: int = None,
        rerank_metric: str = 'frechet'
    ) -> List[Tuple[str, float]]:
        """Two-stage curve-based search."""
        # Stage 1: FPCA → Hilbert → range query (fast approximation)
        # Stage 2: Curve distance reranking (accurate)
```

**Algorithm**:
1. Query curve → FPCA scores
2. FPCA scores → Hilbert distance
3. Range query → candidate curves
4. Compute curve distances
5. Rerank and return top-K

**Performance**:
- Stage 1: O(log N) range query
- Stage 2: O(candidates × curve_eval_cost)
- Total: much faster than O(N) exhaustive

---

### Day 12-13: Main CurveDB Class

**File**: `src/level2/curve_db.py`

**Core Class**: `CurveDB`

**Methods**:
```python
class CurveDB:
    def __init__(
        self,
        db_path: str,
        n_fpca_components: int = 10,
        curve_degree: int = 3,
        model_name: str = "bert-base-uncased"
    ):
        """Initialize curve-based database."""

    def fit_fpca(self, texts: List[str]):
        """Fit FPCA on corpus."""
        # 1. Get token embeddings
        # 2. Fit curves
        # 3. Fit FPCA
        # 4. Save FPCA model

    def add_batch(self, doc_ids: List[str], texts: List[str], metadata: List[dict] = None):
        """Add documents with curve representation."""
        # 1. Token embeddings → curves
        # 2. FPCA transformation
        # 3. Hilbert mapping
        # 4. Store curve + FPCA + metadata

    def search(self, query: str, k: int = 10, rerank: bool = True) -> List[Tuple[str, str, float]]:
        """Search with curve-based similarity."""

    def get_stats(self) -> dict:
        """Database statistics."""
```

**Integration**:
- Combine all Level 2 modules
- Maintain similar API to Level 1
- Add curve-specific functionality

---

## Phase 3: Testing and Benchmarking (Week 3)

### Day 14-15: Unit Tests

**File**: `tests/test_level2.py`

**Test Coverage**:
- [ ] CurveFitter: token sequences → curves
- [ ] FPCA: fit/transform/inverse_transform
- [ ] Distances: Fréchet, DTW, edge cases
- [ ] Storage: put/get/range queries
- [ ] Retrieval: two-stage search
- [ ] CurveDB: end-to-end pipeline

**Target**: ≥70% coverage

---

### Day 16-17: Benchmark Suite

**File**: `benchmarks/level2_bench.py`

**Metrics**:
- Recall@K (K=1,5,10,20)
- MRR@10
- nDCG@10
- Latency (P50, P95, Mean)
- QPS
- Indexing speed
- Memory usage
- Storage size

**Comparisons**:
- Level 2 vs Level 1
- Level 2 vs Baseline (exhaustive)
- Different curve distance metrics

**Datasets**:
- MS MARCO 1K (quick test)
- MS MARCO 10K (main evaluation)

**Expected Results**:
- Recall@10: ≥75% (vs 0.7% Level 1)
- Latency: <200ms (vs <20ms Level 1, acceptable tradeoff)
- Quality significantly better than Level 1

---

### Day 18: Visualization Tools

**File**: `src/level2/viz.py`

**Functions**:
```python
def plot_curve(curve: Curve, title: str = ""):
    """Plot single curve in 3D."""

def plot_curve_comparison(curve1: Curve, curve2: Curve):
    """Compare two curves visually."""

def plot_fpca_components(fpca: FunctionalPCA, n_components: int = 3):
    """Visualize FPCA principal components."""

def plot_curve_space_tsne(fpca_scores: np.ndarray, labels: List[str]):
    """t-SNE visualization of curve space."""
```

**Deliverables**:
- Example visualizations in docs/
- Jupyter notebook: examples/level2_visualization.ipynb

---

### Day 19-20: Documentation

**File**: `docs/ru/level2.md`

**Sections**:
1. Overview and motivation
2. Curve representation theory
3. FPCA methodology
4. Architecture and components
5. API reference
6. Benchmark analysis
7. Level 2 vs Level 1 comparison
8. Performance tuning guide
9. When to use Level 2
10. Limitations and future work

**Additional**:
- Update README.md with Level 2
- Update SPECIFY.md with actual results
- Create migration guide from Level 1 to Level 2

---

### Day 21: Polish and Release

**Tasks**:
- [ ] Code review and refactoring
- [ ] Fix any remaining bugs
- [ ] Optimize performance bottlenecks
- [ ] Final documentation pass
- [ ] Create examples
- [ ] Git commit and push
- [ ] Tag release: v0.2.0

---

## Success Criteria

### Must Have ✅:
- [ ] All 6 core modules implemented and tested
- [ ] Recall@10 ≥75% on MS MARCO 10K
- [ ] Complete documentation
- [ ] Working end-to-end demo
- [ ] Unit tests with ≥70% coverage

### Nice to Have ⭐:
- [ ] Multiple distance metrics implemented
- [ ] Visualization tools
- [ ] Jupyter notebooks with examples
- [ ] Performance comparison plots

### Stretch Goals 🚀:
- [ ] Incremental FPCA updates
- [ ] Curve clustering for faster search
- [ ] Integration with Level 1 as fallback

---

## Risk Assessment

### High Risk:
1. **FPCA complexity**: scikit-fda has learning curve
   - *Mitigation*: Start with simple examples, read documentation thoroughly

2. **Curve distance computation slow**: Could bottleneck search
   - *Mitigation*: Profile early, optimize sampling, consider caching

3. **RocksDB installation issues**: System-dependent
   - *Mitigation*: Have plyvel as fallback, document installation

### Medium Risk:
4. **Variable-length sequences**: Curve fitting may struggle
   - *Mitigation*: Normalize sequence length, test edge cases early

5. **FPCA reconstruction quality**: May lose information
   - *Mitigation*: Tune n_components, monitor reconstruction error

### Low Risk:
6. **Storage format**: Pickle vs msgpack tradeoffs
   - *Mitigation*: Easy to switch, not critical path

---

## Dependencies

### Required Packages:
```python
# Core Level 2
scipy>=1.11.0              # Spline interpolation
scikit-fda>=0.8.0          # Functional PCA
similaritymeasures>=0.5.0  # Fréchet, DTW
python-rocksdb>=0.7.0      # Storage (or plyvel)
transformers>=4.30.0       # Token-level embeddings

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
umap-learn>=0.5.0          # For curve space visualization

# Inherited from Level 1
sentence-transformers>=2.2.0
numpy>=1.24.0
lmdb>=1.4.0
hilbertcurve>=2.0.0
pytest>=7.4.0
```

---

## Estimated LOC

| Module | Estimated LOC |
|--------|---------------|
| curve_fitter.py | 150-200 |
| fpca.py | 120-150 |
| distances.py | 80-100 |
| storage.py | 150-180 |
| retrieval.py | 120-150 |
| curve_db.py | 180-220 |
| viz.py | 100-120 |
| **Total Core** | **~900-1120** |
| Tests | 300-400 |
| Benchmarks | 200-250 |
| **Grand Total** | **~1400-1770** |

---

## Next Steps

1. Review and approve roadmap
2. Install dependencies
3. Begin Day 1-2: Setup and dependencies
4. Daily standup: progress, blockers, next steps

**Ready to start? 🚀**
