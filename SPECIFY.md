# CurvaDB Technical Specification

## Project Overview

**CurvaDB** is an experimental semantic search database that uses mathematical curves and space-filling curve indexing instead of traditional vector representations. The project explores whether curve-based representations can achieve better search quality, lower memory usage, and faster retrieval compared to standard vector databases.

## Scope of Work

The project is divided into **three development levels**, each building upon the previous one. Each level represents a complete, functional system that can be independently tested and evaluated.

---

## Level 1: Minimal Hilbert Vector DB

### Objective
Build a minimal working prototype that uses Hilbert curve indexing for vector embeddings. This establishes the baseline infrastructure and validates the core space-filling curve approach.

### Scope

#### 1.1 Core Components

**Text Embedding Module** (`src/level1/embedder.py`)
- Wrapper around SentenceTransformer
- Model: `all-MiniLM-L6-v2` (384 dimensions)
- Batch encoding support
- Input: raw text strings
- Output: numpy arrays of embeddings

**Dimensionality Reduction** (`src/level1/reducer.py`)
- PCA-based dimension reduction
- Configurable target dimensions (default: 8)
- Fit on initial batch, transform on all subsequent data
- Normalization to [0, 2^p - 1] range for Hilbert mapping

**Hilbert Indexing** (`src/level1/hilbert_index.py`)
- N-dimensional Hilbert curve implementation
- Conversion of N-D points to 1-D distance values
- Configurable precision (p) and dimensions (n)
- Default: p=16 (65,536 values per dimension), n=8

**Storage Layer** (`src/level1/storage.py`)
- LMDB key-value store
- Key: 64-bit Hilbert index (packed binary)
- Value: document ID + text + metadata (serialized)
- Configurable database size (default: 10GB)

**Search Engine** (`src/level1/search.py`)
- Range query on Hilbert index
- Configurable search radius
- Two-stage retrieval:
  1. Approximate search via Hilbert range
  2. Reranking by cosine similarity
- Return top-K results with scores

**Main Database Class** (`src/level1/minimal_db.py`)
- Public API combining all components
- Methods:
  - `__init__(db_path, p, n)`: Initialize database
  - `add_batch(doc_ids, texts)`: Insert documents
  - `search(query, k, radius)`: Search for similar documents
  - `close()`: Cleanup resources

#### 1.2 Features

✅ **Must Have**:
- Add documents (batch insertion)
- Semantic search (top-K retrieval)
- Persistence (survive restart)
- Configurable parameters (dimensions, precision)

❌ **Explicitly Out of Scope**:
- Update/delete operations
- Metadata filtering
- Multi-language support (English only for testing)
- Distributed deployment
- True curve representation (deferred to Level 2)

#### 1.3 Performance Targets

- **Indexing Speed**: ≥100 docs/second (single thread)
- **Search Latency**: <100ms for 10K documents
- **Memory Usage**: <500MB for 10K documents
- **Storage Size**: <50MB for 10K documents
- **Recall@10**: ≥70% vs exhaustive cosine search

#### 1.4 Deliverables

1. **Source Code**:
   - `src/level1/` - all modules listed above
   - ~150-200 lines of core logic
   - Type-annotated, documented

2. **Tests** (`tests/test_level1.py`):
   - Unit tests for each component
   - Integration test for end-to-end workflow
   - Minimum 70% code coverage

3. **Benchmark Suite** (`benchmarks/level1_bench.py`):
   - Synthetic dataset (1K, 5K, 10K documents)
   - Comparison with baseline (exhaustive cosine search)
   - Metrics: recall@K, latency, memory, storage

4. **Example Usage** (`examples/level1_demo.py`):
   - Simple demo script
   - ~30 lines showing basic usage

5. **Documentation** (`docs/ru/level1.md`):
   - Architecture overview
   - Usage guide
   - Benchmark results
   - Known limitations

#### 1.5 Dependencies

```
sentence-transformers>=2.2.0
hilbertcurve>=2.0.0
lmdb>=1.4.0
numpy>=1.24.0
scikit-learn>=1.3.0
pytest>=7.4.0  # dev dependency
```

#### 1.6 Estimated Effort

- **Development**: 2-3 days
- **Testing**: 1 day
- **Benchmarking**: 1 day
- **Documentation**: 0.5 day
- **Total**: ~5 days (1 work week)

---

## Level 2: Full Curve-Based DB

### Objective
Implement true curve-based representation using splines and Functional PCA (FPCA). This is where we test the core hypothesis: do curves provide better semantic representation than vectors?

### Scope

#### 2.1 Core Components

**Curve Fitting Module** (`src/level2/curve_fitter.py`)
- Convert token embeddings to parametric curves
- Use scipy.interpolate.splprep for spline fitting
- Handle variable-length sequences
- Store curve parameters (knots, coefficients)

**Functional Data Analysis** (`src/level2/fpca.py`)
- Functional PCA for dimensionality reduction
- Use scikit-fda library
- Extract low-dimensional curve features
- Fit on corpus, transform on new documents

**Curve Distance Metrics** (`src/level2/distances.py`)
- Frechet distance implementation
- Dynamic Time Warping (DTW)
- Use similaritymeasures library
- Configurable distance metric

**Enhanced Storage** (`src/level2/storage.py`)
- RocksDB for better performance
- Store both curve parameters and FPCA scores
- Support for metadata
- Efficient serialization with pickle/msgpack

**Two-Stage Retrieval** (`src/level2/retrieval.py`)
- Stage 1: Hilbert index lookup on FPCA scores (fast)
- Stage 2: Curve distance reranking (accurate)
- Configurable candidate pool size
- Return results with distance scores

**Main Database Class** (`src/level2/curve_db.py`)
- Extended API from Level 1
- Methods:
  - `fit_fpca(texts)`: Fit FPCA on corpus
  - `add(doc_id, text, metadata)`: Insert with metadata
  - `search_curve(query, k, radius, rerank)`: Curve-based search
  - `get_stats()`: Database statistics

#### 2.2 Features

✅ **Must Have**:
- True curve representation (splines)
- FPCA-based feature extraction
- Curve distance reranking
- Metadata storage and retrieval
- Corpus-based FPCA fitting

⚠️ **Nice to Have** (if time permits):
- Multiple curve distance metrics
- Curve visualization utilities
- Incremental FPCA updates

❌ **Out of Scope**:
- Real-time updates (batch-oriented)
- Multi-modal data (text only)
- Distributed architecture
- GPU acceleration

#### 2.3 Performance Targets

- **Indexing Speed**: ≥50 docs/second (curve fitting is slower)
- **Search Latency**: <200ms for 10K documents
- **Memory Usage**: <800MB for 10K documents
- **Storage Size**: <30MB for 10K documents (curve compression)
- **Recall@10**: ≥75% vs exhaustive search
- **Quality**: Better than Level 1 in at least ONE metric

#### 2.4 Deliverables

1. **Source Code**:
   - `src/level2/` - all modules
   - ~400-600 lines of core logic
   - Clean separation from Level 1 code

2. **Tests** (`tests/test_level2.py`):
   - Unit tests for curve fitting, FPCA, distances
   - Integration tests
   - Edge cases (short texts, long texts)
   - 70% coverage minimum

3. **Enhanced Benchmark Suite** (`benchmarks/level2_bench.py`):
   - Real dataset (MS MARCO subset or similar)
   - Comparison with Level 1 AND baseline
   - Additional metrics:
     - Curve fitting time
     - FPCA transformation time
     - Storage efficiency (compression ratio)
   - Quality metrics:
     - nDCG@10
     - MRR (Mean Reciprocal Rank)

4. **Visualization Tools** (`src/level2/viz.py`):
   - Plot curves for sample documents
   - Visualize FPCA components
   - t-SNE/UMAP of curve features

5. **Documentation** (`docs/ru/level2.md`):
   - Curve representation theory
   - FPCA methodology
   - Benchmark analysis
   - When to use Level 2 vs Level 1
   - Performance tuning guide

#### 2.5 Dependencies

Inherits from Level 1, plus:

```
scipy>=1.11.0
scikit-fda>=0.8.0
similaritymeasures>=0.5.0
python-rocksdb>=0.7.0  # or plyvel as alternative
matplotlib>=3.7.0  # for visualization
torch>=2.0.0  # for token-level embeddings
```

#### 2.6 Estimated Effort

- **Development**: 1-1.5 weeks
- **Testing**: 2-3 days
- **Benchmarking**: 2-3 days (more complex)
- **Visualization**: 1-2 days
- **Documentation**: 1 day
- **Total**: ~2.5-3 weeks

---

## Level 3: Optimized Production System

### Objective
Transform the experimental prototype into a production-ready system with performance optimizations, scalability improvements, and comprehensive monitoring.

### Scope

#### 3.1 Core Enhancements

**Fast Curve Distances** (`src/level3/fast_distances.py`)
- Integration with Fred-Frechet C++ library
- 10-100x speedup for curve comparisons
- Fallback to Python implementation if Fred unavailable

**Parallel Processing** (`src/level3/parallel.py`)
- Multiprocessing for batch operations
- Parallel curve fitting
- Parallel distance computation
- Configurable worker pool size

**Multi-Probe Hilbert Search** (`src/level3/multi_probe.py`)
- Multiple Hilbert curves with different rotations
- LSH-style multi-probe technique
- Better coverage of the search space
- Configurable number of probes

**Curve Compression** (`src/level3/compression.py`)
- Compress spline parameters
- Quantization of coefficients
- Reduce storage footprint by 2-5x
- Configurable compression level

**Caching Layer** (`src/level3/cache.py`)
- LRU cache for frequent queries
- Cache FPCA transformations
- Precomputed curve distances for popular documents
- Configurable cache size

**Monitoring & Metrics** (`src/level3/metrics.py`)
- Prometheus-style metrics
- Track: query latency, throughput, cache hit rate
- Resource usage monitoring
- Query distribution analysis

**API Server** (`src/level3/api.py`)
- REST API using FastAPI
- Endpoints:
  - `POST /add` - Insert documents
  - `POST /search` - Search query
  - `GET /stats` - Database statistics
  - `GET /health` - Health check
- OpenAPI documentation
- Request validation with Pydantic

#### 3.2 Features

✅ **Must Have**:
- 10-100x faster curve distances (Fred)
- Parallel batch processing
- Multi-probe search
- Curve parameter compression
- REST API
- Monitoring dashboard
- Configuration management

⚠️ **Nice to Have**:
- gRPC API option
- WebSocket for streaming results
- Admin UI for management
- Docker containerization

❌ **Still Out of Scope**:
- Multi-node distribution
- GPU acceleration
- Real-time streaming updates
- Multi-tenancy

#### 3.3 Performance Targets

- **Indexing Speed**: ≥500 docs/second (with parallelization)
- **Search Latency**: <50ms for 100K documents (p95)
- **Throughput**: ≥100 queries/second (concurrent)
- **Memory Usage**: <5GB for 100K documents
- **Storage Size**: <200MB for 100K documents (compressed)
- **Recall@10**: ≥85% vs exhaustive search
- **Uptime**: 99.9% (monitoring and graceful degradation)

#### 3.4 Deliverables

1. **Optimized Source Code**:
   - `src/level3/` - production modules
   - ~1500-2000 lines total
   - Clean API design
   - Extensive error handling
   - Logging throughout

2. **Comprehensive Tests**:
   - Unit, integration, and stress tests
   - Performance regression tests
   - API endpoint tests
   - 80% coverage minimum

3. **Production Benchmark Suite**:
   - Standard datasets (MS MARCO, Natural Questions)
   - Comparison with real vector DBs (FAISS, Annoy)
   - Scalability tests (10K → 100K → 1M documents)
   - Stress testing (concurrent queries)
   - Detailed performance report

4. **API Documentation**:
   - OpenAPI/Swagger specs
   - Client examples (Python, curl)
   - Authentication guide (if implemented)

5. **Deployment Guide** (`docs/deployment.md`):
   - System requirements
   - Installation steps
   - Configuration options
   - Performance tuning
   - Monitoring setup
   - Troubleshooting

6. **Architecture Documentation** (`docs/architecture.md`):
   - System design
   - Component interaction
   - Data flow diagrams
   - Design decisions (ADRs)

7. **Benchmark Report** (`docs/benchmarks.md`):
   - Comprehensive results
   - Comparison with vector DBs
   - When to use CurvaDB
   - Performance vs accuracy trade-offs

#### 3.5 Dependencies

Inherits from Level 2, plus:

```
Fred-Frechet>=2.0.0
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.4.0
prometheus-client>=0.18.0
msgpack>=1.0.0
python-multipart>=0.0.6
httpx>=0.25.0  # for testing API
```

#### 3.6 Estimated Effort

- **Core Optimization**: 1 week
- **Parallelization**: 3-4 days
- **API Development**: 4-5 days
- **Monitoring**: 2-3 days
- **Testing**: 1 week
- **Benchmarking**: 1 week
- **Documentation**: 3-4 days
- **Total**: ~4-5 weeks

---

## Cross-Level Requirements

### Code Quality

All levels must maintain:
- **Type hints** for all public APIs
- **Google-style docstrings** for classes and methods
- **Black** formatting (line length: 100)
- **Pylint score** ≥8.0/10
- **No unused imports or variables**

### Testing

- **pytest** as test framework
- **pytest-cov** for coverage reporting
- **Test naming**: `test_<functionality>_<scenario>`
- **Fixtures** for common setup
- **Parametrized tests** for multiple scenarios

### Documentation

- **README.md** with quick start
- **Module docstrings** explaining purpose
- **Inline comments** for complex logic only
- **Type hints** as primary documentation
- **Examples** in docstrings for public APIs

### Version Control

- **Git** for version control
- **Conventional commits**: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`
- **Semantic versioning**: v0.1.0 (Level 1), v0.2.0 (Level 2), v1.0.0 (Level 3)
- **.gitignore** for Python, data files, notebooks

### Configuration

- **YAML or TOML** for configuration files
- **Environment variables** for sensitive data
- **Defaults** for all configuration options
- **Validation** of configuration on load

---

## Success Metrics Summary

| Metric | Level 1 | Level 2 | Level 3 |
|--------|---------|---------|---------|
| **Code Size** | ~200 lines | ~600 lines | ~2000 lines |
| **Max Documents** | 10K | 50K | 100K+ |
| **Search Latency** | <100ms | <200ms | <50ms |
| **Indexing Speed** | 100 doc/s | 50 doc/s | 500 doc/s |
| **Recall@10** | ≥70% | ≥75% | ≥85% |
| **Development Time** | 5 days | 3 weeks | 5 weeks |

---

## Risk Assessment

### Technical Risks

| Risk | Level | Mitigation |
|------|-------|------------|
| Curve fitting is too slow | Medium | Use Fred library, parallel processing |
| FPCA doesn't preserve semantics | High | Validate with benchmarks, fallback to PCA |
| Hilbert mapping loses locality | Medium | Multi-probe search, tune parameters |
| Storage overhead from curves | Low | Compression, parameter quantization |
| Dependencies are unstable | Low | Pin versions, vendor if needed |

### Project Risks

| Risk | Level | Mitigation |
|------|-------|------------|
| Curves don't outperform vectors | High | Accept negative results, publish findings |
| Scope creep | Medium | Strict adherence to this spec |
| Over-optimization too early | Medium | Follow level progression strictly |
| Benchmark bias | Medium | Use multiple standard datasets |

---

## Out of Scope (All Levels)

The following features are **explicitly excluded** from all three levels:

❌ Multi-modal data (images, audio)
❌ Distributed/cluster deployment
❌ GPU acceleration
❌ Real-time streaming updates
❌ Multi-tenancy and access control
❌ Advanced query languages (SQL, GraphQL)
❌ Integration with external systems (Elasticsearch, etc.)
❌ Mobile/edge deployment
❌ Federated learning
❌ Differential privacy

These may be considered for **future work** after Level 3 if the core hypothesis proves successful.

---

## Appendix: Dataset Sources

### For Testing (Levels 1-2)
- **Synthetic**: Generated Lorem Ipsum with semantic clustering
- **20 Newsgroups**: sklearn.datasets.fetch_20newsgroups
- **AG News**: Hugging Face datasets (ag_news)
- **SQuAD**: Question-answer pairs for retrieval

### For Production Benchmarks (Level 3)
- **MS MARCO**: Microsoft Machine Reading Comprehension
- **Natural Questions**: Google Q&A dataset
- **BEIR**: Benchmark for Information Retrieval
- **Wikipedia subset**: Standard IR corpus

---

**Version**: 1.0
**Last Updated**: 2025-11-18
**Status**: Active
**Estimated Total Effort**: ~9-10 weeks (Level 1 → Level 3)
