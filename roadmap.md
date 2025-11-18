# CurvaDB Development Roadmap

## Overview

This roadmap outlines the development plan for **Level 1: Minimal Hilbert Vector DB**. The goal is to build a working prototype in ~5 days that validates the core space-filling curve indexing approach.

---

## Level 1: Minimal Hilbert Vector DB

**Duration**: 5 days
**Goal**: Functional prototype with benchmark validation
**Success Criteria**: Working search on 10K docs with ≥70% recall@10

### Day 1: Project Setup & Core Infrastructure

#### Morning (2-3 hours)
- [x] Create project structure
- [x] Set up `constitution.md`, `SPECIFY.md`, `roadmap.md`
- [ ] Initialize Python package structure
  - [ ] Create `pyproject.toml` with dependencies
  - [ ] Set up `src/level1/__init__.py`
  - [ ] Configure development tools (black, pylint, pytest)
- [ ] Set up version control
  - [ ] Initialize git repository
  - [ ] Create `.gitignore` for Python
  - [ ] Initial commit with spec files

#### Afternoon (3-4 hours)
- [ ] **Component 1: Text Embedder** (`src/level1/embedder.py`)
  - [ ] Create `TextEmbedder` class
  - [ ] Wrap SentenceTransformer model
  - [ ] Implement batch encoding
  - [ ] Add type hints and docstrings
  - [ ] Write unit tests (`tests/test_embedder.py`)

- [ ] **Component 2: Dimensionality Reducer** (`src/level1/reducer.py`)
  - [ ] Create `DimensionalityReducer` class
  - [ ] PCA implementation with sklearn
  - [ ] Normalization to [0, 2^p-1] range
  - [ ] Persistence (save/load PCA model)
  - [ ] Write unit tests (`tests/test_reducer.py`)

#### Evening (1-2 hours)
- [ ] Integration test for embedder + reducer pipeline
- [ ] Verify model downloads work correctly
- [ ] Documentation: `docs/level1_architecture.md` (draft)

**Deliverables**:
- ✅ Project structure
- ✅ Embedder module with tests
- ✅ Reducer module with tests
- ✅ 2/6 core components complete

---

### Day 2: Hilbert Indexing & Storage

#### Morning (3-4 hours)
- [ ] **Component 3: Hilbert Index** (`src/level1/hilbert_index.py`)
  - [ ] Create `HilbertIndexer` class
  - [ ] Wrap hilbertcurve library
  - [ ] Point-to-distance conversion
  - [ ] Distance-to-point conversion (for debugging)
  - [ ] Configure p (precision) and n (dimensions)
  - [ ] Write unit tests (`tests/test_hilbert.py`)
  - [ ] Test with sample 2D/3D points for visualization

#### Afternoon (3-4 hours)
- [ ] **Component 4: Storage Layer** (`src/level1/storage.py`)
  - [ ] Create `LMDBStorage` class
  - [ ] Initialize LMDB environment
  - [ ] Implement `put(key, value)` method
  - [ ] Implement `get(key)` method
  - [ ] Implement `range_query(start_key, end_key)` method
  - [ ] Serialization (struct for keys, msgpack/pickle for values)
  - [ ] Write unit tests (`tests/test_storage.py`)
  - [ ] Test persistence (write, close, reopen, read)

#### Evening (1 hour)
- [ ] Integration test: Hilbert + Storage
  - [ ] Store points with Hilbert indices
  - [ ] Perform range queries
  - [ ] Verify spatial locality preservation

**Deliverables**:
- ✅ Hilbert indexing module with tests
- ✅ Storage layer with tests
- ✅ 4/6 core components complete

---

### Day 3: Search Engine & Main Database Class

#### Morning (3-4 hours)
- [ ] **Component 5: Search Engine** (`src/level1/search.py`)
  - [ ] Create `SearchEngine` class
  - [ ] Implement query embedding pipeline
  - [ ] Hilbert range query logic
  - [ ] Configurable search radius
  - [ ] Cosine similarity reranking
  - [ ] Return top-K results with scores
  - [ ] Write unit tests (`tests/test_search.py`)

#### Afternoon (3-4 hours)
- [ ] **Component 6: Main Database** (`src/level1/minimal_db.py`)
  - [ ] Create `MinimalCurveDB` class
  - [ ] Integrate all components
  - [ ] Implement `__init__` (initialization)
  - [ ] Implement `add_batch(doc_ids, texts)` method
  - [ ] Implement `search(query, k, radius)` method
  - [ ] Implement `close()` cleanup method
  - [ ] Configuration management
  - [ ] Write integration tests (`tests/test_minimal_db.py`)

#### Evening (1-2 hours)
- [ ] End-to-end integration test
  - [ ] Create small test dataset (100 documents)
  - [ ] Add documents to database
  - [ ] Perform multiple search queries
  - [ ] Verify results make sense
  - [ ] Test persistence (restart and search again)
- [ ] Fix any bugs discovered

**Deliverables**:
- ✅ Search engine module with tests
- ✅ Main database class with tests
- ✅ 6/6 core components complete
- ✅ Fully functional prototype

---

### Day 4: Testing & Example Code

#### Morning (2-3 hours)
- [ ] **Comprehensive Testing**
  - [ ] Review test coverage (run `pytest --cov`)
  - [ ] Write additional tests for edge cases:
    - [ ] Empty queries
    - [ ] Very short texts (<5 words)
    - [ ] Very long texts (>1000 words)
    - [ ] Special characters and Unicode
    - [ ] Duplicate documents
  - [ ] Fix any bugs discovered
  - [ ] Ensure ≥70% code coverage

#### Afternoon (2-3 hours)
- [ ] **Example Code** (`examples/level1_demo.py`)
  - [ ] Simple demo script (~30 lines)
  - [ ] Show basic usage:
    - [ ] Initialize database
    - [ ] Add sample documents
    - [ ] Perform searches
    - [ ] Display results
  - [ ] Add comments explaining each step

- [ ] **Advanced Example** (`examples/level1_rag_demo.py`)
  - [ ] RAG (Retrieval-Augmented Generation) pipeline
  - [ ] Integrate with simple text generation
  - [ ] Show practical use case

#### Evening (2 hours)
- [ ] **Code Quality**
  - [ ] Run black formatter on all code
  - [ ] Run pylint and fix issues (target ≥8.0/10)
  - [ ] Check type hints coverage
  - [ ] Review docstrings for completeness
  - [ ] Clean up any TODO comments

**Deliverables**:
- ✅ Comprehensive test suite (≥70% coverage)
- ✅ Example scripts demonstrating usage
- ✅ Clean, well-formatted code

---

### Day 5: Benchmarking & Documentation

#### Morning (3-4 hours)
- [ ] **Benchmark Suite** (`benchmarks/level1_bench.py`)
  - [ ] Create benchmark framework
  - [ ] Generate synthetic test datasets:
    - [ ] 1,000 documents (small)
    - [ ] 5,000 documents (medium)
    - [ ] 10,000 documents (large)
  - [ ] Implement baseline (exhaustive cosine search)
  - [ ] Measure metrics:
    - [ ] **Recall@K** (K=1, 5, 10, 20)
    - [ ] **Search latency** (p50, p95, p99)
    - [ ] **Indexing time** (total and per-doc)
    - [ ] **Memory usage** (process RSS)
    - [ ] **Storage size** (database file size)
  - [ ] Run benchmarks and collect results
  - [ ] Create results visualization (matplotlib)

#### Afternoon (2-3 hours)
- [ ] **Documentation** (`docs/level1.md`)
  - [ ] Architecture overview with diagrams
  - [ ] Component descriptions
  - [ ] Usage guide with examples
  - [ ] API reference (from docstrings)
  - [ ] Configuration options
  - [ ] Performance characteristics
  - [ ] Known limitations
  - [ ] Troubleshooting section

- [ ] **README.md** (project root)
  - [ ] Project description
  - [ ] Quick start guide
  - [ ] Installation instructions
  - [ ] Basic usage example
  - [ ] Link to detailed docs
  - [ ] Contributing guidelines
  - [ ] License (MIT)

#### Evening (1-2 hours)
- [ ] **Benchmark Results Analysis** (`docs/benchmark_results.md`)
  - [ ] Write up findings
  - [ ] Compare against success criteria
  - [ ] Identify bottlenecks
  - [ ] Recommendations for Level 2
  - [ ] Decision: proceed to Level 2 or pivot?

- [ ] **Final Review**
  - [ ] Run all tests one more time
  - [ ] Verify examples work
  - [ ] Check documentation links
  - [ ] Git commit all work
  - [ ] Tag release: `v0.1.0`

**Deliverables**:
- ✅ Complete benchmark suite with results
- ✅ Comprehensive documentation
- ✅ README with quick start
- ✅ Analysis and recommendations

---

## Level 1 Checklist

Before advancing to Level 2, verify:

### Functionality
- [ ] Database can insert documents (batch operation)
- [ ] Database can search and return top-K results
- [ ] Results are semantically relevant (manual inspection)
- [ ] Database persists data across restarts
- [ ] Configuration parameters work correctly

### Performance
- [ ] Handles 10,000 documents without crashing
- [ ] Search latency <100ms (p95) for 10K docs
- [ ] Recall@10 ≥70% vs exhaustive search
- [ ] Memory usage <500MB for 10K docs
- [ ] Storage size <50MB for 10K docs

### Code Quality
- [ ] All modules have type hints
- [ ] All public APIs have docstrings
- [ ] Code passes black formatting
- [ ] Pylint score ≥8.0/10
- [ ] No critical bugs or crashes

### Testing
- [ ] Test coverage ≥70%
- [ ] All tests pass
- [ ] Edge cases covered
- [ ] Integration tests work end-to-end

### Documentation
- [ ] Architecture documented
- [ ] Usage guide complete
- [ ] Examples work and are clear
- [ ] Benchmark results documented
- [ ] README is informative

### Deliverables
- [ ] Source code in `src/level1/`
- [ ] Tests in `tests/`
- [ ] Benchmarks in `benchmarks/`
- [ ] Examples in `examples/`
- [ ] Documentation in `docs/`
- [ ] README.md in root
- [ ] Git tagged as `v0.1.0`

---

## Dependencies Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install --upgrade pip
pip install sentence-transformers>=2.2.0
pip install hilbertcurve>=2.0.0
pip install lmdb>=1.4.0
pip install numpy>=1.24.0
pip install scikit-learn>=1.3.0
pip install msgpack>=1.0.0

# Development dependencies
pip install pytest>=7.4.0
pip install pytest-cov>=4.1.0
pip install black>=23.0.0
pip install pylint>=3.0.0
pip install matplotlib>=3.7.0

# Or install everything from requirements.txt
pip install -r requirements.txt
```

---

## Risk Mitigation

### Potential Blockers

| Risk | Impact | Mitigation | Status |
|------|--------|------------|--------|
| Model download fails | High | Pre-download models, cache locally | Pending |
| LMDB setup issues | Medium | Test on Day 2, have plyvel as backup | Pending |
| Poor recall@10 | High | Tune radius parameter, compare methods | Pending |
| Hilbert locality breaks | Medium | Test with synthetic clustered data | Pending |
| Memory leaks | Medium | Profile early, fix as discovered | Pending |

### Fallback Plans

- **If Hilbert indexing fails**: Fall back to simple PCA + cosine search (still useful baseline)
- **If LMDB is problematic**: Use SQLite or simple file-based storage
- **If recall is too low**: Increase search radius or candidate pool size
- **If performance is poor**: Reduce dimensions or use simpler embedder

---

## Communication & Reporting

### Daily Standup (End of Day)
Report:
1. **What was completed today?**
2. **What's blocking progress?**
3. **What's planned for tomorrow?**
4. **Any changes to timeline or scope?**

### End of Level 1 Report
Include:
1. **What worked well?**
2. **What didn't work as expected?**
3. **Key metrics vs targets**
4. **Recommendation: Proceed to Level 2?**
5. **Lessons learned for Level 2**

---

## Success Criteria

Level 1 is considered **successful** if:

1. ✅ All core components implemented and tested
2. ✅ Database works end-to-end (add + search)
3. ✅ Benchmark shows ≥70% recall@10
4. ✅ Search latency <100ms for 10K docs
5. ✅ Code quality meets standards
6. ✅ Documentation is complete

Level 1 is **successful enough to proceed** if:
- Even if recall is slightly below 70%, we understand why
- Performance is acceptable for a prototype
- Architecture is sound and extensible to Level 2

Level 1 requires **pivot or stop** if:
- Fundamental architectural flaws discovered
- Hilbert approach completely fails (no spatial locality)
- Better alternatives found during research
- Effort vs value doesn't justify continuing

---

## Next Steps After Level 1

If Level 1 succeeds:
1. **Review benchmark results** with stakeholders
2. **Plan Level 2 roadmap** (detailed 3-week plan)
3. **Set up Level 2 development branch**
4. **Research curve fitting approaches** in more depth
5. **Begin Level 2 implementation**

If Level 1 needs iteration:
1. **Identify specific issues** from benchmarks
2. **Create improvement plan** (1-2 days max)
3. **Re-run benchmarks** after fixes
4. **Re-evaluate** proceed/pivot decision

---

**Version**: 1.0
**Created**: 2025-11-18
**Status**: Active
**Next Review**: After Day 5 completion

---

## Appendix: Detailed Task Breakdown

### Day 1 Detailed Tasks

```
[ ] 1.1 Project Setup (30 min)
    [ ] Create directory structure
    [ ] Initialize git
    [ ] Create .gitignore
    [ ] First commit

[ ] 1.2 Python Package (45 min)
    [ ] Create pyproject.toml
    [ ] Set up src/level1/__init__.py
    [ ] Configure black, pylint
    [ ] Install dependencies

[ ] 1.3 Text Embedder (90 min)
    [ ] Create embedder.py
    [ ] TextEmbedder class
    [ ] encode() method
    [ ] encode_batch() method
    [ ] Type hints + docstrings
    [ ] Unit tests

[ ] 1.4 Dimensionality Reducer (90 min)
    [ ] Create reducer.py
    [ ] DimensionalityReducer class
    [ ] fit() method
    [ ] transform() method
    [ ] normalize() method
    [ ] save/load methods
    [ ] Unit tests

[ ] 1.5 Integration Test (30 min)
    [ ] Test embedder → reducer pipeline
    [ ] Verify shapes and ranges

[ ] 1.6 Documentation Draft (30 min)
    [ ] Start architecture doc
    [ ] Document design decisions
```

### Day 2 Detailed Tasks

```
[ ] 2.1 Hilbert Indexer (120 min)
    [ ] Create hilbert_index.py
    [ ] HilbertIndexer class
    [ ] points_to_distances() method
    [ ] distances_to_points() method
    [ ] Configuration
    [ ] Unit tests
    [ ] Visualization test (optional)

[ ] 2.2 Storage Layer (120 min)
    [ ] Create storage.py
    [ ] LMDBStorage class
    [ ] put() method
    [ ] get() method
    [ ] range_query() method
    [ ] Serialization helpers
    [ ] Unit tests

[ ] 2.3 Integration Test (60 min)
    [ ] Hilbert + Storage together
    [ ] Test spatial locality
    [ ] Test persistence
```

### Day 3 Detailed Tasks

```
[ ] 3.1 Search Engine (150 min)
    [ ] Create search.py
    [ ] SearchEngine class
    [ ] Query processing pipeline
    [ ] Range query execution
    [ ] Reranking logic
    [ ] Unit tests

[ ] 3.2 Main Database (150 min)
    [ ] Create minimal_db.py
    [ ] MinimalCurveDB class
    [ ] __init__() integration
    [ ] add_batch() implementation
    [ ] search() implementation
    [ ] close() cleanup
    [ ] Integration tests

[ ] 3.3 E2E Test (60 min)
    [ ] Full workflow test
    [ ] Persistence test
    [ ] Bug fixes
```

### Day 4 Detailed Tasks

```
[ ] 4.1 Edge Case Testing (120 min)
    [ ] Empty/malformed inputs
    [ ] Boundary conditions
    [ ] Unicode handling
    [ ] Coverage analysis
    [ ] Bug fixes

[ ] 4.2 Examples (120 min)
    [ ] Basic demo script
    [ ] RAG demo script
    [ ] Comments and explanations

[ ] 4.3 Code Quality (90 min)
    [ ] Black formatting
    [ ] Pylint fixes
    [ ] Docstring review
    [ ] Type hint coverage
```

### Day 5 Detailed Tasks

```
[ ] 5.1 Benchmark Suite (180 min)
    [ ] Framework setup
    [ ] Dataset generation
    [ ] Baseline implementation
    [ ] Metric collection
    [ ] Visualization
    [ ] Run benchmarks

[ ] 5.2 Documentation (120 min)
    [ ] docs/level1.md (architecture, usage)
    [ ] README.md (quick start)
    [ ] API reference
    [ ] Configuration guide

[ ] 5.3 Results Analysis (60 min)
    [ ] Write benchmark report
    [ ] Analyze results
    [ ] Make recommendations
    [ ] Final review and commit
```

---

**Total Estimated Time**: ~40-45 hours (1 work week)
