# CurvaDB Constitution

## Project Vision

CurvaDB is an experimental database system that represents data as mathematical curves instead of static vectors, aiming to achieve superior semantic search performance through innovative curve-based indexing and retrieval.

## Core Principles

### 1. Research-First Mindset
- This is an **experimental research project**, not a production replacement for existing vector databases
- Every design decision must be **measured and validated** with benchmarks
- We prioritize **learning and discovery** over premature optimization
- Negative results are valuable - document what doesn't work

### 2. Incremental Development
- Build in **clear levels** of complexity: Level 1 → Level 2 → Level 3
- Each level must be **fully functional and tested** before advancing
- No level is skipped - we validate assumptions at each stage
- **Fail fast**: If a level doesn't show promise, we pivot or stop

### 3. Benchmark-Driven Validation
- **Every claim must be proven** with reproducible benchmarks
- Compare against established baselines (cosine similarity, FAISS, etc.)
- Measure what matters:
  - Search quality: Recall@K, Precision@K, nDCG
  - Performance: Latency (p50, p95, p99), throughput
  - Resources: Memory usage, storage size, indexing time
- Maintain honest comparison - no cherry-picking favorable metrics

### 4. Code Quality Standards
- **Simplicity over cleverness** - readable code that works
- **Type hints** for all public APIs (Python 3.10+)
- **Docstrings** for all classes and public methods
- **Tests** for core functionality (pytest)
- **No premature optimization** - measure first, optimize second

### 5. Scientific Integrity
- Document **all experiments**, including failures
- Share **raw benchmark data**, not just summaries
- Be **honest about limitations** and trade-offs
- If curve-based approach doesn't outperform vectors, we admit it

### 6. Minimal Dependencies
- Prefer **well-established libraries** (scipy, numpy, scikit-learn)
- Every dependency must be **justified** and **documented**
- Pin versions for reproducibility
- No exotic or unmaintained packages

### 7. Open Science
- All code is **open source** (MIT License)
- Benchmark datasets are **publicly available**
- Results are **reproducible** by anyone
- Document methodology clearly

## Non-Goals

❌ **Not** trying to replace production vector databases immediately
❌ **Not** building a distributed system (single-node focus)
❌ **Not** supporting every feature (no multi-modal, no GPU acceleration in v1)
❌ **Not** optimizing for write-heavy workloads (read-optimized)

## Success Criteria

### Level 1 Success:
- ✅ Working prototype with <200 lines of core code
- ✅ Functional search on 1K+ documents
- ✅ Benchmark suite comparing to baseline
- ✅ Clear understanding of performance characteristics

### Level 2 Success:
- ✅ True curve-based representation (splines + FPCA)
- ✅ Handles 10K+ documents
- ✅ Measurable improvement in at least ONE metric vs baseline
- ✅ Documented trade-offs and use cases

### Level 3 Success:
- ✅ Optimized implementation (Fred library, parallelization)
- ✅ Scales to 100K+ documents
- ✅ Production-ready code quality
- ✅ Published benchmark results and findings

## Decision-Making Framework

### When to advance to next level:
1. Current level meets all success criteria
2. Benchmarks show promising results OR interesting insights
3. Core hypothesis is validated (even partially)
4. Clear path forward identified

### When to pivot or stop:
1. Fundamental performance issues with no solution
2. Curve approach demonstrably worse than vectors across all metrics
3. Complexity doesn't justify marginal gains
4. Better alternatives discovered

## Code Organization

```
CurvaDB/
├── constitution.md          # This file - project principles
├── SPECIFY.md              # Detailed specifications
├── roadmap.md              # Development roadmap
├── src/                    # Source code
│   ├── level1/            # Minimal Hilbert Vector DB
│   ├── level2/            # Full Curve-Based DB
│   └── level3/            # Optimized Production System
├── tests/                  # Unit and integration tests
├── benchmarks/             # Benchmark suite
│   ├── datasets/          # Test datasets
│   └── results/           # Benchmark results
├── docs/                   # Documentation
└── examples/               # Usage examples
```

## Coding Standards

### Python Style
- **PEP 8** compliance (use `black` formatter)
- **Type hints** required for public APIs
- **Docstrings** in Google style
- **Maximum line length**: 100 characters

### Testing
- Minimum **70% code coverage** for core modules
- Unit tests for algorithmic correctness
- Integration tests for end-to-end workflows
- Benchmark tests for performance tracking

### Git Workflow
- **Descriptive commit messages** (conventional commits style)
- **Small, focused commits** - one logical change per commit
- **Feature branches** for major changes
- **Main branch** always in working state

### Documentation
- **README.md** with quick start guide
- **API documentation** auto-generated from docstrings
- **Benchmark reports** in markdown
- **Architecture decisions** documented in docs/adr/

## Performance Philosophy

1. **Measure first** - no optimization without profiling
2. **Simple is fast** - clean code often outperforms clever code
3. **Good enough is good** - 80/20 rule applies
4. **Benchmark everything** - intuition lies, data doesn't

## Community Principles

- **Respectful discourse** - critique ideas, not people
- **Assume good intent** - we're all learning
- **Share knowledge** - document what you learn
- **Give credit** - acknowledge contributions and prior art

---

**Last Updated**: 2025-11-18
**Version**: 1.0
**Status**: Active
