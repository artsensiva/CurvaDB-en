> Архивная версия README до перехода проекта на trajectory-pivot
> (хранилище GPS-траекторий на сплайнах с поиском по Фреше). Код
> Level 1 (Hilbert-индекс над эмбеддингами, `src/level1`) не трогается
> и не расширяется, но описание оставлено здесь как контекст.

# CurvaDB - Curve-Based Semantic Search Database

An experimental database that uses mathematical curves and space-filling curve indexing for semantic search, exploring whether curve-based representations can outperform traditional vector databases.

## Overview

**CurvaDB** represents documents as mathematical curves instead of static vectors, using:
- **Hilbert curve indexing** for efficient spatial search
- **Spline fitting** for continuous data representation (Level 2+)
- **Functional PCA** for dimensionality reduction (Level 2+)
- **Curve distance metrics** (Frechet, DTW) for semantic similarity

## Current Status

🚧 **In Development - Level 1**

- [x] Project specification complete
- [x] Architecture designed
- [ ] Core implementation (in progress)
- [ ] Benchmarking suite
- [ ] Production-ready

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/CurvaDB.git
cd CurvaDB

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage (Level 1)

```python
from src.level1.minimal_db import MinimalCurveDB

# Initialize database
db = MinimalCurveDB(db_path='./my_db', p=16, n=8)

# Add documents
db.add_batch(
    doc_ids=['doc1', 'doc2', 'doc3'],
    texts=[
        'Machine learning is fascinating',
        'Deep neural networks are powerful',
        'I love cooking pasta'
    ]
)

# Search
results = db.search('AI and neural networks', k=2)
for doc_id, text, score in results:
    print(f"{doc_id}: {text[:50]}... (score: {score:.3f})")

# Close database
db.close()
```

## Project Structure

```
CurvaDB/
├── constitution.md          # Project principles and guidelines
├── SPECIFY.md              # Technical specifications (Levels 1-3)
├── roadmap.md              # Development roadmap
├── README.md               # This file
├── src/
│   ├── level1/            # Minimal Hilbert Vector DB
│   ├── level2/            # Full Curve-Based DB
│   └── level3/            # Optimized Production System
├── tests/                  # Unit and integration tests
├── benchmarks/             # Benchmark suite and results
├── examples/               # Usage examples
└── docs/                   # Documentation

```

## Development Levels

### Level 1: Minimal Hilbert Vector DB ✅ Current
- Space-filling curve indexing of vector embeddings
- ~200 lines of code
- 10K documents
- 5 days development

### Level 2: Full Curve-Based DB 🔜 Next
- True curve representation with splines
- Functional PCA feature extraction
- ~600 lines of code
- 50K documents
- 3 weeks development

### Level 3: Optimized Production System 🔮 Future
- Fast C++ curve distances (Fred library)
- Parallel processing
- REST API
- ~2000 lines of code
- 100K+ documents
- 5 weeks development

## Why Curves?

Potential advantages over traditional vector databases:
- **Compression**: Curve parameters use less memory than full vectors
- **Smoothness**: Built-in noise reduction through continuous representation
- **Derivatives**: Analytical derivatives available for free
- **Temporal data**: Natural representation of time-series semantics
- **Mathematical richness**: Wide array of curve operations

## Benchmarks

Performance targets for Level 1:
- **Search Latency**: <100ms (p95) for 10K documents
- **Recall@10**: ≥70% vs exhaustive cosine search
- **Memory Usage**: <500MB for 10K documents
- **Storage Size**: <50MB for 10K documents

Detailed benchmark results will be available after Level 1 completion.

## Documentation

- [Constitution](constitution.md) - Project principles and standards
- [Technical Specification](SPECIFY.md) - Detailed scope and requirements
- [Roadmap](roadmap.md) - Development plan and timeline
- [Architecture](docs/level1_architecture.md) - System design (coming soon)
- [API Reference](docs/api.md) - API documentation (coming soon)

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_minimal_db.py
```

### Code Quality

```bash
# Format code
black src/ tests/ examples/

# Lint code
pylint src/

# Type checking
mypy src/
```

### Running Benchmarks

```bash
# Run Level 1 benchmarks
python benchmarks/level1_bench.py

# Generate visualizations
python benchmarks/visualize_results.py
```

## Contributing

This is currently an experimental research project. Contributions, ideas, and feedback are welcome!

### Development Workflow
1. Read [constitution.md](constitution.md) for project principles
2. Check [roadmap.md](roadmap.md) for current priorities
3. Create feature branch from `main`
4. Follow code quality standards
5. Write tests for new functionality
6. Submit pull request with clear description

## Research Philosophy

This project follows a **research-first mindset**:
- Every claim must be validated with benchmarks
- Negative results are valuable and will be published
- We prioritize learning over premature optimization
- Honest comparison with existing solutions

If curve-based representations don't outperform vectors, we will document why and share our findings.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with:
- [sentence-transformers](https://www.sbert.net/) - Text embeddings
- [hilbertcurve](https://pypi.org/project/hilbertcurve/) - Space-filling curves
- [scipy](https://scipy.org/) - Scientific computing
- [scikit-fda](https://fda.readthedocs.io/) - Functional data analysis
- [LMDB](https://lmdb.readthedocs.io/) - Key-value storage

Inspired by research in:
- Space-filling curves for indexing
- Functional data analysis
- Curve similarity metrics
- Semantic search and information retrieval

## Contact

For questions, suggestions, or collaboration:
- GitHub Issues: [Create an issue](https://github.com/yourusername/CurvaDB/issues)
- Email: your.email@example.com

---

**Status**: Level 1 Development
**Version**: 0.1.0-dev
**Last Updated**: 2025-11-18
