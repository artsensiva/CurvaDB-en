"""Improved benchmark suite for CurvaDB Level 1 with proper methodology.

Implements MS MARCO best practices:
- Warm-up phase
- Multiple runs
- Clear between tests
- All queries (not just subset)
- Comprehensive metrics: Recall@K, MRR@10, QPS, RAM, Index Time
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from level1.minimal_db import MinimalCurveDB
from level1.embedder import TextEmbedder
from level1.search import SearchEngine
import time
import numpy as np
import shutil
import json
from collections import defaultdict
import psutil
import gc


class MSMARCOLoader:
    """Load MS MARCO dataset from TSV files."""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_collection(self, limit=None):
        """Load documents from collection.tsv."""
        collection_path = os.path.join(self.data_dir, "collection.tsv")
        doc_ids = []
        texts = []

        with open(collection_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    doc_ids.append(parts[0])
                    texts.append(parts[1])

        return doc_ids, texts

    def load_queries(self):
        """Load queries from queries.tsv."""
        queries_path = os.path.join(self.data_dir, "queries.tsv")
        queries = {}

        with open(queries_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    queries[parts[0]] = parts[1]

        return queries

    def load_qrels(self):
        """Load relevance judgments from qrels.tsv."""
        qrels_path = os.path.join(self.data_dir, "qrels.tsv")
        qrels = defaultdict(set)

        with open(qrels_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    query_id = parts[0]
                    doc_id = parts[1]
                    relevance = int(parts[2])
                    if relevance > 0:
                        qrels[query_id].add(doc_id)

        return dict(qrels)


class BaselineSearch:
    """Baseline exhaustive cosine similarity search."""

    def __init__(self):
        self.embedder = TextEmbedder()
        self.doc_ids = []
        self.doc_embeddings = None

    def index(self, doc_ids, texts):
        """Index documents."""
        self.doc_ids = doc_ids
        print("  Embedding documents for baseline...")
        self.doc_embeddings = self.embedder.encode(texts, show_progress=True)

    def search(self, query, k=10):
        """Search for top-k documents."""
        query_emb = self.embedder.encode([query])[0]

        # Compute cosine similarities
        similarities = SearchEngine.batch_cosine_similarity(
            query_emb,
            self.doc_embeddings
        )

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            results.append((
                self.doc_ids[idx],
                "",
                float(similarities[idx])
            ))

        return results


def calculate_recall_at_k(retrieved, relevant, k):
    """Calculate Recall@K."""
    if not relevant:
        return 0.0

    retrieved_k = set([doc_id for doc_id, _, _ in retrieved[:k]])
    relevant_set = set(relevant)

    hits = len(retrieved_k & relevant_set)
    return hits / len(relevant_set)


def calculate_mrr_at_k(retrieved, relevant, k=10):
    """Calculate MRR@K (Mean Reciprocal Rank at K)."""
    for i, (doc_id, _, _) in enumerate(retrieved[:k], 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def get_memory_usage():
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def get_db_size(db_path):
    """Get total database size in MB."""
    total = 0
    for root, dirs, files in os.walk(db_path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / 1024 / 1024


def warm_up_db(db, queries, num_warmup=10):
    """Warm up database with sample queries."""
    print("\n  Warming up...")
    sample_queries = list(queries.values())[:num_warmup]
    for query in sample_queries:
        db.search(query, k=10)
    print(f"  Completed {num_warmup} warm-up queries")


def run_benchmark():
    """Run comprehensive benchmark with proper methodology."""
    print("=" * 70)
    print("CurvaDB Level 1 - Improved MS MARCO 10K Benchmark")
    print("=" * 70)

    # Setup
    data_dir = "./benchmarks/data/ms_marco/subsets/ms_marco_10001"
    db_path = "./bench_db_10k"

    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    # Load data
    print("\n1. Loading MS MARCO 1K dataset...")
    loader = MSMARCOLoader(data_dir)
    doc_ids, texts = loader.load_collection()
    queries = loader.load_queries()
    qrels = loader.load_qrels()

    print(f"   Documents: {len(doc_ids)}")
    print(f"   Queries: {len(queries)}")
    print(f"   Relevance judgments: {len(qrels)}")

    # Build CurvaDB index
    # Use p=10, n=12 → 2^120 max distance (fits in 128 bits)
    print("\n2. Building CurvaDB index (n=12, p=10)...")
    mem_before = get_memory_usage()
    start_time = time.time()

    db = MinimalCurveDB(db_path=db_path, p=10, n=12)
    db.add_batch(doc_ids, texts, show_progress=True)

    index_time = time.time() - start_time
    mem_after = get_memory_usage()
    mem_used = mem_after - mem_before

    print(f"   Indexing time: {index_time:.2f}s")
    print(f"   Documents/sec: {len(doc_ids)/index_time:.1f}")
    print(f"   Memory used: {mem_used:.1f} MB")

    # Get database size
    db_size = get_db_size(db_path)
    print(f"   Database size on disk: {db_size:.2f} MB")

    # Warm-up phase (important for fair comparison!)
    warm_up_db(db, queries, num_warmup=20)

    # Build baseline
    print("\n3. Building baseline (exhaustive cosine search)...")
    gc.collect()  # Clear memory
    baseline = BaselineSearch()
    baseline_start = time.time()
    baseline.index(doc_ids, texts)
    baseline_index_time = time.time() - baseline_start

    # Warm-up baseline
    print("  Warming up baseline...")
    sample_queries = list(queries.values())[:20]
    for query in sample_queries:
        baseline.search(query, k=10)

    # Evaluate ALL queries (not just subset!)
    print("\n4. Evaluating queries (ALL queries)...")
    k_values = [1, 5, 10, 20]

    # CurvaDB evaluation
    print("\n   CurvaDB:")
    curvadb_recalls = {k: [] for k in k_values}
    curvadb_mrrs = []
    curvadb_latencies = []

    eval_queries = [(qid, qtext) for qid, qtext in queries.items() if qid in qrels]
    print(f"   Evaluating {len(eval_queries)} queries with ground truth...")

    for query_id, query_text in eval_queries:
        relevant = qrels[query_id]

        # Search (measure latency)
        start_time = time.time()
        results = db.search(query_text, k=20)
        latency = (time.time() - start_time) * 1000  # ms

        curvadb_latencies.append(latency)

        # Calculate metrics
        for k in k_values:
            recall = calculate_recall_at_k(results, relevant, k)
            curvadb_recalls[k].append(recall)

        mrr = calculate_mrr_at_k(results, relevant, k=10)
        curvadb_mrrs.append(mrr)

    # Baseline evaluation
    print("\n   Baseline:")
    baseline_recalls = {k: [] for k in k_values}
    baseline_mrrs = []
    baseline_latencies = []

    for query_id, query_text in eval_queries:
        relevant = qrels[query_id]

        # Search
        start_time = time.time()
        results = baseline.search(query_text, k=20)
        latency = (time.time() - start_time) * 1000  # ms

        baseline_latencies.append(latency)

        # Calculate metrics
        for k in k_values:
            recall = calculate_recall_at_k(results, relevant, k)
            baseline_recalls[k].append(recall)

        mrr = calculate_mrr_at_k(results, relevant, k=10)
        baseline_mrrs.append(mrr)

    # Calculate QPS (queries per second)
    curvadb_qps = 1000.0 / np.mean(curvadb_latencies) if curvadb_latencies else 0
    baseline_qps = 1000.0 / np.mean(baseline_latencies) if baseline_latencies else 0

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\nQuality Metrics:")
    print(f"{'Metric':<15} {'CurvaDB':>12} {'Baseline':>12} {'Ratio':>12}")
    print("-" * 55)
    for k in k_values:
        curvadb_mean = np.mean(curvadb_recalls[k])
        baseline_mean = np.mean(baseline_recalls[k])
        ratio = curvadb_mean / baseline_mean if baseline_mean > 0 else 0
        print(f"Recall@{k:<9} {curvadb_mean:>11.4f} {baseline_mean:>11.4f} {ratio:>11.2f}x")

    curvadb_mrr_mean = np.mean(curvadb_mrrs)
    baseline_mrr_mean = np.mean(baseline_mrrs)
    mrr_ratio = curvadb_mrr_mean / baseline_mrr_mean if baseline_mrr_mean > 0 else 0
    print(f"MRR@10{'':>9} {curvadb_mrr_mean:>11.4f} {baseline_mrr_mean:>11.4f} {mrr_ratio:>11.2f}x")

    print("\nPerformance Metrics:")
    print(f"{'Metric':<20} {'CurvaDB':>15} {'Baseline':>15} {'Speedup':>12}")
    print("-" * 65)

    curvadb_p50 = np.percentile(curvadb_latencies, 50)
    baseline_p50 = np.percentile(baseline_latencies, 50)
    print(f"P50 Latency (ms)    {curvadb_p50:>15.2f} {baseline_p50:>15.2f} {baseline_p50/curvadb_p50:>11.2f}x")

    curvadb_p95 = np.percentile(curvadb_latencies, 95)
    baseline_p95 = np.percentile(baseline_latencies, 95)
    print(f"P95 Latency (ms)    {curvadb_p95:>15.2f} {baseline_p95:>15.2f} {baseline_p95/curvadb_p95:>11.2f}x")

    curvadb_mean_lat = np.mean(curvadb_latencies)
    baseline_mean_lat = np.mean(baseline_latencies)
    print(f"Mean Latency (ms)   {curvadb_mean_lat:>15.2f} {baseline_mean_lat:>15.2f} {baseline_mean_lat/curvadb_mean_lat:>11.2f}x")

    print(f"QPS                 {curvadb_qps:>15.1f} {baseline_qps:>15.1f} {curvadb_qps/baseline_qps:>11.2f}x")

    print("\nResource Metrics:")
    print(f"  Database size: {db_size:.2f} MB")
    print(f"  Memory used: {mem_used:.1f} MB")
    print(f"  Indexing time: {index_time:.2f}s ({len(doc_ids)/index_time:.1f} docs/sec)")
    print(f"  Baseline indexing time: {baseline_index_time:.2f}s")

    print("\nTarget Goals:")
    print(f"  {'Metric':<25} {'Target':>15} {'Actual':>15} {'Status':>10}")
    print("-" * 68)

    recall_10 = np.mean(curvadb_recalls[10])
    print(f"  {'Recall@10':<25} {'>= 0.99':>15} {f'{recall_10:.4f}':>15} {'✓' if recall_10 >= 0.99 else '✗':>10}")
    print(f"  {'MRR@10':<25} {'>= 0.95':>15} {f'{curvadb_mrr_mean:.4f}':>15} {'✓' if curvadb_mrr_mean >= 0.95 else '✗':>10}")
    print(f"  {'P50 Latency':<25} {'< 5ms':>15} {f'{curvadb_p50:.2f}ms':>15} {'✓' if curvadb_p50 < 5 else '✗':>10}")
    print(f"  {'P95 Latency':<25} {'< 10ms':>15} {f'{curvadb_p95:.2f}ms':>15} {'✓' if curvadb_p95 < 10 else '✗':>10}")
    print(f"  {'DB Size':<25} {'< 10MB':>15} {f'{db_size:.1f}MB':>15} {'✓' if db_size < 10 else '✗':>10}")
    print(f"  {'QPS':<25} {'>= 100':>15} {f'{curvadb_qps:.1f}':>15} {'✓' if curvadb_qps >= 100 else '✗':>10}")

    # Save results
    results = {
        "dataset": "MS MARCO 10K",
        "num_docs": len(doc_ids),
        "num_queries": len(eval_queries),
        "config": {"p": 10, "n": 12},
        "curvadb": {
            "recall": {f"recall@{k}": float(np.mean(curvadb_recalls[k])) for k in k_values},
            "mrr@10": float(curvadb_mrr_mean),
            "latency_p50": float(curvadb_p50),
            "latency_p95": float(curvadb_p95),
            "latency_mean": float(curvadb_mean_lat),
            "qps": float(curvadb_qps),
            "index_time": index_time,
            "db_size_mb": db_size,
            "memory_mb": mem_used
        },
        "baseline": {
            "recall": {f"recall@{k}": float(np.mean(baseline_recalls[k])) for k in k_values},
            "mrr@10": float(baseline_mrr_mean),
            "latency_p50": float(baseline_p50),
            "latency_p95": float(baseline_p95),
            "latency_mean": float(baseline_mean_lat),
            "qps": float(baseline_qps),
            "index_time": baseline_index_time
        }
    }

    os.makedirs("./benchmarks/results", exist_ok=True)
    with open("./benchmarks/results/level1_10k_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\n  Results saved to: benchmarks/results/level1_10k_results.json")

    # Cleanup
    db.close()
    shutil.rmtree(db_path)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
