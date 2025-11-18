"""Benchmark suite for CurvaDB Level 1 using MS MARCO 1K dataset."""

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
                "",  # No text needed for evaluation
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


def calculate_mrr(retrieved, relevant):
    """Calculate Mean Reciprocal Rank."""
    for i, (doc_id, _, _) in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def run_benchmark():
    """Run comprehensive benchmark."""
    print("=" * 70)
    print("CurvaDB Level 1 - MS MARCO 1K Benchmark")
    print("=" * 70)

    # Setup
    data_dir = "./benchmarks/data/ms_marco/subsets/ms_marco_1k"
    db_path = "./bench_db"

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
    print("\n2. Building CurvaDB index...")
    start_time = time.time()

    db = MinimalCurveDB(db_path=db_path, p=10, n=8)
    db.add_batch(doc_ids, texts, show_progress=True)

    index_time = time.time() - start_time
    print(f"   Indexing time: {index_time:.2f}s")
    print(f"   Documents/sec: {len(doc_ids)/index_time:.1f}")

    # Get database size
    db_size = sum(
        os.path.getsize(os.path.join(db_path, f))
        for f in os.listdir(db_path)
        if os.path.isfile(os.path.join(db_path, f))
    )
    print(f"   Database size: {db_size / 1024 / 1024:.2f} MB")

    # Build baseline
    print("\n3. Building baseline (exhaustive cosine search)...")
    baseline = BaselineSearch()
    baseline.index(doc_ids, texts)

    # Evaluate queries
    print("\n4. Evaluating queries...")
    k_values = [1, 5, 10, 20]

    # CurvaDB evaluation
    print("\n   CurvaDB:")
    curvadb_recalls = {k: [] for k in k_values}
    curvadb_mrrs = []
    curvadb_latencies = []

    for query_id, query_text in list(queries.items())[:20]:  # First 20 queries
        if query_id not in qrels:
            continue

        relevant = qrels[query_id]

        # Search
        start_time = time.time()
        results = db.search(query_text, k=20)
        latency = (time.time() - start_time) * 1000  # ms

        curvadb_latencies.append(latency)

        # Calculate metrics
        for k in k_values:
            recall = calculate_recall_at_k(results, relevant, k)
            curvadb_recalls[k].append(recall)

        mrr = calculate_mrr(results, relevant)
        curvadb_mrrs.append(mrr)

    # Baseline evaluation
    print("\n   Baseline:")
    baseline_recalls = {k: [] for k in k_values}
    baseline_mrrs = []
    baseline_latencies = []

    for query_id, query_text in list(queries.items())[:20]:
        if query_id not in qrels:
            continue

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

        mrr = calculate_mrr(results, relevant)
        baseline_mrrs.append(mrr)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\nQuality Metrics (Recall@K):")
    print(f"{'Metric':<15} {'CurvaDB':>12} {'Baseline':>12} {'Ratio':>12}")
    print("-" * 55)
    for k in k_values:
        curvadb_mean = np.mean(curvadb_recalls[k])
        baseline_mean = np.mean(baseline_recalls[k])
        ratio = curvadb_mean / baseline_mean if baseline_mean > 0 else 0
        print(f"Recall@{k:<9} {curvadb_mean:>11.3f} {baseline_mean:>11.3f} {ratio:>11.2f}x")

    curvadb_mrr_mean = np.mean(curvadb_mrrs)
    baseline_mrr_mean = np.mean(baseline_mrrs)
    mrr_ratio = curvadb_mrr_mean / baseline_mrr_mean if baseline_mrr_mean > 0 else 0
    print(f"MRR{'':>12} {curvadb_mrr_mean:>11.3f} {baseline_mrr_mean:>11.3f} {mrr_ratio:>11.2f}x")

    print("\nPerformance Metrics (Latency in ms):")
    print(f"{'Metric':<15} {'CurvaDB':>12} {'Baseline':>12} {'Speedup':>12}")
    print("-" * 55)

    curvadb_p50 = np.percentile(curvadb_latencies, 50)
    baseline_p50 = np.percentile(baseline_latencies, 50)
    print(f"P50 Latency    {curvadb_p50:>11.2f} {baseline_p50:>11.2f} {baseline_p50/curvadb_p50:>11.2f}x")

    curvadb_p95 = np.percentile(curvadb_latencies, 95)
    baseline_p95 = np.percentile(baseline_latencies, 95)
    print(f"P95 Latency    {curvadb_p95:>11.2f} {baseline_p95:>11.2f} {baseline_p95/curvadb_p95:>11.2f}x")

    curvadb_mean_lat = np.mean(curvadb_latencies)
    baseline_mean_lat = np.mean(baseline_latencies)
    print(f"Mean Latency   {curvadb_mean_lat:>11.2f} {baseline_mean_lat:>11.2f} {baseline_mean_lat/curvadb_mean_lat:>11.2f}x")

    print("\nResource Metrics:")
    print(f"  Database size: {db_size / 1024 / 1024:.2f} MB")
    print(f"  Indexing speed: {len(doc_ids)/index_time:.1f} docs/sec")

    print("\nSuccess Criteria (from SPECIFY.md):")
    print(f"  {'Metric':<25} {'Target':>12} {'Actual':>12} {'Status':>10}")
    print("-" * 62)
    print(f"  {'Recall@10':<25} {'>= 0.70':>12} {curvadb_mean:>12.3f} {'✓' if curvadb_mean >= 0.70 else '✗':>10}")
    print(f"  {'Search Latency (P95)':<25} {'< 100ms':>12} {f'{curvadb_p95:.1f}ms':>12} {'✓' if curvadb_p95 < 100 else '✗':>10}")
    print(f"  {'DB Size':<25} {'< 50MB':>12} {f'{db_size/1024/1024:.1f}MB':>12} {'✓' if db_size < 50*1024*1024 else '✗':>10}")

    # Save results
    results = {
        "dataset": "MS MARCO 1K",
        "num_docs": len(doc_ids),
        "num_queries": len(queries),
        "curvadb": {
            "recall": {f"recall@{k}": np.mean(curvadb_recalls[k]) for k in k_values},
            "mrr": curvadb_mrr_mean,
            "latency_p50": curvadb_p50,
            "latency_p95": curvadb_p95,
            "latency_mean": curvadb_mean_lat,
            "index_time": index_time,
            "db_size_mb": db_size / 1024 / 1024
        },
        "baseline": {
            "recall": {f"recall@{k}": np.mean(baseline_recalls[k]) for k in k_values},
            "mrr": baseline_mrr_mean,
            "latency_p50": baseline_p50,
            "latency_p95": baseline_p95,
            "latency_mean": baseline_mean_lat
        }
    }

    os.makedirs("./benchmarks/results", exist_ok=True)
    with open("./benchmarks/results/level1_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\n  Results saved to: benchmarks/results/level1_results.json")

    # Cleanup
    db.close()
    shutil.rmtree(db_path)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
