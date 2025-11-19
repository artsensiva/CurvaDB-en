"""MVP Benchmark for Level 2 CurveDB.

Compares Level 2 (curve-based) approach with Level 1 (PCA + Hilbert).
Uses MS MARCO 1K dataset.
"""

import sys
import os
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from level2.curve_db import CurveDB


class MSMARCOLoader:
    """Load MS MARCO dataset from TSV files."""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_collection(self):
        """Load document collection."""
        collection_path = os.path.join(self.data_dir, "collection.tsv")

        corpus = {}
        with open(collection_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc_id, text = line.strip().split('\t')
                corpus[doc_id] = text

        return corpus

    def load_queries(self):
        """Load queries."""
        queries_path = os.path.join(self.data_dir, "queries.tsv")

        queries = {}
        with open(queries_path, 'r', encoding='utf-8') as f:
            for line in f:
                query_id, query_text = line.strip().split('\t')
                queries[query_id] = query_text

        return queries

    def load_qrels(self):
        """Load relevance judgments."""
        qrels_path = os.path.join(self.data_dir, "qrels.tsv")

        qrels = {}
        with open(qrels_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                query_id = parts[0]
                doc_id = parts[1]  # Fixed: doc_id is in column 1, not 2

                if query_id not in qrels:
                    qrels[query_id] = []
                qrels[query_id].append(doc_id)

        return qrels


def compute_recall_at_k(results, ground_truth, k=10):
    """Compute recall@k.

    Args:
        results: List of retrieved document indices
        ground_truth: List of relevant document indices
        k: Cutoff

    Returns:
        Recall@k score
    """
    results_at_k = set(results[:k])
    relevant = set(ground_truth)

    if len(relevant) == 0:
        return 0.0

    hits = len(results_at_k.intersection(relevant))
    recall = hits / len(relevant)

    return recall


def main():
    print("=" * 80)
    print("Level 2 MVP Benchmark - CurveDB")
    print("=" * 80)

    # Load data
    print("\n[1/5] Loading MS MARCO 1K dataset...")
    data_dir = "./benchmarks/data/ms_marco/subsets/ms_marco_1k"
    loader = MSMARCOLoader(data_dir)

    corpus = loader.load_collection()
    queries = loader.load_queries()
    qrels = loader.load_qrels()

    print(f"Loaded {len(corpus)} documents, {len(queries)} queries")

    # Build database
    print("\n[2/5] Building CurveDB index...")
    db_path = Path(__file__).parent.parent / 'data' / 'level2_mvp_db'
    db_path.mkdir(parents=True, exist_ok=True)

    # Create CurveDB with improved parameters
    db = CurveDB(
        db_path=str(db_path),
        embedding_model="all-MiniLM-L6-v2",
        n_fpca_components=20,  # Increased from 10 to capture more variance
        hilbert_order=8,
        curve_degree=3,
        distance_metric='dtw',  # DTW instead of Fréchet for better matching
        n_sample_points=50
    )

    # Check if already built
    if db.fitted and len(db.texts) == len(corpus):
        print("Database already built and fitted!")
    else:
        # Add documents
        start_time = time.time()

        # Get document texts
        doc_ids = sorted(corpus.keys())
        doc_texts = [corpus[doc_id] for doc_id in doc_ids]

        db.add_documents(doc_texts, show_progress=True)

        # Fit FPCA
        db.fit()

        # Save
        db.save()

        build_time = time.time() - start_time
        print(f"Index built in {build_time:.2f} seconds")

    print(f"\nDatabase stats: {db.stats()}")

    # Run queries
    print("\n[3/5] Running queries...")

    all_recalls = []
    query_times = []

    # Map doc_id to index
    doc_ids = sorted(corpus.keys())
    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

    for query_id, query_text in queries.items():
        # Get ground truth
        if query_id not in qrels:
            continue

        relevant_doc_ids = qrels[query_id]

        # Run search with increased rerank factor for better candidate pool
        start_time = time.time()
        results = db.search(query_text, k=10, rerank_factor=20)
        query_time = time.time() - start_time

        # Get result indices
        result_texts = [text for text, _ in results]
        result_indices = []
        for text in result_texts:
            try:
                idx = doc_texts.index(text)
                result_indices.append(idx)
            except ValueError:
                pass

        # Compute recall
        ground_truth_indices = [
            doc_id_to_idx[doc_id]
            for doc_id in relevant_doc_ids
            if doc_id in doc_id_to_idx
        ]

        recall = compute_recall_at_k(result_indices, ground_truth_indices, k=10)
        all_recalls.append(recall)
        query_times.append(query_time)

    # Compute metrics
    print("\n[4/5] Computing metrics...")

    mean_recall = np.mean(all_recalls)
    median_recall = np.median(all_recalls)
    std_recall = np.std(all_recalls)

    mean_query_time = np.mean(query_times)
    median_query_time = np.median(query_times)

    # Results
    print("\n[5/5] Results:")
    print("=" * 80)
    print(f"Recall@10:")
    print(f"  Mean:   {mean_recall:.4f} ({mean_recall*100:.2f}%)")
    print(f"  Median: {median_recall:.4f} ({median_recall*100:.2f}%)")
    print(f"  Std:    {std_recall:.4f}")
    print(f"\nQuery time:")
    print(f"  Mean:   {mean_query_time*1000:.2f}ms")
    print(f"  Median: {median_query_time*1000:.2f}ms")

    # Compare with Level 1
    print("\n" + "=" * 80)
    print("Comparison with Level 1:")
    print("=" * 80)
    print("Level 1 (PCA + Hilbert):")
    print("  Recall@10: ~0.08 (8%)")
    print(f"\nLevel 2 (Curves + FPCA):")
    print(f"  Recall@10: {mean_recall:.4f} ({mean_recall*100:.2f}%)")

    improvement = (mean_recall - 0.08) / 0.08 * 100 if mean_recall > 0.08 else 0
    if mean_recall > 0.08:
        print(f"\n✓ Improvement: +{improvement:.1f}%")
    else:
        print(f"\n✗ Degradation: {improvement:.1f}%")

    # Save results
    results_file = Path(__file__).parent.parent / 'data' / 'level2_mvp_results.json'
    results_data = {
        'recall_at_10': {
            'mean': float(mean_recall),
            'median': float(median_recall),
            'std': float(std_recall),
            'all_scores': [float(r) for r in all_recalls]
        },
        'query_time': {
            'mean': float(mean_query_time),
            'median': float(median_query_time)
        },
        'config': {
            'n_fpca_components': db.n_fpca_components,
            'hilbert_order': db.hilbert_order,
            'distance_metric': db.distance_metric,
            'curve_degree': db.curve_degree
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT:")
    print("=" * 80)

    if mean_recall >= 0.75:
        print("✓ SUCCESS: Level 2 MVP achieves target Recall@10 ≥ 75%")
        print("  → Proceed with full Level 2 implementation")
    elif mean_recall > 0.08:
        print("⚠ PARTIAL SUCCESS: Level 2 improves over Level 1")
        print(f"  → Recall@10: {mean_recall*100:.2f}% (target: 75%)")
        print("  → Consider tuning parameters or alternative approaches")
    else:
        print("✗ FAILURE: Level 2 does not improve over Level 1")
        print("  → Need to reconsider approach or investigate issues")


if __name__ == '__main__':
    main()
