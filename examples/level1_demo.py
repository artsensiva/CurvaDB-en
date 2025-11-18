"""Simple demo of CurvaDB Level 1.

This example shows how to:
1. Initialize a database
2. Add documents
3. Perform semantic search
4. Close the database
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from level1.minimal_db import MinimalCurveDB
import shutil


def main():
    print("=" * 60)
    print("CurvaDB Level 1 - Simple Demo")
    print("=" * 60)

    # Clean up any existing demo database
    db_path = "./demo_db"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    # 1. Initialize database
    print("\n1. Initializing database...")
    db = MinimalCurveDB(db_path=db_path, p=10, n=6)
    print(f"   Created: {db}")

    # 2. Add documents
    print("\n2. Adding documents...")
    doc_ids = [
        "tech1", "tech2", "tech3",
        "health1", "health2",
        "travel1", "travel2"
    ]

    texts = [
        "Artificial intelligence is transforming software development",
        "Machine learning models require large datasets to train effectively",
        "Cloud computing enables scalable infrastructure for modern applications",
        "Regular exercise and balanced nutrition are essential for good health",
        "Mental wellness practices like meditation can reduce stress",
        "Exploring new cultures broadens your perspective on life",
        "Sustainable tourism helps preserve natural environments"
    ]

    db.add_batch(doc_ids, texts)
    print(f"   Added {db.count()} documents")

    # 3. Perform searches
    print("\n3. Performing semantic searches...")

    queries = [
        "software engineering with AI",
        "staying healthy and fit",
        "vacation and discovering new places"
    ]

    for query in queries:
        print(f"\n   Query: \"{query}\"")
        results = db.search(query, k=3)

        if results:
            print("   Results:")
            for i, (doc_id, text, score) in enumerate(results, 1):
                print(f"      {i}. [{doc_id}] {text[:50]}... (score: {score:.3f})")
        else:
            print("   No results found.")

    # 4. Show database stats
    print("\n4. Database statistics:")
    stats = db.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # 5. Close database
    print("\n5. Closing database...")
    db.close()
    print("   ✓ Database closed")

    # Optional: Clean up demo database
    print("\n6. Cleaning up demo database...")
    shutil.rmtree(db_path)
    print("   ✓ Demo database removed")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
