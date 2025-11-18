"""Quick end-to-end test of MinimalCurveDB."""

import sys
sys.path.insert(0, 'src')

from level1.minimal_db import MinimalCurveDB
import shutil
import os

# Clean up any existing test database
test_db_path = "./test_db_e2e"
if os.path.exists(test_db_path):
    shutil.rmtree(test_db_path)

print("=" * 60)
print("CurvaDB Level 1 - End-to-End Test")
print("=" * 60)

# Initialize database
print("\n1. Initializing database...")
db = MinimalCurveDB(db_path=test_db_path, p=12, n=6)
print(f"   ✓ {db}")

# Add documents
print("\n2. Adding documents...")
doc_ids = [
    "ml1", "ml2", "ml3",
    "cooking1", "cooking2",
    "sports1", "sports2",
    "science1", "science2", "science3"
]

texts = [
    "Machine learning is a subset of artificial intelligence",
    "Deep neural networks can learn complex patterns from data",
    "Natural language processing enables computers to understand text",
    "Pasta carbonara is a delicious Italian dish",
    "The secret to perfect risotto is constant stirring",
    "Football is the most popular sport in the world",
    "Basketball requires teamwork and individual skill",
    "Quantum physics explores the behavior of subatomic particles",
    "Einstein's theory of relativity changed our understanding of space and time",
    "The Large Hadron Collider is the world's most powerful particle accelerator"
]

db.add_batch(doc_ids, texts, show_progress=True)
print(f"   ✓ Added {db.count()} documents")

# Test searches
print("\n3. Testing searches...")

queries = [
    ("AI and neural networks", "ml"),
    ("cooking recipes", "cooking"),
    ("team sports", "sports"),
    ("physics theories", "science")
]

for query, expected_category in queries:
    print(f"\n   Query: '{query}'")
    print(f"   Expected category: {expected_category}")

    results = db.search(query, k=3)

    print(f"   Results:")
    for i, (doc_id, text, score) in enumerate(results, 1):
        category = doc_id.rstrip('0123456789')
        marker = "✓" if category == expected_category else "✗"
        print(f"      {i}. [{marker}] {doc_id}: {text[:50]}... (score: {score:.3f})")

# Stats
print("\n4. Database statistics:")
stats = db.get_stats()
for key, value in stats.items():
    print(f"   {key}: {value}")

# Cleanup
print("\n5. Cleaning up...")
db.close()
shutil.rmtree(test_db_path)
print("   ✓ Database closed and cleaned up")

print("\n" + "=" * 60)
print("END-TO-END TEST COMPLETE!")
print("=" * 60)
