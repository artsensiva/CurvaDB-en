"""Debug test to understand search issue."""

import sys
sys.path.insert(0, 'src')

from level1.minimal_db import MinimalCurveDB
import shutil
import os

# Clean up
test_db_path = "./test_db_debug"
if os.path.exists(test_db_path):
    shutil.rmtree(test_db_path)

# Initialize
db = MinimalCurveDB(db_path=test_db_path, p=10, n=4)  # Smaller for debugging

# Add a few documents (need at least n documents for PCA)
docs = [
    "Machine learning is great",
    "Deep learning is powerful",
    "Cooking pasta is fun",
    "Physics is interesting",
    "Sports are exciting"
]
doc_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]

print("Adding documents...")
db.add_batch(doc_ids, docs)
print(f"Total docs: {db.count()}")

# Check what's in storage directly
print("\nChecking storage contents:")
with db.storage.env.begin() as txn:
    cursor = txn.cursor()
    count = 0
    for key_bytes, value_bytes in cursor:
        key = int.from_bytes(key_bytes, byteorder='big', signed=False)
        print(f"  Key: {key}")
        count += 1
    print(f"  Total entries in LMDB: {count}")

# Try a search
print("\nSearching for 'artificial intelligence'...")
query = "artificial intelligence"

# Get query embedding and Hilbert distance
query_embedding = db.embedder.encode([query])[0]
query_reduced = db.reducer.transform(query_embedding.reshape(1, -1))[0]
query_distance = db.hilbert.points_to_distances([query_reduced.tolist()])[0]

print(f"  Query Hilbert distance: {query_distance}")
print(f"  Max possible distance: {db.hilbert.get_max_distance()}")

# Try search with FULL range
start_dist = 0
end_dist = db.hilbert.get_max_distance()
print(f"  Search range: {start_dist} to {end_dist}")

# Direct range query across ALL documents
candidates = db.storage.range_query(start_dist, end_dist, limit=10)
print(f"  Candidates found in FULL range: {len(candidates)}")

if candidates:
    for hilbert_dist, doc_data in candidates[:3]:
        print(f"    Hilbert: {hilbert_dist}, Doc: {doc_data['doc_id']}")

# Try actual search with auto radius
results = db.search(query, k=3)
print(f"\nSearch results (with default auto radius): {len(results)}")
for doc_id, text, score in results:
    print(f"  {doc_id}: {text} (score: {score:.3f})")

# Try with explicit large radius
results2 = db.search(query, k=3, radius=db.hilbert.get_max_distance() // 10)
print(f"\nSearch results (with radius=max/10): {len(results2)}")
for doc_id, text, score in results:
    print(f"  {doc_id}: {text} (score: {score:.3f})")

# Cleanup
db.close()
shutil.rmtree(test_db_path)
