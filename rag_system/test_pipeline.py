from pipeline import ask, all_chunks
print(f"Pipeline loaded: {len(all_chunks)} chunks")

# Test 1: answerable
r = ask("What is RAG?")
print("\nQ: What is RAG?")
print("Status:", r["status"])
print("Answer:", r["answer"][:300])

# Test 2: unanswerable
r2 = ask("Who won IPL 2025?")
print("\nQ: Who won IPL 2025?")
print("Status:", r2["status"])
print("Answer:", r2["answer"][:200])

# Test 3: another doc question
r3 = ask("What is BM25?")
print("\nQ: What is BM25?")
print("Status:", r3["status"])
print("Answer:", r3["answer"][:300])
print("\nCitations:", r3.get("citations", []))
