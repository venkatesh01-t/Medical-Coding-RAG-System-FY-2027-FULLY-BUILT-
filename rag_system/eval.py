"""
eval.py — Test your RAG with a golden question set
Run: python eval.py
"""

from pipeline import ask
import json, time

# ─── Golden Test Set ──────────────────────────────────────────────────────────
# Format: question, expected (or "NOT_IN_DOCS" for unanswerable), type
GOLDEN = [
    # Simple factual (should answer)
    {"q": "What is RAG?",
     "expected": "Retrieval-Augmented Generation",
     "type": "factual"},

    {"q": "What is the purpose of chunking in a RAG system?",
     "expected": "split documents",
     "type": "factual"},

    {"q": "What is BM25?",
     "expected": "keyword",
     "type": "factual"},

    {"q": "What is a vector store?",
     "expected": "store",
     "type": "factual"},

    {"q": "What does RRF stand for?",
     "expected": "Reciprocal Rank Fusion",
     "type": "exact"},

    {"q": "What is the recommended chunk size range?",
     "expected": "300",
     "type": "factual"},

    {"q": "What is the ABSTAIN response for?",
     "expected": "insufficient",
     "type": "factual"},

    {"q": "What embedding model is used as a baseline in the skills?",
     "expected": "bge-base",
     "type": "exact"},

    {"q": "What is faithfulness in RAG evaluation?",
     "expected": "supported",
     "type": "factual"},

    {"q": "What tool is recommended for parsing PDFs?",
     "expected": "Docling",
     "type": "exact"},

    # Unanswerable (should say "I don't know")
    {"q": "What is the population of Chennai in 2026?",
     "expected": "NOT_IN_DOCS",
     "type": "unanswerable"},

    {"q": "Who won the IPL 2025?",
     "expected": "NOT_IN_DOCS",
     "type": "unanswerable"},

    {"q": "What is the price of a MacBook Pro in India?",
     "expected": "NOT_IN_DOCS",
     "type": "unanswerable"},

    {"q": "How do I make biryani?",
     "expected": "NOT_IN_DOCS",
     "type": "unanswerable"},

    {"q": "What is the best movie of 2024?",
     "expected": "NOT_IN_DOCS",
     "type": "unanswerable"},
]

# ─── Run Evaluation ───────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  🧪 RAG Evaluation — Golden Test Set")
print("="*65)

results = []
correct = 0
abstain_correct = 0
abstain_wrong   = 0
answered_correct = 0
answered_wrong   = 0

for i, test in enumerate(GOLDEN):
    print(f"\n[{i+1}/{len(GOLDEN)}] {test['type'].upper()}: {test['q'][:60]}...")
    start = time.time()
    out   = ask(test["q"])
    elapsed = time.time() - start

    answered = out["status"] == "answered"
    unanswerable = test["expected"] == "NOT_IN_DOCS"

    if unanswerable:
        # Should say "I don't know"
        ok = not answered
        if ok:
            abstain_correct += 1
            correct += 1
            verdict = "✅ Correctly said I don't know"
        else:
            abstain_wrong += 1
            verdict = "❌ Hallucinated — should have said I don't know!"
    else:
        # Should answer and expected keyword should appear
        keyword = test["expected"].lower()
        ok = answered and keyword in out["answer"].lower()
        if ok:
            answered_correct += 1
            correct += 1
            verdict = "✅ Correct answer"
        elif not answered:
            answered_wrong += 1
            verdict = "⚠️  Refused to answer (false abstain)"
        else:
            answered_wrong += 1
            verdict = f"❌ Wrong answer (expected '{test['expected']}')"

    print(f"  {verdict}")
    print(f"  Answer: {out['answer'][:120]}...")
    print(f"  Time: {elapsed:.1f}s")

    results.append({
        "question": test["q"],
        "type"    : test["type"],
        "expected": test["expected"],
        "got"     : out["answer"][:200],
        "status"  : out["status"],
        "correct" : ok,
        "time_s"  : round(elapsed, 2),
    })

# ─── Summary ─────────────────────────────────────────────────────────────────
total      = len(GOLDEN)
answerable = sum(1 for t in GOLDEN if t["expected"] != "NOT_IN_DOCS")
unanswerable_total = total - answerable

print("\n" + "="*65)
print("  📊 RESULTS SUMMARY")
print("="*65)
print(f"  Total questions   : {total}")
print(f"  ✅ Correct overall : {correct}/{total} = {correct/total*100:.0f}%")
print(f"")
print(f"  Answerable questions ({answerable}):")
print(f"    ✅ Answered correctly : {answered_correct}")
print(f"    ❌ Wrong / false abstain: {answered_wrong}")
print(f"")
print(f"  Unanswerable questions ({unanswerable_total}):")
print(f"    ✅ Correctly said I don't know: {abstain_correct}")
print(f"    ❌ Hallucinated (bad!): {abstain_wrong}")
print("="*65)

if abstain_wrong == 0:
    print("  🎉 GREAT: System never hallucinated on unanswerable questions!")
else:
    print(f"  ⚠️  ISSUE: Hallucinated {abstain_wrong} times. Lower MIN_SCORE in pipeline.py")

# Save results
with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to eval_results.json")
