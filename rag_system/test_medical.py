import json
from medical_pipeline import ask_medical

test_queries = [
    # 1. Exact ICD-10 Code with dot
    "E11.9",
    # 2. Exact ICD-10 Code without dot
    "A000",
    # 3. Clinical Description
    "Type 2 diabetes mellitus without complications",
    # 4. HCPCS Level II code
    "A1001",
    # 5. Conversion Table lookup (new code)
    "A04.71",
    # 6. Unanswerable query (Strict Anti-Hallucination Guardrail)
    "Who won the IPL 2025 cricket trophy?",
    # 7. Unanswerable non-existent code
    "Z999.99",
]

print("="*65)
print("  🧪 MEDICAL PIPELINE VERIFICATION SUITE")
print("="*65)

for i, q in enumerate(test_queries, 1):
    res = ask_medical(q)
    print(f"\n[{i}] Query: {q}")
    print(f"    Status: {res['status']}")
    print(f"    Match Type: {res.get('match_type', 'none')}")
    print(f"    Answer preview:\n{res['answer'][:200]}...")
    if res.get('citations'):
        print(f"    Citations: {res['citations']}")
