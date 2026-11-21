"""
medical_pipeline.py — Hybrid Medical Coding RAG Engine (FY 2027)
Features:
- Instant SQLite Exact Code Lookup (ICD-10-CM + HCPCS)
- Billability Verification (Billable vs Category Header)
- Conversion Table Mapping (Previous vs 2027 New Code)
- SQLite FTS5 Full-Text Search (BM25 ranking for Clinical Descriptions)
- Strict Anti-Hallucination Guardrail (says "I don't know" when not in CMS data)
"""

import os, sqlite3, re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "medical_codes.db")

ABSTAIN_MSG = "I don't know. No matching FY 2027 ICD-10-CM diagnosis or HCPCS Level II code found in official records."

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Clean code input helper: E11.9 -> ("E11.9", "E119")
def normalize_code_input(text: str):
    cleaned = text.strip().upper()
    # remove punctuation
    raw = cleaned.replace(".", "").replace("-", "").replace(" ", "")
    # formatted version
    formatted = raw[:3] + "." + raw[3:] if len(raw) > 3 else raw
    return formatted, raw

# ─── 1. Exact Code Lookup ────────────────────────────────────────────────────
def lookup_exact_code(query: str):
    q = query.strip().upper()
    formatted, raw = normalize_code_input(q)

    conn = get_db()
    cur = conn.cursor()

    # Check ICD-10
    cur.execute("""
    SELECT code, is_billable, short_desc, long_desc, fiscal_year
    FROM icd10
    WHERE code = ? OR code_raw = ?
    LIMIT 1
    """, (formatted, raw))
    icd_row = cur.fetchone()

    if icd_row:
        # Check conversion table
        cur.execute("""
        SELECT previous_code, effective_year
        FROM conversions
        WHERE current_code = ?
        """, (icd_row["code"],))
        conv_rows = cur.fetchall()

        conn.close()

        billable_status = "✅ Valid Billable Code" if icd_row["is_billable"] == 1 else "⚠️ Category Header / Non-Billable (Requires higher specificity subcode)"

        conv_info = ""
        if conv_rows:
            prev_codes = ", ".join(f"{r['previous_code']} (eff. {r['effective_year']})" for r in conv_rows)
            conv_info = f"\n🔄 **FY 2027 Conversion**: Updated from previous code {prev_codes}"

        answer = (
            f"### ICD-10-CM Code: `{icd_row['code']}`\n\n"
            f"**Description:** {icd_row['long_desc']}\n\n"
            f"**Status:** {billable_status}\n"
            f"**Short Label:** {icd_row['short_desc']}\n"
            f"**Fiscal Year:** FY {icd_row['fiscal_year']} Official CMS Tabular List"
            f"{conv_info}"
        )

        return {
            "status": "answered",
            "match_type": "exact_code",
            "code_type": "ICD-10-CM",
            "code": icd_row["code"],
            "billable": bool(icd_row["is_billable"]),
            "answer": answer,
            "citations": [f"CMS FY 2027 Tabular Order: {icd_row['code']}"]
        }

    # Check HCPCS
    cur.execute("""
    SELECT code, description, fiscal_year
    FROM hcpcs
    WHERE code = ?
    LIMIT 1
    """, (raw,))
    hcpcs_row = cur.fetchone()
    conn.close()

    if hcpcs_row:
        answer = (
            f"### HCPCS Level II Code: `{hcpcs_row['code']}`\n\n"
            f"**Description:** {hcpcs_row['description']}\n\n"
            f"**Category:** Medical Supplies, Equipment & Services\n"
            f"**Fiscal Year:** January 2027 Official CMS Alpha-Numeric File"
        )
        return {
            "status": "answered",
            "match_type": "exact_code",
            "code_type": "HCPCS Level II",
            "code": hcpcs_row["code"],
            "billable": True,
            "answer": answer,
            "citations": [f"CMS January 2027 HCPCS: {hcpcs_row['code']}"]
        }

    return None

# ─── 2. Clinical Description Search (FTS5 BM25) ──────────────────────────────
def search_clinical_description(query: str, limit: int = 5):
    # Clean tokens for FTS5 syntax
    tokens = re.findall(r"\w+", query)
    # filter stop words
    stop = {"patient", "has", "with", "without", "diagnosed", "due", "to", "and", "or", "the", "a", "an", "is", "of", "for", "in"}
    cleaned_tokens = [t for t in tokens if t.lower() not in stop and len(t) > 1]
    if not cleaned_tokens:
        cleaned_tokens = tokens

    if not cleaned_tokens:
        return []

    # Build FTS5 match query
    match_query = " ".join(f'"{t}"' for t in cleaned_tokens[:6])

    conn = get_db()
    cur = conn.cursor()

    results = []
    try:
        # Search ICD-10
        cur.execute("""
        SELECT i.code, i.is_billable, i.short_desc, i.long_desc, i.fiscal_year, rank
        FROM icd10_fts f
        JOIN icd10 i ON i.rowid = f.rowid
        WHERE icd10_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """, (match_query, limit))

        for row in cur.fetchall():
            results.append({
                "code": row["code"],
                "code_type": "ICD-10-CM",
                "is_billable": row["is_billable"],
                "short_desc": row["short_desc"],
                "long_desc": row["long_desc"],
                "rank": row["rank"]
            })
    except Exception:
        pass

    # If no or few ICD matches, also try HCPCS
    if len(results) < limit:
        try:
            cur.execute("""
            SELECT h.code, h.description, rank
            FROM hcpcs_fts f
            JOIN hcpcs h ON h.rowid = f.rowid
            WHERE hcpcs_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """, (match_query, limit - len(results)))

            for row in cur.fetchall():
                results.append({
                    "code": row["code"],
                    "code_type": "HCPCS Level II",
                    "is_billable": 1,
                    "short_desc": row["description"],
                    "long_desc": row["description"],
                    "rank": row["rank"]
                })
        except Exception:
            pass

    conn.close()
    return results

# ─── 3. Main Medical Ask Pipeline ─────────────────────────────────────────────
def ask_medical(question: str) -> dict:
    q = question.strip()
    if not q:
        return {
            "status": "insufficient_context",
            "answer": "Please enter an ICD-10 code, HCPCS code, or a clinical disease description.",
            "citations": [],
            "results": []
        }

    # Step 1: Detect if user typed a specific code (e.g. "E11.9", "E119", "A00.0", "J0129", "What is code E11.9")
    code_match = re.search(r"\b([A-TV-Z][0-9]{2}(\.[0-9A-Z]{1,4})?|[A-V][0-9]{4})\b", q.upper())
    if code_match:
        found_code = code_match.group(1)
        res = lookup_exact_code(found_code)
        if res:
            return res

    # If the whole query is short and alphanumeric, try direct code lookup anyway
    if len(q) <= 10 and re.match(r"^[A-Za-z0-9\.\-\s]+$", q):
        res = lookup_exact_code(q)
        if res:
            return res

    # Step 2: Clinical Description Search
    matches = search_clinical_description(q, limit=5)
    if not matches:
        return {
            "status": "insufficient_context",
            "answer": ABSTAIN_MSG,
            "citations": [],
            "results": []
        }

    # Format the top matches
    formatted_items = []
    citations = []
    for idx, m in enumerate(matches, 1):
        billable_badge = "🟢 Valid Billable" if m["is_billable"] == 1 else "🟡 Category Header (Needs subcode)"
        formatted_items.append(
            f"**{idx}. [{m['code_type']}] `{m['code']}`** — {billable_badge}\n"
            f"   {m['long_desc']}"
        )
        citations.append(f"{m['code_type']}: {m['code']}")

    answer = (
        f"Found **{len(matches)} matching FY 2027 code(s)** for your query:\n\n" +
        "\n\n".join(formatted_items) +
        "\n\n*Source: Official CMS FY 2027 ICD-10-CM Tabular Order & HCPCS Files.*"
    )

    return {
        "status": "answered",
        "match_type": "clinical_search",
        "answer": answer,
        "citations": citations,
        "results": matches
    }
