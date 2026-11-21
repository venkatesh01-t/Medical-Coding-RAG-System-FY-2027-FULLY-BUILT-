"""
ingest_medical.py — Official FY 2027 Medical Coding Ingestion Pipeline
Parses:
1. 2027-code-descriptions-in-tabular-order.zip (98,400+ ICD-10-CM codes + headers)
2. hcpc2027_jan_anweb_0.zip (17,200+ HCPCS Level II codes)
3. icd-10-cm-conversion-table-fy2027-october-1-2026-final.zip (6,000+ code mappings)

Stores in:
- SQLite Database: medical_codes.db (with FTS5 full-text BM25 search)
"""

import os, zipfile, sqlite3, csv, io, re, time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOC_DIR = os.path.join(BASE_DIR, "doc")
DB_PATH = os.path.join(BASE_DIR, "medical_codes.db")

print("="*65)
print("  🏥 Medical Coding Ingestion Pipeline (FY 2027)")
print("="*65)

# Connect / Create SQLite database
if os.path.exists(DB_PATH):
    try:
        os.remove(DB_PATH)
        print("🗑️  Cleaned existing medical_codes.db")
    except Exception:
        pass

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Create Tables
cur.executescript("""
CREATE TABLE icd10 (
    code TEXT PRIMARY KEY,
    code_raw TEXT,
    is_billable INTEGER,
    short_desc TEXT,
    long_desc TEXT,
    fiscal_year INTEGER DEFAULT 2027
);

CREATE TABLE hcpcs (
    code TEXT PRIMARY KEY,
    description TEXT,
    fiscal_year INTEGER DEFAULT 2027
);

CREATE TABLE conversions (
    current_code TEXT,
    effective_year TEXT,
    previous_code TEXT,
    PRIMARY KEY (current_code, previous_code)
);

-- Fast lookup indexes
CREATE INDEX idx_icd10_raw ON icd10(code_raw);
CREATE INDEX idx_icd10_billable ON icd10(is_billable);
CREATE INDEX idx_conv_prev ON conversions(previous_code);

-- FTS5 Full-Text Search Virtual Tables (BM25 ranking built-in)
CREATE VIRTUAL TABLE icd10_fts USING fts5(
    code,
    long_desc,
    content='icd10',
    content_rowid='rowid'
);

CREATE VIRTUAL TABLE hcpcs_fts USING fts5(
    code,
    description,
    content='hcpcs',
    content_rowid='rowid'
);
""")
conn.commit()

# Helper: format raw code (e.g. A000 -> A00.0, E119 -> E11.9)
def format_icd_code(raw: str) -> str:
    raw = raw.strip()
    if len(raw) > 3:
        return raw[:3] + "." + raw[3:]
    return raw

# ─── 1. Ingest ICD-10-CM Tabular Order ──────────────────────────────────────
icd_zip = os.path.join(DOC_DIR, "2027-code-descriptions-in-tabular-order.zip")
print(f"\n📦 [1/3] Parsing ICD-10-CM Codes from {os.path.basename(icd_zip)}...")
t0 = time.time()

icd_rows = []
if os.path.exists(icd_zip):
    with zipfile.ZipFile(icd_zip, "r") as z:
        with z.open("Code Descriptions/icd10cm_order_2027.txt") as f:
            for line in f:
                decoded = line.decode("latin1", errors="ignore")
                if len(decoded) < 78:
                    continue
                code_raw = decoded[6:14].strip()
                if not code_raw:
                    continue
                billable_flag = decoded[14:16].strip()
                is_billable = 1 if billable_flag == "1" else 0
                short_desc = decoded[16:77].strip()
                long_desc = decoded[77:].strip()
                code_formatted = format_icd_code(code_raw)

                icd_rows.append((code_formatted, code_raw, is_billable, short_desc, long_desc, 2027))

    cur.executemany("""
    INSERT OR REPLACE INTO icd10 (code, code_raw, is_billable, short_desc, long_desc, fiscal_year)
    VALUES (?, ?, ?, ?, ?, ?)
    """, icd_rows)

    # Populate FTS5
    cur.execute("""
    INSERT INTO icd10_fts(rowid, code, long_desc)
    SELECT rowid, code, long_desc FROM icd10
    """)
    conn.commit()
    print(f"  ✅ Indexed {len(icd_rows):,} ICD-10-CM codes & headers in {time.time()-t0:.2f}s")
else:
    print(f"  ❌ File not found: {icd_zip}")

# ─── 2. Ingest HCPCS Level II ───────────────────────────────────────────────
hcpcs_zip = os.path.join(DOC_DIR, "hcpc2027_jan_anweb_0.zip")
print(f"\n📦 [2/3] Parsing HCPCS Level II from {os.path.basename(hcpcs_zip)}...")
t0 = time.time()

hcpcs_rows = []
if os.path.exists(hcpcs_zip):
    with zipfile.ZipFile(hcpcs_zip, "r") as z:
        with z.open("HCPC2027_JAN_ANWEB.txt") as f:
            for line in f:
                decoded = line.decode("latin1", errors="ignore")
                # Format: code in first columns, description starts around col 10-15
                m = re.match(r"^\s*([A-Z0-9]{4,5})\s*(\d{0,4})\s*(.*)", decoded)
                if m:
                    code = m.group(1).strip()
                    desc = m.group(3).strip()
                    if code and desc:
                        # Clean repetitive trailing words
                        clean_desc = desc
                        if len(desc) > 80:
                            parts = desc.split("  ")
                            clean_desc = parts[0].strip() if parts[0].strip() else desc[:80]
                        hcpcs_rows.append((code, clean_desc, 2027))

    cur.executemany("""
    INSERT OR REPLACE INTO hcpcs (code, description, fiscal_year)
    VALUES (?, ?, ?)
    """, hcpcs_rows)

    cur.execute("""
    INSERT INTO hcpcs_fts(rowid, code, description)
    SELECT rowid, code, description FROM hcpcs
    """)
    conn.commit()
    print(f"  ✅ Indexed {len(hcpcs_rows):,} HCPCS Level II codes in {time.time()-t0:.2f}s")
else:
    print(f"  ❌ File not found: {hcpcs_zip}")

# ─── 3. Ingest Conversion Table ──────────────────────────────────────────────
conv_zip = os.path.join(DOC_DIR, "icd-10-cm-conversion-table-fy2027-october-1-2026-final.zip")
print(f"\n📦 [3/3] Parsing Conversion Table from {os.path.basename(conv_zip)}...")
t0 = time.time()

conv_rows = []
if os.path.exists(conv_zip):
    with zipfile.ZipFile(conv_zip, "r") as z:
        for name in z.namelist():
            if name.endswith(".csv"):
                with z.open(name) as f:
                    reader = csv.reader(io.TextIOWrapper(f, "latin1"))
                    for row in reader:
                        if len(row) >= 3:
                            curr = row[0].strip()
                            eff = row[1].strip()
                            prev = row[2].strip()
                            if curr and prev and curr != "Current code assignment":
                                conv_rows.append((curr, eff, prev))

    cur.executemany("""
    INSERT OR IGNORE INTO conversions (current_code, effective_year, previous_code)
    VALUES (?, ?, ?)
    """, conv_rows)
    conn.commit()
    print(f"  ✅ Indexed {len(conv_rows):,} code conversions in {time.time()-t0:.2f}s")
else:
    print(f"  ❌ File not found: {conv_zip}")

conn.close()

print("\n" + "="*65)
print("  🎉 INGESTION COMPLETE!")
print(f"  Database created: {DB_PATH}")
print(f"  Total ICD-10-CM : {len(icd_rows):,}")
print(f"  Total HCPCS     : {len(hcpcs_rows):,}")
print(f"  Total Mappings  : {len(conv_rows):,}")
print("="*65)
