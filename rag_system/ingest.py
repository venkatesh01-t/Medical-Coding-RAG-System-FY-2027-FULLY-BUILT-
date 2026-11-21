"""
ingest.py — Documents படிச்சு, chunks ஆக்கி, index பண்ணு
Uses: sklearn TF-IDF + BM25 (pure Python, no torch/onnx needed)
Run: python ingest.py
"""

import os, re, hashlib, glob, pickle, json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

# ─── Config ───────────────────────────────────────────────────────────────────
DOCS_PATH   = "../rag-skills/rag-skills"   # Markdown files folder
INDEX_PATH  = "./rag_index.pkl"            # All indexes saved here
CHUNK_SIZE  = 800                          # characters per chunk
OVERLAP     = 100                          # overlap between chunks

print("🚀 RAG Ingest starting...")
print(f"📁 Looking for docs in: {os.path.abspath(DOCS_PATH)}")

# ─── Helper: chunk a long text ────────────────────────────────────────────────
def chunk_text(text, source, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start, idx = 0, 0
    while start < len(text):
        end   = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if len(piece) > 60:
            cid = hashlib.md5(f"{source}#{idx}".encode()).hexdigest()[:10]
            chunks.append({"id": cid, "source": os.path.basename(source),
                           "text": piece, "idx": idx})
            idx += 1
        start += chunk_size - overlap
    return chunks

# ─── Load all .md files ───────────────────────────────────────────────────────
md_files = glob.glob(os.path.join(DOCS_PATH, "*.md"))
if not md_files:
    # Try subdirectory
    md_files = glob.glob(os.path.join(DOCS_PATH, "**", "*.md"), recursive=True)

print(f"📄 Found {len(md_files)} documents")
if not md_files:
    print("❌ No .md files found! Check DOCS_PATH in ingest.py")
    exit(1)

all_chunks = []
for filepath in md_files:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    # Strip YAML frontmatter
    raw = re.sub(r"^---.*?---\n", "", raw, flags=re.DOTALL)
    chunks = chunk_text(raw, filepath)
    all_chunks.extend(chunks)
    print(f"  ✓ {os.path.basename(filepath)} → {len(chunks)} chunks")

print(f"\n📦 Total chunks: {len(all_chunks)}")

# ─── Build TF-IDF vector index ───────────────────────────────────────────────
print("🔢 Building TF-IDF vector index...")
texts  = [c["text"] for c in all_chunks]
tfidf  = TfidfVectorizer(max_features=20000, ngram_range=(1,2), sublinear_tf=True)
matrix = tfidf.fit_transform(texts)   # sparse matrix: chunks × vocab
print(f"   Vocabulary size: {len(tfidf.vocabulary_)}")

# ─── Build BM25 keyword index ─────────────────────────────────────────────────
print("🔍 Building BM25 keyword index...")
tokenize = lambda s: re.findall(r"\w+", s.lower())
bm25     = BM25Okapi([tokenize(t) for t in texts])

# ─── Save everything ─────────────────────────────────────────────────────────
print("💾 Saving indexes...")
with open(INDEX_PATH, "wb") as f:
    pickle.dump({"chunks": all_chunks, "tfidf": tfidf,
                 "matrix": matrix, "bm25": bm25}, f)

print(f"\n✅ Indexing complete!")
print(f"   Index saved: {INDEX_PATH}")
print(f"   Total chunks indexed: {len(all_chunks)}")
print(f"\n👉 Now run: python app.py  then open http://localhost:8000")
