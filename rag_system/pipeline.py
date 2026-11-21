"""
pipeline.py — Hybrid Retrieve + Dual-Engine Answer
Modes:
1. Local LLM (Ollama) if running
2. Grounded Extractive QA (Pure Python, instant, zero-install) when Ollama is offline
Strictly says "I don't know" when documents don't have the answer.
"""

import re, pickle, json, urllib.request
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ─── Config ───────────────────────────────────────────────────────────────────
INDEX_PATH   = "./rag_index.pkl"
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"       # or "mistral" or "phi3"
TOP_K        = 30               # retrieve wide
RERANK_TOP   = 5                # send this many chunks to answerer
MIN_SCORE    = 0.08             # below this → "I don't know"

# ─── Load index ───────────────────────────────────────────────────────────────
print("Loading RAG index...")
with open(INDEX_PATH, "rb") as f:
    _idx = pickle.load(f)

all_chunks = _idx["chunks"]
tfidf      = _idx["tfidf"]
matrix     = _idx["matrix"]
bm25       = _idx["bm25"]
tokenize   = lambda s: re.findall(r"\w+", s.lower())

print(f"✅ Loaded {len(all_chunks)} chunks")

# ─── ABSTAIN response ─────────────────────────────────────────────────────────
ABSTAIN = {
    "status"   : "insufficient_context",
    "answer"   : "I don't know. The documents I have don't contain enough information to answer this question.",
    "citations": [],
    "chunks"   : [],
}

STOP_WORDS = {
    "what", "is", "the", "how", "to", "in", "a", "an", "and", "of", "for",
    "does", "do", "did", "are", "were", "was", "can", "could", "should", "would",
    "explain", "tell", "me", "about", "which", "who", "whom", "where", "when", "why"
}

# ─── Reciprocal Rank Fusion ───────────────────────────────────────────────────
def rrf(rank_lists, k=60):
    scores = {}
    for ranks in rank_lists:
        for r, idx in enumerate(ranks, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + r)
    return sorted(scores, key=scores.get, reverse=True)

# ─── Retrieve: TF-IDF dense + BM25 keyword → Fuse ────────────────────────────
def retrieve(question: str):
    # TF-IDF vector search
    qv           = tfidf.transform([question])
    sims         = cosine_similarity(qv, matrix).flatten()
    tfidf_ranked = list(np.argsort(-sims)[:TOP_K])
    tfidf_scores = {i: float(sims[i]) for i in tfidf_ranked}

    # BM25 keyword search
    bm25_scores  = bm25.get_scores(tokenize(question))
    bm25_ranked  = list(np.argsort(-bm25_scores)[:TOP_K])

    # Fuse with RRF
    fused = rrf([tfidf_ranked, bm25_ranked])[:RERANK_TOP]

    results = []
    for idx in fused:
        score = tfidf_scores.get(idx, 0.0)
        results.append({**all_chunks[idx], "score": score})

    return results

# ─── Score Gate ───────────────────────────────────────────────────────────────
def gate(chunks):
    return [c for c in chunks if c["score"] >= MIN_SCORE]

# ─── Grounded prompt for Ollama ───────────────────────────────────────────────
SYSTEM = """You are a helpful assistant. You answer questions ONLY using the provided source documents below.

STRICT RULES:
1. Use ONLY facts that appear in the sources. Never use outside knowledge.
2. Cite every claim with a tag like [S1], [S2].
3. If the sources do not contain enough information, reply EXACTLY with this sentence:
   I don't know. The documents I have don't contain enough information to answer this question.
4. Never guess. Never make up information. When in doubt, say you don't know.
5. Be concise and clear."""

def build_prompt(question: str, chunks: list) -> str:
    sources = "\n\n".join(
        f"[S{i+1}] (from: {c['source']})\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    return f"{SYSTEM}\n\nSOURCES:\n{sources}\n\nQUESTION: {question}\n\nANSWER:"

# ─── Call Ollama (returns None if offline) ────────────────────────────────────
def is_ollama_online() -> bool:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.2)
    try:
        s.connect(('127.0.0.1', 11434))
        s.close()
        return True
    except Exception:
        return False

def call_ollama(prompt: str):
    if not is_ollama_online():
        return None
    data = json.dumps({
        "model": OLLAMA_MODEL, "prompt": prompt,
        "stream": False, "options": {"temperature": 0, "num_predict": 512},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read()).get("response", "").strip()
    except Exception:
        return None

# ─── Grounded Extractive Answerer (Pure Python Fallback) ──────────────────────
def extract_grounded_answer(question: str, top_chunks: list):
    q_tokens = set(tokenize(question)) - STOP_WORDS
    if not q_tokens:
        q_tokens = set(tokenize(question))

    candidates = []
    for c_idx, c in enumerate(top_chunks):
        raw_sents = re.split(r"(?<=[.!?\n])\s+", c["text"])
        for s in raw_sents:
            s_clean = s.strip()
            if len(s_clean) < 20:
                continue
            s_toks = set(tokenize(s_clean))
            overlap = len(q_tokens & s_toks)
            if overlap > 0:
                score = overlap / (len(q_tokens) ** 0.5)
                candidates.append((score, s_clean, c_idx + 1, c["source"]))

    if not candidates:
        return None

    candidates.sort(key=lambda x: -x[0])
    
    # Must have solid keyword overlap with question intent
    if candidates[0][0] < 0.6:
        return None

    # Pick top 2-3 unique sentences
    selected_sents = []
    seen = set()
    used_sources = set()
    for score, s_clean, s_idx, src in candidates[:4]:
        norm = s_clean.lower()[:40]
        if norm not in seen:
            seen.add(norm)
            selected_sents.append(f"[S{s_idx}] {s_clean}")
            used_sources.add(f"S{s_idx}: {src}")

    return {
        "text": "\n\n".join(selected_sents),
        "citations": sorted(list(used_sources))
    }

# ─── Simple grounding verify ──────────────────────────────────────────────────
def verify(answer: str, chunks: list) -> bool:
    if "don't know" in answer.lower():
        return True
    source_words = set(tokenize(" ".join(c["text"] for c in chunks)))
    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 20]
    if not sentences:
        return True
    supported = sum(
        1 for s in sentences
        if len(set(tokenize(s)) & source_words) / max(len(tokenize(s)), 1) > 0.25
    )
    return supported / len(sentences) >= 0.5

# ─── Main pipeline ────────────────────────────────────────────────────────────
def ask(question: str) -> dict:
    if not question.strip():
        return {**ABSTAIN, "answer": "Please type a question."}

    # 1. Retrieve
    candidates = retrieve(question)
    if not candidates:
        return ABSTAIN

    # 2. Gate — throw away low-score chunks
    top = gate(candidates)
    if not top:
        return ABSTAIN

    # 3. Try Generative LLM via Ollama first
    prompt = build_prompt(question, top)
    ollama_answer = call_ollama(prompt)

    if ollama_answer is not None:
        if "don't know" in ollama_answer.lower() or "do not have" in ollama_answer.lower():
            return {
                **ABSTAIN,
                "chunks": [{"source": c["source"], "text": c["text"][:200],
                            "score": round(c["score"], 3)} for c in top]
            }

        if verify(ollama_answer, top):
            return {
                "status"   : "answered",
                "answer"   : ollama_answer,
                "citations": [f"S{i+1}: {c['source']}" for i, c in enumerate(top)],
                "chunks"   : [{"source": c["source"], "text": c["text"][:200],
                               "score": round(c["score"], 3)} for c in top],
            }

    # 4. If Ollama is offline or unverified, use Grounded Extractive Engine!
    extracted = extract_grounded_answer(question, top)
    if extracted is not None:
        return {
            "status"   : "answered",
            "answer"   : extracted["text"],
            "citations": extracted["citations"],
            "chunks"   : [{"source": c["source"], "text": c["text"][:200],
                           "score": round(c["score"], 3)} for c in top],
        }

    # 5. Abstain if no direct evidence found
    return {
        **ABSTAIN,
        "chunks": [{"source": c["source"], "text": c["text"][:200],
                    "score": round(c["score"], 3)} for c in top]
    }
