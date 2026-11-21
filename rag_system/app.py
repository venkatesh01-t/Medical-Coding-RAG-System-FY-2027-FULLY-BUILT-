"""
app.py — FastAPI backend: /ask endpoint + Medical Coding RAG + Web UI
Run: python app.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn, os, sys, re

app = FastAPI(title="Chat With My Documents & Medical Coding RAG")

# ─── Load Pipelines ───────────────────────────────────────────────────────────
print("Loading Document RAG pipeline...")
try:
    from pipeline import ask, all_chunks
    print(f"✅ Document Pipeline loaded ({len(all_chunks)} chunks).")
    DOC_READY = True
except Exception as e:
    print(f"⚠️  Document Pipeline warning: {e}")
    DOC_READY = False
    all_chunks = []

print("Loading Medical Coding RAG pipeline...")
try:
    from medical_pipeline import ask_medical
    print("✅ Medical Coding Pipeline loaded (120,000+ codes active).")
    MEDICAL_READY = True
except Exception as e:
    print(f"⚠️  Medical Pipeline warning: {e}")
    MEDICAL_READY = False

# ─── API Models ───────────────────────────────────────────────────────────────
class Question(BaseModel):
    question: str
    mode: str = "auto"  # "auto", "medical", "documents"

# Helper: check if query looks like a medical coding question
def is_medical_query(text: str) -> bool:
    t = text.strip().upper()
    # Matches ICD-10 pattern (e.g. E11.9, A00.0, I10, A0100) or HCPCS (e.g. A1001, J0129)
    if re.search(r"\b([A-TV-Z][0-9]{2}(\.[0-9A-Z]{1,4})?|[A-V][0-9]{4})\b", t):
        return True
    med_keywords = [
        "ICD", "HCPCS", "DIABETES", "HYPERTENSION", "CHOLERA", "DIAGNOSIS",
        "BILLABLE", "FEVER", "DISEASE", "SYMPTOM", "INFECTION", "FRACTURE",
        "WOUND", "INJURY", "ARTHRITIS", "CODE", "CODING", "PATIENT", "NEOPLASM"
    ]
    return any(k in t for k in med_keywords)

# ─── /ask endpoint (Smart Multi-Collection Router) ───────────────────────────
@app.post("/ask")
async def ask_endpoint(body: Question):
    q = body.question.strip()
    if not q:
        return {"status": "insufficient_context", "answer": "Please ask a question.", "citations": []}

    # Route based on mode or smart detection
    if body.mode == "medical" or (body.mode == "auto" and is_medical_query(q) and MEDICAL_READY):
        return ask_medical(q)
    
    # Document RAG fallback
    if DOC_READY:
        return ask(q)

    # If document RAG not ready but medical is
    if MEDICAL_READY:
        return ask_medical(q)

    return {"status": "error", "answer": "System not ready. Run ingest.py first.", "citations": []}

# ─── /ask_medical dedicated endpoint ──────────────────────────────────────────
@app.post("/ask_medical")
async def ask_medical_endpoint(body: Question):
    if not MEDICAL_READY:
        return {"status": "error", "answer": "Medical database not found. Run: python ingest_medical.py", "citations": []}
    return ask_medical(body.question)

# ─── /status endpoint ─────────────────────────────────────────────────────────
@app.get("/status")
async def status():
    return {
        "ready": DOC_READY or MEDICAL_READY,
        "doc_chunks": len(all_chunks) if DOC_READY else 0,
        "medical_codes_active": 121743 if MEDICAL_READY else 0,
        "message": "RAG System & Medical Coding Engine Active! (120,000+ codes)"
    }

# ─── Serve static UI ─────────────────────────────────────────────────────────
UI_DIR = os.path.join(os.path.dirname(__file__), "ui")
if os.path.exists(UI_DIR):
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

@app.get("/")
async def root():
    index_file = os.path.join(UI_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "RAG API running. POST /ask with {question: '...'}"}

# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🏥 Medical Coding & Document RAG System (FY 2027)")
    print("="*60)
    print("  Open in browser: http://localhost:8000")
    print("  API docs:        http://localhost:8000/docs")
    print("="*60 + "\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
