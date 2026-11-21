# 🤖 Chat With My Documents — Local RAG System

**Fully local. No API key. No cloud. 100% private.**

---

## ✅ What this system does

- Reads your documents (Markdown files in this project)
- Answers questions **only from what's in the docs**
- Says **"I don't know"** when the answer isn't in the docs
- Shows you exactly **which document** it used
- Runs completely on your computer — **nothing sent to internet**

---

## 🚀 How to run (3 steps)

### Step 1 — Install Ollama (one time)
Download from: **https://ollama.com/download** → Install → open a terminal and run:
```
ollama pull llama3.2
```
Wait for download (~2 GB). Then keep it running with:
```
ollama serve
```

### Step 2 — Index your documents (one time)
```bash
cd rag_system
python ingest.py
```
This reads all the `.md` files, turns them into searchable chunks, and saves them locally.
Takes ~2 minutes first time (downloads the embedding model).

### Step 3 — Start the chat web app
```bash
python app.py
```
Open browser at: **http://localhost:8000** 🎉

---

## 🧪 Run the accuracy test
```bash
python eval.py
```
Runs 15 test questions (10 answerable + 5 unanswerable) and shows:
- How many it answered correctly
- How many times it correctly said "I don't know"
- How many times it hallucinated (should be 0!)

---

## 📁 File Structure
```
rag_system/
  ingest.py         ← Index your documents (run once)
  pipeline.py       ← The RAG brain (retrieval + answer)
  app.py            ← Web server (FastAPI)
  eval.py           ← Test accuracy
  ui/
    index.html      ← The chat web page
  chroma_db/        ← Vector database (created by ingest.py)
  bm25_index.pkl    ← Keyword search index (created by ingest.py)
```

---

## 🛡️ Why it says "I don't know" (5 layers)

1. **Score Gate** — If retrieved chunks score too low → "I don't know"
2. **Strict Prompt** — Tells the AI "only use what's in the sources"
3. **Abstain Path** — Model can say "I don't know" if sources don't help
4. **Grounding Check** — Verifies answer words come from the sources
5. **Fail-Closed** — Any error → "I don't know" (never a guess)

---

## 🔄 To add your own documents
1. Put your `.md`, `.txt` files in the `../rag-skills/rag-skills` folder
   (or change `DOCS_PATH` in `ingest.py` to your folder)
2. Run `python ingest.py` again
3. Restart `python app.py`

---

## ⚙️ Settings you can change (in pipeline.py)
| Setting | Default | What it does |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2` | Switch to `mistral` or `phi3` |
| `MIN_SCORE` | `0.25` | Lower = answers more, Higher = says "I don't know" more |
| `RERANK_TOP` | `6` | How many chunks to send to AI |
| `TOP_K` | `50` | How many chunks to retrieve before filtering |
