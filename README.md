# 🧠 RAG Chat Assistant

A production-ready **Retrieval-Augmented Generation (RAG)** chat application built with Python, FastAPI, and Claude AI.

## ✨ Features

| Feature | Details |
|---|---|
| 📄 Document ingestion | Upload `.txt`, `.pdf`, `.md` files |
| 🔍 Semantic retrieval | Cosine-similarity search over embeddings |
| 🤖 AI responses | Powered by Claude (Anthropic) |
| 💬 Multi-turn chat | Persistent session history |
| 🎭 Custom personas | Switch assistant personality on-the-fly |
| ⚡ REST API | Clean FastAPI backend with auto-docs |
| 🎨 Chat UI | Single-page frontend served by FastAPI |

---

## 🏗️ Architecture

```
rag-chat/
├── backend/
│   ├── main.py              # FastAPI app, routes
│   ├── core/
│   │   ├── rag_engine.py    # Chunking, embedding, vector search
│   │   └── chat_manager.py  # Sessions, LLM calls
│   └── frontend/
│       └── index.html       # Chat UI (served as static file)
└── requirements.txt
```

### RAG Pipeline

```
Upload document
     │
     ▼
Extract text  (pdfplumber for PDFs, UTF-8 for text)
     │
     ▼
Chunk text    (sliding window, 400 chars, 80 overlap)
     │
     ▼
Embed chunks  (sentence-transformers all-MiniLM-L6-v2)
     │
     ▼
Store vectors (in-memory cosine-similarity store)

── At query time ──

User query → embed → cosine search → top-k chunks
     │
     ▼
Inject context into Claude prompt → stream reply
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run the server

```bash
cd backend
uvicorn main:app --reload
```

### 4. Open the UI

Visit **http://localhost:8000** in your browser.

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/sessions` | Create a chat session |
| `POST` | `/api/chat` | Send a message |
| `GET`  | `/api/sessions/{id}/history` | Get chat history |
| `DELETE` | `/api/sessions/{id}` | Clear session |
| `POST` | `/api/documents` | Upload & index a document |
| `GET`  | `/api/documents` | List indexed documents |
| `DELETE` | `/api/documents/{id}` | Remove a document |
| `GET`  | `/api/health` | Health check |

Auto-generated docs: **http://localhost:8000/docs**

---

## 🔧 Configuration

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `CHUNK_SIZE` | `400` | Characters per chunk |
| `CHUNK_OVERLAP` | `80` | Overlap between chunks |
| `MAX_HISTORY` | `20` | Max conversation turns kept |

---

## 🚀 Production Upgrades

- **Vector DB**: Swap in-memory store for [ChromaDB](https://www.trychroma.com/) or [Pinecone](https://www.pinecone.io/)
- **Persistence**: Add SQLite/PostgreSQL for session storage
- **Auth**: Add JWT authentication via FastAPI middleware
- **Streaming**: Use SSE for streamed token responses
- **Deployment**: Docker + Railway / Render / AWS

---

## 🛠️ Tech Stack

- **Backend**: Python 3.11+, FastAPI, Uvicorn
- **AI/LLM**: Anthropic Claude (`claude-sonnet-4-20250514`)
- **Embeddings**: `sentence-transformers` (all-MiniLM-L6-v2)
- **PDF parsing**: `pdfplumber`
- **Frontend**: Vanilla HTML/CSS/JS (zero dependencies)

---

*Built as a portfolio project demonstrating RAG architecture, API design, and AI integration.*
