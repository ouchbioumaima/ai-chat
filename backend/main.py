"""
RAG Chat Assistant - FastAPI Backend (v2 - with SQLite persistence)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os

from core.rag_engine import RAGEngine
from core.chat_manager import ChatManager
from core.database import init_db, get_sessions, get_session_messages, delete_session, save_message, save_session

app = FastAPI(title="RAG Chat Assistant", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

rag_engine = RAGEngine()
chat_manager = ChatManager()


@app.on_event("startup")
async def startup():
    init_db()


class ChatRequest(BaseModel):
    session_id: str
    message: str
    use_rag: bool = True

class ChatResponse(BaseModel):
    reply: str
    sources: list[str]
    session_id: str

class SessionCreate(BaseModel):
    persona: Optional[str] = "helpful assistant"


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


@app.post("/api/sessions")
async def create_session(body: SessionCreate):
    session_id = chat_manager.create_session(persona=body.persona)
    save_session(session_id, body.persona)   # ← saves to DB
    return {"session_id": session_id, "persona": body.persona}


@app.get("/api/sessions")
async def list_sessions():
    return {"sessions": get_sessions()}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not chat_manager.session_exists(req.session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    context_chunks, sources = [], []
    if req.use_rag and rag_engine.has_documents():
        results = rag_engine.retrieve(req.message, top_k=3)
        context_chunks = [r["text"] for r in results]
        sources = list({r["source"] for r in results})

    save_message(req.session_id, "user", req.message, [])

    reply = await chat_manager.generate_reply(
        session_id=req.session_id,
        user_message=req.message,
        context_chunks=context_chunks,
    )

    save_message(req.session_id, "assistant", reply, sources)

    return ChatResponse(reply=reply, sources=sources, session_id=req.session_id)


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    allowed = {".txt", ".pdf", ".md"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    content = await file.read()
    doc_id = rag_engine.ingest(filename=file.filename, content=content, ext=ext)
    return {"doc_id": doc_id, "filename": file.filename, "status": "indexed"}


@app.get("/api/documents")
async def list_documents():
    return {"documents": rag_engine.list_documents()}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    if not rag_engine.delete(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted"}


@app.get("/api/sessions/{session_id}/history")
async def get_history(session_id: str):
    return {"history": get_session_messages(session_id)}


@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str):
    chat_manager.clear_session(session_id)
    delete_session(session_id)
    return {"status": "deleted"}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "documents_indexed": rag_engine.count(),
        "active_sessions": chat_manager.count(),
        "total_saved_sessions": len(get_sessions()),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)