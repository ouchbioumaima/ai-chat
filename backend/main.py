from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import os

from core.rag_engine import RAGEngine
from core.chat_manager import ChatManager
from core.database import (init_db, create_user, get_user_by_username,
    save_session, save_message, get_user_sessions, get_session_messages,
    get_session_owner, delete_session)
from core.auth import hash_password, verify_password, create_token, decode_token

app = FastAPI(title="RAG Voice Chat", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=True)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

rag_engine = RAGEngine()
chat_manager = ChatManager()

@app.on_event("startup")
async def startup():
    init_db()

# ── Auth schemas ──────────────────────────────────────────────────────────────
class AuthRequest(BaseModel):
    username: str
    password: str

class SessionCreate(BaseModel):
    persona: Optional[str] = "a helpful assistant"

class ChatRequest(BaseModel):
    session_id: str
    message: str
    use_rag: bool = True

# ── Auth dependency ───────────────────────────────────────────────────────────
def get_current_user(token: Optional[str] = Cookie(None)):
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

@app.post("/api/register")
async def register(body: AuthRequest):
    if len(body.username) < 3:
        raise HTTPException(400, "Username must be at least 3 characters")
    if len(body.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    if get_user_by_username(body.username):
        raise HTTPException(400, "Username already taken")
    user_id = create_user(body.username, hash_password(body.password))
    token = create_token(user_id, body.username)
    resp = JSONResponse({"message": "Account created!", "username": body.username})
    resp.set_cookie("token", token, httponly=True, max_age=86400*7, samesite="lax")
    return resp

@app.post("/api/login")
async def login(body: AuthRequest):
    user = get_user_by_username(body.username)
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(401, "Invalid username or password")
    token = create_token(user["id"], user["username"])
    resp = JSONResponse({"message": "Logged in!", "username": body.username})
    resp.set_cookie("token", token, httponly=True, max_age=86400*7, samesite="lax")
    return resp

@app.post("/api/logout")
async def logout():
    resp = JSONResponse({"message": "Logged out"})
    resp.delete_cookie("token")
    return resp

@app.get("/api/me")
async def me(user=Depends(get_current_user)):
    return {"username": user["username"], "user_id": user["user_id"]}

@app.post("/api/sessions")
async def create_session(body: SessionCreate, user=Depends(get_current_user)):
    session_id = chat_manager.create_session(persona=body.persona)
    save_session(session_id, user["user_id"], body.persona)
    return {"session_id": session_id, "persona": body.persona}

@app.get("/api/sessions")
async def list_sessions(user=Depends(get_current_user)):
    return {"sessions": get_user_sessions(user["user_id"])}

@app.post("/api/chat")
async def chat(req: ChatRequest, user=Depends(get_current_user)):
    # Verify session belongs to this user
    owner = get_session_owner(req.session_id)
    if owner and owner != user["user_id"]:
        raise HTTPException(403, "Not your session")
    if not chat_manager.session_exists(req.session_id):
        # Restore session if server restarted
        session_id = chat_manager.create_session()
        # remap
        chat_manager._sessions[req.session_id] = chat_manager._sessions.pop(session_id)

    context_chunks, sources = [], []
    if req.use_rag and rag_engine.has_documents():
        results = rag_engine.retrieve(req.message, top_k=3)
        context_chunks = [r["text"] for r in results]
        sources = list({r["source"] for r in results})

    save_message(req.session_id, "user", req.message, [])
    reply = await chat_manager.generate_reply(
        session_id=req.session_id, user_message=req.message, context_chunks=context_chunks)
    save_message(req.session_id, "assistant", reply, sources)
    return {"reply": reply, "sources": sources, "session_id": req.session_id}

@app.get("/api/sessions/{session_id}/history")
async def get_history(session_id: str, user=Depends(get_current_user)):
    owner = get_session_owner(session_id)
    if owner and owner != user["user_id"]:
        raise HTTPException(403, "Not your session")
    return {"history": get_session_messages(session_id)}

@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str, user=Depends(get_current_user)):
    owner = get_session_owner(session_id)
    if owner and owner != user["user_id"]:
        raise HTTPException(403, "Not your session")
    chat_manager.clear_session(session_id)
    delete_session(session_id)
    return {"status": "deleted"}

@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...), user=Depends(get_current_user)):
    allowed = {".txt", ".pdf", ".md"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported: {ext}")
    content = await file.read()
    doc_id = rag_engine.ingest(filename=file.filename, content=content, ext=ext)
    return {"doc_id": doc_id, "filename": file.filename, "status": "indexed"}

@app.get("/api/documents")
async def list_documents(user=Depends(get_current_user)):
    return {"documents": rag_engine.list_documents()}

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str, user=Depends(get_current_user)):
    if not rag_engine.delete(doc_id):
        raise HTTPException(404, "Not found")
    return {"status": "deleted"}

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "3.0.0"}
