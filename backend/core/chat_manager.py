import uuid, os
from typing import Optional
from groq import Groq

SYSTEM_TEMPLATE = """You are {persona}.
IMPORTANT: Always respond in the SAME language the user writes in.
If they write French → respond French. Arabic → Arabic. Never switch unless asked.
When context from documents is provided, use it and cite the source.
Be concise, helpful, and honest."""

class Session:
    def __init__(self, session_id, persona):
        self.session_id = session_id
        self.persona = persona
        self.history = []
    def add_turn(self, role, content):
        self.history.append({"role": role, "content": content})
    def get_messages(self):
        return self.history.copy()

class ChatManager:
    MAX_HISTORY = 20
    def __init__(self):
        self._sessions = {}
        api_key = os.getenv("GROQ_API_KEY", "")
        self._client = Groq(api_key=api_key) if api_key else None
    def create_session(self, persona="a helpful assistant"):
        sid = str(uuid.uuid4())[:12]
        self._sessions[sid] = Session(sid, persona)
        return sid
    def session_exists(self, sid): return sid in self._sessions
    def get_history(self, sid): return self._sessions[sid].get_messages()
    def clear_session(self, sid):
        if sid in self._sessions: self._sessions[sid].history = []
    def count(self): return len(self._sessions)
    async def generate_reply(self, session_id, user_message, context_chunks=None):
        session = self._sessions[session_id]
        msg = user_message
        if context_chunks:
            ctx = "\n\n".join(f"[Context {i+1}]:\n{c}" for i,c in enumerate(context_chunks))
            msg = f"{user_message}\n\n---\nRelevant context:\n{ctx}"
        session.add_turn("user", msg)
        messages = session.get_messages()[-self.MAX_HISTORY:]
        system = SYSTEM_TEMPLATE.format(persona=session.persona)
        reply = self._call_groq(system, messages) if self._client else f"[Set GROQ_API_KEY] You asked: {user_message}"
        session.add_turn("assistant", reply)
        return reply
    def _call_groq(self, system, messages):
        r = self._client.chat.completions.create(
            model="llama-3.3-70b-versatile", max_tokens=1024,
            messages=[{"role":"system","content":system}] + messages)
        return r.choices[0].message.content
