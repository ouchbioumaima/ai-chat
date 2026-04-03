"""
Chat Manager — Session handling and LLM response generation.
Uses Groq API (free) with llama-3.3-70b model.
Get your free key at: https://console.groq.com
"""

import uuid
import os
from typing import Optional
from groq import Groq


SYSTEM_TEMPLATE = """You are {persona}.

You have access to a knowledge base of documents. When relevant context is provided below,
use it to give accurate, grounded answers. Always cite which document you're drawing from
when using retrieved context. If the context doesn't help, rely on your general knowledge
and say so transparently.

Be concise, helpful, and honest. If you don't know something, say so.
"""


class Session:
    def __init__(self, session_id: str, persona: str):
        self.session_id = session_id
        self.persona = persona
        self.history: list[dict] = []

    def add_turn(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        return self.history.copy()


class ChatManager:
    MAX_HISTORY = 20

    def __init__(self):
        self._sessions: dict[str, Session] = {}
        api_key = os.getenv("GROQ_API_KEY", "")
        self._client = Groq(api_key=api_key) if api_key else None

    def create_session(self, persona: str = "a helpful assistant") -> str:
        session_id = str(uuid.uuid4())[:12]
        self._sessions[session_id] = Session(session_id=session_id, persona=persona)
        return session_id

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def get_history(self, session_id: str) -> list[dict]:
        return self._sessions[session_id].get_messages()

    def clear_session(self, session_id: str):
        if session_id in self._sessions:
            self._sessions[session_id].history = []

    def count(self) -> int:
        return len(self._sessions)

    async def generate_reply(
        self,
        session_id: str,
        user_message: str,
        context_chunks: Optional[list[str]] = None,
    ) -> str:
        session = self._sessions[session_id]

        if context_chunks:
            context_block = "\n\n".join(
                f"[Context {i+1}]:\n{chunk}" for i, chunk in enumerate(context_chunks)
            )
            augmented_message = (
                f"{user_message}\n\n---\nRelevant context:\n{context_block}"
            )
        else:
            augmented_message = user_message

        session.add_turn("user", augmented_message)
        messages = session.get_messages()[-self.MAX_HISTORY:]
        system_prompt = SYSTEM_TEMPLATE.format(persona=session.persona)

        if self._client:
            reply = self._call_groq(system_prompt, messages)
        else:
            reply = self._mock_reply(user_message, context_chunks)

        session.add_turn("assistant", reply)
        return reply

    def _call_groq(self, system: str, messages: list[dict]) -> str:
        response = self._client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1024,
            messages=[{"role": "system", "content": system}] + messages,
        )
        return response.choices[0].message.content

    @staticmethod
    def _mock_reply(user_message: str, context_chunks: Optional[list[str]]) -> str:
        if context_chunks:
            preview = context_chunks[0][:120].strip()
            return f"[DEMO MODE — set GROQ_API_KEY]\n\nFound {len(context_chunks)} passage(s). Preview:\n\"{preview}...\""
        return f"[DEMO MODE — set GROQ_API_KEY]\n\nYou asked: \"{user_message}\"\nSet your GROQ_API_KEY to get real AI answers."
