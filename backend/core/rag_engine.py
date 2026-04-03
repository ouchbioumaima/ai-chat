"""
RAG Engine — Document ingestion, chunking, embedding & retrieval.

Uses sentence-transformers for local embeddings and a simple in-memory
vector store (no external DB required for the demo).
Swap VectorStore for ChromaDB / Pinecone in production.
"""

import uuid
import re
import math
from typing import Optional
from collections import defaultdict


class VectorStore:
    """
    Lightweight cosine-similarity vector store (pure Python).
    Drop-in replacement target: ChromaDB or FAISS for production.
    """

    def __init__(self):
        self._docs: dict[str, dict] = {}  # id -> {text, embedding, source, doc_id}

    def add(self, chunk_id: str, text: str, embedding: list[float], source: str, doc_id: str):
        self._docs[chunk_id] = {
            "text": text,
            "embedding": embedding,
            "source": source,
            "doc_id": doc_id,
        }

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[dict]:
        if not self._docs:
            return []
        scored = []
        for chunk_id, doc in self._docs.items():
            score = self._cosine(query_embedding, doc["embedding"])
            scored.append({**doc, "chunk_id": chunk_id, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def delete_by_doc(self, doc_id: str) -> int:
        to_delete = [k for k, v in self._docs.items() if v["doc_id"] == doc_id]
        for k in to_delete:
            del self._docs[k]
        return len(to_delete)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x ** 2 for x in a))
        mag_b = math.sqrt(sum(x ** 2 for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)


class EmbeddingModel:
    """
    Wrapper around sentence-transformers.
    Falls back to a TF-IDF-style bag-of-words if the library isn't installed,
    so the project runs without GPU / heavy deps during a demo.
    """

    def __init__(self):
        self._model = None
        self._vocab: dict[str, int] = {}
        self._use_st = False
        self._try_load()

    def _try_load(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self._use_st = True
            print("[EmbeddingModel] Using sentence-transformers (all-MiniLM-L6-v2)")
        except ImportError:
            print("[EmbeddingModel] sentence-transformers not found — using BoW fallback")

    def encode(self, text: str) -> list[float]:
        if self._use_st:
            return self._model.encode(text).tolist()
        return self._bow(text)

    def _bow(self, text: str) -> list[float]:
        tokens = re.findall(r"\w+", text.lower())
        for t in tokens:
            if t not in self._vocab:
                self._vocab[t] = len(self._vocab)
        dim = max(len(self._vocab), 1)
        vec = [0.0] * dim
        for t in tokens:
            vec[self._vocab[t]] += 1.0
        # L2 normalise
        mag = math.sqrt(sum(x ** 2 for x in vec)) or 1.0
        return [x / mag for x in vec]


class RAGEngine:
    """
    Orchestrates document ingestion and semantic retrieval.

    Pipeline:
        upload → extract text → chunk → embed → store
        query  → embed → cosine search → return top-k chunks
    """

    CHUNK_SIZE = 400      # characters per chunk
    CHUNK_OVERLAP = 80    # overlap between consecutive chunks

    def __init__(self):
        self._store = VectorStore()
        self._embedder = EmbeddingModel()
        self._documents: dict[str, dict] = {}  # doc_id -> metadata

    # ── Ingestion ─────────────────────────────────────────────────────────

    def ingest(self, filename: str, content: bytes, ext: str) -> str:
        """Process and index a document. Returns its doc_id."""
        text = self._extract_text(content, ext)
        chunks = self._chunk(text)
        doc_id = str(uuid.uuid4())[:8]

        for i, chunk in enumerate(chunks):
            embedding = self._embedder.encode(chunk)
            # Pad/trim to consistent dim if using BoW
            self._store.add(
                chunk_id=f"{doc_id}-{i}",
                text=chunk,
                embedding=embedding,
                source=filename,
                doc_id=doc_id,
            )

        self._documents[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "chunks": len(chunks),
            "size_bytes": len(content),
        }
        print(f"[RAGEngine] Indexed '{filename}' → {len(chunks)} chunks (id={doc_id})")
        return doc_id

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Return the top-k most relevant chunks for a query."""
        q_emb = self._embedder.encode(query)
        results = self._store.search(q_emb, top_k=top_k)
        return [{"text": r["text"], "source": r["source"], "score": round(r["score"], 4)} for r in results]

    def delete(self, doc_id: str) -> bool:
        if doc_id not in self._documents:
            return False
        self._store.delete_by_doc(doc_id)
        del self._documents[doc_id]
        return True

    def list_documents(self) -> list[dict]:
        return list(self._documents.values())

    def has_documents(self) -> bool:
        return bool(self._documents)

    def count(self) -> int:
        return len(self._documents)

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _extract_text(content: bytes, ext: str) -> str:
        if ext == ".pdf":
            try:
                import pdfplumber
                import io
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    return "\n".join(p.extract_text() or "" for p in pdf.pages)
            except ImportError:
                return content.decode("utf-8", errors="ignore")
        return content.decode("utf-8", errors="ignore")

    def _chunk(self, text: str) -> list[str]:
        """Sliding-window character chunker."""
        text = re.sub(r"\s+", " ", text).strip()
        chunks, start = [], 0
        while start < len(text):
            end = start + self.CHUNK_SIZE
            chunks.append(text[start:end])
            start += self.CHUNK_SIZE - self.CHUNK_OVERLAP
        return [c for c in chunks if len(c.strip()) > 20]
