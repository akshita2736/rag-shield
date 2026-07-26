"""
RAG pipeline: load documents, chunk them, embed chunks, and retrieve the
top-k most relevant chunks for a query via FAISS.

Deliberately simple and inspectable: one class, no abstraction layers,
no vector-DB-agnostic interface. This is the pipeline the firewall sits
in front of — Layer 1/2/3 (heuristics.py / ml_classifier.py / llm_judge.py)
never touch this file directly; firewall.py is the glue between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ragshield import config


@dataclass
class Chunk:
    chunk_id: int
    text: str
    source: str  # filename the chunk came from


def load_document(path: Path) -> str:
    """Read a document's raw text. Supports .txt and .pdf."""
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    raise ValueError(f"Unsupported document type: {suffix} ({path.name})")


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping character-based chunks.

    Simple sliding window — no sentence/paragraph-aware splitting. Good
    enough for a portfolio project; see README limitations re: attacks
    split across a chunk boundary.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)  # guard against overlap >= chunk_size

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += step

    return [c for c in chunks if c]  # drop empty chunks


class RagPipeline:
    """
    Holds an in-memory FAISS index over document chunks and an embedding
    model. Not persisted between runs by design (v1 is single-session:
    upload documents, ask questions, done) — persistence can be added
    later if needed.
    """

    def __init__(self) -> None:
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self.chunks: list[Chunk] = []
        self.index: faiss.IndexFlatL2 | None = None

    def add_documents(self, paths: list[Path]) -> int:
        """Load, chunk, and index one or more documents. Returns chunk count added."""
        new_chunks = []
        for path in paths:
            text = load_document(path)
            for chunk_str in chunk_text(text):
                new_chunks.append(
                    Chunk(
                        chunk_id=len(self.chunks) + len(new_chunks),
                        text=chunk_str,
                        source=path.name,
                    )
                )

        if not new_chunks:
            return 0

        embeddings = self._embed([c.text for c in new_chunks])

        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])

        self.index.add(embeddings)
        self.chunks.extend(new_chunks)
        return len(new_chunks)

    def retrieve(self, query: str, top_k: int = config.RETRIEVAL_TOP_K) -> list[Chunk]:
        """Return the top_k chunks most similar to the query."""
        if self.index is None or not self.chunks:
            return []

        query_embedding = self._embed([query])
        top_k = min(top_k, len(self.chunks))
        _distances, indices = self.index.search(query_embedding, top_k)

        return [self.chunks[i] for i in indices[0] if i != -1]

    def _embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self.embedder.encode(texts, convert_to_numpy=True)
        return embeddings.astype("float32")  # FAISS requires float32