from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sentence_transformers import SentenceTransformer

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    words = text.split()
    step = chunk_size - overlap
    return [
        " ".join(words[start : start + chunk_size])
        for start in range(0, len(words), step)
    ]

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)

@dataclass
class VectorStore:
    vectors: np.ndarray | None = None
    texts: list[str] | None = None

    def build(self, embedder: Embedder, texts: list[str]) -> None:
        if not texts:
            raise ValueError("Cannot build the vector store without text chunks")
        self.vectors = embedder.embed(texts)
        self.texts = texts

    def search(
        self, query_vec: np.ndarray, top_k: int = 3
    ) -> list[tuple[str, float]]:
        if self.vectors is None or self.texts is None:
            raise RuntimeError("Build the vector store before searching")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        scores = self.vectors @ query_vec
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.texts[i], float(scores[i])) for i in top_indices]


class RAGPipeline:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm_call: Callable[[str], str] | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.llm_call = llm_call

    def answer(self, query: str, top_k: int = 3) -> str:
        if not query.strip():
            raise ValueError("The question cannot be empty")

        query_vector = self.embedder.embed([query])[0]
        retrieved = self.store.search(query_vector, top_k=top_k)
        context = "\n---\n".join(text for text, _ in retrieved)

        if self.llm_call is None:
            return context

        prompt = (
            "Answer the question using only the context below. "
            "If the context does not contain the answer, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        return self.llm_call(prompt)

def main() -> None:
    handbook_path = Path(__file__).with_name("company_handbook.txt")
    if not handbook_path.exists():
        handbook_path = Path(__file__).with_name("company_handbook")
    if not handbook_path.is_file():
        raise FileNotFoundError(
            "Could not find company_handbook.txt or company_handbook "
            "next to main.py"
        )

    document = handbook_path.read_text(encoding="utf-8")
    chunks = chunk_text(document)
    if not chunks:
        raise ValueError(f"The handbook is empty: {handbook_path}")

    embedder = Embedder()
    store = VectorStore()
    store.build(embedder, chunks)
    rag = RAGPipeline(embedder, store)

    question = input("Ask a question about the handbook: ")
    print(rag.answer(question))

if __name__ == "__main__":
    main()
