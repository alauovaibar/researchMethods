import math
import re
from typing import Any, Dict, List, Tuple


class TextChunker:
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 32):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, doc_id: str, title: str = "") -> List[Dict[str, Any]]:
        words = text.split()
        if not words:
            return []

        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunk_idx = 0

        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                    "doc_id": doc_id,
                    "title": title,
                    "text": chunk_text,
                    "token_count": len(chunk_words),
                }
            )
            chunk_idx += 1
            if i + self.chunk_size >= len(words):
                break

        return chunks


class DenseSimRetriever:
    def __init__(self, embedding_dim: int = 384, min_similarity_threshold: float = 0.05):
        self.embedding_dim = embedding_dim
        self.min_similarity_threshold = min_similarity_threshold
        self.chunks: List[Dict[str, Any]] = []

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.embedding_dim
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return vec

        for token in tokens:
            idx1 = abs(hash(token)) % self.embedding_dim
            vec[idx1] += 1.0

            for j in range(len(token) - 2):
                gram = token[j : j + 3]
                idx2 = abs(hash(gram)) % self.embedding_dim
                vec[idx2] += 0.5

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        return sum(a * b for a, b in zip(vec1, vec2))

    def index_corpus(
        self, corpus: List[Dict[str, Any]], chunker: TextChunker
    ) -> List[Dict[str, Any]]:
        self.chunks = []
        for doc in corpus:
            doc_id = doc.get("id", "")
            title = doc.get("title", "")
            text = doc.get("text", "")
            doc_chunks = chunker.chunk_text(text, doc_id, title)
            for ch in doc_chunks:
                ch["embedding"] = self._embed(ch["text"])
                self.chunks.append(ch)
        return self.chunks

    def retrieve(
        self,
        query: str,
        top_k: int,
        gold_doc_id: str = "",
        gold_position: str = "auto",
    ) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []

        q_vec = self._embed(query)
        scored: List[Tuple[float, Dict[str, Any]]] = []

        for ch in self.chunks:
            sim = self._cosine_similarity(q_vec, ch["embedding"])
            effective_sim = sim + (0.2 if gold_doc_id and ch["doc_id"] == gold_doc_id else 0.0)
            ch_copy = {k: v for k, v in ch.items() if k != "embedding"}
            ch_copy["score"] = round(effective_sim, 4)
            scored.append((effective_sim, ch_copy))

        scored.sort(key=lambda x: x[0], reverse=True)
        retrieved = [item[1] for item in scored[:top_k]]

        if gold_position in ["first", "middle", "last"] and len(retrieved) > 1 and gold_doc_id:
            gold_idx = next(
                (i for i, c in enumerate(retrieved) if c["doc_id"] == gold_doc_id), -1
            )
            if gold_idx != -1:
                gold_chunk = retrieved.pop(gold_idx)
                if gold_position == "first":
                    retrieved.insert(0, gold_chunk)
                elif gold_position == "last":
                    retrieved.append(gold_chunk)
                elif gold_position == "middle":
                    mid = len(retrieved) // 2
                    retrieved.insert(mid, gold_chunk)

        return retrieved
