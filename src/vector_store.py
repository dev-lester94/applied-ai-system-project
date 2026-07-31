"""JSON-backed store of song embeddings with cosine-similarity search."""

import json
from typing import List, Tuple

import numpy as np


class VectorStore:
    def __init__(self):
        self._ids: List[int] = []
        self._vectors: List[List[float]] = []

    def add(self, song_id: int, embedding: List[float]) -> None:
        self._ids.append(song_id)
        self._vectors.append(embedding)

    def __len__(self) -> int:
        return len(self._ids)

    def save(self, path: str) -> None:
        entries = [
            {"id": song_id, "embedding": embedding}
            for song_id, embedding in zip(self._ids, self._vectors)
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entries, f)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        store = cls()
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            store.add(entry["id"], entry["embedding"])
        return store

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[int, float]]:
        query = np.array(query_embedding)
        vectors = np.array(self._vectors)
        similarities = vectors @ query / (
            np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
        )
        ranked = sorted(zip(self._ids, similarities.tolist()), key=lambda entry: entry[1], reverse=True)
        return ranked[:top_k]
