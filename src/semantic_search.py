"""Semantic song search: Gemini-generated descriptions + embeddings + match explanations."""

import csv
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

from gemini_client import GeminiClient
from vector_store import VectorStore


@dataclass
class Song:
    """Represents a song and its attributes."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    description: Optional[str] = None


def load_songs(csv_path: str) -> List[Dict]:
    """Loads songs from a CSV file."""
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            songs.append({
                "id": int(row["id"]),
                "title": row["title"],
                "artist": row["artist"],
                "genre": row["genre"],
                "mood": row["mood"],
                "energy": float(row["energy"]),
                "tempo_bpm": float(row["tempo_bpm"]),
                "valence": float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
                "description": row.get("description") or None,
            })
    return songs


def describe_song(song: Dict, client: GeminiClient) -> str:
    prompt = (
        "Write one short, vivid description of this song for a music search engine, "
        "under 20 words. Do not include the song title or artist name in the description. "
        "Do not repeat the raw numbers, translate them into descriptive language.\n"
        f"Title: {song['title']}\n"
        f"Artist: {song['artist']}\n"
        f"Genre: {song['genre']}\n"
        f"Mood: {song['mood']}\n"
        f"Energy (0-1): {song['energy']}\n"
        f"Tempo (BPM): {song['tempo_bpm']}\n"
        f"Valence (0-1): {song['valence']}\n"
        f"Danceability (0-1): {song['danceability']}\n"
        f"Acousticness (0-1): {song['acousticness']}\n"
    )
    return client.generate(prompt).strip()


def build_descriptions(
    csv_path: str,
    client: GeminiClient,
    force: bool = False,
) -> Dict[int, str]:
    """Reads songs from csv_path, generating and persisting a 'description' column
    the first time it's missing (or whenever force=True)."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    has_descriptions = (
        fieldnames is not None
        and "description" in fieldnames
        and all(row.get("description") for row in rows)
    )
    if has_descriptions and not force:
        return {int(row["id"]): row["description"] for row in rows}

    for i, row in enumerate(rows, start=1):
        row["description"] = describe_song(row, client)
        print(f"  Generated description {i}/{len(rows)}: {row['title']}")

    if fieldnames is not None and "description" not in fieldnames:
        fieldnames = list(fieldnames) + ["description"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {int(row["id"]): row["description"] for row in rows}


def build_vector_store(
    songs: List[Dict],
    client: GeminiClient,
    cache_path: str = "data/song_embeddings.json",
    force: bool = False,
) -> VectorStore:
    if not force and os.path.exists(cache_path):
        store = VectorStore.load(cache_path)
        print(f"  Loaded {len(store)} cached embeddings from {cache_path}")
        return store

    store = VectorStore()
    for i, song in enumerate(songs, start=1):
        text = (
            f"{song['description']} "
            f"Genre: {song['genre']}. Mood: {song['mood']}. "
            f"Energy: {song['energy']}. Tempo: {song['tempo_bpm']} BPM. "
            f"Valence: {song['valence']}. Danceability: {song['danceability']}. "
            f"Acousticness: {song['acousticness']}."
        )
        embedding = client.embed_text(text, task_type="RETRIEVAL_DOCUMENT")
        store.add(song["id"], embedding)
        preview = ", ".join(f"{v:.4f}" for v in embedding[:5])
        print(
            f"  Generated embedding {i}/{len(songs)}: {song['title']} "
            f"(dim={len(embedding)}, first 5 values=[{preview}, ...])"
        )
    store.save(cache_path)
    return store


def explain_match(query: str, song: Dict, client: GeminiClient) -> str:
    prompt = (
        f"A user searched for: \"{query}\"\n"
        f"This song was recommended as a match:\n"
        f"Title: {song['title']} by {song['artist']}\n"
        f"Description: {song['description']}\n"
        "In one short sentence, explain why this song fits the user's search."
    )
    return client.generate(prompt).strip()


def semantic_recommend(
    query: str,
    songs: List[Dict],
    store: VectorStore,
    client: GeminiClient,
    k: int = 5,
) -> Iterator[Tuple[Dict, float, str]]:
    songs_by_id = {song["id"]: song for song in songs}
    query_embedding = client.embed_text(query, task_type="RETRIEVAL_QUERY")
    matches = store.search(query_embedding, top_k=k)

    for song_id, score in matches:
        song = songs_by_id[song_id]
        explanation = explain_match(query, song, client)
        yield song, score, explanation
