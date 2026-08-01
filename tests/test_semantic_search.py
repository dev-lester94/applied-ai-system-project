import csv
import json

import pytest

from semantic_search import build_descriptions, build_vector_store, semantic_recommend
from vector_store import VectorStore

SONG_FIELDS = [
    "id", "title", "artist", "genre", "mood", "energy",
    "tempo_bpm", "valence", "danceability", "acousticness",
]


class FakeGeminiClient:
    def generate(self, prompt):
        return "A fake generated description."

    def embed_text(self, text, task_type="RETRIEVAL_DOCUMENT", model="gemini-embedding-001"):
        return [0.1, 0.2, 0.3]


class FakeQueryClient:
    """Always embeds the query as [1.0, 0.0, 0.0], so whichever song's stored
    embedding points in that same direction is the closest match."""

    def generate(self, prompt):
        return "This song is the closest semantic match to the query."

    def embed_text(self, text, task_type="RETRIEVAL_QUERY", model="gemini-embedding-001"):
        return [1.0, 0.0, 0.0]


def test_build_descriptions_generates_when_missing(tmp_path):
    csv_path = tmp_path / "songs.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SONG_FIELDS)  # no "description" column
        writer.writeheader()
        writer.writerow({
            "id": "1", "title": "Test Song", "artist": "Test Artist",
            "genre": "pop", "mood": "happy", "energy": "0.8",
            "tempo_bpm": "120", "valence": "0.7", "danceability": "0.8",
            "acousticness": "0.2",
        })

    descriptions = build_descriptions(str(csv_path), FakeGeminiClient())

    assert descriptions == {1: "A fake generated description."}

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["description"] == "A fake generated description."


def test_build_descriptions_regenerates_when_forced(tmp_path):
    csv_path = tmp_path / "songs.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SONG_FIELDS + ["description"])
        writer.writeheader()
        writer.writerow({
            "id": "1", "title": "Test Song", "artist": "Test Artist",
            "genre": "pop", "mood": "happy", "energy": "0.8",
            "tempo_bpm": "120", "valence": "0.7", "danceability": "0.8",
            "acousticness": "0.2", "description": "An old, stale description.",
        })

    descriptions = build_descriptions(str(csv_path), FakeGeminiClient(), force=True)

    assert descriptions == {1: "A fake generated description."}

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["description"] == "A fake generated description."


def test_build_descriptions_regenerates_all_when_partially_filled(tmp_path):
    csv_path = tmp_path / "songs.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SONG_FIELDS + ["description"])
        writer.writeheader()
        writer.writerow({
            "id": "1", "title": "Has Description", "artist": "Artist A",
            "genre": "pop", "mood": "happy", "energy": "0.8",
            "tempo_bpm": "120", "valence": "0.7", "danceability": "0.8",
            "acousticness": "0.2", "description": "An existing description that should be overwritten.",
        })
        writer.writerow({
            "id": "2", "title": "Missing Description", "artist": "Artist B",
            "genre": "rock", "mood": "intense", "energy": "0.9",
            "tempo_bpm": "150", "valence": "0.4", "danceability": "0.5",
            "acousticness": "0.1",
        })

    descriptions = build_descriptions(str(csv_path), FakeGeminiClient(), force=True)

    # Not every row already had a description, so the whole file gets regenerated --
    # including the row that already had one, not just the empty one.
    assert descriptions == {
        1: "A fake generated description.",
        2: "A fake generated description.",
    }

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["description"] == "A fake generated description."
    assert rows[1]["description"] == "A fake generated description."


def test_build_vector_store_generates_when_cache_missing(tmp_path):
    songs = [{
        "id": 1, "title": "Test Song", "artist": "Test Artist",
        "genre": "pop", "mood": "happy", "energy": 0.8,
        "tempo_bpm": 120, "valence": 0.7, "danceability": 0.8,
        "acousticness": 0.2, "description": "A fake generated description.",
    }]
    cache_path = tmp_path / "song_embeddings.json"
    assert not cache_path.exists()  # descriptions exist, but no embeddings cache yet

    store = build_vector_store(songs, FakeGeminiClient(), cache_path=str(cache_path))

    assert len(store) == 1
    assert cache_path.exists()

    with open(cache_path, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved == [{"id": 1, "embedding": [0.1, 0.2, 0.3]}]


def test_build_vector_store_regenerates_when_forced(tmp_path):
    songs = [{
        "id": 1, "title": "Test Song", "artist": "Test Artist",
        "genre": "pop", "mood": "happy", "energy": 0.8,
        "tempo_bpm": 120, "valence": 0.7, "danceability": 0.8,
        "acousticness": 0.2, "description": "A fake generated description.",
    }]
    cache_path = tmp_path / "song_embeddings.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump([{"id": 1, "embedding": [9.9, 9.9, 9.9]}], f)  # stale cached embedding

    store = build_vector_store(songs, FakeGeminiClient(), cache_path=str(cache_path), force=True)

    assert len(store) == 1

    with open(cache_path, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved == [{"id": 1, "embedding": [0.1, 0.2, 0.3]}]


def test_semantic_recommend_returns_closest_match():
    songs = [
        {
            "id": 1, "title": "Close Match", "artist": "Artist A",
            "genre": "pop", "mood": "happy", "energy": 0.8,
            "tempo_bpm": 120, "valence": 0.7, "danceability": 0.8,
            "acousticness": 0.2, "description": "A fake generated description.",
        },
        {
            "id": 2, "title": "Far Match", "artist": "Artist B",
            "genre": "rock", "mood": "intense", "energy": 0.9,
            "tempo_bpm": 150, "valence": 0.4, "danceability": 0.5,
            "acousticness": 0.1, "description": "A fake generated description.",
        },
    ]

    store = VectorStore()
    store.add(1, [1.0, 0.0, 0.0])  # same direction as the query embedding below
    store.add(2, [0.0, 1.0, 0.0])  # orthogonal to the query embedding

    results = list(semantic_recommend("upbeat happy pop song", songs, store, FakeQueryClient(), k=2))

    assert len(results) == 2

    top_song, top_score, top_explanation = results[0]
    assert top_song["id"] == 1
    assert top_score == pytest.approx(1.0)
    assert top_explanation == "This song is the closest semantic match to the query."

    second_song, second_score, _ = results[1]
    assert second_song["id"] == 2
    assert second_score == pytest.approx(0.0)


def test_semantic_recommend_with_k_larger_than_available_songs():
    songs = [
        {
            "id": 1, "title": "Song One", "artist": "Artist A",
            "genre": "pop", "mood": "happy", "energy": 0.8,
            "tempo_bpm": 120, "valence": 0.7, "danceability": 0.8,
            "acousticness": 0.2, "description": "A fake generated description.",
        },
        {
            "id": 2, "title": "Song Two", "artist": "Artist B",
            "genre": "rock", "mood": "intense", "energy": 0.9,
            "tempo_bpm": 150, "valence": 0.4, "danceability": 0.5,
            "acousticness": 0.1, "description": "A fake generated description.",
        },
    ]

    store = VectorStore()
    store.add(1, [1.0, 0.0, 0.0])
    store.add(2, [0.0, 1.0, 0.0])

    results = list(semantic_recommend("upbeat happy pop song", songs, store, FakeQueryClient(), k=5))

    # Only 2 songs exist even though k=5 was requested -- no error, no padding.
    assert len(results) == 2
    assert {song["id"] for song, _, _ in results} == {1, 2}
