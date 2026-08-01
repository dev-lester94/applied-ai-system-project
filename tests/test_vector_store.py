import math

import pytest

from vector_store import VectorStore


def test_vector_store_search_returns_top_k_closest():
    store = VectorStore()
    query = [1.0, 0.0]

    for i in range(10):
        # id 1 sits at 0 degrees (identical to the query), id 10 at 90 degrees (orthogonal).
        angle_degrees = i * 10
        angle_radians = math.radians(angle_degrees)
        vector = [math.cos(angle_radians), math.sin(angle_radians)]
        store.add(i + 1, vector)

    results = store.search(query, top_k=5)

    assert len(results) == 5
    assert [song_id for song_id, _ in results] == [1, 2, 3, 4, 5]

    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1.0)


def test_vector_store_save_and_load_round_trip(tmp_path):
    store = VectorStore()
    store.add(1, [0.1, 0.2, 0.3])
    store.add(2, [0.4, 0.5, 0.6])
    store.add(3, [0.7, 0.8, 0.9])

    path = tmp_path / "vectors.json"
    store.save(str(path))

    loaded = VectorStore.load(str(path))

    assert len(loaded) == len(store)
    assert loaded._ids == store._ids
    assert loaded._vectors == store._vectors
