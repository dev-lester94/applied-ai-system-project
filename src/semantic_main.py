"""Console app: semantic song search powered by Gemini embeddings."""

from pathlib import Path

from gemini_client import GeminiClient
from recommender import load_songs
from semantic_search import build_descriptions, build_vector_store, semantic_recommend

SONGS_CSV = Path(__file__).resolve().parent.parent / "data" / "songsV2.csv"
EMBEDDINGS_CACHE = Path(__file__).resolve().parent.parent / "data" / "song_embeddings.json"


def main() -> None:
    songs = load_songs(str(SONGS_CSV))
    client = GeminiClient()


    print("Loading song descriptions (generating any that are missing)...")
    build_descriptions(str(SONGS_CSV), client)


    songs = load_songs(str(SONGS_CSV))
    if EMBEDDINGS_CACHE.exists():
        print("Loading song embeddings from cache...")
    else:
        print("Generating song embeddings with Gemini...")
    store = build_vector_store(songs, client, cache_path=str(EMBEDDINGS_CACHE))
    
    print("\nReady. Describe the kind of songs you're looking for.\n")
    
    while True:
        query = input("Enter a query (or 'quit' to exit): ").strip()
        if not query or query.lower() in ("quit", "exit"):
            break
    
        results = semantic_recommend(query, songs, store, client, k=5)
    
        print(f"\nQuery: {query}\n")
        for i, (song, score, explanation) in enumerate(results, start=1):
            print(f"{i}. {song['title']} by {song['artist']} - similarity: {score:.2f}")
            print(f"   Because: {explanation}")
        print()


if __name__ == "__main__":
    main()
