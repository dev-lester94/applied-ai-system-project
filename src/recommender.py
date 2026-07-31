import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
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

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        user_prefs = asdict(user)
        song_dicts = [asdict(song) for song in self.songs]
        ranked = recommend_songs(user_prefs, song_dicts, k)
        songs_by_id = {song.id: song for song in self.songs}
        return [songs_by_id[song_dict["id"]] for song_dict, _score, _explanation in ranked]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        user_prefs = asdict(user)
        song_dict = asdict(song)
        _score, reasons = score_song(user_prefs, song_dict)
        if not reasons:
            return "This song does not strongly match your preferences."
        return "Recommended because: " + "; ".join(reasons)

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
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

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py

    EXPERIMENT: energy doubled, genre halved (rounded to 15) — see README
    Experiments section. Original weights kept below, commented out.
    """
    reasons = []

    genre_score = 0
    if song["genre"] == user_prefs["favorite_genre"]:
        genre_score = 35  # original weight
        # genre_score = 15   # experimental weight, see README Experiments section
        reasons.append(f"Genre matches your favorite ({song['genre']})")

    mood_score = 0
    if song["mood"] == user_prefs["favorite_mood"]:
        mood_score = 25
        reasons.append(f"Mood matches your favorite ({song['mood']})")

    energy_diff = abs(song["energy"] - user_prefs["target_energy"])
    # Original banding (max 20), commented out for the experiment:
    if energy_diff <= 0.1:
        energy_score = 20
    elif energy_diff <= 0.2:
        energy_score = 15
    elif energy_diff <= 0.3:
        energy_score = 10
    elif energy_diff <= 0.4:
        energy_score = 5
    else:
        energy_score = 0

    #Experiment: energy doubled (max 40), see README Experiments section
    # if energy_diff <= 0.1:
    #     energy_score = 40
    # elif energy_diff <= 0.2:
    #     energy_score = 30
    # elif energy_diff <= 0.3:
    #     energy_score = 20
    # elif energy_diff <= 0.4:
    #     energy_score = 10
    # else:
    #     energy_score = 0
    if energy_score > 0:
        reasons.append(
            f"Energy ({song['energy']:.2f}) is close to your target ({user_prefs['target_energy']:.2f})"
        )

    if user_prefs["likes_acoustic"]:
        acoustic_score = song["acousticness"] * 20
        if acoustic_score > 0:
            reasons.append("Matches your preference for acoustic tracks")
    else:
        acoustic_score = (1 - song["acousticness"]) * 20
        if acoustic_score > 0:
            reasons.append("Matches your preference for non-acoustic tracks")

    total_score = genre_score + mood_score + energy_score + acoustic_score
    return total_score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    scored = [(song, *score_song(user_prefs, song)) for song in songs]
    scored.sort(key=lambda entry: entry[1], reverse=True)
    return [(song, score, "; ".join(reasons)) for song, score, reasons in scored[:k]]
