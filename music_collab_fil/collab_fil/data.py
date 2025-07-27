"""This module features functions and classes to manipulate data for the
collaborative filtering algorithm.
"""

from pathlib import Path
import scipy
import pandas as pd
from typing import Optional

def load_user_artists(user_artists_file: Path):
    """Load the user artists file and return a user-artists matrix in csr format, along with user and artist id mappings."""
    user_artists = pd.read_csv(user_artists_file, sep="\t")
    user_ids = user_artists['userID'].unique()
    artist_ids = user_artists['artistID'].unique()
    user_id_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
    artist_id_map = {artist_id: idx for idx, artist_id in enumerate(artist_ids)}
    user_artists['user_index'] = user_artists['userID'].apply(lambda x: user_id_map[x])
    user_artists['artist_index'] = user_artists['artistID'].apply(lambda x: artist_id_map[x])
    coo = scipy.sparse.coo_matrix(
        (
            user_artists.weight.astype(float),
            (
                user_artists['user_index'],
                user_artists['artist_index'],
            ),
        )
    )
    return coo.tocsr(), user_id_map, artist_id_map

class ArtistRetriever:
    """The ArtistRetriever class gets the artist name from the artist ID."""
    def __init__(self):
        self._artists_df: Optional[pd.DataFrame] = None
    def get_artist_name_from_id(self, artist_id: int) -> str:
        if self._artists_df is None:
            raise ValueError("Artists data not loaded. Call load_artists() first.")
        return self._artists_df.loc[artist_id, "name"]
    def load_artists(self, artists_file: Path) -> None:
        artists_df = pd.read_csv(artists_file, sep="\t")
        artists_df = artists_df.set_index("id")
        self._artists_df = artists_df

if __name__ == "__main__":
    import os
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
    try:
        user_artists_matrix, user_id_map, artist_id_map = load_user_artists(
            Path(os.path.join(data_dir, "user_artists.dat"))
        )
        print(f"User-artists matrix shape: {user_artists_matrix.shape}")
        print(f"Number of users: {len(user_id_map)}")
        print(f"Number of artists: {len(artist_id_map)}")
    except Exception as e:
        print(f"Error loading user artists: {e}")
    try:
        artist_retriever = ArtistRetriever()
        artist_retriever.load_artists(Path(os.path.join(data_dir, "artists.dat")))
        test_artist_id = 1
        if artist_retriever._artists_df is not None and test_artist_id in artist_retriever._artists_df.index:
            artist = artist_retriever.get_artist_name_from_id(test_artist_id)
            print(f"Artist ID {test_artist_id}: {artist}")
        else:
            if artist_retriever._artists_df is not None:
                first_artist_id = artist_retriever._artists_df.index[0]
                if isinstance(first_artist_id, (int, float)):
                    first_artist_id = int(first_artist_id)
                else:
                    first_artist_id = int(str(first_artist_id))
                artist = artist_retriever.get_artist_name_from_id(first_artist_id)
                print(f"First artist (ID {first_artist_id}): {artist}")
    except Exception as e:
        print(f"Error loading artists: {e}") 