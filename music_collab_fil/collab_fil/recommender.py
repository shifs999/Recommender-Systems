"""This module features the ImplicitRecommender class that performs
recommendation using the implicit library.
"""

from pathlib import Path
from typing import Tuple, List
import os
import implicit
import scipy
from collab_fil.data import load_user_artists, ArtistRetriever

class ImplicitRecommender:
    """The ImplicitRecommender class computes recommendations for a given user
    using the implicit library.
    """
    def __init__(self, artist_retriever: ArtistRetriever, implicit_model):
        self.artist_retriever = artist_retriever
        self.implicit_model = implicit_model
    def fit(self, user_artists_matrix: scipy.sparse.csr_matrix) -> None:
        self.implicit_model.fit(user_artists_matrix)
    def recommend(
        self,
        user_index: int,
        user_artists_matrix: scipy.sparse.csr_matrix,
        n: int = 10,
    ) -> Tuple[List[str], List[float]]:
        user_items = user_artists_matrix[user_index]
        artist_ids, scores = self.implicit_model.recommend(
            user_index, user_items, N=n
        )
        artists = []
        valid_scores = []
        for artist_id, score in zip(artist_ids, scores):
            try:
                artist_name = self.artist_retriever.get_artist_name_from_id(artist_id)
                artists.append(artist_name)
                valid_scores.append(score)
            except (KeyError, ValueError):
                continue
        return artists, valid_scores

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
    user_artists, user_id_map, artist_id_map = load_user_artists(Path(os.path.join(data_dir, "user_artists.dat")))
    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path(os.path.join(data_dir, "artists.dat")))
    implicit_model = implicit.als.AlternatingLeastSquares(
        factors=50, iterations=10, regularization=0.01
    )
    recommender = ImplicitRecommender(artist_retriever, implicit_model)
    recommender.fit(user_artists)
    user_id = 2
    if user_id in user_id_map:
        user_index = user_id_map[user_id]
        print(f"User-item matrix shape: {user_artists.shape}")
        print(f"User index for user ID {user_id}: {user_index}")
        artists, scores = recommender.recommend(user_index, user_artists, n=5)
        for artist, score in zip(artists, scores):
            print(f"{artist}: {score}")
    else:
        print(f"User ID {user_id} not found in the dataset.") 