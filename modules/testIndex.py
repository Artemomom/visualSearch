from typing import Tuple

import nmslib
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors


class Index(object):
    def __init__(
            self,
            dimension: int,
    ):
        # metric is a list of numbers for recall@k
        # vectors need to create without shuffle
        self.dimension = dimension

    @staticmethod
    def l2_normalize(v) -> np.array:
        if len(v.shape) == 1:
            # if only one vector
            norm = np.linalg.norm(v)
            return np.asarray(v) / norm
        elif len(v.shape) == 2:
            # if v == matrix
            print("v")
            return v / np.expand_dims(np.linalg.norm(v, axis=1, ord=2), axis=1)

    def predict(self, embedding: np.array, n_images: int):
        pass

    def load(self, path):
        raise NotImplementedError

class KNN(Index):
    def __init__(self, dimension: int, embeddings_path):
        super().__init__(dimension)

        self.labels, embeddings = self.load_embeddings(embeddings_path)
        print("loaded embeddings")
        print("you are here")
        embeddings = list(map(self.l2_normalize, embeddings))

        # print("[INFO] Loading Embeddings")

        print("[INFO] Training KNN Model")

        self.neighbors = NearestNeighbors(n_neighbors=3,algorithm="brute", metric="cosine"
        ).fit(np.array(embeddings))
        print("[INFO] Index Loaded")

    def predict(self, embedding: np.array, n_images: int):
        embeddings = list(map(self.l2_normalize, embedding))
        if len(embeddings) == 1:
            embedding = embeddings[0]
        print("[INF] Finding Simillar Images")
        distances, indices = self.neighbors.kneighbors(embedding, n_images)
        idx = np.asarray(indices)
        dist = np.asarray(distances)
        return dist, idx

    def load_embeddings(self,embeddings_path):
        with open(embeddings_path, "rb") as f:
            items = pickle.load(f)
        labels, embeddings = zip(*items.items())
        return labels, embeddings

if __name__=="__main__":
    index = KNN(128, "/Users/aivashchenko/Documents/floralFiles/model/imgs2style.pickle")