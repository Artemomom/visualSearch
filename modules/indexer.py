from typing import Tuple

import nmslib
import numpy as np
import pickle
from sklearn.preprocessing import normalize
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
            return v / np.expand_dims(np.linalg.norm(v, axis=1, ord=2), axis=1)

    def predict(self, embedding: np.array, n_images: int):
        pass

    def load(self, path):
        raise NotImplementedError


class KNN(Index):
    def __init__(self, dimension: int, style_path,content_path, style_coef, content_coef):
        super().__init__(dimension)
        self.coeficients = (style_coef, content_coef)
        self.labels, style_emb = self.preprocess_embeddings(style_path, style_coef)
        _, content_emb = self.preprocess_embeddings(content_path,content_coef)
        embeddings = np.concatenate((style_emb, content_emb), axis=1)


        print("[INFO] Training KNN Model")

        self.neighbors = NearestNeighbors(algorithm="brute", metric="cosine"
        ).fit(embeddings)
        print("[INFO] Index Loaded")

    def predict(self, embedding: np.array, n_images: int):
        res = []
        for v,c in zip(embedding,self.coeficients):
            embedding = normalize(v)*c
            res.append(embedding)
        embedding = np.concatenate(res, axis=1)
        print("[INF] Finding Simillar Images")
        distances, indices = self.neighbors.kneighbors(embedding, n_images)
        idx = np.asarray(indices)
        dist = np.asarray(distances)
        return dist, idx

    def preprocess_embeddings(self,embeddings_path,coefficient):
        with open(embeddings_path, "rb") as f:
            items = pickle.load(f)
            labels, embeddings = zip(*items.items())
        embeddings = normalize(embeddings)
        embeddings = list(map(lambda x: x * coefficient, embeddings))
        return labels, np.asarray(embeddings)


class NmsLibIndex(Index):


    def __init__(self, dimension: int, embeddings_path):
        super().__init__(dimension)
        self.index_params = {
            'M': 64,
            'efConstruction': 512
        }
        self.labels, embeddings = self.load_embeddings(embeddings_path)
        embeddings = list(map(self.l2_normalize, embeddings))

        self.index = nmslib.init(method='hnsw', space='cosinesimil',
                                data_type=nmslib.DataType.DENSE_VECTOR, dtype=nmslib.DistType.FLOAT)
        self.index.addDataPointBatch(data=np.asarray(embeddings))
        self.index.createIndex(self.index_params,print_progress=True)

    def predict(self, embedding: np.array, n_images: int) -> Tuple[np.ndarray, np.ndarray]:
        embeddings = list(map(self.l2_normalize, embedding))
        if len(embeddings)==1:
            embedding = embeddings[0]

        result = self.index.knnQueryBatch(embedding, k=n_images)
        idx, dist = list(map(list, zip(*result)))
        idx = np.asarray(idx)
        dist = np.asarray(dist)
        return dist, idx

    def load(self, path):
        self.index.loadIndex(path)

    def load_embeddings(self,embeddings_path):
        with open(embeddings_path, "rb") as f:
            items = pickle.load(f)
        labels, embeddings = zip(*items.items())
        return labels, embeddings

