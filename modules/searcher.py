import joblib

from modules.embedder import Embedder
from modules.indexer import Index
import numpy as np

from modules.timer import Timer


class SimilaritySearcher(object):
    def __init__(self, embedder: Embedder, index: Index):
        self.embedder = embedder
        self.index = index
        self.labels = index.labels

    def search_image(self, image: np.array, n_images: int, timer: Timer):
        """
        if return_classes=True, then func will return labels, else return distances and paths of predicted objects
        """
        with timer.measure('model time'):
            embedding = self.embedder.embed_image(image)

        with timer.measure('knn time'):
            dists, indexes = self.index.predict(embedding, n_images)
        dists, indexes = dists[0], indexes[0]
        return dists, self._get_paths_from_indexes(indexes)

    def _get_paths_from_indexes(self, indexes):
        return [self.labels[k] for k in indexes]
