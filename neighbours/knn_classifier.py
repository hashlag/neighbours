import numpy as np

from .rp_neighbours import *
from .exceptions import *


class KNNClassifier:
    def __init__(self, features, classes_count, trees_count, rpt_m):
        self.features = features
        self.forest = RPTForest(features, trees_count, rpt_m)
        self.classes = None
        self.classes_count = classes_count

    def load(self, points, classes):
        if not isinstance(points, np.ndarray):
            raise InvalidType("points should be represented as np.ndarray")

        if not isinstance(classes, np.ndarray) and not isinstance(classes, list):
            raise InvalidType("classes should be represented as np.ndarray or list")

        self.classes = classes

        if points.ndim != 2:
            raise InvalidDimensionError("points array should be two-dimensional")

        if points.shape[1] != self.features:
            raise InvalidDimensionError(
                "invalid number of features in sample (expected {}, got {})".format(self.features, points.shape[1])
            )

        self.forest.load(points)

    def predict(self, point: np.ndarray, distance, kernel, h):
        nearest_point_indexes = self.forest.get_neighbours(point)

        votes = np.zeros(self.classes_count)

        for point_ix in nearest_point_indexes:
            votes[self.classes[point_ix]] += kernel(distance(point, self.forest.get_point(point_ix)) / h)

        return np.argmax(votes)
