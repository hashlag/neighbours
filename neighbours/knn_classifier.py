import numpy as np

from .rp_neighbours import *
from .exceptions import *


class KNNClassifier:
    def __init__(self, features, trees_count, rpt_m):
        self.features = features
        self.forest = RPTForest(features, trees_count, rpt_m)
        self.classes = None

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

    def predict(self, point: np.ndarray):
        pass
