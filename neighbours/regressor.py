import numpy as np

from .rp_neighbours import *
from .exceptions import *


class KNNRegressor:
    def __init__(self, features, trees_count, rpt_m):
        self.features = features
        self.forest = RPTForest(features, trees_count, rpt_m)
        self.targets = None

    def load(self, points, targets):
        """Loads train data, builds a corresponding forest

        :param points: np.ndarray of train samples
        :param targets: an array of target values corresponding to loaded train points
        """

        if not isinstance(points, np.ndarray):
            raise InvalidType("points should be represented as np.ndarray")

        if not isinstance(targets, np.ndarray) and not isinstance(targets, list):
            raise InvalidType("targets should be represented as np.ndarray or list")

        self.targets = targets

        if points.ndim != 2:
            raise InvalidDimensionError("points array should be two-dimensional")

        if points.shape[1] != self.features:
            raise InvalidDimensionError(
                "invalid number of features in sample (expected {}, got {})".format(self.features, points.shape[1])
            )

        self.forest.load(points)

    def predict(self, point: np.ndarray, distance, kernel, h):
        nearest_point_indexes = self.forest.get_neighbours(point)

        # Nadaraya-Watson estimator

        numerator = float(0)
        denominator = float(0.0000001)

        for point_ix in nearest_point_indexes:
            weight = kernel(distance(point, self.forest.get_point(point_ix)) / h)
            numerator += weight * self.targets[point_ix]
            denominator += weight

        return numerator / denominator
