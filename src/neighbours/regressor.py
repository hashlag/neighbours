import numpy as np

from .rp_neighbours import *
from .exceptions import *


class KNNRegressor:
    """K-nearest neighbors regressor

    Nadaraya-Watson kNN regressor based on random projection forest.

    Supports different (including custom) smoothing kernels and distance metrics.

    Attributes:
        features: number of features in each sample
        forest: an instance of RPTForest
        targets: an array of target values corresponding to loaded train points
    """

    def __init__(self, features, trees_count, rpt_m):
        """Initializes new regressor

        :param features: number of features in each sample
        :param trees_count: number of trees in the forest
        :param rpt_m: maximum number of samples in one leaf of an RP tree
        """

        self.features = features
        self.forest = RPForest(features, trees_count, rpt_m)
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
        """Predict target value for given point

        :param point: target point as np.ndarray
        :param distance: distance metric function (e.g., from neighbours.distance)
        :param kernel: smoothing kernel function (e.g., from neighbours.kernel)
        :param h: bandwidth
        :return: predicted value or numpy.nan if unable to obtain a prediction
        """

        nearest_point_indexes = self.forest.get_neighbours(point)

        # Nadaraya-Watson estimator

        numerator = float(0)
        denominator = float(0)

        for point_ix in nearest_point_indexes:
            weight = kernel(distance(point, self.forest.get_point(point_ix)) / h)
            numerator += weight * self.targets[point_ix]
            denominator += weight

        return np.nan if denominator == 0 else numerator / denominator
