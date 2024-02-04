import numpy as np

from .rp_neighbours import *
from .exceptions import *


class KNNClassifier:
    """K-nearest neighbors classifier

    Weighted kNN classifier based on random projection forest.

    Supports different (including custom) smoothing kernels and distance metrics.

    Attributes:
        features: number of features in each sample
        forest: an instance of RPTForest
        classes: an array of labels (integers from 0 to N) corresponding to loaded train points
        classes_count: number of classes used
    """

    def __init__(self, features, classes_count, trees_count, rpt_m):
        """Initializes new classifier

        :param features: number of features in each sample
        :param classes_count: number of classes used
        :param trees_count: number of trees in the forest
        :param rpt_m: maximum number of samples in one leaf of an RP tree
        """

        self.features = features
        self.forest = RPForest(features, trees_count, rpt_m)
        self.classes = None
        self.classes_count = classes_count

    def load(self, points, classes):
        """Loads train data, builds a corresponding forest

        :param points: np.ndarray of train samples
        :param classes: an array of labels (integers from 0 to N) corresponding to loaded train points
        """

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
        """Predict class of given sample

        :param point: target point as np.ndarray
        :param distance: distance metric function (from neighbours.distance)
        :param kernel: smoothing kernel function (from neighbours.kernel)
        :param h: bandwidth
        :return: predicted class
        """

        nearest_point_indexes = self.forest.get_neighbours(point)

        votes = np.zeros(self.classes_count)

        for point_ix in nearest_point_indexes:
            votes[self.classes[point_ix]] += kernel(distance(point, self.forest.get_point(point_ix)) / h)

        return np.argmax(votes)
