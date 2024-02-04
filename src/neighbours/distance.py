import numpy as np


def euclidean(x: np.ndarray, y: np.ndarray):
    """Calculate the Euclidean distance between two vectors

    :param x: vector as np.ndarray
    :param y: vector as np.ndarray
    :return: Euclidean distance between two vectors
    """

    return np.linalg.norm(x - y)


def manhattan(x: np.ndarray, y: np.ndarray):
    """Calculate the Manhattan distance between two vectors

    :param x: vector as np.ndarray
    :param y: vector as np.ndarray
    :return: Manhattan distance between two vectors
    """

    return np.sum(np.abs(x - y))
