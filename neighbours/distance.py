import numpy as np


def euclidean(x: np.ndarray, y: np.ndarray):
    return np.linalg.norm(x - y)


def manhattan(x: np.ndarray, y: np.ndarray):
    return np.sum(np.abs(x - y))


def cosine(x: np.ndarray, y: np.ndarray):
    return 1 - (x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y)))
