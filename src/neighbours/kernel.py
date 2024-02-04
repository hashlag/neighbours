import numpy as np


def rectangular(x):
    return (np.abs(x) <= 1) / 2


def gaussian(x):
    return 0.3989422804014326779 * np.exp(-2 * x * x)


def epanechnikov(x):
    return (np.abs(x) <= 1) * 0.75 * (1 - (x * x))
