import neighbours as ns

import matplotlib.pyplot as plt
import numpy as np
import random
import math


# function for generating a synthetic regression problem
def f(x):
    if x > 40:
        return math.log(x, 2) - 6
    else:
        return math.cos(x * 0.1)


# generate x coordinates
X = [[i + random.uniform(-1, 1)] for i in np.arange(start=1, stop=100, step=1)]

# calculate corresponding y coordinates
y = [f(i[0]) + random.uniform(-0.1, 0.1) for i in X]

# convert to numpy arrays
X = np.array(X)
y = np.array(y)

# generate x coordinates for demo plot
x_points = np.arange(start=0, stop=100, step=0.1)
X_demo = np.array([[x] for x in x_points])

# create a regressor then load data
regressor = ns.KNNRegressor(1, 10, 7)
regressor.load(X, y)

# create an array to store predicted y values for demo plot
y_predicted = []

# get predictions for all samples in X_demo
for sample in X_demo:
    predicted_value = regressor.predict(sample, ns.distance.euclidean, ns.kernel.gaussian, 3)
    y_predicted.append(predicted_value)

# plot train points
plt.plot(X, y, 'bo')

# plot predicted y against x
plt.plot(x_points, y_predicted, 'r')

plt.show()
