import neighbours as ns

import numpy as np
import pandas as pd

df = pd.read_csv("IRIS.csv")
df["species"] = pd.factorize(df["species"])[0]

iris = np.array(df)
np.random.shuffle(iris)

test = iris[:15]
train = iris[15:]

class_labels = train[:, 4].astype(int)
train = np.delete(train, (4,), axis=1)

classifier = ns.KNNClassifier(4, 3, 10, 7)
classifier.load(train, class_labels)

correct_predictions = 0

for sample in test:
    predicted_class = classifier.predict(sample[:4], ns.distance.manhattan, ns.kernel.epanechnikov, 3)
    correct_predictions += predicted_class == sample[4]

print("accuracy: ", correct_predictions / len(test))
