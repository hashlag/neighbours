# Neighbours

Weighted kNN classifier implementation based on random projection forest.

### Classifier description

This project implements weighted kNN classifier with kernel smoothing.

Built-in smoothing kernels:

* rectangular
* gaussian
* epanechnikov

Built-in distance metrics:

* euclidean
* manhattan

Custom distance metrics and smoothing kernels are supported.

### Neighbors search algorithm

Random projection forest is used to search for neighbours.

RP trees are built by recursive binary splits of space by selected hyperplanes.

On each step algorithm chooses two random objects from train set and calculates a hyperplane symmetrically separating these two objects in feature space.
Such hyperplanes become splitting nodes of an RP tree.

Each resulting subset is split again if it contains more than `m` (hyperparameter) objects. 

Hyperplane in each splitting node is represented with an equation of the form
w · x + b = 0, where w is a vector normal to the hyperplane and b is an offset.

For simplicity, we assume what points may be located either "to the right"
or "to the left" of the hyperplane.

We also define points belonging to a hyperplane as located "to the left."

So for any point x we can clearly determine its location:

```
w · x + b > 0 point x is located "to the right"
w · x + b ≤ 0 point x is located "to the left"
```

Since nodes are organized into a tree, we can perform search by evaluating the expression and proceeding to the corresponding child node.

