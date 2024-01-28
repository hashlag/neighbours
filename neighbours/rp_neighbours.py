import numpy as np
import random


class SplittingNode:
    """Hyperplane splitting points into two subsets

    Represents a hyperplane with an equation of
    w · x + b = 0, where w is a vector normal to the hyperplane and b is an offset.

    For simplicity, we assume what points may be located either "to the right"
    or "to the left" of the hyperplane.

    We also define points belonging to a hyperplane as located "to the left".

    So for any point x:
        w · x + b > 0 --> point x is located "to the right"
        w · x + b <= 0 --> point x is located "to the left"

    Attributes:
        w: a vector normal to the hyperplane
        b: an offset
        right: right child node (an instance SplittingNode or LeafNode)
        left: left child node (an instance SplittingNode or LeafNode)
    """

    def __init__(self, w: np.ndarray, b: float, left=None, right=None):
        self.w = w
        self.b = b
        self.right = right
        self.left = left

    def locate(self, point: np.ndarray) -> bool:
        """Gets point location of a point relative to the splitting hyperplane

        :param point: numpy.ndarray representing a point
        :return: True if point is located "to the right", False otherwise
        """

        return self.w.dot(point) + self.b > 0


class LeafNode:
    """Set of points in one particular region of space

    Attributes:
        indexes: set of point indexes
    """

    def __init__(self, ixs):
        """Creates new LeafNode with given set of point indexes

        :param ixs: list of point indexes
        """

        self.indexes = set(ixs)


class RPTForest:
    """Forest of random projection trees

    Attributes:
        features: number of features in each sample
        trees_count: number of trees in the forest
        m: hyperparameter representing the maximum number of points in one region of space
        points: numpy.ndarray of samples
        trees: roots of random projection trees in the forest
    """

    def __init__(self, features, trees_count, m):
        """Creates new RPTForest

        :param features: number of features in each sample
        :param trees_count: number of trees in the forest
        :param m: hyperparameter representing the maximum number of points in one region of space
        """

        self.features = features
        self.trees_count = trees_count
        self.m = m

        self.points = None
        self.trees = []

    def get_point(self, ix):
        """Returns stored point by index

        :param ix: point index
        :return: np.ndarray representing the point
        """

        return self.points[ix]

    def load(self, points: list) -> None:
        """Loads a list of points and builds the corresponding forest

        :param points: list of points
        """

        self.trees.clear()
        self.points = np.array(points)

        ixs = list(range(len(self.points)))

        for _ in range(self.trees_count):
            self.trees.append(self._build_tree(ixs, self.m))

    def find_bucket(self, root, point):
        """Find a set of points from the region of space to which a given point belongs

        :param root: root of a random projection tree to search
        :param point: target point
        :return: set of points located in the same region of space
        """

        while isinstance(root, SplittingNode):
            if root.locate(point):
                root = root.right
            else:
                root = root.left

        return root.indexes

    def get_neighbours(self, point) -> set:
        """Retrieves the nearest neighbors
        of a given point by aggregating data
        from all trees in the random projection forest

        :param point: target point
        :return: set of nearest point indexes
        """

        neighbours = set()

        for tree_root in self.trees:
            neighbours.update(self.find_bucket(tree_root, point))

        return neighbours

    def _split_set(self, s: list):
        """Splits a set of points with randomly selected hyperplane

        :param s: list of indexes of points to split
        :return: a SplittingNode instance
        """

        first_point_ix, second_point_ix = random.sample(range(len(s)), 2)

        first_point = self.get_point(s[first_point_ix])
        second_point = self.get_point(s[second_point_ix])

        normal_vector = []
        for i in range(self.features):
            normal_vector.append(second_point[i] - first_point[i])
        normal_vector = np.array(normal_vector)

        middle_point = []
        for i in range(self.features):
            middle_point.append((first_point[i] + second_point[i]) / 2)
        middle_point = np.array(middle_point)

        b = -normal_vector.dot(middle_point)

        new_node = SplittingNode(normal_vector, b)

        # Here we write indexes of "left" and "right" points to temporary attributes
        # of a SplittingNode. It is needed for tree building algorithm, see _build_tree(s, m).
        # _build_tree(s, m) will delete the attributes after using them.

        new_node._right_ixs = []
        new_node._left_ixs = []

        for point_ix in s:
            point = self.get_point(point_ix)
            if new_node.locate(point):
                new_node._right_ixs.append(point_ix)
            else:
                new_node._left_ixs.append(point_ix)

        return new_node

    def _build_tree(self, s: list, m: int):
        """Builds a random projection tree

        :param s: set of point indexes
        :param m: hyperparameter representing the maximum number of points in one region of space
        :return: root of the constructed tree
        """

        root = self._split_set(s)
        stack = [root]

        while stack:
            node = stack.pop()

            if len(node._right_ixs) > m:
                node.right = self._split_set(node._right_ixs)
                stack.append(node.right)
            else:
                node.right = LeafNode(node._right_ixs)

            del node._right_ixs

            if len(node._left_ixs) > m:
                node.left = self._split_set(node._left_ixs)
                stack.append(node.left)
            else:
                node.left = LeafNode(node._left_ixs)

            del node._left_ixs

        return root
