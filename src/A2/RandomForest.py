from operator import index

import numpy as np
import pandas as pd
from collections import Counter


# Gini Impurity Function
def gini_impurity(y):
    classes = np.unique(y)
    gini = 1
    for cls in classes:
        prob = np.sum(y == cls) / len(y)
        gini -= prob ** 2
    return gini


# Information Gain Function (based on Gini Impurity)
def split(x, y):
    m, n = x.shape
    best_gini = float('inf')
    best_split = None
    best_left_y = None
    best_right_y = None

    for feature_index in range(n):
        thresholds = np.unique(x[:, feature_index])
        for threshold in thresholds:
            left_mask = x[:, feature_index] <= threshold
            right_mask = ~left_mask

            left_y = y[left_mask]
            right_y = y[right_mask]

            if len(left_y) == 0 or len(right_y) == 0:
                continue

            gini_left = gini_impurity(left_y)
            gini_right = gini_impurity(right_y)

            gini = (len(left_y) / m) * gini_left + (len(right_y) / m) * gini_right

            if gini < best_gini:
                best_gini = gini
                best_split = (feature_index, threshold)
                best_left_y = left_y
                best_right_y = right_y

    return best_split, best_left_y, best_right_y


# Decision Tree Implementation
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, x, y, depth=0):
        if type(x) is pd.DataFrame:
            x = x.values
        if type(y) is pd.DataFrame:
            y = y.values.squeeze()

        if len(np.unique(y)) == 1:
            self.tree = np.unique(y)[0]
            return self.tree

        if self.max_depth and depth >= self.max_depth:
            self.tree = Counter(y).most_common(1)[0][0]
            return self.tree

        best_split, left_y, right_y = split(x, y)

        if best_split is None:
            self.tree = Counter(y).most_common(1)[0][0]
            return self.tree

        feature_index, threshold = best_split
        left_mask = x[:, feature_index] <= threshold
        right_mask = ~left_mask

        self.tree = {
            'feature_index': feature_index,
            'threshold': threshold,
            'left': DecisionTree(self.max_depth).fit(x[left_mask], left_y, depth + 1),
            'right': DecisionTree(self.max_depth).fit(x[right_mask], right_y, depth + 1)
        }
        return self

    def predict(self, x, indexes=None):
        if isinstance(x, pd.DataFrame):
            x = x.values  # Convert DataFrame to ndarray for compatibility

        if isinstance(self.tree, dict):  # Tree node is a dictionary
            if indexes is None:
                indexes = np.arange(x.shape[0])  # Generate default indexes

            feature_index = self.tree['feature_index']
            threshold = self.tree['threshold']
            left_node = self.tree['left']
            right_node = self.tree['right']

            # Split data based on the threshold
            left_mask = x[:, feature_index] <= threshold
            right_mask = ~left_mask  # Inverse of left_mask

            left_x, right_x = x[left_mask], x[right_mask]
            left_indexes, right_indexes = indexes[left_mask], indexes[right_mask]

            # Predict for left and right nodes
            left_pred = self._predict_subtree(left_x, left_node, left_indexes)
            right_pred = self._predict_subtree(right_x, right_node, right_indexes)

            # Concatenate predictions
            if left_pred is not None and right_pred is not None:
                pred = np.concatenate((left_pred, right_pred))
                pred_index = np.concatenate((left_indexes, right_indexes))
                sorted_indices = np.argsort(pred_index)
                return pred[sorted_indices]
            return left_pred if left_pred is not None else right_pred

        else:  # Base case: the tree is a leaf node
            return self.tree

    @staticmethod
    def _predict_subtree(x_subtree, node, indexes_subtree):
        if len(x_subtree) > 0:
            if isinstance(node, DecisionTree):
                return node.predict(x_subtree, indexes_subtree)
            else:
                return np.full(x_subtree.shape[0], node)  # Return constant value
        return None


# Random Forest Class Implementation
class RandomForest:
    def __init__(self, n_trees, max_depth):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = None
        self.trees = []

    def fit(self, x, y):
        if type(x) is pd.DataFrame:
            x = x.values
        if type(y) is pd.DataFrame:
            y = y.values.squeeze()

        m, n = x.shape
        self.max_features = int(np.sqrt(n))
        for _ in range(self.n_trees):
            # Bootstrap sampling
            bootstrap_idx = np.random.choice(m, m, replace=True)
            X_sample, y_sample = x[bootstrap_idx], y[bootstrap_idx]

            # Feature selection (if max_features is set)
            if self.max_features is not None:
                features_idx = np.random.choice(n, self.max_features, replace=False)
                X_sample = X_sample[:, features_idx]
            else:
                features_idx = np.arange(n)

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append((tree, features_idx))

    def predict(self, x):
        if type(x) is pd.DataFrame:
            x = x.values

        predictions = np.stack([tree.predict(x[:, features_idx]) for (tree, features_idx) in self.trees])
        return np.array([Counter(predictions[:, i].tolist()).most_common(1)[0][0] for i in range(predictions.shape[1])])