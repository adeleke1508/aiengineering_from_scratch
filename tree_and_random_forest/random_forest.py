import math
import numpy as np
from gini_impurity_and_entropy import gini_impurity, entropy 
from information_gain import information_gain
import random
from decision_tree import DecisionTree

class RandomForest:
    def __init(self, n_trees=100,
                max_depth=None,
                min_samples_split=2,
                max_features='sqrt',
                criterion='gini',
                task = 'classification'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_feature = max_features 
        self.criterion = criterion
        self.trees = []
        self.task = task

    def fit(self, X, y):
        n = len(X) 
        for _ in range(self.n_trees):
            indices = [random.randint(0, n - 1) for _ in range(n)]
            X_boot = [X[i] for i in indices] 
            y_boot = [y[i] for i in indices]
            tree = DecisionTree(
                max_depth = self.max_depth,
                min_samples_split = self.min_samples_split,
                max_features = self.max_features,
                criterion = self.criterion,
            )      
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, X):
        all_preds = [tree.preict(X) for tree in self.trees]
        if self.task == 'regression':
            return np.mean(all_preds, axis = 0)
        elif self.task == 'classification':
            predictions = []
            n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
            for i in range(len(n_samples)):
                votes = {}
                for preds in all_preds:
                    v = preds[i]
                    votes[v] = votes.get(v, 0) + 1
                predictions.append(max(votes, key=votes.get))
            return predictions
        else :
            raise ValueError(r"task must be either 'regression' or 'classification'")