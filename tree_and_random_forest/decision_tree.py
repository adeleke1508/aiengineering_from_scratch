import math
from gini_impurity_and_entropy import gini_impurity, entropy 
from information_gain import information_gain
import random

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, criterion='gini', 
                 max_features=None, task='classification'):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.criterion = criterion 
        self.max_features = max_features
        self.task = task
        self.tree = None
        self.feature_importances = None

    def fit(self, X, y):
        self.n_features = len(X[0])
        self.feature_importances = [0.0] * self.n_features 
        self.n_samples = len(X)
        self.tree = self._build(X, y, depth=0)   
        total = sum(self.feature_importances)
        if total > 0 :
            self.feature_importances = [fi / total for fi in self.feature_impoirtances]

    def predict(self, X):
        return [self._predict_one(X, self.tree) for x in X]

    def _build(self, X, y, depth):
        if len(set(y)) ==  1:
            return {'leaf': True, 'value': y[0]}

        if self.max_depth is not None and depth >= self.max_depth:
            return self._make_leaf(y) 

        if len(y) < self.min_samples_split:
            return self._make_leaf(y)

        best_feature, best_threshold, best_gain = self._best_split(X, y)

        if best_feature is None or best_gain <=0:
            return self._make_leaf(y)
        left_X, left_y, right_X, right_y = self._split_data(X, y, best_feature, best_threshold)
        if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
            return self._make_leaf(y)

        weight = len(y) / self.n_samples
        self.feature_importances[best_feature] += weight * best_gain

        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self._build(left_X, left_y, depth + 1),
            'right': self._build(right_X, right_y, depth + 1),
        }
    def _make_leaf(self, y):
        if self.task == 'regression':
            return {'leaf':True, 'value':sum(y) /len(y)}
        else:
            counts = {}
            for label in y:
                counts[label] = counts.get(label, 0) + 1
            return {'leaf': True, 'value':max(counts, key = counts.get)}

    def _best_split(self, X, y):
        best_feature = None 
        best_threshold = None 
        best_gain = -1.0

        if self.max_features == 'sqrt':
            k = max(1, int(math.sqrt(self.n_features))) 
            feature_indices = random.sample(range(self.n_features), k)
        elif isinstance(self.max_features, int):
            if self.max_features < 1:
                raise ValueError('max_features must be at least when given as an integer')
            k = min(self.max_features, self.n_features)
            feature_indices = random.sample(range(self.n_features), k)
        else:
            feature_indices = list(range(self.n_features))

        for feature_idx in feature_indices:
            values = sorted(set(X[i][feature_idx] for i in range(len(X))))
            if len(values) <= 1:
                continue

            for i in range(len(values) - 1):
                threshold = (values[i] + values[i+1]) / 2.0
                left_y = [y[j] for j in range(len(X)) if X[j][feature_idx] <=threshold]
                right_y = [y[j] for j in range(len(X)) if X[j][feature_idx] > threshold]

                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf: 
                    continue

                gain = information_gain(y, left_y, right_y, self.criterion, self.task)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _split_data(self, X, y, feature, threshold):
        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(len(X)):
            if X[i][feature] <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])

        return left_X, left_y, right_X, right_y

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['value']
        if [node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])