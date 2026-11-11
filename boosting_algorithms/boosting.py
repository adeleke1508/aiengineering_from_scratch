import numpy as np
import pandas as pd

from decisions_stump import DecisionStump

def _to_numpy(X):
     return X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)

class AdaBoostScratch:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators 
        self.stumps = []
        self.alphas = []
        self.columns_ = None

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.columns_ = list(X.columns)
        X = _to_numpy(X)
        y = np.asarray(y)
        n = X.shape[0]
        weights = np.full(n, 1 /n)
        self.stumps, self.alphas = [], []

        for _ in range(self.n_estimators):
            stump = DecisionStump().fit(X, y, weights)
            pred = stump.predict(X)

            err = np.sum(weights[pred != y])
            err = np.clip(err, 1e-10, 1 - 1e-10)

            alpha = 0.5 * np.log((1 - err) / err)
            weights *= np.exp(-alpha * y * pred)
            weights /= weights.sum()

            stump.alpha = alpha
            self.stumps.append(stump)
            self.alphas.append(alpha)
        return self
    def decision_function(self, X):
        if isinstance(X, pd.DataFrame) and self.columns_ is not None:
            X = X[self.columns_]
        X = _to_numpy(X)
        return sum(a * s.predict(X) for a, s in zip(self.alphas, self.stumps))

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-2 * self.decision_function(X)))
        return np.column_stack([1 - p, p])
'''    def predict(self, X):
        if isinstance(X, pd.DataFrame) and self.columns_ is not None:
            X = X[self.columns_]
            X = _to_numpy(X)
        total = sum(a * s.predict(X) for a, s in zip(self.alphas, self.stumps))
        return np.where(total >= 0, 1, -1)'''

class SimpleRegressionTree:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        X, y = _to_numpy(X), np.asarray(y, dtype=float)
        self.tree = self._build(X, y, 0)
        return self

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.all(y == y[0]):
             return {'value': y.mean()}
        f, t = self._best_split(X, y)
        if f is None:
             return {'value': y.mean()}
        mask = X[:, f] <= t
        return {'f': f, 't':t,
                'left': self._build(X[mask], y[mask], depth + 1),
                'right': self._build(X[~mask], y[~mask], depth + 1)}
    @staticmethod
    def _best_split(X, y):
        n, d = X.shape
        best_score , best  = -np.inf, (None, None)
        for f in range(d):
              order = np.argsort(X[:, f], kind='stable')
              xs, ys = X[order, f], y[order]
              csum = np.cumsum(ys)
              n1 = np.arange(1, n)
              s1 = csum[:-1]
              sr = csum[-1] - s1       
              score = s1**2 / n1 + sr**2 / (n - n1)
              score = np.where(xs[:-1] != xs[1:], score, -np.inf)
              i = np.argmax(score)
              if score[i] > best_score:
                   best_score, best = score[i], (f, (xs[i] + xs[i + 1]) / 2)
        return best

    def _predict_row(self, x, node):
        while 'value' not in node:
            node = node['left'] if x[node['f']] <= node['t'] else node['right']
        return node['value']

    def predict(self, X):
         X = _to_numpy(X)
         return np.array([self.predict_row(x, self.tree_) for x in X])

class GradientBoostingScratch:
    def __init(self, n_estimators=100, learning_rate=0.1, max_depth=3):
            self.n_estimators = n_estimators
            self.lr = learning_rate
            self.max_depth = max_depth
            self.trees = []
            self.initial_pred = None
            self.columns_ = None

    def fit(self, X, y):
            if isinstance(X, pd.DataFrame):
                self.columns_ = list(X.columns)
            X = _to_numpy(X)
            y = np.asarray(u, dtype=float)
            self.initial_pred = np.mean(y)
            current_pred = np.full(len(y), self.initial_pred)
            self.trees = []

            for _ in range(self.n_estimators):
                residuals = y - current_pred
                tree = SimpleRegressionTree(max_depth = self.max_depth).fit(X, residuals)
                update = tree.predict(X)
                current_pred += self.lr * update
                self.trees.append(tree)
            return self

    def predict(self, X):
            if isinstance(X, pd.DataFrame) and self.columns_ is not None:
                X = X[self.columns_]
            X = _to_numpy(X)
            pred = np.full(X.shape[0], self.initial_pred)
            for tree in self.trees:
                pred += self.lr * tree.predict(X)
            return pred
        