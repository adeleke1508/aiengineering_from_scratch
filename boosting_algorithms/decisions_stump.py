import numpy as np
import pandas as pd

class DecisionStump:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None
        self.feature_name = None
        self.columns_ = None
        self.error = None 

    @staticmethod
    def _to_numpy(X):
        return X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    def fit(self, X, y, weights=None):
        if isinstance(X, pd.DataFrame):
            self.columns_ = list(X.columns)
        X_arr = self._to_numpy(X).astype(float)
        y_arr = np.asarray(y)
        n_samples , n_features = X_arr.shape
        if weights is None:
            weights = np.full(n_samples, 1/n_samples)
        weights = np.asarray(weights)

        best_error = float('inf')
        for f in range(n_features):
            order = np.argsort(X_arr[:, f], kind='stable')
            xs, ys, ws = X_arr[order, f], y_arr[order], weights[order]

            wpos , wneg = ws * (ys == 1), ws * (ys == -1)
            cpos, cneg = np.cumsum(wpos), np.cumsum(wneg)
            totpos , totneg = cpos[-1], cneg[-1] 

            uniq, first = np.unique(xs, return_index=True)
            last = np.r_[first[1:] -1 , n_samples - 1]

            err1 = (cpos[first] - wpos[first]) + (totneg - (cneg[first] - wneg[first]))
            err2 = (totpos - cpos[last]) + cneg[last]

            for err, pol in ((err1, 1), (err2, -1)):
                i = np.argmin(err)
                if err[i] < best_error:
                    best_error = err[i]
                    self.feature_idx , self.threshold, self.polarity = f, uniq[i], pol
            '''thresholds = np.unique(X_arr[:, f])
            for thresh in thresholds:
                for polarity in [1, -1]:
                    pred = np.ones(n_samples)
                    pred[polarity * X_arr[:, f] < polarity * thresh] = -1
                    error = np.sum(weights[pred != y_arr])
                    if error < best_error:
                        best_error = error
                        self.feature_idx = f
                        self.threshold = thresh
                        self.polarity = polarity
'''
        self.error = best_error
        if self.columns_ is not None:
            self.feature_name = self.columns_[self.feature_idx]
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame) and self.columns_ is not None:
            X_arr = X[self.columns_]._to_numpy(dtype=float)
        else:
            X_arr = self._to_numpy(X).astype(float)

        n = X_arr.shape[0]
        pred = np.ones(n)
        idx = self.polarity + X_arr[:, self.feature_idx] < self.polarity * self.threshold
        pred[idx] = -1 
        return pred