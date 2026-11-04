import numpy as np
from distance_functions import l2_distance, l1_distance, cosine_distance, minowski_distance
class KNN:
    def __init__(self, k= 5, distance_fn=l2_distance, weighted=False, task = 'classification'):
        self.k = k
        self.distance_fn = distance_fn
        self.weighted = weighted 
        self.task = task
        self.X_train = None
        self.y_train = None

    def fit(self, X, Y):
        self.X_train =np.asarray(X, dtype=float) 
        self.y_train = np.asarray(Y)
        return self

    def _predict_one(self, X):
        distances = []
        for i, x_train_pt in enumerate(self.X_train):
            dist = self.distance_fn(X, x_train_pt)
            distances.append((dist, self.y_train[i]))

        distances.sort(key=lambda item: item[0])
        k_nearest = distances[:self.k]

        if self.task == 'classification':
            if not self.weighted:
                labels = [label for _, label in k_nearest]
                return max(set(labels), key=labels.count)
            else:
                class_weights = {}
                for dist, label in k_nearest:
                    weight = 1.0 / (dist + 1e-10) 
                    class_weights[label] = class_weights.get(label, 0.0) + weight
                return max(class_weights, key=class_weights.get)
        else: 
            if not self.weighted:
                values = [val for _, val in k_nearest]
                return sum(values) / len(values)
            else:
                total_weight = 0.0
                weighted_sum = 0.0
                for dist, val in k_nearest:
                    weight = 1.0 / (dist + 1e-10)
                    weighted_sum += val * weight
                    total_weight += weight
                return weighted_sum / total_weight if total_weight > 0 else 0.0
    def predict(self, X):
        if hasattr(X, 'to_numpy'):
            X = X.to_numpy()
        elif not isinstance(X, np.ndarray):
            X = np.array(X, dtype=float)
        return [self._predict_one(x) for x in X]