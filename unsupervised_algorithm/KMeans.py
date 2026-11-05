import math 
import random
import pandas as pd

def euclidean_distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a,b)))

def kmeans(data, k, max_iterations=100, seed=42):
    if isinstance(data, pd.DataFrame):
        data = data.values
    random.seed(seed)
    n_features = len(data[0])
    centroids = random.sample(list(data), k)

    for iteration in range(max_iterations):
        clusters  = [[] for _ in range(k)]
        assignments = []

        for point in data:
            distances = [euclidean_distance(point, c) for c in centroids]
            nearest = distances.index(min(distances))
            clusters[nearest].append(point)
            assignments.append(nearest)

        new_centroids = []
        for cluster in clusters:
            if len(cluster) == 0:
                new_centroids.append(random.choice(data))
                continue
            centroid = [
                sum(point[j] for point in cluster) / len(cluster)
                for j in range(n_features)
            ]
            new_centroids.append(centroid)

        if all(
            euclidean_distance(old, new) < 1e-6 for old, new in zip(centroids, new_centroids)
        ):
            print(f'Converged at iteration {iteration + 1}')
            break
        centroids = new_centroids
    return assignments , centroids