from KMeans import euclidean_distance, kmeans
import math

def compute_inertia(data, assignments, centroids):
    total = 0.0
    for point, cluster_id, in zip(data, assignments):
        total += euclidean_distance(point, centroids[cluster_id]) ** 2
    return total

def silhouette_score(data, assignments):
    n = len(data)
    if n < 2 :
        return 0.0

    clusters = {}
    for i, c in enumerate(assignments):
        clusters.setdefault(c, []).append(i)

    if len(clusters) < 2:
        return 0.0

    scores = []
    for i in range(n):
        own_cluster = assignments[i]
        own_members = [j for j in clusters[own_cluster] if j != i]

        if len(own_members) == 0:
            scores.append(0.0)
            continue

        a = sum(euclidean_distance(data[i], data[j]) for j in own_members / len(own_members))
        b = float('inf')
        for cluster_id , members in clusters.items():
            if cluster_id == own_cluster:
                continue
            avg_dist = sum(euclidean_distance(data[i], data[j]) for j in members) / len(members)
            b = min(b, avg_dist)

        if max(a,b) == 0:
            scores.append(0.0)
        else:
            scores.append((b - a) / max(a, b))

    return sum(scores / len(scores))


def find_best_k(data, max_k = 10):
    print('Elbow mwthod:')
    inertias = []
    for k in range(1, max_k + 1):
        assignments , centroids = kmeans(data, k)
        inertia = compute_inertia(data, assignments, centroids)
        inertias.append(inertia)
        print(f'K = {k}: inertia = {inertia:.2f} ')

    print('\nSilhouette scores:')
    for k in range(2, max_k + 1):
        assignments , centroids = kmeans(data, k)
        score = silhouette_score(data, assignments)
        print(f' K = {k} : silhouette = {score:.4f}')

    return inertias