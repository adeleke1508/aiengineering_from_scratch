import heapq

class KDTree:
    def __init__(self, X, indices=None, depth=0):
        self.X = X
        if indices is None:
            indices = list(range(len(X)))

        self.indices = indices
        if not indices:
            self.left = None
            self.right = None
            self.median_idx = None
            return

        self.k_dim = len(X[0])
        self.axis = depth % self.k_dim

        indices_sorted = sorted(indices, key=lambda i: X[i][self.axis])
        mid = len(indices_sorted) // 2

        self.median_idx = indices_sorted[mid]

        self.left = KDTree(X, indices_sorted[:mid], depth + 1) if mid else None
        self.right = KDTree(X, indices_sorted[mid+1:], depth + 1) if mid + 1 < len(indices_sorted) else None

    def query(self, point, k =1):
        best_neighbors = []

        def search(node):
            if node is None or node.median_idx is None:
                return 
            current_point = self.X[node.median_idx]
            dist = sum((q-p) ** 2 for q, p in zip(point, current_point))

            if len(best_neighbors) < k:
                heapq.heappush(best_neighbors , (-dist, node.median_idx))

            diff = point[node.axis] - current_point[node.axis]
            nearer = node.left if diff < 0 else node.right
            further = node.right if diff < 0 else node.left

            search(nearer)

            max_dist_in_heap = -best_neighbors[0][0] if len(best_neighbors) == k else float('inf')
            if len(best_neighbors) < k or (diff ** 2) < max_dist_in_heap:
                search(further)
        search(self)

        results = [(idx, (-neg_dist) ** 0.5) for neg_dist, idx in best_neighbors]
        results.sort(key=lambda x: x[1])
        return results