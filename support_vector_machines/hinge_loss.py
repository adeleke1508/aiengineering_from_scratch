import math 
import numpy as np

def hinge_loss(X, y, w, b):
    n = len(X)
    total_loss = 0.0
    for i in range(n):
        margin = y[i] *(np.dot(w, X[i]) + b )
        total_loss += max(0.0, 1.0 - margin)
    return total_loss / n