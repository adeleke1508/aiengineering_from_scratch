import math
import numpy as np

def linear_kernel(x, z):
    return np.dot(x, z)

def polynomial_kernel(x, z, degree=3, c=1.0):
    return (np.dot(x, z) + c) ** degree

def rbf_kernel(x, z, gamma = 0.5):
    diff = [xi - zi for xi, zi in zip(x, z)] 
    return math.exp(-gamma * np.dot(diff, diff))