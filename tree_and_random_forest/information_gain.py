import math 
from gini_impurity_and_entropy import gini_impurity, entropy

def variance(labels):
    if len(labels) == 0:
        return 0.0
    mean = sum(labels) / len(labels)
    return sum((l - mean) ** 2 for l in labels) / len(labels)

def information_gain(parent_labels, left_labels, right_labels, criterion = 'gini', task = 'classification'):
    n = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)
    if n_left == 0 or n_right == 0:
        return 0.0

    if task == 'regression' or criterion in ['mse', 'variance']:
        parent_impurity = variance(parent_labels)
        child_impurity = (n_left / n ) * variance(left_labels) + (n_right / n) * variance(right_labels)
    else:
        measure = gini_impurity if criterion == 'gini' else entropy
    parent_impurity = measure(parent_labels)
    child_impurity = (
        (n_left / n) * measure(left_labels) + (n_right / n) * measure(right_labels)
        )
    return parent_impurity - child_impurity