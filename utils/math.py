from scipy import linalg, stats
import numpy as np


def cos(vec1, vec2):
    norm1 = linalg.norm(vec1)
    norm2 = linalg.norm(vec2)

    return vec1.dot(vec2) / (norm1 * norm2)


def rho(vec1,vec2):
    return stats.stats.spearmanr(vec1, vec2)[0]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)
