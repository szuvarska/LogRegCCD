import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

def generate_synthetic_data(p:float, n:int, d:int, g:float):
    """
    Generate a synthetic dataset based on the given parameters.

    Parameters:
        p (float): Prior probability for Y=1 in Bernoulli distribution.
        n (int): Number of observations to generate.
        d (int): Dimensionality of the feature vector X.
        g (float): Parameter controlling covariance decay.

    Returns:
        X (numpy.ndarray): Generated feature vectors of shape (n, d).
        Y (numpy.ndarray): Generated class labels of shape (n,).
    """
    Y = np.random.binomial(1, p, size=n)

    if abs(g) > 1:
        raise ValueError("g must be less than or equal to 1.")

    mean_0 = np.zeros(d)
    mean_1 = np.array([1 / (i + 1) for i in range(d)])

    S = np.fromfunction(lambda i, j: g ** np.abs(i - j), (d, d), dtype=int)

    X = np.array([
        np.random.multivariate_normal(mean_1 if y == 1 else mean_0, S) for y in Y])

    return X, Y


