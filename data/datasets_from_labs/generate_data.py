import random
from typing import Dict, Any, List, Union
import numpy as np
import pandas as pd


def generate_data(n: int, p: int, distributions: List[Dict], is_balanced: bool = True) -> Union[np.ndarray, np.ndarray]:
    """
    Generate data for classification
    :param n: Number of samples to generate
    :param p: Number of features
    :param distributions: List of dictionaries specifying the distribution parameters for each class
    :param is_balanced: If True, the classes will be balanced
    :return: Tuple of feature matrix X and label vector y
    """
    if is_balanced:
        y = np.random.binomial(1, 0.5, size=n)
    else:
        y = np.random.binomial(1, 1/len(distributions), size=n)

    X = np.zeros((n, p))
    for i in range(n):
        class_val = y[i]

        if class_val == 0:
            dist_params = distributions[random.randint(1, len(distributions)-1)]
        else:
            dist_params = distributions[class_val-1]

        if dist_params['type'] == 'normal':
            mean = dist_params['mean']
            var = dist_params['var']
            X[i, :] = np.random.normal(mean, np.sqrt(var), p)

        elif dist_params['type'] == 'multi_normal':
            mean = dist_params['mean']
            var = dist_params['var']
            corr = dist_params['corr']
            translation = dist_params['translation'] if 'translation' in dist_params else 0
            cov = np.full((p, p), corr * var)
            np.fill_diagonal(cov, var)
            X[i, :] = np.random.multivariate_normal([mean] * p, cov)
            X[i, 1] = X[i, 1] + translation

    return X, y


def generate_data_scheme_1(a: float) -> Union[np.ndarray, np.ndarray]:
    """
    Generate data for scheme 1
    :param a: Mean value for the second class in the normal distribution
    :return: Tuple of feature matrix X and label vector y
    """
    return generate_data(n=1000, p=2, distributions=[{'type': 'normal', 'mean': 0, 'var': 1},
                                                     {'type': 'normal', 'mean': a, 'var': 1}])


def generate_data_scheme_2(a: float, rho: float) -> Union[np.ndarray, np.ndarray]:
    """
    Generate data for scheme 2
    :param a: Mean value for the second class in the multivariate normal distribution
    :param rho: Correlation coefficient for the multivariate normal distribution
    :return: Tuple of feature matrix X and label vector y
    """
    return generate_data(n=1000, p=2, distributions=[{'type': 'multi_normal', 'mean': 0, 'var': 1, 'corr': rho},
                                                     {'type': 'multi_normal', 'mean': a, 'var': 1, 'corr': -rho}])


# def generate_data_2(n: int):
#     X_sim = np.random.randn(n, 5)
#     beta = np.array([1, 1, 1, 1, 1])
#     p = 1 / (1 + np.exp(-(0.5 + X_sim @ beta)))
#     y_sim = np.random.binomial(1, p, size=n)
#     return X_sim, y_sim
#
# def add_noise(x: pd.DataFrame, noise_prob: float):
#     for col in range(x.shape[1]):
#         for row in range(x.shape[0]):
#             if np.random.binomial(1, noise_prob):
#                 x[row, col] = x[row, col] + np.random.normal()
#     return x


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

    mean_0 = np.zeros(d)
    mean_1 = np.array([1 / (i + 1) for i in range(d)])

    S = np.fromfunction(lambda i, j: g ** np.abs(i - j), (d, d), dtype=int)

    X = np.array([
        np.random.multivariate_normal(mean_1 if y == 1 else mean_0, S) for y in Y])

    return X, Y
