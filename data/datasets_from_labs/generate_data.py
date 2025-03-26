from typing import Dict, Any, List, Union
import numpy as np


def generate_data(n: int, p: int, distributions: List[Dict]) -> Union[np.ndarray, np.ndarray]:
    """
    Generate data for classification
    :param n: Number of samples to generate
    :param p: Number of features
    :param distributions: List of dictionaries specifying the distribution parameters for each class
    :return: Tuple of feature matrix X and label vector y
    """
    y = np.random.binomial(1, 0.5, size=n)
    X = np.zeros((n, p))
    for i in range(n):
        class_val = y[i]
        dist_params = distributions[class_val]

        if dist_params['type'] == 'normal':
            mean = dist_params['mean']
            var = dist_params['var']
            X[i, :] = np.random.normal(mean, np.sqrt(var), p)

        elif dist_params['type'] == 'multi_normal':
            mean = dist_params['mean']
            var = dist_params['var']
            corr = dist_params['corr']
            cov = np.full((p, p), corr * var)
            np.fill_diagonal(cov, var)
            X[i, :] = np.random.multivariate_normal([mean] * p, cov)

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


def generate_data_2(n: int):
    X_sim = np.random.randn(n, 5)
    beta = np.array([1, 1, 1, 1, 1])
    p = 1 / (1 + np.exp(-(0.5 + X_sim @ beta)))
    y_sim = np.random.binomial(1, p, size=n)
    return X_sim, y_sim
