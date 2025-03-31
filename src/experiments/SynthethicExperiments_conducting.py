import pandas as pd
import numpy as np
np.random.seed(42)

from typing import List
import itertools

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# This script generates synthetic data for a binary classification problem using the specified parameters.
from src.data_help.synthetic_data_generator import generate_synthetic_data
from src.LogRegCCD import LogRegCCD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, roc_curve, auc

def explore_parameters(lambda_vals:List[float],
                    p:List[float] = [0.5],
                    n:List[int] = [500],
                    d:List[int] = [10],
                    g:List[float] = [0.5],
                    num_repeats:int = 1,
                    save_plots = False
                    ) -> None:
    """
    Explore combinations of parameters for synthetic data experiments
    """
    results_dataframe = None
    sklearn_results_dataframe = None
    nr_of_combinations = len(p) * len(n) * len(d) * len(g) * num_repeats
    i = 1
    # Iterate over all combinations of parameters
    for p_val, n_val, d_val, g_val in tqdm(itertools.product(p, n, d, g), desc="Exploring parameters"):
        for repeat in range(num_repeats):
            print(f"\nCurrent iteration nr: {i}/{nr_of_combinations}")
            i += 1
            print(f"p: {p_val}, n: {n_val}, d: {d_val}, g: {g_val}")
            print("---" * 10)
            X, y = generate_synthetic_data(p_val, n_val, d_val, g_val)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2871)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            log_reg = LogRegCCD(lambda_vals=lambda_vals, max_iter=100)
            log_reg.fit(X_train, y_train)
            
            if save_plots:
                img_folder = "results/synthetic_data_results/imgs"
                if not os.path.exists(img_folder):
                    os.makedirs(img_folder)
                filename = f"{img_folder}/p_{p_val}_n_{n_val}_d_{d_val}_g_{g_val}.png"
                log_reg.validate_and_plot_all(X_test, y_test, save_path=filename)
            else:
                log_reg.validate_all(X_test, y_test)
                # Find best lambda value
                best_lambda_index = np.argmax(log_reg.results["roc_auc"])
                log_reg.results = log_reg.results.iloc[[best_lambda_index]]
            # Drop values below from results dataframe
            # "betas": betas,
            # "intercepts": intercepts,
            # "log_likelihoods": log_likelihoods,
            log_reg.results.drop(columns=["betas", "intercepts", "log_likelihoods"], inplace=True)
            log_reg.results["p"] = p_val
            log_reg.results["n"] = n_val
            log_reg.results["d"] = d_val
            log_reg.results["g"] = g_val

            if results_dataframe is None:
                results_dataframe = log_reg.results
            else:
                results_dataframe = pd.concat([results_dataframe, log_reg.results], ignore_index=True)

            log_reg_sklearn = LogisticRegression(penalty='l1', solver='liblinear', max_iter=100, random_state=42)
            log_reg_sklearn.fit(X_train, y_train)

            sklearn_results = pd.DataFrame({
                "Betas": [log_reg_sklearn.coef_],
                "Intercept": [log_reg_sklearn.intercept_],
                "Iterations": [log_reg_sklearn.n_iter_],
                "Balanced Accuracy": [balanced_accuracy_score(y_test, log_reg_sklearn.predict(X_test))],
                "Roc_Auc": [roc_auc_score(y_test, log_reg_sklearn.predict(X_test))],
                "p": [p_val],
                "n": [n_val],
                "d": [d_val],
                "g": [g_val]
            })

            if sklearn_results_dataframe is None:
                sklearn_results_dataframe = sklearn_results
            else:
                sklearn_results_dataframe = pd.concat([sklearn_results_dataframe, sklearn_results], ignore_index=True)


    # Save results to CSV
    results_folder = "results/synthetic_data_results/LogRegCCD"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    results_dataframe.to_csv(f"{results_folder}/logregCDD.csv", index=False)

    sklearn_results_folder = "results/synthetic_data_results/LogRegSklearn"
    if not os.path.exists(sklearn_results_folder):
        os.makedirs(sklearn_results_folder)
    sklearn_results_dataframe.to_csv(f"{sklearn_results_folder}/logresSklearn.csv", index=False)

if __name__ == "__main__":
    # Define the parameters for the synthetic data generation
    lambda_vals = np.logspace(-3, 1, 10)  # Regularization parameters for LogRegCCD
    # p = np.arange(0.1, 1.1, 0.2).tolist()  # Proportion of positive samples
    # n = np.arange(100, 1001, 100).tolist()  # Number of samples
    # d = [1, 2, 8, 16]  # Number of features
    # g = np.arange(-1.0, 1.1, 0.25).tolist()  # Correlation between features

    # High dimensionality
    p = [0.5]
    n = [50]
    d = [100, 200, 300, 400, 500]
    g = [0, 1]

    # Imbalanced classes
    # p = [0.05, 0.5]
    # n = [1000]
    # d = [32]
    # g = [0, 1]
    
    # Explore the parameter space
    explore_parameters(lambda_vals, p, n, d, g, num_repeats=50, save_plots=False)

                    

