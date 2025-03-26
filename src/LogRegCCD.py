import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, balanced_accuracy_score


class LogRegCCD:
    def __init__(self, lambda_vals=None, max_iter: int = 100, stop_tol: float = 1e-5):
        """
        Initialize the CCD-based logistic regression model.

        """
        if lambda_vals is None:
            lambda_vals = [0.0, 0.1, 1.0, 10.0]
        self.lambda_vals = lambda_vals
        self.max_iter = max_iter
        self.stop_tol = stop_tol
        self.mse_values = []
        self.accuracy_values = []
        self.best_lambda = None
        self.best_beta = None
        self.best_intercept = None
        self.results = None

    def sigmoid(self, z):
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def soft_threshold(self, rho: float, lambda_val: float):
        if rho < -lambda_val:
            return rho + lambda_val
        elif rho > lambda_val:
            return rho - lambda_val
        else:
            return 0.0

    def compute_log_likelihood(self, X, y, beta, intercept):
        """
        Compute the log-likelihood of the logistic regression model.

        :param X: Feature matrix.
        :param y: Target vector.
        :param beta: Coefficients.
        :return: Log-likelihood value.
        """
        z = intercept + X @ beta
        log_likelihood = np.sum(y * z - np.log(1 + np.exp(z)))
        return log_likelihood

    def compute_mse(self, X, y, beta):
        """
        Compute Mean Squared Error (MSE) for the given model parameters.
        """
        y_pred = self.sigmoid(X @ beta)
        return np.mean((y - y_pred) ** 2)

    def compute_accuracy(self, X, y, beta):
        """
        Compute accuracy for the given model parameters.
        """
        y_pred = (self.sigmoid(X @ beta) >= 0.5).astype(int)
        return np.mean(y_pred == y)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Fit the logistic regression model using Cyclic Coordinate Descent (CCD) for all lambda values.
        """
        results = []
        for lambda_val in self.lambda_vals:
            results_dict = self.fit_lambda(X, y, lambda_val)
            results.append(results_dict)
        self.results = pd.DataFrame(results)

    def fit_lambda(self, X: pd.DataFrame, y: pd.DataFrame, lambda_val: float) -> dict:
        """
        Fit the logistic regression model using Cyclic Coordinate Descent (CCD) for given lambda value.
        """
        n_samples, n_features = X.shape
        beta = np.zeros(n_features)
        intercept = 0.0
        log_likelihoods = []
        betas = []
        intercepts = []
        converged_iter = 0

        for iteration in range(self.max_iter):
            # Compute linear predictor and probabilities
            z = intercept + X @ beta
            p = self.sigmoid(z)

            # IRLS weights and working response
            W = p * (1 - p)
            z_working = z + (y - p) / np.maximum(W, 1e-5)  # Avoid divide-by-zero

            # Precompute weighted residuals
            residual = z_working - intercept - X @ beta

            beta_old = beta.copy()

            # Coordinate-wise updates
            for j in range(n_features):
                # Compute the partial residual
                r_j = residual + X[:, j] * beta[j]

                # Update rule based on weighted least squares
                rho = np.dot(W * X[:, j], r_j)
                z_j = np.dot(W * X[:, j], X[:, j])

                if z_j != 0:
                    beta[j] = self.soft_threshold(rho, lambda_val) / z_j
                else:
                    beta[j] = 0.0

                # Update residual
                residual = r_j - X[:, j] * beta[j]

            # Update intercept (not penalized)
            intercept = np.sum(W * (z_working - X @ beta)) / np.sum(W)

            # Compute log-likelihood (without L1 penalty)
            log_likelihood = self.compute_log_likelihood(X, y, beta, intercept)

            # Print log-likelihood and coefficients
            print(f"Iteration {iteration + 1}")
            print(f"Log-Likelihood: {log_likelihood:.6f}")
            print(f"Intercept: {intercept:.6f}")
            print(f"Coefficients: {beta}\n")
            log_likelihoods.append(log_likelihood)
            betas.append(beta.copy())
            intercepts.append(intercept)

            # Convergence check
            if np.linalg.norm(beta - beta_old, ord=1) < self.stop_tol:
                print(f'Converged at iteration {iteration}')
                converged_iter = iteration
                break

        results = {
            "lambda": lambda_val,
            "betas": betas,
            "intercepts": intercepts,
            "log_likelihoods": log_likelihoods,
            "beta": beta,
            "intercept": intercept,
            "converged_iter": converged_iter
        }
        return results

    def predict_proba_best(self, X_test):
        """
        Predict probability estimates.

        :param X_test: Test feature matrix.
        :return: Predicted probabilities.
        """
        z = self.best_intercept + X_test @ self.best_beta
        probs = self.sigmoid(z)
        return probs

    def predict_proba(self, X_test):
        # Extract all intercepts and coefficients into NumPy arrays
        intercepts = self.results["intercept"].values
        betas = np.stack(self.results["beta"].values)  # Shape: (n_lambdas, n_features)

        # Compute probabilities
        z = intercepts + X_test @ betas.T  # Shape: (n_samples, n_lambdas)
        probs = self.sigmoid(z)  # Apply sigmoid
        return probs

    @staticmethod
    def calculate_measure_value(y_test, y_pred_proba, measure):
        """
        Calculate the evaluation measure for the given test data.

        :param y_test: Test labels.
        :param y_pred_proba: Predicted probabilities.
        :param measure: Evaluation measure ("roc_auc", "precision", "recall", "f1", "balanced_accuracy").
        :return: Computed evaluation score.
        """
        y_pred = (y_pred_proba >= 0.5).astype(int)

        if measure == "roc_auc":
            return roc_auc_score(y_test, y_pred_proba)
        elif measure in ["precision", "recall", "f1"]:
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary",
                                                                       zero_division=1.0)
            return locals()[measure]
        elif measure == "balanced_accuracy":
            return balanced_accuracy_score(y_test, y_pred)
        else:
            raise ValueError("Invalid evaluation measure.")

    def validate(self, X_valid, y_valid, measure="precision"):
        """
        Validate the model using the given evaluation measure.

        :param X_valid: Validation feature matrix.
        :param y_valid: Validation labels.
        :param measure: Evaluation measure ("roc_auc", "precision", "recall", "f1", "balanced_accuracy").
        :return: Computed evaluation score.
        """
        probs = self.predict_proba(X_valid)
        # Compute the measure for all lambdas in a vectorized way
        scores = [self.calculate_measure_value(y_valid, probs[:, i], measure) for i in range(len(self.lambda_vals))]
        self.results[measure] = scores  # Store the measure in the results DataFrame

        # Find the best lambda based on the evaluation measure
        best_idx = np.argmax(scores)
        best_result = self.results.iloc[best_idx]
        self.best_lambda = best_result["lambda"]
        self.best_beta = best_result["beta"]
        self.best_intercept = best_result["intercept"]
        print(f"Best lambda: {self.best_lambda}\n"
              f"Best {measure}: {scores[best_idx]}\n"
              f"Best coefficients: {self.best_beta}\n"
              f"Best intercept: {self.best_intercept}")

    def plot_score(self, measure):
        """
        Plot how the evaluation measure changes with lambda.

        :param measure: Evaluation measure to plot.
        """
        scores = self.results[measure].values

        # Plot the measure against lambda (log scale for lambda)
        plt.plot(self.lambda_vals, scores, marker='o', linestyle='-', label=measure, color='b')

        # Formatting
        plt.xscale("log")  # Log scale for lambda
        plt.xlabel("Lambda (log scale)")
        plt.ylabel(measure.capitalize())
        plt.title(f"{measure.capitalize()} vs. Lambda")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Show the plot
        plt.show()

    def plot_coeff(self):
        """
        Plot the coefficient values as a function of lambda.
        """
        plt.figure(figsize=(12, 6))
        coef_matrix = np.column_stack([
            self.results["intercept"].values,  # Intercept as first column
            np.vstack(self.results["beta"].values)  # Feature coefficients
        ])
        n_coeffs = coef_matrix.shape[1]
        labels = ["Intercept"] + [f"β{j}" for j in range(1, n_coeffs)]

        # Plot each coefficient path
        for j in range(n_coeffs):
            plt.plot(self.lambda_vals, coef_matrix[:, j], marker='o', linestyle='-', label=labels[j])

        # Formatting
        plt.xscale("log")  # Log scale for lambda
        plt.xlabel("Lambda (log scale)")
        plt.ylabel("Coefficient values")
        plt.title("Coefficients for Different Lambda Values")
        plt.axhline(0, color="black", linestyle="--", linewidth=1)  # Zero line
        plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, title="Coefficients")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout(rect=[0, 0, 0.75, 1])

        # Show the plot
        plt.show()

    def plot_likelihoods(self):
        """
        Plot likelihood function values depending on iteration.
        """
        plt.figure(figsize=(12, 6))
        for i, row in self.results.iterrows():
            lambd = row["lambda"]
            likelihoods = row["log_likelihoods"]
            plt.plot(range(1, len(likelihoods) + 1), likelihoods, marker='o', linestyle='-', label=f"λ={lambd}")

        # Formatting
        plt.xlabel("Iteration")
        plt.ylabel("Log-Likelihood")
        plt.title("Log-Likelihood Convergence for Different Lambda Values")
        plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, title="Lambda")
        plt.grid(True, linestyle="--", linewidth=0.5)

        # Adjust layout to fit legend outside
        plt.tight_layout(rect=[0, 0, 0.75, 1])

        # Show the plot
        plt.show()
