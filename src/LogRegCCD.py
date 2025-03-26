import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, balanced_accuracy_score


class LogRegCCD:
    def __init__(self, lambda_val: float, max_iter: int = 100, stop_tol: float = 1e-5):
        """
        Initialize the CCD-based logistic regression model.

        """
        self.lambda_val = lambda_val
        self.max_iter = max_iter
        self.stop_tol = stop_tol
        self.beta = None
        self.intercept = None
        self.coefficients = []
        self.likelihoods = []
        self.mse_values = []
        self.accuracy_values = []

    def sigmoid(self, z):
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def soft_threshold(self, rho):
        if rho < -self.lambda_val:
            return rho + self.lambda_val
        elif rho > self.lambda_val:
            return rho - self.lambda_val
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

    def fit(self, X, y):
        """
        Fit the logistic regression model using Cyclic Coordinate Descent (CCD).
        """
        n_samples, n_features = X.shape
        self.beta = np.zeros(n_features)
        self.intercept = 0.0

        for iteration in range(self.max_iter):
            # Compute linear predictor and probabilities
            z = self.intercept + X @ self.beta
            p = self.sigmoid(z)

            # IRLS weights and working response
            W = p * (1 - p)
            z_working = z + (y - p) / np.maximum(W, 1e-5)  # Avoid divide-by-zero

            # Precompute weighted residuals
            residual = z_working - self.intercept - X @ self.beta

            beta_old = self.beta.copy()

            # Coordinate-wise updates
            for j in range(n_features):
                # Compute the partial residual
                r_j = residual + X[:, j] * self.beta[j]

                # Update rule based on weighted least squares
                rho = np.dot(W * X[:, j], r_j)
                z_j = np.dot(W * X[:, j], X[:, j])

                if z_j != 0:
                    self.beta[j] = self.soft_threshold(rho) / z_j
                else:
                    self.beta[j] = 0.0

                # Update residual
                residual = r_j - X[:, j] * self.beta[j]

            # Update intercept (not penalized)
            self.intercept = np.sum(W * (z_working - X @ self.beta)) / np.sum(W)

            # Compute log-likelihood (without L1 penalty)
            log_likelihood = self.compute_log_likelihood(X, y, self.beta, self.intercept)

            # Print log-likelihood and coefficients
            print(f"Iteration {iteration + 1}")
            print(f"Log-Likelihood: {log_likelihood:.6f}")
            print(f"Intercept: {self.intercept:.6f}")
            print(f"Coefficients: {self.beta}\n")

            # Convergence check
            if np.linalg.norm(self.beta - beta_old, ord=1) < self.stop_tol:
                print(f'Converged at iteration {iteration}')
                break

        # for iteration in range(self.max_iter):  # Max iterations
        #     # Compute linear predictor and probabilities
        #     z = intercept + X @ beta
        #     p = self.sigmoid(z)
        #     for j in range(p + 1):
        #         residual = y_train - self.sigmoid(X @ beta)
        #         rho_j = np.dot(X[:, j], residual)  # Partial residual sum
        #         if j == 0:  # Intercept update
        #             beta[j] += rho_j / n
        #         else:
        #             beta_j_new = np.sign(rho_j) * max(abs(rho_j) - lambda_val, 0) / n
        #             beta[j] = beta_j_new
        #     likelihood = self.compute_likelihood(X, y_train, beta)
        #     mse = self.compute_mse(X, y_train, beta)
        #     accuracy = self.compute_accuracy(X, y_train, beta)
        #     print(f"Iteration {iteration + 1}: Likelihood={likelihood:.4f}, MSE={mse:.4f}, Accuracy={accuracy:.4f}")
        #     likelihoods_per_lambda.append(likelihood)
        #     coefficients_per_lambda.append(beta.copy())
        #     if iteration > 1 and abs(likelihoods_per_lambda[-1] - likelihood) < self.stop_tol:
        #         break

    def predict_proba(self, X):
        """
        Predict probability estimates.

        :param X_test: Test feature matrix.
        :return: Predicted probabilities.
        """
        z = self.intercept + X @ self.beta
        probs = self.sigmoid(z)
        return probs

    def validate(self, X_valid, y_valid, measure="roc_auc"):
        """
        Validate the model using the given evaluation measure.

        :param X_valid: Validation feature matrix.
        :param y_valid: Validation labels.
        :param measure: Evaluation metric ("roc_auc", "precision", "recall", "f1", "balanced_accuracy").
        :return: Computed evaluation score.
        """
        y_pred_proba = self.predict_proba(X_valid)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        if measure == "roc_auc":
            return roc_auc_score(y_valid, y_pred_proba)
        elif measure in ["precision", "recall", "f1"]:
            precision, recall, f1, _ = precision_recall_fscore_support(y_valid, y_pred, average="binary")
            return locals()[measure]
        elif measure == "balanced_accuracy":
            return balanced_accuracy_score(y_valid, y_pred)
        else:
            raise ValueError("Invalid evaluation measure.")

    def plot_score(self, measure):
        """
        Plot how the evaluation measure changes with lambda.

        :param measure: Evaluation measure to plot.
        """
        scores = [self.validate(X_valid, y_valid, measure) for _ in self.lambda_vals]
        plt.plot(self.lambda_vals, scores, marker='o')
        plt.xlabel("Lambda")
        plt.ylabel(measure)
        plt.title(f"{measure} vs. Lambda")
        plt.show()

    def plot_coeff(self):
        """
        Plot the coefficient values as a function of lambda.
        """
        for i, lambda_val in enumerate(self.lambda_vals):
            plt.plot(range(len(self.coefficients[i][-1])), self.coefficients[i][-1], marker='o',
                     label=f'Lambda={lambda_val}')
        plt.xlabel("Coefficient Index")
        plt.ylabel("Coefficient Values")
        plt.title("Coefficient values vs. Lambda")
        plt.legend()
        plt.show()

    def plot_likelihoods(self):
        """
        Plot likelihood function values depending on iteration.
        """
        for i, lambda_val in enumerate(self.lambda_vals):
            plt.plot(range(len(self.likelihoods[i])), self.likelihoods[i], marker='o', label=f'Lambda={lambda_val}')
        plt.xlabel("Iteration")
        plt.ylabel("Log-Likelihood")
        plt.title("Likelihood Function Values over Iterations")
        plt.legend()
        plt.show()
