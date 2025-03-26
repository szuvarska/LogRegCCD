import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, balanced_accuracy_score


class LogRegCCD:
    def __init__(self, lambda_vals):
        """
        Initialize the CCD-based logistic regression model.

        :param lambda_vals: List of regularization parameters (lambda) for the L1 penalty.
        """
        self.lambda_vals = lambda_vals
        self.coefficients = []
        self.likelihoods = []
        self.mse_values = []
        self.accuracy_values = []

    def sigmoid(self, z):
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def compute_likelihood(self, X, y, beta):
        """
        Compute the log-likelihood of the logistic regression model.

        :param X: Feature matrix.
        :param y: Target vector.
        :param beta: Coefficients.
        :return: Log-likelihood value.
        """
        p = self.sigmoid(X @ beta)
        return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

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

    def fit(self, X_train, y_train):
        """
        Fit the logistic regression model using Cyclic Coordinate Descent (CCD).

        :param X_train: Training feature matrix.
        :param y_train: Training labels.
        """
        n, d = X_train.shape
        X = np.c_[np.ones(n), X_train]  # Add intercept term
        beta = np.zeros(d + 1)

        for lambda_val in self.lambda_vals:
            likelihoods_per_lambda = []
            coefficients_per_lambda = []
            for iteration in range(100):  # Max iterations
                for j in range(d + 1):
                    residual = y_train - self.sigmoid(X @ beta)
                    rho_j = np.dot(X[:, j], residual)  # Partial residual sum
                    if j == 0:  # Intercept update
                        beta[j] += rho_j / n
                    else:
                        beta_j_new = np.sign(rho_j) * max(abs(rho_j) - lambda_val, 0) / n
                        beta[j] = beta_j_new
                likelihood = self.compute_likelihood(X, y_train, beta)
                mse = self.compute_mse(X, y_train, beta)
                accuracy = self.compute_accuracy(X, y_train, beta)
                print(f"Iteration {iteration + 1}: Likelihood={likelihood:.4f}, MSE={mse:.4f}, Accuracy={accuracy:.4f}")
                likelihoods_per_lambda.append(likelihood)
                coefficients_per_lambda.append(beta.copy())
            self.likelihoods.append(likelihoods_per_lambda)
            self.coefficients.append(coefficients_per_lambda)

    def predict_proba(self, X_test):
        """
        Predict probability estimates.

        :param X_test: Test feature matrix.
        :return: Predicted probabilities.
        """
        X = np.c_[np.ones(X_test.shape[0]), X_test]
        return self.sigmoid(X @ self.coefficients[-1][-1])

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
