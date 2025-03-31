#  Cyclic Coordinate Descent for Regularized Logistic Regression

This repository contains a project that aims to implement the Cyclic Coordinate Descent algorithm for Regularized Logistic Regression. The project is part of the course "Advanced Machine Learning" at Warsaw University of Technology.

# Team members
* Łukasz Grabarski ([@LukaszGrabarski](https://github.com/LukaszGrabarski))
* Łukasz Lepianka ([@Luki308](https://github.com/Luki308))
* Marta Szuwarska ([@szuvarska](https://github.com/szuvarska))

# Algorithm execution guide

This guide provides instructions on how to run the algorithm on the provided dataset.

## Requirements

The project requires Python 3.7 or higher. To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```
## Running the algorithm

### 1. Import the model
    
    ```python
    from src.LogRegCCD import LogRegCCD
    ```

### 2. Load the dataset

    ```python
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Example dataset
    df = pd.read_csv('data/diabetes.csv')
    y = df['Outcome']
    X = df.drop(columns=['Outcome'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

### 3. Initialize and train the model

The model accepts the following parameters:
* `lambda_vals` - list of regularization parameters. The model will be trained for all of them. Default: `[0.01, 0.1, 1.0, 10.0]`.
* `max_iter` - maximum number of iterations. Default: `100`.
* `stop_tol` - tolerance for stopping criterion. Default: `1e-5`.


    ```python
    model = LogRegCCD(lambda_vals=[0.01, 0.1, 1.0, 10.0], max_iter=100, stop_tol=1e-5)
    model.fit(X_train, y_train)  
    ```

### 4. Validate and select the best model

The model can be validated using the `validate` method. The method accepts the following parameters:
* `X_valid` - validation dataset features.
* `y_valid` - validation dataset labels.
* `measure` - evaluation metric. Default: `roc_auc`. Available metrics: `roc_auc`, `pr_auc`, `precision`, `recall`, `f1`, `balanced_accuracy`.

The method returns the best model based on the selected metric.

    ```python
    model.validate(X_valid, y_valid, measure="roc_auc", find_best=True)
    ```

### 5. Predict

The model can be used to make predictions using the `predict_proba` or `predict_proba_best` method. The `predict_proba` method returns the probabilities for all models trained with different regularization parameters. The `predict_proba_best` method returns the probabilities for the best model chosen by `validate` function.

    ```python
    probs = model.predict_proba(X_test)
    ```

### 6. Plot the results

The results can be plotted using the following methods:
* `plot_scores` - plots the evaluation metric score for the selected measure.
* `plot_coeff` - plots the coefficients values for all regularization parameter values.
* `plot_likelihoods` - plots the log-likelihood values depending on iteration number for all regularization parameter values.

All the functions have an optional parameter `save_path` that allows saving the plot to a file.

    ```python
    model.plot_scores()
    model.plot_coeff()
    model.plot_likelihoods()
    ```

### 7. Valuate and plot for all measures

If you want to evaluate the model for all available measures and plot the results, you can use the `validate_and_plot_all` method.

    ```python
    model.validate_and_plot_all(X_valid, y_valid)
    ```