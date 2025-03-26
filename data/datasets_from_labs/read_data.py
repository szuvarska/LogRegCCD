import pandas as pd


def read_diabetes_dataset(filepath: str) -> tuple:
    """
    Read the Pima Indians Diabetes dataset from the given filepath
    :param filepath: The path to the CSV file containing the dataset.
    :return: A tuple containing the features (X) and the target variable (y).
    """
    df = pd.read_csv(filepath)
    y = df['Outcome']
    X = df.drop(columns=['Outcome'])
    return X, y


def read_haberman_dataset(filepath: str) -> tuple:
    """
    Read the Haberman's Survival dataset from the given filepath
    :param filepath: The path to the CSV file containing the dataset.
    :return: A tuple containing the features (X) and the target variable (y).
    """
    df = pd.read_csv(filepath, header=None)
    y = df.iloc[:, -1]
    y = y.map({1: 1, 2: 0})  # Encoding the target variable with 0 and 1
    X = df.drop(df.columns[-1], axis=1)
    return X, y


def read_wdbc_dataset(filepath: str, drop_correlated: bool = False) -> tuple:
    """
    Read the Wisconsin Diagnostic Breast Cancer dataset from the given filepath
    :param filepath: The path to the CSV file containing the dataset.
    :param drop_correlated: If True, drop highly correlated features.
    :return: A tuple containing the features (X) and the target variable (y).
    """
    df = pd.read_csv(filepath, header=None)
    y = df.iloc[:, 1]
    X = df.drop(columns=[0, 1])
    y = y.map({'M': 1, 'B': 0})  # Encoding the target variable with 0 and 1

    if drop_correlated:
        corr_wdbc = X.corr().abs()
        upper = corr_wdbc.where(np.triu(np.ones(corr_wdbc.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        X = X.drop(columns=to_drop)
    return X, y


def read_earthquake_dataset(filepath: str) -> tuple:
    """
    Read the earthquake dataset from the given filepath
    :param filepath: The path to the TXT file containing the dataset.
    :return: A tuple containing the features (X) and the target variable (y).
    """
    df = pd.read_csv(filepath, sep=" ")
    df['popn'] = df['popn'].map({'equake': 0, 'explosn': 1})  # Encoding the target variable with 0 and 1
    X = df[['body', 'surface']]
    y = df['popn']
    return X, y
