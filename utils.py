import numpy as np
import pandas as pd


def random_train_test_split(df, train_col='is_train', train_split=0.75):
    """
    :param df: pandas.DataFrame
        DataFrame with train and test data
    :param train_col: str
        String of the name of the DataFrame column that indicates
        whether the row is part of the training data
    :param train_split: float
        Fraction of data that is training data
    :return: training DataFrame, test Dataframe
    """
    df[train_col] = np.random.uniform(0, 1, len(df)) <= train_split
    train, test = df[df[train_col] == True], df[df[train_col] == False]
    return train, test