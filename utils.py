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


def load_data(full_csv_file=None, train_csv_file=None, test_csv_file=None,
              random=True,
              train_col='is_train', train_split=0.75,
              **read_csv_kwargs):
    """
    Returns a training data set and a test data set
    in 2 pandas data frames from csv files
    """
    if random:
        df = pd.read_csv(full_csv_file, **read_csv_kwargs)
        train, test = random_train_test_split(df, train_col, train_split)
        return train, test
    else:
        return read_csv_files(train_csv_file, test_csv_file, **read_csv_kwargs)


def read_csv_files(*csv_files, **read_csv_kwargs):
    """
    :param csv_files: str
    :param read_csv_kwargs: keyword-only arguments for read_csv pandas
    :rtype: tuple
    """
    df_list = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, **read_csv_kwargs)
        df_list.append(df)
    return tuple(df_list)

