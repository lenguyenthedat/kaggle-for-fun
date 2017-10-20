import pandas as pd
import datetime
import csv
import numpy as np
import os
import scipy as sp
import xgboost as xgb
import itertools
import operator
import warnings
import utils

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import cross_validation, metrics

seed = 1337

goal = 'QuoteConversion_Flag'
myid = 'QuoteNumber'

if not os.path.exists('./result/'):
    os.makedirs('./result/')


def load_data():
    """
        Load data and specified features of the data sets
    """
    train, test = utils.read_csv_files('./data/train1000.csv', './data/test1000.csv')
    features = test.columns.tolist()
    return train, test, features


def process_data(train, test, features):
    """
        Feature engineering and selection.
    """
    # # FEATURE ENGINEERING

    train['Original_Quote_Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
    test['Original_Quote_Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))

    train['Year'] = train['Original_Quote_Date'].apply(lambda x: int(str(x)[:4]))
    train['Month'] = train['Original_Quote_Date'].apply(lambda x: int(str(x)[5:7]))
    train['Weekday'] = train['Original_Quote_Date'].dt.dayofweek

    test['Year'] = test['Original_Quote_Date'].apply(lambda x: int(str(x)[:4]))
    test['Month'] = test['Original_Quote_Date'].apply(lambda x: int(str(x)[5:7]))
    test['Weekday'] = test['Original_Quote_Date'].dt.dayofweek

    grams_binary = ['Field12', 'PersonalField7', 'PropertyField3', 'PropertyField4', 'PropertyField5',
                    'PropertyField30', 'PropertyField32', 'PropertyField34', 'PropertyField36', 'PropertyField37',
                    'PropertyField38', 'GeographicField63', 'Month', 'Weekday']

    two_grams_binary = []
    for i in range(len(grams_binary)):
        for j in range(i + 1, len(grams_binary)):
            two_gram = grams_binary[i] + '_' + grams_binary[j]
            two_grams_binary += [two_gram]
            train[two_gram] = train[grams_binary[i]].map(str) + '_' + train[grams_binary[j]].map(str)
            test[two_gram] = test[grams_binary[i]].map(str) + '_' + test[grams_binary[j]].map(str)

    grams = ['Field6', 'CoverageField8', 'CoverageField9', 'SalesField7', 'PersonalField16',
             'PersonalField17', 'PersonalField18', 'PersonalField19', 'PropertyField7',
             'PropertyField14', 'PropertyField28', 'PropertyField31', 'PropertyField33',
             'GeographicField64', 'Month', 'Weekday']

    two_grams = []
    for i in range(len(grams)):
        for j in range(i + 1, len(grams)):
            two_gram = grams[i] + '_' + grams[j]
            two_grams += [two_gram]
            train[two_gram] = train[grams[i]].map(str) + '_' + train[grams[j]].map(str)
            test[two_gram] = test[grams[i]].map(str) + '_' + test[grams[j]].map(str)

    arithmetics = ['SalesField5', 'PersonalField9', 'Field7', 'PersonalField2', 'PersonalField1',
                   'SalesField1A', 'SalesField4', 'PersonalField10A', 'SalesField1B', 'PersonalField10B',
                   'PersonalField13', 'PersonalField4A', 'PersonalField27']

    two_arithmetics = []
    for i in range(len(arithmetics)):
        for j in range(i + 1, len(arithmetics)):
            # products
            two_arithmetic1 = arithmetics[i] + 'times' + arithmetics[j]
            two_arithmetics += [two_arithmetic1]
            train[two_arithmetic1] = train[arithmetics[i]] * train[arithmetics[j]]
            test[two_arithmetic1] = test[arithmetics[i]] * test[arithmetics[j]]
            # divisions
            two_arithmetic2 = arithmetics[i] + 'divs' + arithmetics[j]
            two_arithmetics += [two_arithmetic2]
            train[two_arithmetic2] = train[arithmetics[i]] / train[arithmetics[j]]
            test[two_arithmetic2] = test[arithmetics[i]] / test[arithmetics[j]]

    noisy_features = [myid, 'Original_Quote_Date'] + grams + grams_binary + arithmetics

    features = [c for c in features if c not in noisy_features]
    features.extend(['Year', 'Month', 'Weekday'])
    features.extend(two_grams_binary)
    features.extend(two_grams)
    features.extend(two_arithmetics)
    features = list(set(features))

    # Fill NA
    train['PropertyField29'] = train['PropertyField29'].fillna(-1)
    test['PropertyField29'] = test['PropertyField29'].fillna(-1)
    train = train.fillna(0)
    test = test.fillna(0)

    # Pre-processing non-numberic values
    for f in train.columns:
        if train[f].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(train[f].values) + list(test[f].values))
            train[f] = lbl.transform(list(train[f].values))
            test[f] = lbl.transform(list(test[f].values))
    # Scale features
    scaler = StandardScaler()
    for col in ['Field8', 'Field9', 'Field10', 'Field11', 'SalesField8']:  # need to be scaled
        scaler.fit(np.reshape((list(train[col])+list(test[col])), (-1, 1))
        train[col] = scaler.transform(np.reshape(train[col], (-1,1)))
        test[col] = scaler.transform(np.reshape(test[col], (-1,1)))

    return (train, test, features)


def XGB_native(train, test, features):
    depth = 5
    eta = 0.01
    ntrees = 6238
    mcw = 1
    params = {"objective": "multi:softprob", 'num_class': 2,
              "eval_metric": "auc",
              "eta": eta,
              "max_depth": depth,
              "min_child_weight": mcw,
              "subsample": 0.9,
              "colsample_bytree": 0.77,
              "silent": 1
              }

    print "Running with params: " + str(params)
    print "Running with ntrees: " + str(ntrees)
    print "Running with " + str(len(features)) + " features ..."

    print str(datetime.datetime.now())
    xgbtrain = xgb.DMatrix(train[features], label=train[goal])
    classifier = xgb.train(params, xgbtrain, ntrees)

    pred = np.compress([False, True], classifier.predict(xgb.DMatrix(test[features])), axis=1)
    predictions_df = pd.DataFrame()
    predictions_df[myid] = test[myid]
    predictions_df[goal] = pred
    predictions_df = predictions_df.sort_values(by=myid, axis = 'index', ascending=True)
    csvfile = "./result/dat-xgb_d%s_eta%s_ntree%s_mcw%s_%sfeatures.csv" % (
    str(depth), str(eta), str(ntrees), str(mcw), str(len(features)))
    predictions_df.to_csv(csvfile, index=False)

                   
def main():
    print "=> Loading data - " + str(datetime.datetime.now())
    train, test, features = load_data()
    print "=> Processing data - " + str(datetime.datetime.now())
    train, test, features = process_data(train, test, features)
    print "=> ML in action - " + str(datetime.datetime.now())
    XGB_native(train, test, features)


if __name__ == "__main__":
    main()
