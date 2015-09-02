import pandas as pd
import time
import csv
import numpy as np
import os
import scipy as sp
import xgboost as xgb
import itertools
import warnings
warnings.filterwarnings("ignore")

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import TransformerMixin
from sklearn import cross_validation

sample = True

features = ['T1_V1','T1_V2','T1_V3','T1_V4','T1_V5','T1_V6','T1_V7','T1_V8','T1_V9','T1_V10',
            'T1_V11','T1_V12','T1_V13','T1_V14','T1_V15','T1_V16','T1_V17',
            'T2_V1','T2_V2','T2_V3','T2_V4','T2_V5','T2_V6','T2_V7','T2_V8','T2_V9','T2_V10',
            'T2_V11','T2_V12','T2_V13','T2_V14','T2_V15']
features_non_numeric = ['T1_V4','T1_V5','T1_V6','T1_V7','T1_V8','T1_V9',
            'T1_V11','T1_V12','T1_V15','T1_V16','T1_V17',
            'T2_V3','T2_V5','T2_V11','T2_V12','T2_V13']
goal = 'Hazard'
myid = 'Id'

noisy_features = []
features = [c for c in features if c not in noisy_features]
features_non_numeric = [c for c in features_non_numeric if c not in noisy_features]

# https://www.kaggle.com/jpopham91/liberty-mutual-group-property-inspection-prediction/gini-scoring-simple-and-efficient
def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.values.shape == y_pred.shape
    n_samples = y_true.values.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_true.values, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(0, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred/G_true

# Load data
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# Pre-processing non-number values
le = LabelEncoder()
for col in features_non_numeric:
    le.fit(list(train[col])+list(test[col]))
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

# Neural Network, Stochastic Gradient Descent is sensitive to feature scaling, so it is highly recommended to scale your data.
scaler = StandardScaler()
for col in set(features) - set(features_non_numeric):
    scaler.fit(list(train[col])+list(test[col]))
    train[col] = scaler.transform(train[col])
    test[col] = scaler.transform(test[col])

# XGB Params
params = {'max_depth':9, 'eta':0.01, 'silent':1,'objective':'reg:linear',
          'min_child_weight':6, 'subsample':0.7,'colsample_bytree':0.7, 'nthread':4}
num_rounds = 555

# TRAINING / GRIDSEARCH
if sample:
    cv = cross_validation.KFold(len(train), n_folds=5, shuffle=True, indices=False, random_state=1337)
    results = []
    for traincv, testcv in cv:
        xgbtrain = xgb.DMatrix(train[traincv][list(features)], label=train[traincv][goal])
        regressor = xgb.train(params, xgbtrain, num_rounds)
        score = Gini(train[testcv][goal], regressor.predict(xgb.DMatrix(train[testcv][features])))
        print score
        results.append(score)
    print "Results: " + str(results)
    print "Mean: " + str(np.array(results).mean())

# EVAL OR EXPORT
if not sample: # Export result
    xgbtrain = xgb.DMatrix(train[features], label=train[goal])
    regressor = xgb.train(params, xgbtrain, num_rounds)
    if not os.path.exists('result/'):
        os.makedirs('result/')
    csvfile = 'result/Booster-submit.csv'
    with open(csvfile, 'w') as output:
      predictions = []
      for i in test[myid].tolist():
          predictions += [[i,regressor.predict(xgb.DMatrix(test[test[myid]==i][features])).item()]]
      writer = csv.writer(output, lineterminator='\n')
      writer.writerow([myid,goal])
      writer.writerows(predictions)