import pandas as pd
import time
import csv
import numpy as np
import os

from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sknn.mlp import Regressor, Layer
from sklearn.grid_search import GridSearchCV
from xgboost import XGBRegressor

pd.options.mode.chained_assignment = None

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

gini_scorer = make_scorer(Gini, greater_is_better = True)

sample = True
gridsearch = True

features = ['T1_V1','T1_V2','T1_V3','T1_V4','T1_V5','T1_V6','T1_V7','T1_V8','T1_V9',
            'T1_V10','T1_V11','T1_V12','T1_V13','T1_V14','T1_V15','T1_V16','T1_V17',
            'T2_V1','T2_V2','T2_V3','T2_V4','T2_V5','T2_V6','T2_V7','T2_V8','T2_V9',
            'T2_V10','T2_V11','T2_V12','T2_V13','T2_V14','T2_V15']
features_non_numeric = ['T1_V4','T1_V5','T1_V6','T1_V7','T1_V8','T1_V9',
            'T1_V11','T1_V12','T1_V15','T1_V16','T1_V17',
            'T2_V3','T2_V5','T2_V11','T2_V12','T2_V13']
goal = 'Hazard'
myid = 'Id'

# Load data
if sample: # To train with 75% data
    df = pd.read_csv('./data/train.csv',dtype={'Category':pd.np.string_})
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
    train, test = df[df['is_train']==True], df[df['is_train']==False]
else:
    # To run with real data
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')

# Pre-processing non-number values
le = LabelEncoder()
for col in features_non_numeric:
    le.fit(list(train[col])+list(test[col]))
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

# Define regressors
if sample:
    regressors = [
        Regressor(layers=[
                    Layer("Sigmoid", units=100),
                    Layer("Sigmoid", units=100),
                    Layer("Linear")],
                  learning_rate=0.01,learning_rule='adadelta',learning_momentum=0.9,
                  batch_size=100,valid_size=0.01,
                  n_stable=50,n_iter=200,verbose=False),
        GradientBoostingRegressor(n_estimators=10, learning_rate=1.0,max_depth=5, random_state=0),
        RandomForestRegressor(),
        XGBRegressor()
    ]
else:
    regressors = [# Other methods are underperformed yet take very long training time for this data set
        RandomForestRegressor(max_depth=8,n_estimators=128),
        XGBRegressor(max_depth=2,n_estimators=512)
    ]

# Train
for regressor in regressors:
    print regressor.__class__.__name__
    start = time.time()
    if (gridsearch & sample): # only do gridsearch if we run with sampled data.
        try: # depth & estimator: usually fits for RF and XGB
            if (regressor.__class__.__name__ == "GradientBoostingRegressor"):
                print "Attempting GridSearchCV for GB model"
                gscv = GridSearchCV(regressor, {
                    'max_depth': [2, 8, 16],
                    'n_estimators': [32, 64, 128, 256, 512],
                    'learning_rate': [0.1, 0.05, 0.01],
                    'subsample': [0.6,0.8,1]},
                    verbose=1, n_jobs=2, scoring=gini_scorer)
            if (regressor.__class__.__name__ == "XGBRegressor"):
                print "Attempting GridSearchCV for XGB model"
                gscv = GridSearchCV(regressor, {
                    'max_depth': [2, 8, 16],
                    'n_estimators': [32, 64, 128, 256, 512],
                    'min_child_weight': [3,5],
                    'subsample': [0.6,0.8,1]},
                    verbose=1, n_jobs=2, scoring=gini_scorer)
            if (regressor.__class__.__name__ == "RandomForestRegressor"):
                print "Attempting GridSearchCV for RF model"
                gscv = GridSearchCV(regressor, {
                    'max_depth': [2, 8, 16],
                    'n_estimators': [32, 64, 128, 256, 512],
                    'bootstrap':[True,False],
                    'oob_score': [True,False]},
                    verbose=1, n_jobs=2, scoring=gini_scorer)
            if (regressor.__class__.__name__ == "Regressor"): # NN Regressor
                print "Attempting GridSearchCV for Neural Network model"
                gscv = GridSearchCV(regressor, {
                    'hidden0__units': [4, 16, 64, 128],
                    'hidden0__type': ["Rectifier", "Sigmoid", "Tanh"]},
                    verbose=1, n_jobs=1)
            regressor = gscv.fit(np.array(train[list(features)]), train[goal])
            print(regressor.best_score_)
            print(regressor.best_params_)
        except:
            regressor.fit(np.array(train[list(features)]), train[goal]) # just fit regular one
    else:
        regressor.fit(np.array(train[list(features)]), train[goal])
    print '  -> Training time:', time.time() - start

# Evaluation and export result
if sample:
    # Test results
    for regressor in regressors:
        print regressor.__class__.__name__
        try:
            print 'Root mean_squared_error:'
            print mean_squared_error(test[goal],regressor.predict(np.array(test[features])))**0.5
            print 'Gini:'
            print Gini(test[goal],regressor.predict(np.array(test[features])))
        except:
            pass

else: # Export result
    count = 0
    for regressor in regressors:
        count += 1
        if not os.path.exists('result/'):
            os.makedirs('result/')
        # TODO: fix this shit
        # test[myid] values will get converted to float since column_stack will result in array
        predictions = np.column_stack((test[myid], regressor.predict(np.array(test[features])))).tolist()
        predictions = [[int(i[0])] + i[1:] for i in predictions]
        csvfile = 'result/' + regressor.__class__.__name__ + '-'+ str(count) + '-submit.csv'
        with open(csvfile, 'w') as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow([myid,goal])
            writer.writerows(predictions)
