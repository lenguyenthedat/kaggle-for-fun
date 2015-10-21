import pandas as pd
import time
import csv
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
from xgboost import XGBRegressor

pd.options.mode.chained_assignment = None
if not os.path.exists('result/'): os.makedirs('result/')
sample = True
gridsearch = False
goal = 'Sales'
myid = 'Id'
features = ['Store','DayOfWeek','Date','Open','Promo','StateHoliday','SchoolHoliday',
            'StoreType','Assortment','CompetitionDistance','CompetitionOpenSinceMonth',
            'CompetitionOpenSinceYear','Promo2','Promo2SinceWeek','Promo2SinceYear','PromoInterval']

def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w

def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe

def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe

store = pd.read_csv('./data/store.csv')

# Load data
if sample: # To train with 75% data
    if gridsearch:
        train_org = pd.read_csv('./data/train.csv',dtype={'StateHoliday':pd.np.string_})
        test_org = pd.read_csv('./data/test.csv',dtype={'StateHoliday':pd.np.string_})
    else:
        df = pd.read_csv('./data/train.csv',dtype={'StateHoliday':pd.np.string_})
        df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
        train_org, test_org = df[df['is_train']==True], df[df['is_train']==False]
else:
    # To run with real data
    train_org = pd.read_csv('./data/train.csv',dtype={'StateHoliday':pd.np.string_})
    test_org = pd.read_csv('./data/test.csv',dtype={'StateHoliday':pd.np.string_})

train = pd.merge(train_org,store, on='Store', how='left')
test = pd.merge(test_org,store, on='Store', how='left')

train = train.fillna(-1)
test = test.fillna(-1)

for f in train[features]:
    if train[f].dtype=='object':
        lbl = LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

regressor = XGBRegressor(n_estimators=100, nthread=-1, max_depth=6,
                   learning_rate=0.01, silent=False, subsample=0.8, colsample_bytree=0.7)

start = time.time()
if (gridsearch & sample): # only do gridsearch if we run with sampled data.
    print "Attempting GridSearchCV for XGB model"
    gscv = GridSearchCV(regressor, {
        'max_depth': [3, 5, 7, 11, 13, 17, 23],
        'n_estimators': [32, 64, 128, 512, 1024, 2048, 4096],
        'learning_rate': [0.005,0.01,0.025],
        'min_child_weight': [1,2,3,4,5],
        'subsample': [0.6,0.7,0.8],
        'colsample_bytree': [0.6,0.7,0.8]},
        verbose=1, n_jobs=2)
    regressor = gscv.fit(np.array(train), train[goal])
    print(regressor.best_score_)
    print(regressor.best_params_)
else:
    regressor.fit(np.array(train[features]), train[goal])
print '  -> Training time:', time.time() - start

# Evaluation and export result
if sample:
    if not gridsearch:
        # Test results
        print "RMSPE: " + str(rmspe(regressor.predict(np.array(test)),test[goal].values))
else:
    predictions = np.column_stack((test[myid], regressor.predict(np.array(test[features])))).tolist()
    predictions = [[int(i[0])] + i[1:] for i in predictions]
    csvfile = 'result/' + regressor.__class__.__name__ + '-submit.csv'
    with open(csvfile, 'w') as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow([myid,goal])
        writer.writerows(predictions)
