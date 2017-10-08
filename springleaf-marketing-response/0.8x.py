# coding: utf-8
"""
Beating the benchmark @ Kaggle Springleaf
@author: Abhishek Thakur
"""


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing, linear_model


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

y = train.target.values
train = train.drop(['ID', 'target'], axis=1)
test = test.drop('ID', axis=1)

train = train.dropna(axis=1)
test = test.dropna(axis=1)

train = train.fillna(-1)
test = test.fillna(-1)

for f in train.columns:
    if train[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))


X = np.array(train)
X_test = np.array(test)
clf = xgb.XGBClassifier(n_estimators=5000, nthread=-1, max_depth=17,
                        learning_rate=0.01, silent=False, subsample=0.8, colsample_bytree=0.7)


clf.fit(X, y)

preds = clf.predict_proba(X_test)[:,1]
sample = pd.read_csv('data/sample_submission.csv')
sample.target = preds
sample.to_csv('benchmark.csv', index=False)
