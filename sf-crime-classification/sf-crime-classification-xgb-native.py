from __future__ import division

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
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import TransformerMixin
from sklearn import cross_validation
from sklearn.metrics import log_loss
from time import strptime

sample = True

goal = 'Category'
myid = 'Id'

# Create random 5000 lines train and test files:
# brew install coreutils
# tail -n +2 train.csv | gsort -R | head -5000 > train-5000.csv
# tail -n +2 test.csv | gsort -R | head -5000 > test-5000.csv
# rmb to copy over the headers

# When we want to stop the script at some point:
# import pdb;pdb.set_trace()

if not os.path.exists('result/'): os.makedirs('result/')

def load_data():
    """
        Load data and specified features of the data sets
    """
    # Load data
    if sample:
        train = pd.read_csv('./data/train-100000R')
        test = pd.read_csv('./data/test-100000R')
    else:
        # train = pd.read_csv('./data/mini-train.csv')
        # test = pd.read_csv('./data/mini-test.csv')
        train = pd.read_csv('./data/train.csv')
        test = pd.read_csv('./data/test.csv')

    features = ['Dates','DayOfWeek','PdDistrict','Address','X','Y']
    features_non_numeric = ['Dates','DayOfWeek','PdDistrict','Address']
    return (train,test,features,features_non_numeric)

def process_data(train,test,features,features_non_numeric):
    train['StreetNo'] = train['Address'].apply(lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)
    test['StreetNo'] = test['Address'].apply(lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)
    train['Address'] = train['Address'].apply(lambda x: x.split(' ', 1)[1] if x.split(' ', 1)[0].isdigit() else x)
    test['Address'] = test['Address'].apply(lambda x: x.split(' ', 1)[1] if x.split(' ', 1)[0].isdigit() else x)
    train['hour'] = train['Dates'].apply(lambda x: x[11:13] if len(x) > 4 else 12)
    test['hour'] = test['Dates'].apply(lambda x: x[11:13] if len(x) > 4 else 12)
    train['dark'] = train['Dates'].apply(lambda x: 1 if (len(x) > 4 and x[11:13] >= 18 and x[11:13] < 6) else 0)
    test['dark'] = test['Dates'].apply(lambda x: 1 if (len(x) > 4 and x[11:13] >= 18 and x[11:13] < 6) else 0)
    features += ['hour','dark','StreetNo']

    print "Filling N/As: " + str(datetime.datetime.now())
    train = train.fillna(train.mode().iloc[0])
    test = test.fillna(test.mode().iloc[0])
    # Pre-processing non-numberic values
    print "Label Encoder: " + str(datetime.datetime.now())
    le = LabelEncoder()
    for col in features:
        # print col
        le.fit(list(train[col])+list(test[col]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
    # Xgb requires goal to be numeric...
    le.fit(list(train[goal]))
    train[goal] = le.transform(train[goal])

    # Neural Network, Stochastic Gradient Descent is sensitive to feature scaling, so it is highly recommended to scale your data.
    print "Standard Scaler: " + str(datetime.datetime.now())
    scaler = StandardScaler()
    for col in set(features) - set(features_non_numeric):
        # print col
        scaler.fit(list(train[col])+list(test[col]))
        train[col] = scaler.transform(train[col])
        test[col] = scaler.transform(test[col])
    return (train,test,features)

def XGB_native(train,test,features):
    # XGB Params
    params = {'max_depth':8, 'eta':0.05, 'silent':1,
              'objective':'multi:softprob', 'num_class':39, 'eval_metric':'mlogloss',
              'min_child_weight':3, 'subsample':0.65,'colsample_bytree':0.65, 'nthread':4}
    num_rounds = 200
    # Training / Cross Validation
    if sample:
        # import pdb;pdb.set_trace()
        xgbtrain = xgb.DMatrix(train[list(features)], label=train[goal])
        classifier = xgb.train(params, xgbtrain, num_rounds)
        # xgval = xgb.DMatrix(train[testcv][list(features)], label=train[testcv][goal])
        # watchlist  = [ (xgbtrain,'train'),(xgval,'eval')]
        # classifier = xgb.train(params, xgbtrain, num_rounds, watchlist,early_stopping_rounds=50)
        # import pdb;pdb.set_trace()
        score = log_loss(test[goal].values, classifier.predict(xgb.DMatrix(test[features])))
        print score

        # cv = cross_validation.KFold(len(train), n_folds=5, shuffle=True, indices=False, random_state=1337)
        # results = []
        # for traincv, testcv in cv:
        #     # import pdb;pdb.set_trace()
        #     xgbtrain = xgb.DMatrix(train[traincv][list(features)], label=train[traincv][goal])
        #     classifier = xgb.train(params, xgbtrain, num_rounds)
        #     # xgval = xgb.DMatrix(train[testcv][list(features)], label=train[testcv][goal])
        #     # watchlist  = [ (xgbtrain,'train'),(xgval,'eval')]
        #     # classifier = xgb.train(params, xgbtrain, num_rounds, watchlist,early_stopping_rounds=50)
        #     import pdb;pdb.set_trace()
        #     score = log_loss(train[testcv][goal].values, classifier.predict(xgb.DMatrix(train[testcv][features])))
        #     print score
        #     results.append(score)
        # print "Results: " + str(results)
        # print "Mean: " + str(np.array(results).mean())
    else: # Export result
        print "Training: " + str(datetime.datetime.now())
        xgbtrain = xgb.DMatrix(train[features], label=train[goal])
        classifier = xgb.train(params, xgbtrain, num_rounds)
        csvfile = 'result/' + classifier.__class__.__name__ + '-submit.csv'
        with open(csvfile, 'w') as output:
            predictions = []
            print "Predicting: " + str(datetime.datetime.now())
            for i in test[myid].tolist():
                # import pdb;pdb.set_trace()
                predictions += [[i]+classifier.predict(xgb.DMatrix(test[test[myid]==i][features])).tolist()[0]]
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow([myid,'ARSON','ASSAULT','BAD CHECKS','BRIBERY','BURGLARY','DISORDERLY CONDUCT',
                             'DRIVING UNDER THE INFLUENCE','DRUG/NARCOTIC','DRUNKENNESS','EMBEZZLEMENT',
                             'EXTORTION','FAMILY OFFENSES','FORGERY/COUNTERFEITING','FRAUD','GAMBLING',
                             'KIDNAPPING','LARCENY/THEFT','LIQUOR LAWS','LOITERING','MISSING PERSON',
                             'NON-CRIMINAL','OTHER OFFENSES','PORNOGRAPHY/OBSCENE MAT','PROSTITUTION',
                             'RECOVERED VEHICLE','ROBBERY','RUNAWAY','SECONDARY CODES','SEX OFFENSES FORCIBLE',
                             'SEX OFFENSES NON FORCIBLE','STOLEN PROPERTY','SUICIDE','SUSPICIOUS OCC','TREA',
                             'TRESPASS','VANDALISM','VEHICLE THEFT','WARRANTS','WEAPON LAWS'])
            writer.writerows(predictions)
            print "Predicting done: " + str(datetime.datetime.now())
        # Feature importance
        outfile = open('result/xgb.fmap', 'w')
        i = 0
        for feat in features:
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))
            i = i + 1
        outfile.close()
        importance = classifier.get_fscore(fmap='result/xgb.fmap')
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        df = pd.DataFrame(importance, columns=['feature', 'fscore'])
        df.to_csv('result/importance.csv',index=False)

def main():
    print "=> Loading data - " + str(datetime.datetime.now())
    train,test,features,features_non_numeric = load_data()
    print "=> Processing data - " + str(datetime.datetime.now())
    train,test,features = process_data(train,test,features,features_non_numeric)
    print "=> XGBoost in action - " + str(datetime.datetime.now())
    XGB_native(train,test,features)

if __name__ == "__main__":
    main()
