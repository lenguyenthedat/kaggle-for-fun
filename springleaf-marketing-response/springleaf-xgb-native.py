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
from sklearn.metrics import roc_auc_score
from time import strptime

sample = True

goal = 'target'
myid = 'ID'

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
        train = pd.read_csv('./data/train-5000.csv')
        test = pd.read_csv('./data/test-5000.csv')
    else:
        train = pd.read_csv('./data/train.csv')
        test = pd.read_csv('./data/test.csv')
    # Features set
    features = test.columns.tolist()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    features_numeric = test.select_dtypes(include=numerics).columns.tolist()
    features_non_numeric = [f for f in features if f not in features_numeric]

    # # Features set.
    noisy_features = [myid,goal]
    # for f in features:
    #     if len(train[f].value_counts()) < 1: # only 0 or 1 class: noisy
    #         print f + ' only has ' + str(len(train[f].value_counts())) + ' class...'
    #         noisy_features += [f]
    #     if sum(train[f].value_counts()) < 0.5 * len(train[f]): # more than 50% is emty
    #         print f + ' is ' + str(100 - 100*sum(train[f].value_counts()) / len(train[f])) + ' percent empty...'
    #         noisy_features += [f]
    #     if len(train[f].value_counts()) > 0.5 * len(train[f]): # too many classes
    #         print f + ' has ' + str(len(train[f].value_counts())) + ' classes...'
    #         noisy_features += [f]
    # This list is the result of the above calculation, generated from the 5000 rows - training set.
    # To be hardcoded here mainly because it took too long to re-create from the above for loop
    noisy_features += [ 'VAR_0073','VAR_0074','VAR_0156','VAR_0157','VAR_0158','VAR_0159','VAR_0166','VAR_0167',
                        'VAR_0168','VAR_0169','VAR_0176','VAR_0177','VAR_0178','VAR_0179','VAR_0205','VAR_0206',
                        'VAR_0207','VAR_0207','VAR_0208','VAR_0209','VAR_0210','VAR_0211','VAR_0212','VAR_0213',
                        'VAR_0213','VAR_0214','VAR_0214','VAR_0227','VAR_0228','VAR_0241','VAR_0293','VAR_0313',
                        'VAR_0541','VAR_0543','VAR_0609','VAR_0648','VAR_0649','VAR_0652','VAR_0704','VAR_0840',
                        'VAR_0840','VAR_0887','VAR_0893','VAR_0896','VAR_0898','VAR_0899','VAR_0908','VAR_0920',
                        'VAR_0921','VAR_0931','VAR_0950','VAR_0951','VAR_0970','VAR_1081','VAR_1082','VAR_1087',
                        'VAR_1088','VAR_1089','VAR_1130','VAR_1179','VAR_1180','VAR_1181','VAR_1199','VAR_1200',
                        'VAR_1201','VAR_1202','VAR_1203','VAR_1220','VAR_1227','VAR_1228','VAR_1241','VAR_1243',
                        'VAR_1312','VAR_1313','VAR_1353','VAR_1354','VAR_1371','VAR_1372','VAR_1489','VAR_1494',
                        'VAR_1495','VAR_1496','VAR_1497','VAR_1801','VAR_1802']
    features = [c for c in features if c not in noisy_features]
    features_numeric = [c for c in features_numeric if c not in noisy_features]
    features_non_numeric = [c for c in features_non_numeric if c not in noisy_features]
    return (train,test,features,features_non_numeric)

def process_data(train,test,features,features_non_numeric):
    #14JUL11:00:00:00
    for v in ['VAR_0073','VAR_0075','VAR_0156','VAR_0157','VAR_0158',
              'VAR_0159','VAR_0166','VAR_0167','VAR_0168','VAR_0169',
              'VAR_0176','VAR_0177','VAR_0178','VAR_0179']:
        try:
            train[v+'D'] = train[v].apply(lambda x: x[:2] if pd.notnull(x) else np.nan)
            train[v+'M'] = train[v].apply(lambda x: strptime(x[2:5],'%b').tm_mon if pd.notnull(x) else np.nan)
            train[v+'Y'] = train[v].apply(lambda x: x[5:7] if pd.notnull(x) else np.nan)
            train[v+'WD'] = train[v].apply(lambda x: datetime.datetime.strptime(x[:7],'%d%b%y').weekday() if pd.notnull(x) else np.nan)
            test[v+'D'] = test[v].apply(lambda x: x[:2] if pd.notnull(x) else np.nan)
            test[v+'M'] = test[v].apply(lambda x: strptime(x[2:5],'%b').tm_mon if pd.notnull(x) else np.nan)
            test[v+'Y'] = test[v].apply(lambda x: x[5:7] if pd.notnull(x) else np.nan)
            test[v+'WD'] = test[v].apply(lambda x: datetime.datetime.strptime(x[:7],'%d%b%y').weekday() if pd.notnull(x) else np.nan)
            features += [v+'D',v+'M',v+'Y',v+'WD']
            features.remove(v)
            features_non_numeric.remove(v)
        except:
            pass
    #29JAN14:21:16:00
    for v in ['VAR_0204','VAR_0217']:
        try:
            train[v+'D'] = train[v].apply(lambda x: x[:2] if pd.notnull(x) else np.nan)
            train[v+'M'] = train[v].apply(lambda x: strptime(x[2:5],'%b').tm_mon if pd.notnull(x) else np.nan)
            train[v+'Y'] = train[v].apply(lambda x: x[5:7] if pd.notnull(x) else np.nan)
            train[v+'H'] = train[v].apply(lambda x: x[8:10] if pd.notnull(x) else np.nan)
            train[v+'WD'] = train[v].apply(lambda x: datetime.datetime.strptime(x[:7],'%d%b%y').weekday() if pd.notnull(x) else np.nan)
            test[v+'D'] = test[v].apply(lambda x: x[:2] if pd.notnull(x) else np.nan)
            test[v+'M'] = test[v].apply(lambda x: strptime(x[2:5],'%b').tm_mon if pd.notnull(x) else np.nan)
            test[v+'Y'] = test[v].apply(lambda x: x[5:7] if pd.notnull(x) else np.nan)
            test[v+'H'] = test[v].apply(lambda x: x[8:10] if pd.notnull(x) else np.nan)
            test[v+'WD'] = test[v].apply(lambda x: datetime.datetime.strptime(x[:7],'%d%b%y').weekday() if pd.notnull(x) else np.nan)
            features += [v+'D',v+'M',v+'Y',v+'H',v+'WD']
            features.remove(v)
            features_non_numeric.remove(v)
        except:
            pass
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
    params = {'max_depth':8, 'eta':0.01, 'silent':1,
              'objective':'multi:softprob', 'num_class':2, 'eval_metric':'auc',
              # 'objective':'multi:softprob', 'num_class':2, 'eval_metric':'mlogloss',
              'min_child_weight':3, 'subsample':0.65,'colsample_bytree':0.65, 'nthread':4}
    num_rounds = 875
    # Training / Cross Validation
    if sample:
        cv = cross_validation.KFold(len(train), n_folds=5, shuffle=True, indices=False, random_state=1337)
        results = []
        for traincv, testcv in cv:
            xgbtrain = xgb.DMatrix(train[traincv][list(features)], label=train[traincv][goal])
            classifier = xgb.train(params, xgbtrain, num_rounds)
            # xgval = xgb.DMatrix(train[testcv][list(features)], label=train[testcv][goal])
            # watchlist  = [ (xgbtrain,'train'),(xgval,'eval')]
            # classifier = xgb.train(params, xgbtrain, num_rounds, watchlist,early_stopping_rounds=50)
            score = roc_auc_score(train[testcv][goal].values, np.compress([False, True],\
                classifier.predict(xgb.DMatrix(train[testcv][features])), axis=1).flatten())
            print score
            results.append(score)
        print "Results: " + str(results)
        print "Mean: " + str(np.array(results).mean())
    else: # Export result
        print "Training: " + str(datetime.datetime.now())
        xgbtrain = xgb.DMatrix(train[features], label=train[goal])
        classifier = xgb.train(params, xgbtrain, num_rounds)
        csvfile = 'result/' + classifier.__class__.__name__ + '-submit.csv'
        with open(csvfile, 'w') as output:
            predictions = []
            print "Predicting: " + str(datetime.datetime.now())
            for i in test[myid].tolist():
                predictions += [[i,classifier.predict(xgb.DMatrix(test[test[myid]==i][features])).tolist()[0][1]]]
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow([myid,goal])
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
