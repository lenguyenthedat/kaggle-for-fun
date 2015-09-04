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

from matplotlib import pylab as plt

sample = False
plot = True # Won't plot if sample = True

goal = 'target'
myid = 'ID'

if not os.path.exists('result/'): os.makedirs('result/')

def load_data():
    """
        Load data and specified features of the data sets
    """
    # Load data
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')
    # Features set
    features = test.columns.tolist()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    features_numeric = test.select_dtypes(include=numerics).columns.tolist()
    features_non_numeric = [f for f in features if f not in features_numeric]
    # # Features set.
    # noisy_features = [myid,'VAR_0207','VAR_1934','VAR_0493']
    noisy_features = [myid,goal]
    features = [c for c in features if c not in noisy_features]
    features_numeric = [c for c in features_numeric if c not in noisy_features]
    features_non_numeric = [c for c in features_non_numeric if c not in noisy_features]
    return (train,test,features,features_non_numeric)

def process_data(train,test,features,features_non_numeric):
    train = train.fillna(train.mode().iloc[0])
    test = test.fillna(test.mode().iloc[0])
    # # filling what wasn't filled yet (wtf / need to fix)
    for col in features_non_numeric:
        train[col] = train[col].fillna('-1') # wtf
        test[col] = test[col].fillna('-1') # wtf
    train = train.fillna(-1)
    test = test.fillna(-1)
    # Pre-processing non-numberic values
    le = LabelEncoder()
    for col in features_non_numeric:
        print col
        le.fit(list(train[col])+list(test[col]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
    # Neural Network, Stochastic Gradient Descent is sensitive to feature scaling, so it is highly recommended to scale your data.
    scaler = StandardScaler()
    for col in set(features) - set(features_non_numeric):
        scaler.fit(list(train[col])+list(test[col]))
        train[col] = scaler.transform(train[col])
        test[col] = scaler.transform(test[col])
    return (train,test,features,features_non_numeric)

def XGB_native(train,test,features,features_non_numeric):
    # XGB Params
    params = {'max_depth':6, 'eta':0.01, 'silent':1,
              'objective':'multi:softprob', 'num_class':2, 'eval_metric':'mlogloss',
              'min_child_weight':3, 'subsample':1,'colsample_bytree':0.55, 'nthread':4}
    num_rounds = 500
    # Training / Cross Validation
    if sample:
        cv = cross_validation.KFold(len(train), n_folds=5, shuffle=True, indices=False, random_state=1337)
        results = []
        for traincv, testcv in cv:
            xgbtrain = xgb.DMatrix(train[traincv][list(features)], label=train[traincv][goal])
            classifier = xgb.train(params, xgbtrain, num_rounds)
            score = roc_auc_score(train[testcv][goal].values, np.compress([False, True],\
                classifier.predict(xgb.DMatrix(train[testcv][features])), axis=1).flatten())
            print score
            results.append(score)
        print "Results: " + str(results)
        print "Mean: " + str(np.array(results).mean())
    # EVAL OR EXPORT
    if not sample: # Export result
        print str(datetime.datetime.now())
        xgbtrain = xgb.DMatrix(train[features], label=train[goal])
        classifier = xgb.train(params, xgbtrain, num_rounds)
        classifier.save_model('model/xgb.model')
        # classifier = xgb.Booster(model_file='result/xgb.model')
        csvfile = 'result/' + classifier.__class__.__name__ + '-submit.csv'
        with open(csvfile, 'w') as output:
            predictions = []
            print str(datetime.datetime.now())
            for i in test[myid].tolist():
              # National ServiceMen always resigned
              if test[test[myid] == i]['EMPLOYEE_GROUP'].item() == 2:
                predictions += [[i,1]]
              else:
                predictions += [[i,classifier.predict(xgb.DMatrix(test[test[myid]==i][features])).tolist()[0][1]]]
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow([myid,goal])
            writer.writerows(predictions)
            print str(datetime.datetime.now())
        # Feature importance
        if plot:
          outfile = open('xgb.fmap', 'w')
          i = 0
          for feat in features:
              outfile.write('{0}\t{1}\tq\n'.format(i, feat))
              i = i + 1
          outfile.close()
          importance = classifier.get_fscore(fmap='xgb.fmap')
          print importance
          importance = sorted(importance.items(), key=operator.itemgetter(1))
          df = pd.DataFrame(importance, columns=['feature', 'fscore'])
          df['fscore'] = df['fscore'] / df['fscore'].sum()
          # Plotitup
          plt.figure()
          df.plot()
          df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(25, 15))
          plt.title('XGBoost Feature Importance')
          plt.xlabel('relative importance')
          plt.gcf().savefig('Feature_Importance_xgb.png')

def main():
    print "=> Loading data - " + str(datetime.datetime.now())
    train,test,features,features_non_numeric = load_data()
    print "=> Processing data - " + str(datetime.datetime.now())
    train,test,features,features_non_numeric = process_data(train,test,features,features_non_numeric)
    print "=> XGBoost in action - " + str(datetime.datetime.now())
    XGB_native(train,test,features,features_non_numeric)

if __name__ == "__main__":
    main()
