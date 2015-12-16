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
from sklearn import cross_validation, metrics
from matplotlib import pylab as plt

seed = 1337331
plot = False
sample = True

goal = 'QuoteConversion_Flag'
myid = 'QuoteNumber'

def load_data():
    """
        Load data and specified features of the data sets
    """
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')
    features = test.columns.tolist()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    return (train,test,features)

def process_data(train,test,features):
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

    # # Features set.
    noisy_features = [myid,'Original_Quote_Date']
    features = [c for c in features if c not in noisy_features]
    features.extend(['Year','Month','Weekday'])
    # Fill NA
    train = train.fillna(-1)
    test = test.fillna(-1)

    # Pre-processing non-numberic values
    for f in train.columns:
        if train[f].dtype=='object':
            lbl = LabelEncoder()
            lbl.fit(list(train[f].values) + list(test[f].values))
            train[f] = lbl.transform(list(train[f].values))
            test[f] = lbl.transform(list(test[f].values))
    # Scale features
    scaler = StandardScaler()
    for col in set(features):
        scaler.fit(list(train[col])+list(test[col]))
        train[col] = scaler.transform(train[col])
        test[col] = scaler.transform(test[col])
    return (train,test,features)

def XGB_native(train,test,features):
    depth = 6
    eta = 0.025
    ntrees = 1800
    mcw = 1
    params = {"objective": "multi:softprob", 'num_class':2,
              "eval_metric":"auc",
              "eta": eta,
              "max_depth": depth,
              "min_child_weight": mcw,
              "subsample": 0.9,
              "colsample_bytree": 0.7,
              "silent": 1
              }
    print "Running with params: " + str(params)
    print "Running with ntrees: " + str(ntrees)
    print "Running with "+ str(len(features)) + " features ..."

    # Training / Cross Validation
    if sample:
        cv = cross_validation.StratifiedKFold(train[goal],5, shuffle=True, random_state=2015)
        results = []
        for traincv, testcv in cv:
            # import pdb; pdb.Pdb().set_trace()
            xgbtrain = xgb.DMatrix(train.iloc[traincv][list(features)], label=train.iloc[traincv][goal])
            classifier = xgb.train(params, xgbtrain, ntrees)
            # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
            y = train.iloc[testcv][goal].values
            pred = np.compress([False, True], classifier.predict(xgb.DMatrix(train.iloc[testcv][features])), axis=1)
            fpr, tpr, thresholds = metrics.roc_curve(y, pred)
            score = metrics.auc(fpr, tpr)
            print score
            results.append(score)
        print "Results: " + str(results)
        print "Mean: " + str(np.array(results).mean())

    # EVAL OR EXPORT
    else: # Export result
        print str(datetime.datetime.now())
        xgbtrain = xgb.DMatrix(train[features], label=train[goal])
        classifier = xgb.train(params, xgbtrain, ntrees)
        if not os.path.exists('result/'):
            os.makedirs('result/')
        csvfile = "./result/dat-xgb_d%s_eta%s_ntree%s_mcw%s_%sfeatures.csv" % (str(depth),str(eta),str(ntrees),str(mcw),str(len(features)))

        with open(csvfile, 'w') as output:
            predictions = []
            print str(datetime.datetime.now())
            for i in test[myid].tolist():
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
    train,test,features = load_data()
    print "=> Processing data - " + str(datetime.datetime.now())
    train,test,features = process_data(train,test,features)
    print "=> XGBoost in action - " + str(datetime.datetime.now())
    XGB_native(train,test,features)

if __name__ == "__main__":
    main()
