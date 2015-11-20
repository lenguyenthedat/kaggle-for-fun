import sys
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

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn import cross_validation
from matplotlib import pylab as plt

plot = True

goal = 'Expected'
myid = 'Id'

def load_data():
    """
        Load data and specified features of the data sets
    """
    train_org = pd.read_csv('./data/train.csv')
    test_org = pd.read_csv('./data/test.csv')
    return (train_org,test_org)

def process_data(train_org,test_org):
    """
        Feature engineering and selection.
    """
    # remove out-liners
    train_org = train_org[train_org['Expected'] < 69]

    # force NA for these Refs
    for data in [train_org,test_org]:
        data['Ref_5x5_50th'] = data['Ref_5x5_50th'].apply(lambda x: np.nan if x < 0 else x)
        data['Ref_5x5_90th'] = data['Ref_5x5_90th'].apply(lambda x: np.nan if x < 0 else x)
        data['RefComposite'] = data['RefComposite'].apply(lambda x: np.nan if x < 0 else x)
        data['RefComposite_5x5_50th'] = data['RefComposite_5x5_50th'].apply(lambda x: np.nan if x < 0 else x)
        data['RefComposite_5x5_90th'] = data['RefComposite_5x5_90th'].apply(lambda x: np.nan if x < 0 else x)
        data['Ref'] = data['Ref'].apply(lambda x: np.nan if x < 0 else x)

    # flatten
    grouped = train_org.groupby('Id')
    train = grouped.agg({'radardist_km' : np.nanmean, 'Ref_5x5_50th' :  np.nanmean,
                         'Ref_5x5_90th' : np.nanmean, 'RefComposite' :  np.nanmean,
                         'RefComposite_5x5_50th' : np.nanmean, 'RefComposite_5x5_90th' :  np.nanmean,
                         'Zdr' : np.nanmean, 'Zdr_5x5_50th' :  np.nanmean,
                         'Zdr_5x5_90th' : np.nanmean, 'Ref' :  [np.nanmean, np.sum, 'count'], 'Id' : [np.mean,'count'],
                         'Expected' : np.mean
                        })
    train.columns = [' '.join(col).strip() for col in train.columns.values]
    train.rename(columns={'Id mean':'Id'}, inplace=True)
    train['Expected'] = train['Expected mean'].apply(lambda x: np.log1p(x))

    grouped = test_org.groupby('Id')
    test = grouped.agg({'radardist_km' : np.nanmean, 'Ref_5x5_50th' :  np.nanmean,
                     'Ref_5x5_90th' : np.nanmean, 'RefComposite' :  np.nanmean,
                     'RefComposite_5x5_50th' : np.nanmean, 'RefComposite_5x5_90th' :  np.nanmean,
                     'Zdr' : np.nanmean, 'Zdr_5x5_50th' :  np.nanmean,
                     'Zdr_5x5_90th' : np.nanmean, 'Ref' :  [np.nanmean, np.sum, 'count'], 'Id' : [np.mean,'count']
                    })
    test.columns = [' '.join(col).strip() for col in test.columns.values]
    test.rename(columns={'Id mean':'Id'}, inplace=True)

    # Features set.
    features = test.columns.tolist()
    noisy_features = ['Id']
    features = [c for c in features if c not in noisy_features]
    features.extend([])
    # Fill NA
    train = train.fillna(train.mean())
    test = test.fillna(test.mean())
    # Neural Network, Stochastic Gradient Descent is sensitive to feature scaling, so it is highly recommended to scale your data.
    scaler = StandardScaler()
    for col in set(features): # TODO: add what not to scale
        scaler.fit(list(train[col])+list(test[col]))
        train[col] = scaler.transform(train[col])
        test[col] = scaler.transform(test[col])
    return (train,test,features)

def XGB_native(train,test,features):
    depth = 8
    eta = 0.007
    ntrees = 1800
    mcw = 1
    params = {"objective": "reg:linear",
              "booster": "gbtree",
              "eta": eta,
              "max_depth": depth,
              "min_child_weight": mcw,
              "subsample": 1,
              "colsample_bytree": 0.8,
              "silent": 1
              }
    print "Running with params: " + str(params)
    print "Running with ntrees: " + str(ntrees)
    print "Running with features: " + str(features)

    # Train model with local split
    tsize = 0.05 # 5% split
    X_train, X_test = cross_validation.train_test_split(train, test_size=tsize)
    dtrain = xgb.DMatrix(X_train[features], X_train[goal])
    dvalid = xgb.DMatrix(X_test[features], X_test[goal])
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    gbm = xgb.train(params, dtrain, ntrees, evals=watchlist, early_stopping_rounds=100, verbose_eval=True)
    train_probs = gbm.predict(xgb.DMatrix(X_test[features]))
    indices = train_probs < 0
    train_probs[indices] = 0
    print mean_absolute_error(train_probs,X_test[goal].values)

    # Predict and Export
    test_probs = gbm.predict(xgb.DMatrix(test[features]))
    indices = test_probs < 0
    test_probs[indices] = 0
    submission = pd.DataFrame({myid: test[myid], goal: np.expm1(test_probs)})

    submission = submission[[myid,goal]]
    if not os.path.exists('result/'):
        os.makedirs('result/')
    result_file = "dat-xgb_d%s_eta%s_ntree%s_mcw%s_tsize%s.csv" % (str(depth),str(eta),str(ntrees),str(mcw),str(tsize))
    submission.to_csv("./result/" + result_file, index=False)
    # Feature importance
    if plot:
      outfile = open('./result/xgb.fmap', 'w')
      i = 0
      for feat in features:
          outfile.write('{0}\t{1}\tq\n'.format(i, feat))
          i = i + 1
      outfile.close()
      importance = gbm.get_fscore(fmap='./result/xgb.fmap')
      importance = sorted(importance.items(), key=operator.itemgetter(1))
      df = pd.DataFrame(importance, columns=['feature', 'fscore'])
      df['fscore'] = df['fscore'] / df['fscore'].sum()
      # Plotitup
      plt.figure()
      df.plot()
      df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(25, 15))
      plt.title('XGBoost Feature Importance')
      plt.xlabel('relative importance')
      plt.gcf().savefig('./result/Feature_Importance_xgb_d%s_eta%s_ntree%s_mcw%s_tsize%s.png' % (str(depth),str(eta),str(ntrees),str(mcw),str(tsize)))
    return result_file

def main():
    print "=> Loading data - " + str(datetime.datetime.now())
    train_org,test_org = load_data()
    print "=> Processing data - " + str(datetime.datetime.now())
    train,test,features = process_data(train_org,test_org)
    print "=> XGBoost in action - " + str(datetime.datetime.now())
    result_file = XGB_native(train,test,features)

if __name__ == "__main__":
    main()
