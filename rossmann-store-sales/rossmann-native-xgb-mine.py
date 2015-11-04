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
from matplotlib import pylab as plt

sample = True
plot = True # Won't plot if sample = True

goal = 'Sales'
myid = 'Id'

# Thanks to Chenglong Chen for providing this in the forum
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

def load_data():
    """
        Load data and specified features of the data sets
    """
    store = pd.read_csv('./data/store.csv')
    train_org = pd.read_csv('./data/train.csv',dtype={'StateHoliday':pd.np.string_})
    test_org = pd.read_csv('./data/test.csv',dtype={'StateHoliday':pd.np.string_})
    train = pd.merge(train_org,store, on='Store', how='left')
    test = pd.merge(test_org,store, on='Store', how='left')
    features = test.columns.tolist()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    features_numeric = test.select_dtypes(include=numerics).columns.tolist()
    features_non_numeric = [f for f in features if f not in features_numeric]
    return (train,test,features,features_non_numeric)

def process_data(train,test,features,features_non_numeric):
    """
        Feature engineering and selection.
    """
    # # FEATURE ENGINEERING
    train = train[train['Sales'] > 0]

    for data in [train,test]:
        # year month day
        data['year'] = data.Date.apply(lambda x: x.split('-')[0])
        data['year'] = data['year'].astype(float)
        data['month'] = data.Date.apply(lambda x: x.split('-')[1])
        data['month'] = data['month'].astype(float)
        data['day'] = data.Date.apply(lambda x: x.split('-')[2])
        data['day'] = data['day'].astype(float)

        # promo interval "Jan,Apr,Jul,Oct"
        data['promojan'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Jan" in x else 0)
        data['promofeb'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Feb" in x else 0)
        data['promomar'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Mar" in x else 0)
        data['promoapr'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Apr" in x else 0)
        data['promomay'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "May" in x else 0)
        data['promojun'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Jun" in x else 0)
        data['promojul'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Jul" in x else 0)
        data['promoaug'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Aug" in x else 0)
        data['promosep'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Sep" in x else 0)
        data['promooct'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Oct" in x else 0)
        data['promonov'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Nov" in x else 0)
        data['promodec'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Dec" in x else 0)

    # # Features set.
    noisy_features = [myid]
    features = [c for c in features if c not in noisy_features]
    features_non_numeric = [c for c in features_non_numeric if c not in noisy_features]
    features.extend(['promojan','promofeb','promomar','promoapr','promomay','promojun',
        'promojul','promoaug','promosep','promooct','promonov','promodec','year','month','day'])
    # Fill NA
    class DataFrameImputer(TransformerMixin):
        # http://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn
        def __init__(self):
            """Impute missing values.
            Columns of dtype object are imputed with the most frequent value
            in column.
            Columns of other types are imputed with mean of column.
            """
        def fit(self, X, y=None):
            self.fill = pd.Series([X[c].value_counts().index[0] # mode
                if X[c].dtype == np.dtype('O') else X[c].mean() for c in X], # mean
                index=X.columns)
            return self
        def transform(self, X, y=None):
            return X.fillna(self.fill)
    train = DataFrameImputer().fit_transform(train)
    test = DataFrameImputer().fit_transform(test)
    # Pre-processing non-numberic values
    le = LabelEncoder()
    for col in features_non_numeric:
        le.fit(list(train[col])+list(test[col]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
    # Neural Network, Stochastic Gradient Descent is sensitive to feature scaling, so it is highly recommended to scale your data.
    scaler = StandardScaler()
    for col in set(features) - set(features_non_numeric) - \
      set([]): # TODO: add what not to scale
        scaler.fit(list(train[col])+list(test[col]))
        train[col] = scaler.transform(train[col])
        test[col] = scaler.transform(test[col])
    return (train,test,features,features_non_numeric)

def XGB_native(train,test,features,features_non_numeric):
    depth = 13
    eta = 0.01
    ntrees = 4000
    mcw = 3
    params = {"objective": "reg:linear",
              "booster": "gbtree",
              "eta": eta,
              "max_depth": depth,
              "min_child_weight": mcw,
              "subsample": 0.9,
              "colsample_bytree": 0.7,
              "silent": 1
              }
    print "Running with params: " + str(params)
    print "Running with features: " + str(features)
    if sample:
        cv = cross_validation.KFold(len(train), n_folds=5, shuffle=True, indices=False, random_state=1337)
        results = []
        for traincv, testcv in cv:
            dtrain = xgb.DMatrix(train[traincv][features], np.log(train[traincv][goal] + 1))
            dvalid = xgb.DMatrix(train[testcv][features], np.log(train[testcv][goal] + 1))
            watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
            gbm = xgb.train(params, dtrain, ntrees, evals=watchlist, early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=False)
            train_probs = gbm.predict(xgb.DMatrix(train[testcv][features]))
            indices = train_probs < 0
            train_probs[indices] = 0
            error = rmspe(np.exp(train_probs) - 1, train[testcv][goal].values)
            print error
            results.append(error)
        print "Results: " + str(results)
        print "Mean: " + str(np.array(results).mean())
    # EVAL OR EXPORT
    else: # Export result
        print str(datetime.datetime.now())
        dtrain = xgb.DMatrix(train[features], np.log(train[goal] + 1))
        gbm = xgb.train(params, dtrain, ntrees, feval=rmspe_xg)
        test_probs = gbm.predict(xgb.DMatrix(test[features]))
        indices = test_probs < 0
        test_probs[indices] = 0
        submission = pd.DataFrame({myid: test[myid], goal: np.exp(test_probs) - 1})
        if not os.path.exists('result/'):
            os.makedirs('result/')
        submission.to_csv("./result/dat-xgb_d%s_eta%s_ntree%s_mcw%s.csv" % (str(depth),str(eta),str(ntrees),str(mcw)) , index=False)
    # Feature importance
        if plot:
          outfile = open('xgb.fmap', 'w')
          i = 0
          for feat in features:
              outfile.write('{0}\t{1}\tq\n'.format(i, feat))
              i = i + 1
          outfile.close()
          importance = gbm.get_fscore(fmap='xgb.fmap')
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
