import pandas as pd
import numpy as np
from sklearn import cross_validation
import xgboost as xgb


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

# Gather some features
def build_features(features, data):
    # remove NaNs
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
    # Use some properties directly
    # features.extend(['Store', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
    #                  'CompetitionOpenSinceYear', 'Promo', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear'])
    features.extend(['Store', 'CompetitionDistance', 'Promo', 'Promo2'])

    # add some more with a bit of preprocessing
    features.append('SchoolHoliday')
    data['SchoolHoliday'] = data['SchoolHoliday'].astype(float)
    #
    #features.append('StateHoliday')
    #data.loc[data['StateHoliday'] == 'a', 'StateHoliday'] = '1'
    #data.loc[data['StateHoliday'] == 'b', 'StateHoliday'] = '2'
    #data.loc[data['StateHoliday'] == 'c', 'StateHoliday'] = '3'
    #data['StateHoliday'] = data['StateHoliday'].astype(float)

    features.append('DayOfWeek')
    features.append('month')
    features.append('day')
    features.append('year')
    data['year'] = data.Date.apply(lambda x: x.split('-')[0])
    data['year'] = data['year'].astype(float)
    data['month'] = data.Date.apply(lambda x: x.split('-')[1])
    data['month'] = data['month'].astype(float)
    data['day'] = data.Date.apply(lambda x: x.split('-')[2])
    data['day'] = data['day'].astype(float)

    features.append('StoreType')
    data.loc[data['StoreType'] == 'a', 'StoreType'] = '1'
    data.loc[data['StoreType'] == 'b', 'StoreType'] = '2'
    data.loc[data['StoreType'] == 'c', 'StoreType'] = '3'
    data.loc[data['StoreType'] == 'd', 'StoreType'] = '4'
    data['StoreType'] = data['StoreType'].astype(float)

    features.append('Assortment')
    data.loc[data['Assortment'] == 'a', 'Assortment'] = '1'
    data.loc[data['Assortment'] == 'b', 'Assortment'] = '2'
    data.loc[data['Assortment'] == 'c', 'Assortment'] = '3'
    data['Assortment'] = data['Assortment'].astype(float)

# print("Load the training, test and store data using pandas")
train = pd.read_csv("./data/train.csv",dtype={'StateHoliday':pd.np.string_})
test = pd.read_csv("./data/test.csv",dtype={'StateHoliday':pd.np.string_})
store = pd.read_csv("./data/store.csv",dtype={'StateHoliday':pd.np.string_})

# print("Assume store open, if not provided")
test.fillna(1, inplace=True)

# print("Consider only open stores for training. Closed stores wont count into the score.")
train = train[train["Open"] != 0]

# print("Join with store")
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

features = []

# print("augment features")
build_features(features, train)
build_features([], test)
# print(features)

depth = 10
eta = 0.02
ntrees = 3000

params = {"objective": "reg:linear",
          "booster": "gbtree",
          "eta": eta,
          "max_depth": depth,
          "subsample": 0.9,
          "colsample_bytree": 0.7,
          "silent": 1
          }

# print("Train a XGBoost model")

#train = train.sort(['Date'])
# print(train.tail(1)['Date'])

# X_train = train[train['Date'] <= '2015-06-25']
# X_test = train[train['Date'] > '2015-06-25']

tsize = 0.05
X_train, X_test = cross_validation.train_test_split(train, test_size=tsize)

# vsize = 10000
# X_train, X_test = train.head(len(train) - vsize), train.tail(vsize)
dtrain = xgb.DMatrix(X_train[features], np.log(X_train["Sales"] + 1))
dvalid = xgb.DMatrix(X_test[features], np.log(X_test["Sales"] + 1))
dtest = xgb.DMatrix(test[features])
watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
gbm = xgb.train(params, dtrain, ntrees, evals=watchlist, early_stopping_rounds=100, feval=rmspe_xg)

# print("Validating")
train_probs = gbm.predict(xgb.DMatrix(X_test[features]))
indices = train_probs < 0
train_probs[indices] = 0
error = rmspe(np.exp(train_probs) - 1, X_test['Sales'].values)
print('error', error)

# print("Make predictions on the test set")
test_probs = gbm.predict(xgb.DMatrix(test[features]))
indices = test_probs < 0
test_probs[indices] = 0
submission = pd.DataFrame({"Id": test["Id"], "Sales": np.exp(test_probs) - 1})

# closed stores sale none
def fix_closed(row):
    if test[test['Id'] == row['Id']]['Open'].values[0] == 0:
        return 0
    else:
        return row['Sales']
submission['Sales'] = submission.apply(fix_closed, axis=1)

print features
print params
print ("./result/xgb_d%s_eta%s_ntree%s.csv" % (str(depth),str(eta),str(ntrees)))
submission.to_csv("./result/xgb_d%s_eta%s_ntree%s.csv" % (str(depth),str(eta),str(ntrees)) , index=False)