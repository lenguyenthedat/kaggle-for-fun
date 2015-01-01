import pandas as pd
import time
import csv
import numpy as np
import scipy as sp
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

sample = True
random = False # disable for testing performance purpose i.e fix train and test dataset.

features = ['hour','day','dow','C1','banner_pos','device_type','device_conn_type','C14','C15','C16','C17','C18','C19','C20','C21','site_id','site_domain','site_category','app_id', 'app_domain','app_category','device_model','device_id','device_ip']

# logloss implementation
def logloss(act, pred):
    ' Vectorised computation of logloss '

    #cap in official Kaggle implementation,
    #per forums/t/1576/r-code-for-logloss
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)

    #compute logloss function (vectorised)
    ll = sum(   act*sp.log(pred) +
                sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

# Load data
if sample:
    #To run with 100k data
    if random:
        df = pd.read_csv('./data/train-100000')
        df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
        train, test = df[df['is_train']==True], df[df['is_train']==False]
    else:
        train = pd.read_csv('./data/train-100000R')
        test = pd.read_csv('./data/test-100000R')

else:
    # To run with real data
    train = pd.read_csv('./data/train.gz',compression='gzip')
    test = pd.read_csv('./data/test.gz',compression='gzip')

# Pre-processing non-number values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

for col in ['site_id','site_domain','site_category','app_id','app_domain','app_category','device_model','device_id','device_ip']:
    le.fit(list(train[col])+list(test[col]))
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

# Stochastic Gradient Descent is sensitive to feature scaling, so it is highly recommended to scale your data.
for col in ['C1','banner_pos','device_type','device_conn_type','C14','C15','C16','C17','C18','C19','C20','C21']:
    scaler = StandardScaler()
    scaler.fit(list(train[col])+list(test[col]))
    train[col] = scaler.transform(train[col])
    test[col] = scaler.transform(test[col])

# Add new features:
train['day'] = train['hour'].apply(lambda x: (x - x%10000)/1000000) # day
train['dow'] = train['hour'].apply(lambda x: ((x - x%10000)/1000000)%7) # day of week
train['hour'] = train['hour'].apply(lambda x: x%10000/100) # hour
test['day'] = test['hour'].apply(lambda x: (x - x%10000)/1000000) # day
test['dow'] = test['hour'].apply(lambda x: ((x - x%10000)/1000000)%7) # day of week
test['hour'] = test['hour'].apply(lambda x: x%10000/100) # hour

# Define regressors
if sample:
    regressors = [
        ExtraTreesRegressor(n_estimators=10),
        RandomForestRegressor(n_estimators=10),
        KNeighborsRegressor(10),
        LDA(),
        QDA(),
        GaussianNB(),
        DecisionTreeRegressor(),
        GradientBoostingRegressor(),0
        SGDRegressor(n_iter=30,verbose=5,learning_rate='invscaling',eta0=0.0000000001)
    ]
else:
    regressors = [# Other methods are underperformed yet take very long training time for this data set
        GradientBoostingRegressor(), # takes ~5 hours to train on a 16GB i5 machine.
        SGDRegressor(n_iter=30,verbose=5,learning_rate='invscaling',eta0=0.0000000001)
    ]

# Train
for regressor in regressors:
    print regressor.__class__.__name__
    start = time.time()
    regressor.fit(train[list(features)], train.click)
    print '  -> Training time:', time.time() - start

# Evaluation and export result
if sample:
    for regressor in regressors:
        print regressor.__class__.__name__
        print logloss(test.click,regressor.predict(test[features]))

else: # Export result
    for regressor in regressors:
        predictions = np.column_stack((test['id'],regressor.predict(test[features])))
        csvfile = 'result/' + regressor.__class__.__name__ + '-submit.csv'
        with open(csvfile, 'w') as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(['id','click'])
            writer.writerows(predictions)
