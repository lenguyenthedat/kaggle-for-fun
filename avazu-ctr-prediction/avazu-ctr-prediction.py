import pandas as pd
import time
import csv
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor

sample = True
features = ["C1","banner_pos","device_type","device_conn_type","C14","C15","C16","C17","C18","C19","C20","C21","site_category","app_id", "app_domain","app_category","device_model"]

if sample:
    #To run with 100k data
    df = pd.read_csv('./data/train-100000')
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
    train, test = df[df['is_train']==True], df[df['is_train']==False]
else:
    # To run with real data
    train = pd.read_csv('./data/train.gz',compression='gzip')
    test = pd.read_csv('./data/test.gz',compression='gzip')


# Pre-processing non-number values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

for col in ['site_category','app_id','app_domain','app_category','device_model']:
    le.fit(list(train[col])+list(test[col]))
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

# Define regressors
regressors = [ 
    ExtraTreesRegressor(n_estimators=10),
    RandomForestRegressor(n_estimators=10),
    KNeighborsRegressor(10),
    LDA(),
    QDA(),
    GaussianNB(),
    DecisionTreeRegressor(),
    SGDRegressor(n_iter=15,verbose=5, learning_rate='invscaling',eta0=0.0000000001)
]

# Train
for regressor in regressors:
    print regressor.__class__.__name__
    start = time.time()
    regressor.fit(train[list(features)], train.click)
    print "  -> Training time:", time.time() - start

# Evaluation and export result
if sample:
    for regressor in regressors:
        print regressor.__class__.__name__
        print np.sqrt(sum(pow(test.click - regressor.predict(test[features]),2)) / float(len(test)))

else: # Export result
    for regressor in regressors:
        predictions = np.column_stack((test["id"],regressor.predict(test[features])))
        csvfile = "result/" + regressor.__class__.__name__ + "-submit.csv"
        with open(csvfile, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(["id","click"])
            writer.writerows(predictions)
