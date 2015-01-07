import pandas as pd
import time
import csv
import numpy as np
import scipy as sp
from sklearn.metrics import log_loss
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

sample = True
random = False # disable for testing performance purpose i.e fix train and test dataset.

features = ['hour','day','dow','C1','banner_pos','device_type','device_conn_type','C14','C15','C16','C17','C18','C19','C20','C21','site_id','site_domain','site_category','app_id', 'app_domain','app_category','device_model','device_id','device_ip']

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

# Define classifiers
if sample:
    classifiers = [
        ExtraTreesClassifier(n_estimators=100),
        RandomForestClassifier(n_estimators=100),
        KNeighborsClassifier(n_neighbors=100, weights='uniform', algorithm='auto', leaf_size=100, p=2, metric='minkowski', metric_params=None),
        LDA(),
        GaussianNB(),
        DecisionTreeClassifier(),
        GradientBoostingClassifier(),
        SGDClassifier(loss='log',n_iter=30,verbose=5,learning_rate='invscaling',eta0=0.0000000001)
    ]
else:
    classifiers = [# Other methods are underperformed yet take very long training time for this data set
        GradientBoostingClassifier(), # takes ~5 hours to train on a 16GB i5 machine.
        SGDClassifier(loss='log',n_iter=30,verbose=5,learning_rate='invscaling',eta0=0.0000000001)
    ]

# Train
for classifier in classifiers:
    print classifier.__class__.__name__
    start = time.time()
    classifier.fit(train[list(features)], train.click)
    print '  -> Training time:', time.time() - start

# Evaluation and export result
if sample:
    for classifier in classifiers:
        print classifier.__class__.__name__
        print log_loss(test.click,np.compress([False, True], classifier.predict_proba(test[features]), axis=1))

else: # Export result
    for classifier in classifiers:
        predictions = np.column_stack((test['id'],np.compress([False, True], classifier.predict_proba(test[features]), axis=1)))
        csvfile = 'result/' + classifier.__class__.__name__ + '-submit.csv'
        with open(csvfile, 'w') as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(['id','click'])
            writer.writerows(predictions)
