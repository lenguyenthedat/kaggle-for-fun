import pandas as pd
import time
import csv
import numpy as np
import os
import utils
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sknn.mlp import Classifier, Layer

pd.options.mode.chained_assignment = None

sample = True
random = False  # disable for testing performance purpose i.e fix train and test dataset.

features = ['hour', 'day', 'dow', 'C1', 'banner_pos', 'device_type', 'device_conn_type',
            'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'site_id', 'site_domain',
            'site_category', 'app_id', 'app_domain', 'app_category', 'device_model',
            'device_id', 'device_ip']

# Load data
if sample:
    train, test = utils.load_data(full_csv_file='./data/train-100000',
                                  train_csv_file='./data/train-100000R',
                                  test_csv_file='./data/test-100000R',
                                  random=random, dtype={'id': pd.np.string_})
else:
    train, test = utils.load_data(train_csv_file='./data/train.gz',
                                  test_csv_file='./data/test.gz',
                                  compression='gzip')

# Pre-processing non-number values
le = LabelEncoder()
for col in ['site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_model',
            'device_id', 'device_ip']:
    le.fit(list(train[col]) + list(test[col]))
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

# Stochastic Gradient Descent is sensitive to feature scaling, so it is highly recommended to scale your data.
scaler = StandardScaler()
for col in ['C1', 'banner_pos', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20',
            'C21']:
    scaler.fit(list(train[col]) + list(test[col]))
    train[col] = scaler.transform(train[col])
    test[col] = scaler.transform(test[col])

# Add new features:
train['day'] = train['hour'].apply(lambda x: (x - x % 10000) / 1000000)  # day
train['dow'] = train['hour'].apply(lambda x: ((x - x % 10000) / 1000000) % 7)  # day of week
train['hour'] = train['hour'].apply(lambda x: x % 10000 / 100)  # hour
test['day'] = test['hour'].apply(lambda x: (x - x % 10000) / 1000000)  # day
test['dow'] = test['hour'].apply(lambda x: ((x - x % 10000) / 1000000) % 7)  # day of week
test['hour'] = test['hour'].apply(lambda x: x % 10000 / 100)  # hour

# Remove outliner
for col in ['C18', 'C20', 'C21']:
    # keep only the ones that are within +3 to -3 standard deviations in the column col,
    train = train[np.abs(train[col] - train[col].mean()) <= (3 * train[col].std())]
    # or if you prefer the other way around
    train = train[~(np.abs(train[col] - train[col].mean()) > (3 * train[col].std()))]

# Define classifiers
if sample:
    classifiers = [
        Classifier(
            layers=[
                # Convolution("Rectifier", channels=10, pool_shape=(2,2), kernel_shape=(3, 3)),
                Layer('Rectifier', units=200),
                Layer('Softmax')],
            learning_rate=0.01,
            learning_rule='momentum',
            learning_momentum=0.9,
            batch_size=25,
            valid_size=0.1,
            # valid_set=(X_test, y_test),
            n_stable=10,
            n_iter=10,
            verbose=True),
        ExtraTreesClassifier(n_estimators=100),
        RandomForestClassifier(n_estimators=100),
        KNeighborsClassifier(n_neighbors=100, weights='uniform', algorithm='auto',
                             leaf_size=100, p=2, metric='minkowski'),
        LDA(),
        GaussianNB(),
        DecisionTreeClassifier(),
        GradientBoostingClassifier(),
        SGDClassifier(loss='log', n_iter=30, verbose=5, learning_rate='invscaling', eta0=0.0000000001),
        XGBClassifier(n_estimators=512, max_depth=4)
    ]
else:
    classifiers = [  # Other methods are underperformed yet take very long training time for this data set
        GradientBoostingClassifier(),  # takes ~5 hours to train on a 16GB i5 machine.
        SGDClassifier(loss='log', n_iter=30, verbose=5, learning_rate='invscaling', eta0=0.0000000001),
        XGBClassifier(n_estimators=512, max_depth=4)
    ]

# Train
for classifier in classifiers:
    print classifier.__class__.__name__
    start = time.time()
    classifier.fit(np.array(train[list(features)]), train.click)
    print '  -> Training time:', time.time() - start

# Evaluation and export result
if sample:
    for classifier in classifiers:
        print classifier.__class__.__name__
        print 'Log Loss:'
        print log_loss(test.click.values, classifier.predict_proba(np.array(test[features])))
        print 'RMSE:'
        print mean_squared_error(test.click.values,
                                 np.compress([False, True], classifier.predict_proba(np.array(test[features])),
                                             axis=1)) ** 0.5  # RMSE

else:  # Export result
    for classifier in classifiers:
        if not os.path.exists('result/'):
            os.makedirs('result/')
        predictions = np.column_stack(
            (test['id'], np.compress([False, True], classifier.predict_proba(np.array(test[features])), axis=1)))
        csvfile = 'result/' + classifier.__class__.__name__ + '-submit.csv'
        with open(csvfile, 'w') as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(['id', 'click'])
            writer.writerows(predictions)
