import pandas as pd
import time
import csv
import numpy as np
import os
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from functools import partial

pd.options.mode.chained_assignment = None

sample = False
random = False # disable for testing performance purpose i.e fix train and test dataset.

features = ['Dates','DayOfWeek','PdDistrict','Address','X','Y']
features_non_numeric = ['Dates','DayOfWeek','PdDistrict','Address']
categories = ['ARSON','ASSAULT','BAD CHECKS','BRIBERY','BURGLARY','DISORDERLY CONDUCT', # 39
              'DRIVING UNDER THE INFLUENCE','DRUG/NARCOTIC','DRUNKENNESS','EMBEZZLEMENT',
              'EXTORTION','FAMILY OFFENSES','FORGERY/COUNTERFEITING','FRAUD','GAMBLING',
              'KIDNAPPING','LARCENY/THEFT','LIQUOR LAWS','LOITERING','MISSING PERSON',
              'NON-CRIMINAL','OTHER OFFENSES','PORNOGRAPHY/OBSCENE MAT','PROSTITUTION',
              'RECOVERED VEHICLE','ROBBERY','RUNAWAY','SECONDARY CODES','SEX OFFENSES FORCIBLE',
              'SEX OFFENSES NON FORCIBLE','STOLEN PROPERTY','SUICIDE','SUSPICIOUS OCC','TREA',
              'TRESPASS','VANDALISM','VEHICLE THEFT','WARRANTS','WEAPON LAWS']

# Load data
if sample: # To run with 100k data
    if random:
        df = pd.read_csv('./data/train-100000',dtype={'Dates':pd.np.string_,'Category':pd.np.string_})
        df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
        train, test = df[df['is_train']==True], df[df['is_train']==False]
    else: # Use data set that is pre-randomized and splitted.
        train = pd.read_csv('./data/train-100000R',dtype={'Dates':pd.np.string_,'Category':pd.np.string_})
        test = pd.read_csv('./data/test-100000R',dtype={'Dates':pd.np.string_,'Category':pd.np.string_})

else:
    # To run with real data
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')

# Pre-processing non-number values
le = LabelEncoder()
for col in features_non_numeric:
    le.fit(list(train[col])+list(test[col]))
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

# Add new features:
# train['year'] = train['Dates'].apply(lambda x: x[:4])
# train['month'] = train['Dates'].apply(lambda x: x[5:7])
# train['day'] = train['Dates'].apply(lambda x: x[8:10])
# train['hour'] = train['Dates'].apply(lambda x: x[11:13])
# test['year'] = test['Dates'].apply(lambda x: x[:4])
# test['month'] = test['Dates'].apply(lambda x: x[5:7])
# test['day'] = test['Dates'].apply(lambda x: x[8:10])
# test['hour'] = test['Dates'].apply(lambda x: x[11:13])

# Define classifiers
if sample:
    classifiers = [
        AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=10)
    ]
else:
    classifiers = [# Other methods are underperformed yet take very long training time for this data set
        AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=10)
    ]

# Train
for classifier in classifiers:
    print classifier.__class__.__name__

    start = time.time()
    classifier.fit(train[list(features)], train.Category)
    # print classifier.classes_ # make sure it's following `features` order
    print '  -> Training time:', time.time() - start
# Evaluation and export result
if sample:
    # Test results
    for classifier in classifiers:
        print classifier.__class__.__name__
        print 'Log Loss:'
        print log_loss(test.Category.values.astype(pd.np.string_), classifier.predict_proba(test[features]))

else: # Export result
    for classifier in classifiers:
        if not os.path.exists('result/'):
            os.makedirs('result/')
        # TODO: fix this shit
        # test['Id'] values will get converted to float since column_stack will result in array
        predictions = np.column_stack((test['Id'], classifier.predict_proba(test[features]))).tolist()
        predictions = [[int(i[0])] + i[1:] for i in predictions]
        csvfile = 'result/' + classifier.__class__.__name__ + '-submit.csv'
        with open(csvfile, 'w') as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(['Id'] + (categories))
            writer.writerows(predictions)
