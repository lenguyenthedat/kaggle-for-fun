import pandas as pd
import time
import csv
import numpy as np
import os

from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sknn.mlp import Classifier, Layer

pd.options.mode.chained_assignment = None

sample = False
random = False # disable for testing performance purpose i.e fix train and test dataset.

features = ['DayOfWeek','PdDistrict','Address','X','Y']
features_non_numeric = ['Dates','DayOfWeek','PdDistrict','Address']
goal = 'Category'
myid = 'Id'

# Load data
if sample: # To run with 100k data
    if random:
        df = pd.read_csv('./data/train-100000',dtype={'Category':pd.np.string_})
        df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
        train, test = df[df['is_train']==True], df[df['is_train']==False]
    else: # Use data set that is pre-randomized and splitted.
        train = pd.read_csv('./data/train-100000R',dtype={'Category':pd.np.string_})
        test = pd.read_csv('./data/test-100000R',dtype={'Category':pd.np.string_})
else:
    # To run with real data
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')

features = ['Dates','hour','dark','DayOfWeek','PdDistrict','StreetNo','Address','X','Y']
train['StreetNo'] = train['Address'].apply(lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)
train['Address'] = train['Address'].apply(lambda x: x.split(' ', 1)[1] if x.split(' ', 1)[0].isdigit() else x)
test['StreetNo'] = test['Address'].apply(lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)
test['Address'] = test['Address'].apply(lambda x: x.split(' ', 1)[1] if x.split(' ', 1)[0].isdigit() else x)
train['hour'] = train['Dates'].apply(lambda x: x[11:13] if len(x) > 4 else 12)
train['dark'] = train['Dates'].apply(lambda x: 1 if (len(x) > 4 and x[11:13] >= 18 and x[11:13] < 6) else 0)
test['hour'] = test['Dates'].apply(lambda x: x[11:13] if len(x) > 4 else 12)
test['dark'] = test['Dates'].apply(lambda x: 1 if (len(x) > 4 and x[11:13] >= 18 and x[11:13] < 6) else 0)

# Pre-processing non-number values
le = LabelEncoder()
for col in features_non_numeric:
    le.fit(list(train[col])+list(test[col]))
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

# Neural Network, Stochastic Gradient Descent is sensitive to feature scaling, so it is highly recommended to scale your data.
scaler = StandardScaler()
for col in features:
    scaler.fit(list(train[col])+list(test[col]))
    train[col] = scaler.transform(train[col])
    test[col] = scaler.transform(test[col])

MyNNClassifier = Classifier(
                    layers=[
                        Layer("Tanh", units=100),
                        Layer("Tanh", units=100),
                        Layer("Tanh", units=100),
                        Layer("Sigmoid", units=100),
                        Layer('Softmax')],
                    learning_rate=0.01,
                    learning_rule='momentum',
                    learning_momentum=0.9,
                    batch_size=100,
                    valid_size=0.01,
                    n_stable=20,
                    n_iter=200,
                    verbose=True)

# Define classifiers
if sample:
    classifiers = [
        RandomForestClassifier(max_depth=16,n_estimators=1024),
        GradientBoostingClassifier(n_estimators=10, learning_rate=1.0,max_depth=5, random_state=0),
        KNeighborsClassifier(n_neighbors=100, weights='uniform', algorithm='auto', leaf_size=100, p=10, metric='minkowski'),
        AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=8), algorithm="SAMME.R", n_estimators=128),
        XGBClassifier(max_depth=8,n_estimators=128),
        MyNNClassifier
    ]
else:
    classifiers = [# Other methods are underperformed yet take very long training time for this data set
        RandomForestClassifier(max_depth=16,n_estimators=1024)
    ]

count = 0
for classifier in classifiers:
    # Train
    print classifier.__class__.__name__
    start = time.time()
    classifier.fit(np.array(train[list(features)]), train[goal])
    print '  -> Training time:', time.time() - start

    # Evaluation
    if sample:
        # Test results
        print classifier.__class__.__name__
        print 'Log Loss:'
        print log_loss(test[goal].values.astype(pd.np.string_),
                       classifier.predict_proba(np.array(test[features])))

    else: # Export result
        count += 1
        if not os.path.exists('result/'):
            os.makedirs('result/')
        # TODO: fix this shit
        # test[myid] values will get converted to float since column_stack will result in array
        predictions = np.column_stack((test[myid], classifier.predict_proba(np.array(test[features])))).tolist()
        predictions = [[int(i[0])] + i[1:] for i in predictions]
        csvfile = 'result/' + classifier.__class__.__name__ + '-'+ str(count) + '-submit.csv'
        with open(csvfile, 'w') as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow([myid,'ARSON','ASSAULT','BAD CHECKS','BRIBERY','BURGLARY','DISORDERLY CONDUCT',
                             'DRIVING UNDER THE INFLUENCE','DRUG/NARCOTIC','DRUNKENNESS','EMBEZZLEMENT',
                             'EXTORTION','FAMILY OFFENSES','FORGERY/COUNTERFEITING','FRAUD','GAMBLING',
                             'KIDNAPPING','LARCENY/THEFT','LIQUOR LAWS','LOITERING','MISSING PERSON',
                             'NON-CRIMINAL','OTHER OFFENSES','PORNOGRAPHY/OBSCENE MAT','PROSTITUTION',
                             'RECOVERED VEHICLE','ROBBERY','RUNAWAY','SECONDARY CODES','SEX OFFENSES FORCIBLE',
                             'SEX OFFENSES NON FORCIBLE','STOLEN PROPERTY','SUICIDE','SUSPICIOUS OCC','TREA',
                             'TRESPASS','VANDALISM','VEHICLE THEFT','WARRANTS','WEAPON LAWS'])
            writer.writerows(predictions)
