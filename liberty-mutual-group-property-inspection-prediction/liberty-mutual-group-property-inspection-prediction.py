import pandas as pd
import time
import csv
import numpy as np
import os
from sklearn.metrics import log_loss, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sknn.mlp import Classifier, Layer

pd.options.mode.chained_assignment = None

sample = False
random = False # disable for testing performance purpose i.e fix train and test dataset.

features = ['T1_V1','T1_V2','T1_V3','T1_V4','T1_V5','T1_V6','T1_V7','T1_V8','T1_V9',
            'T1_V10','T1_V11','T1_V12','T1_V13','T1_V14','T1_V15','T1_V16','T1_V17',
            'T2_V1','T2_V2','T2_V3','T2_V4','T2_V5','T2_V6','T2_V7','T2_V8','T2_V9',
            'T2_V10','T2_V11','T2_V12','T2_V13','T2_V14','T2_V15']
features_non_numeric = features
goal = 'Hazard'
myid = 'Id'

# Load data
if sample: # To run with 100k data
    df = pd.read_csv('./data/train.csv',dtype={'Category':pd.np.string_})
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
    train, test = df[df['is_train']==True], df[df['is_train']==False]
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

# Neural Network, Stochastic Gradient Descent is sensitive to feature scaling, so it is highly recommended to scale your data.
scaler = StandardScaler()
for col in features:
    scaler.fit(list(train[col])+list(test[col]))
    train[col] = scaler.transform(train[col])
    test[col] = scaler.transform(test[col])

MyNNClassifier1 = Classifier(
                    layers=[
                        Layer('Rectifier', units=500),
                        Layer('Rectifier', units=500),
                        Layer('Rectifier', units=500),
                        Layer('Softmax')],
                    learning_rate=0.01,
                    learning_rule='momentum',
                    learning_momentum=0.9,
                    batch_size=100,
                    valid_size=0.01,
                    n_stable=100,
                    n_iter=100,
                    verbose=True)

MyNNClassifier2 = Classifier(
                    layers=[
                        Layer("Tanh", units=500),
                        Layer("Tanh", units=500),
                        Layer('Tanh', units=500),
                        Layer('Softmax')],
                    learning_rate=0.01,
                    learning_rule='momentum',
                    learning_momentum=0.9,
                    batch_size=100,
                    valid_size=0.01,
                    n_stable=100,
                    n_iter=100,
                    verbose=True)

MyNNClassifier3 = Classifier(
                    layers=[
                        Layer("Sigmoid", units=500),
                        Layer("Sigmoid", units=500),
                        Layer('Sigmoid', units=500),
                        Layer('Softmax')],
                    learning_rate=0.01,
                    learning_rule='momentum',
                    learning_momentum=0.9,
                    batch_size=100,
                    valid_size=0.01,
                    n_stable=100,
                    n_iter=100,
                    verbose=True)

MyNNClassifier4 = Classifier(
                    layers=[
                        Layer("Tanh", units=500),
                        Layer("Sigmoid", units=500),
                        Layer('Rectifier', units=500),
                        Layer('Softmax')],
                    learning_rate=0.01,
                    learning_rule='momentum',
                    learning_momentum=0.9,
                    batch_size=100,
                    valid_size=0.01,
                    n_stable=100,
                    n_iter=100,
                    verbose=True)
# Define classifiers
if sample:
    classifiers = [
        RandomForestClassifier(n_estimators=100,verbose=True),
        GradientBoostingClassifier(n_estimators=10, learning_rate=1.0,max_depth=5, random_state=0),
        KNeighborsClassifier(n_neighbors=100, weights='uniform', algorithm='auto', leaf_size=100, p=10, metric='minkowski'),
        AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=20), algorithm="SAMME.R", n_estimators=10),
        MyNNClassifier1,
        MyNNClassifier2,
        MyNNClassifier3,
        MyNNClassifier4
    ]
else:
    classifiers = [# Other methods are underperformed yet take very long training time for this data set
        KNeighborsClassifier(n_neighbors=100, weights='uniform', algorithm='auto', leaf_size=100, p=10, metric='minkowski'),
        RandomForestClassifier(n_estimators=2000,verbose=True),
        MyNNClassifier1,
        MyNNClassifier2,
        MyNNClassifier3,
        MyNNClassifier4
    ]

# Train
for classifier in classifiers:
    print classifier.__class__.__name__
    start = time.time()
    classifier.fit(np.array(train[list(features)]), train[goal])
        # use np.array to avoid this stupid error `IndexError: indices are out-of-bounds`
        # ref: http://stackoverflow.com/questions/27332557/dbscan-indices-are-out-of-bounds-python
    # print classifier.classes_ # make sure it's following `features` order
    print '  -> Training time:', time.time() - start
# Evaluation and export result
if sample:
    # Test results
    for classifier in classifiers:
        print classifier.__class__.__name__
        print 'Accuracy Score:'
        print accuracy_score(test[goal].values,
                       classifier.predict(np.array(test[features])))

else: # Export result
    count = 0
    for classifier in classifiers:
        count += 1
        if not os.path.exists('result/'):
            os.makedirs('result/')
        # TODO: fix this shit
        # test[myid] values will get converted to float since column_stack will result in array
        predictions = np.column_stack((test[myid], classifier.predict(np.array(test[features])))).tolist()
        predictions = [[int(i[0])] + i[1:] for i in predictions]
        csvfile = 'result/' + classifier.__class__.__name__ + '-'+ str(count) + '-submit.csv'
        with open(csvfile, 'w') as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow([myid,goal])
            writer.writerows(predictions)
