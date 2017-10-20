import pandas as pd
import time
import csv
import numpy as np
import os
import itertools
import utils

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sknn.mlp import Classifier, Layer, Convolution

pd.options.mode.chained_assignment = None

sample = True

goal = 'Label' # rmb to change your training data column from `label` to `Label`
myid = 'ImageId'

# Load data
if sample: # To run with 5k data
    train, test = utils.load_data('./data/train-1000.csv')
else:
    # To run with real data
    train, test = utils.read_csv_files('./data/train.csv', './data/test.csv')

features = test.columns.tolist()
if sample:
    features.remove('is_train')
    features.remove('Label')

# Neural Network, Stochastic Gradient Descent is sensitive to feature scaling, so it is highly recommended to scale your data.
scaler = StandardScaler()
for col in features:
    scaler.fit(list(train[col])+list(test[col]))
    train[col] = scaler.transform(train[col])
    test[col] = scaler.transform(test[col])

MyNNClassifier1 = Classifier(
                    layers=[
                        Layer('Rectifier', units=100),
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

MyNNClassifier2 = Classifier(
                    layers=[
                        Convolution("Rectifier", channels=16, kernel_shape=(5,5)),
                        Convolution("Rectifier", channels=16, kernel_shape=(5,5)),
                        Convolution("Rectifier", channels=16, kernel_shape=(5,5)),
                        Layer('Rectifier', units=100),
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
        MyNNClassifier1,
        MyNNClassifier2,
        RandomForestClassifier(n_estimators=256, max_depth=64),
        XGBClassifier(n_estimators=128, max_depth=2)
    ]
else:
    classifiers = [# Other methods are underperformed yet take very long training time for this data set
        MyNNClassifier1,
        MyNNClassifier2,
        RandomForestClassifier(n_estimators=256, max_depth=64),
        XGBClassifier(n_estimators=128, max_depth=2)
    ]

# Train
for classifier in classifiers:
    print classifier.__class__.__name__
    start = time.time()
    classifier.fit(np.array(train[list(features)]), train[goal])
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
        predictions = classifier.predict(np.array(test[features]))
        try: # try to flatten a list that might be flattenable.
            predictions = list(itertools.chain.from_iterable(predictions))
        except:
            pass
        csvfile = 'result/' + classifier.__class__.__name__ + '-'+ str(count) + '-submit.csv'
        with open(csvfile, 'w') as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow([myid,goal])
            for i in range(0, len(predictions)):
                writer.writerow([i+1,predictions[i]])
