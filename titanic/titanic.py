import pandas as pd
import time
import csv
import numpy as np
import os
import math
import xgboost as xgb
import utils
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sknn.mlp import Classifier, Layer
import warnings

warnings.filterwarnings('ignore')

pd.options.mode.chained_assignment = None

sample = True

goal = 'Survived'
myid = 'PassengerId'

# Load data
if sample:  # To run with 100k data
    df = pd.read_csv('./data/train.csv')
    df['is_train'] = (df[myid] % 10) >= 5
    train, test = df[df['is_train'] == True], df[df['is_train'] == False]
else:
    # To run with real data
    train, test = utils.read_csv_files('./data/train.csv', './data/test.csv')

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
features_non_numeric = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Fare', 'Embarked',
                        'firstname', 'titles']
train['firstname'] = train['Name'].apply(lambda x: x.split(',')[0])
test['firstname'] = test['Name'].apply(lambda x: x.split(',')[0])
train['titles'] = train['Name'].apply(lambda x: x.split(',')[1].split('.')[0])
test['titles'] = test['Name'].apply(lambda x: x.split(',')[1].split('.')[0])
train['Age'] = train['Age'].apply(lambda x: 35 if math.isnan(x) else x)
test['Age'] = test['Age'].apply(lambda x: 35 if math.isnan(x) else x)
train['Fare'] = train['Fare'].apply(lambda x: 30 if math.isnan(x) else x)
test['Fare'] = test['Fare'].apply(lambda x: 30 if math.isnan(x) else x)

# Pre-processing non-number values
le = LabelEncoder()
for col in features_non_numeric:
    le.fit(list(train[col]) + list(test[col]))
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

# Neural Network, Stochastic Gradient Descent is sensitive to feature scaling,
# so it is highly recommended to scale your data.
scaler = StandardScaler()
for col in features_non_numeric:
    scaler.fit(list(train[col]) + list(test[col]))
    train[col] = scaler.transform(train[col])
    test[col] = scaler.transform(test[col])

classifiers = [
    xgb.XGBClassifier(gamma=10, max_depth=100, n_estimators=50000),
    Classifier(
        layers=[
            Layer("Tanh", units=200),
            Layer("Sigmoid", units=200),
            Layer('Rectifier', units=200),
            Layer('Softmax')],
        learning_rate=0.05,
        learning_rule='sgd',
        learning_momentum=0.5,
        batch_size=100,
        valid_size=0.05,
        n_stable=100,
        n_iter=100,
        verbose=True)
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

else:  # Export result
    for classifier in classifiers:
        if not os.path.exists('result/'):
            os.makedirs('result/')
        # TODO: fix this shit
        # test[myid] values will get converted to float since column_stack will result in array
        predictions = np.column_stack((test[myid], classifier.predict(np.array(test[features])))).tolist()
        predictions = [[int(i[0])] + i[1:] for i in predictions]
        csvfile = 'result/' + classifier.__class__.__name__ + '-submit.csv'
        with open(csvfile, 'w') as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow([myid, 'Survived'])
            writer.writerows(predictions)
