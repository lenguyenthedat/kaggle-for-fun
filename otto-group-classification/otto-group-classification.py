import pandas as pd
import time
import csv
import numpy as np
import os
import utils
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sknn.mlp import Classifier, Layer
from sklearn.preprocessing import StandardScaler

pd.options.mode.chained_assignment = None

sample = False

features = ['feat_1','feat_2','feat_3','feat_4','feat_5','feat_6','feat_7',
            'feat_8','feat_9','feat_10','feat_11','feat_12','feat_13','feat_14',
            'feat_15','feat_16','feat_17','feat_18','feat_19','feat_20','feat_21',
            'feat_22','feat_23','feat_24','feat_25','feat_26','feat_27','feat_28',
            'feat_29','feat_30','feat_31','feat_32','feat_33','feat_34','feat_35',
            'feat_36','feat_37','feat_38','feat_39','feat_40','feat_41','feat_42',
            'feat_43','feat_44','feat_45','feat_46','feat_47','feat_48','feat_49',
            'feat_50','feat_51','feat_52','feat_53','feat_54','feat_55','feat_56',
            'feat_57','feat_58','feat_59','feat_60','feat_61','feat_62','feat_63',
            'feat_64','feat_65','feat_66','feat_67','feat_68','feat_69','feat_70',
            'feat_71','feat_72','feat_73','feat_74','feat_75','feat_76','feat_77',
            'feat_78','feat_79','feat_80','feat_81','feat_82','feat_83','feat_84',
            'feat_85','feat_86','feat_87','feat_88','feat_89','feat_90','feat_91',
            'feat_92','feat_93']

# Load data
if sample: # To run with 100k data
    df = pd.read_csv('./data/train.csv',dtype={'target':pd.np.string_})
    train, test = utils.random_train_test_split(df)

else:
    # To run with real data
    train, test = utils.read_csv_files('./data/train.csv', './data/test.csv')

# Neural Network, Stochastic Gradient Descent is sensitive to feature scaling, so it is highly recommended to scale your data.
scaler = StandardScaler()
for col in features:
    scaler.fit(list(train[col])+list(test[col]))
    train[col] = scaler.transform(train[col])
    test[col] = scaler.transform(test[col])

# Define classifiers

if sample:
    classifiers = [
        KNeighborsClassifier(n_neighbors=100, weights='uniform', algorithm='auto', leaf_size=100, p=10, metric='minkowski'),
        RandomForestClassifier(n_estimators=100,verbose=True),
        GradientBoostingClassifier(n_estimators=10, learning_rate=1.0,max_depth=5, random_state=0),
        AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=20),
                         algorithm="SAMME.R",
                         n_estimators=10),
        Classifier(
            layers=[
                Layer("Tanh", units=200),
                Layer("Sigmoid", units=200),
                Layer('Rectifier', units=200),
                Layer('Softmax')],
            learning_rate=0.01,
            learning_rule='momentum',
            learning_momentum=0.9,
            batch_size=1000,
            valid_size=0.01,
            n_stable=100,
            n_iter=100,
            verbose=True)
    ]
else:
    classifiers = [# Other methods are underperformed yet take very long training time for this data set
        AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=20),
                     algorithm="SAMME.R",
                     n_estimators=10),
        Classifier(
            layers=[
                Layer("Tanh", units=200),
                Layer("Sigmoid", units=200),
                Layer('Rectifier', units=200),
                Layer('Softmax')],
            learning_rate=0.01,
            learning_rule='momentum',
            learning_momentum=0.9,
            batch_size=1000,
            valid_size=0.01,
            n_stable=100,
            n_iter=100,
            verbose=True)
    ]

# Train
for classifier in classifiers:
    print classifier.__class__.__name__
    start = time.time()
    classifier.fit(np.array(train[list(features)]), train.target)
        # use np.array to avoid this stupid error `IndexError: indices are out-of-bounds`
        # ref: http://stackoverflow.com/questions/27332557/dbscan-indices-are-out-of-bounds-python
    # print classifier.classes_ # make sure it's following `features` order
    print '  -> Training time:', time.time() - start
# Evaluation and export result
if sample:
    # Test results
    for classifier in classifiers:
        print classifier.__class__.__name__
        print 'Log Loss:'
        print log_loss(test.target.values.astype(pd.np.string_),
                       classifier.predict_proba(np.array(test[features])))

else: # Export result
    for classifier in classifiers:
        if not os.path.exists('result/'):
            os.makedirs('result/')
        # TODO: fix this shit
        # test['id'] values will get converted to float since column_stack will result in array
        predictions = np.column_stack((test['id'], classifier.predict_proba(np.array(test[features])))).tolist()
        predictions = [[int(i[0])] + i[1:] for i in predictions]
        csvfile = 'result/' + classifier.__class__.__name__ + '-submit.csv'
        with open(csvfile, 'w') as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(['id','Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9'])
            writer.writerows(predictions)
