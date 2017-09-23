## Kaggle-for-fun

https://www.kaggle.com/lenguyenthedat/competitions

All my submissions for Kaggle contests that I have been, and going to be participating.

I will probably have everything written in Python (utilizing scikit-learn or similar libraries), but occasionally I might also use R or Haskell if I can.

The code written in this repository can be very scrappy since I consider this my playground. Feel free to contribute if you are interested in doing so.

# avazu-ctr-prediction
Avazu's CTR prediction contest - https://www.kaggle.com/c/avazu-ctr-prediction

My solution utilizes GradientBoostingClassifier with a few features preprocessing / engineering processes.

It scored *0.4045696* in term of log-loss, comparing to the first place at *0.3818529* (as of 2nd Feb 2014).

# digit-recognizer
Digit Recognizer - https://www.kaggle.com/c/digit-recognizer/

- 1st version: Random Forest with 2000 estimators. Training time: 841s. Accuracy Score = 0.96800
- 2nd version: 4 layer NN. Training time: 1774s. Accuracy Score = 0.97029
- 3rd version: Convolution NN. Training time: 7674s. Accuracy Score = 0.97243
- 4rd version: better Convo NN. Training time: 47903s. Accuracy Score = 0.98714

# homesite-quote-conversion
Homesite Quote Conversion - https://www.kaggle.com/c/homesite-quote-conversion/

My personal best single model, scoring 0.96896 on the public LB and 0.96832 on the private LB (Rank #37 out of 1762 teams)

# how-much-did-it-rain-ii
How much did it rain II - https://www.kaggle.com/c/how-much-did-it-rain-ii

- 1st version: basic xgboost. MAE: 23.81617
- 2nd version: more features xgboost. MAE: 23.78276 # [1203]  eval-rmse:0.581621  train-rmse:0.407241
- 3rd version: more features better tuning xgboost. MAE 23.77988 # [1772]   eval-rmse:0.553671  train-rmse:0.346084

# liberty-mutual-group-property-inspection-prediction
Liberty Mutual Group: Property Inspection Prediction - https://www.kaggle.com/c/liberty-mutual-group-property-inspection-prediction

- 1st version LB 0.336953: basic RandomForest.
- 2nd version LB 0.366923: xgboost.
- 3rd version LB 0.372686: better xgboost parameter.
- 4rd version LB 0.378719: native xgboost. (Local score: 0.376601296742)
- 5th version LB 0.380626: tuned native xgboost. (Local score: 0.383170885213)
- 6th version LB 0.380916: tuned native xgboost. (Local score: 0.380586744068) (5th seems to be overfit...)

# otto-group-classification
Otto Group Product Classification - https://www.kaggle.com/c/otto-group-product-classification-challenge

- 1st version: basic Adaboost. 28s training time, log_loss = 0.95514
- 2nd version: Neural Network. 96s training time, log_loss = 0.53893
- 3rd version: 4 layers Neural Network, 335s training time, log_loss = 0.50003

# rossmann-store-sales
Rossmann Store Sales - https://www.kaggle.com/c/rossmann-store-sales

- 1st version LB 0.37205: basic xgboost
- 2nd version LB 0.11764 (local: 0.1169): fine-tuning xgboost 1000s training time
- 3nd version LB 0.11021 (local: 0.1097): like 2nd version but with log and exp mod
- 4rd version LB 0.10903 (local: 0.1116): xgboost 3000r d10 lr 0.02 sample 0.9 0.7 - 3600 training time
- 5rd version LB 0.10861 (local: 0.1072): xgboost 3000r d12 lr 0.02 sample 0.9 0.7 - 4500 training time
- 6th version LB 0.10795 (local: 0.1107): native-xgb 3000r d10 lr 0.02 sample 0.9 0.7
- 7th version LB 0.10640 (local: 0.1070): r version xgb 3000r d10 lr 0.02 sample 0.9 0.7
- 8th version LB 0.10568 (local: 0.0993): r version xgb 8000r d13 lr 0.01 sample 0.9 0.7
- 9th version LB 0.10858 my own xgb native implementation d13 eta0.01 ntree4000 mcw3.

# sf-crime-classification
San Francisco Crime Classification - https://www.kaggle.com/c/sf-crime/

- 1st version: basic Adaboost. 47s training time, log_loss = 3.66252
- 2nd version: fine tuned Adaboost. 525s training time, log_loss = 2.72599
- 3rd version: 2-layers Neural Network. 770s training time, log_loss = 2.51524
- 4th version: 2-layers Neural Network with Engineered features. 823s training time, log_loss = 2.47535
- 5th version: 4-layers Neural Network with Engineered features. 3006s training time, log_loss = 2.43479
- 6th version: Fine-tuned Random Forest (1024 trees, depth 16). 3560s training time, log_loss = 2.33752
- 7th version: xgb native. 3000s training time, log_loss = 2.33537

# springleaf-marketing-response
Springleaf Marketing Response - https://www.kaggle.com/c/springleaf-marketing-response

- 1st version: native xgboost. AUC Score = 0.78714 Booster-submit-d8-0.01-mcw3-0.65-0.65-500r
- 2nd version: feature selection and native xgboost. AUC Score = 0.79164. Local: 0.756932926155
- 3rd version: feature selection with ftrl. AUC Score = 0.77348
- 4th version: native xgboost. AUC Score = 0.79200. Local: 0.754258680372

# titanic
Titanic - https://www.kaggle.com/c/titanic/

- 1st version: basic 4-layer Neural Network. Accuracy Score = 0.80383
