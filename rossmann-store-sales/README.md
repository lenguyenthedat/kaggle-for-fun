Rossmann Store Sales
====================

https://www.kaggle.com/c/rossmann-store-sales

- 1st version LB 0.37205: basic xgboost
- 2nd version LB 0.11764 (local: 0.1169): fine-tuning xgboost 1000s training time
- 3nd version LB 0.11021 (local: 0.1097): like 2nd version but with log and exp mod
- 4rd version LB 0.10903 (local: 0.1116): xgboost 3000r d10 lr 0.02 sample 0.9 0.7 - 3600 training time
- 5rd version LB 0.10861 (local: 0.1072): xgboost 3000r d12 lr 0.02 sample 0.9 0.7 - 4500 training time
- 6th version LB 0.10795 (local: 0.1107): native-xgb 3000r d10 lr 0.02 sample 0.9 0.7