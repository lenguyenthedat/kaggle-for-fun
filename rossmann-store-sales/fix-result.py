import pandas as pd
import time
import csv
import numpy as np
import os

result = "./result/To Blend 14/0.10568 rf1-d13-8000r-0.01lr-Dat"
submission = pd.read_csv(result+".csv")

# add x% growth rate into all sales predicted
def fix_growth(row):
    return row['Sales']*1.002
submission['Sales'] = submission.apply(fix_growth, axis=1)

# closed store will have zero sales - doesn't matter for the eval
def fix_closed(row):
    store = pd.read_csv('./data/store.csv')
    train_org = pd.read_csv('./data/train.csv',dtype={'StateHoliday':pd.np.string_})
    test_org = pd.read_csv('./data/test.csv',dtype={'StateHoliday':pd.np.string_})
    train = pd.merge(train_org,store, on='Store', how='left')
    test = pd.merge(test_org,store, on='Store', how='left')
    if test[test['Id'] == row['Id']]['Open'].values[0] == 0:
        return 0
    else:
        return row['Sales']
# submission['Sales'] = submission.apply(fix_closed, axis=1)

# sort by id
def fix_order(df):
    return df.sort(['Id'], ascending=[1])
submission = fix_order(submission)

submission.to_csv(result + "-fixed.csv", index=False)