import pandas as pd
import time
import csv
import numpy as np
import os
import sys

result = sys.argv[1]
submission = pd.read_csv(result)

store = pd.read_csv('./data/store.csv')
test_org = pd.read_csv('./data/test.csv',dtype={'StateHoliday':pd.np.string_})
test = pd.merge(test_org,store, on='Store', how='left')
submission = pd.merge(submission,test, on='Id', how='left')

# add x% growth rate into all sales predicted
def fix_growth_storetype(row):
    if row['StoreType'] == 'a':
        return row['Sales']*0.99
    elif row['StoreType'] == 'b':
        return row['Sales']
    elif row['StoreType'] == 'c':
        return row['Sales']*0.99
    else: #d
        return row['Sales']*0.99
submission['Sales'] = submission.apply(fix_growth_storetype, axis=1)

def fix_growth_assortment(row):
    if row['Assortment'] == 'a':
        return row['Sales']
    elif row['Assortment'] == 'b':
        return row['Sales']
    elif row['Assortment'] == 'c':
        return row['Sales']*1.005
    else: #d
        return row['Sales']
submission['Sales'] = submission.apply(fix_growth_assortment, axis=1)

def fix_growth_promo(row):
    if row['Promo'] == 1:
        return row['Sales']*0.985
    else:
        return row['Sales']*1.015
submission['Sales'] = submission.apply(fix_growth_promo, axis=1)

# closed store will have zero sales - doesn't matter for the eval
def fix_closed(row):
    if row['Open'].values[0] == 0:
        return 0
    else:
        return row['Sales']
# submission['Sales'] = submission.apply(fix_closed, axis=1)

# sort by id
def fix_order(df):
    return df.sort_values(['Id'], ascending=[1])
# submission = fix_order(submission)

submission[['Id','Sales']].to_csv(result + "-fixed.csv", index=False)