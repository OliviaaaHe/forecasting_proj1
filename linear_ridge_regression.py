#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 00:16:05 2022

@author: hanjio
"""
# import modules
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import matplotlib.pyplot as plt  # To visualize
from matplotlib.pyplot import figure
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import RepeatedKFold, GridSearchCV
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error


# import dataset
df = pd.read_csv("ts_day.csv")

# set up new time columns: Day, time, weekday, and month
df['Day'] = pd.to_datetime(df['Day'])
df['time'] = (df['Day'] - min(df['Day'])).astype('timedelta64[D]').astype(int)
df['weekday'] = [df['Day'][i].strftime('%A') for i in range(len(df))]
df['month'] = [df['Day'][i].month for i in range(len(df))]
df['year'] = df['Day'].dt.year

#df2 = pd.get_dummies(data = df, columns = ['weekday']).drop(columns = ['Day'])
df2 = pd.get_dummies(data = df, columns = ['year', 'month', 'weekday']).drop(columns = ['Day'])
#df2.head()
#%%
# Pre modeling
# Mape function:
def calculate_mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

## splitting training and testing
X = df2.drop(['sale'], axis = 1)
y = df2['sale']
X_dev, y_dev, X_test, y_test = X[:8*len(df)//10], y[:8*len(df)//10], X[8*len(df)//10:], y[8*len(df)//10:]
X_train, y_train, X_valid, y_valid = X_dev[:7*len(df)//8], y_dev[:7*len(df)//8], X_dev[1*len(df)//8:], y_dev[1*len(df)//8:]

# split the date for graphing
Date_train, Date_valid, Date_test = df['Day'][:7*len(df)//10], df['Day'][7*len(df)//10:8*len(df)//10], df['Day'][8*len(df)//10:]


#%%
# Modeling

# ?ridge
# ?train, validation, test
# 1. simple linear regression
lr = LinearRegression()
lr.fit(X_dev, y_dev)
y_lr_pred = lr.predict(X_test)
print(f"{lr}:\n Mape: {calculate_mape(y_test, y_lr_pred)}",
      f"\n RMSE: {math.sqrt(mean_squared_error(y_test, y_lr_pred))}")

# 2. OLS: linear regression with interactions

df3 = df.copy()
df3[['month', 'year']] = df[['month', 'year']].astype('category')
df3_train, df3_test = df3[:8*len(df)//10], df3[8*len(df)//10:]

ols = smf.ols(formula='sale ~ time + weekday * month + year', 
              data = df3_train).fit()
#ols.summary().tables[1]
y_ols_pred = ols.predict(df3_test.drop(['sale'], axis = 1))
print(f"{ols}:\n Mape: {calculate_mape(y_test, y_ols_pred)}",
      f"\n RMSE: {math.sqrt(mean_squared_error(y_test, y_ols_pred))}")

#%%
# 3. ridge regression
ridge = Ridge()
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = {"alpha" : np.arange(0, 1, 0.05)}
# define search
search = GridSearchCV(ridge, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X_dev, y_dev)

print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)

ridge_best = results.best_estimator_
ridge_best.fit(X_dev, y_dev)
y_ridge_pred = ridge_best.predict(X_test)
print(f"{ridge}:\n Mape: {calculate_mape(y_test, y_ridge_pred)}",
      f"\n RMSE: {math.sqrt(mean_squared_error(y_test, y_ridge_pred))}")
#%%
# Plotting

plt.plot(Date_test, y_test, label = "y_test")
plt.plot(Date_test, y_lr_pred, label = "linear regression")
plt.plot(Date_test, y_ols_pred, label = "OLS")
plt.plot(Date_test, y_ridge_pred, label = "ridge regression")
plt.legend()
plt.show()

# plot importance
#plt.figure(figsize = (10,10))
#plt.bar(x = X_train.columns, height = lr.coef_, width = 0.8)
#plt.xticks(rotation = 90)