#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 22:15:00 2017

@author: xingxiong
"""

import pandas as pd
from random import randint 

gray = pd.read_csv('input.csv',names = ['X1','X2','X3','X4','X5','X6','X7','X8','X9'])
color = pd.read_csv('color.csv',names = ['red','green','blue'])



# Fit red 0.87
x = gray
y = color['red']
center_point = []
k = 30

for i in range(k):
    center_point[i] = [randint(0,255),randint(0,255),randint(0,255)]





x_train, x_test, y_train, y_test = train_test_split(x, y ,test_size=0.3,random_state = 233)

from sklearn.metrics import explained_variance_score
#print(explained_variance_score(y_pred,y_test))

kf = KFold(len(x_train), n_folds=5,random_state = 233)


def xgb_c(n_estimators,max_depth,subsample,learning_rate,gamma,colsample_bytree):
    xgb_param ={'learning_rate' : learning_rate,'n_estimators' : int(n_estimators),'max_depth' : int(max_depth),'subsample' : subsample,
    'colsample_bytree' : colsample_bytree,'gamma': gamma}
    auc_list = []
    xgb1 = xgb.XGBRegressor(**xgb_param)
    for a,b in kf:
        xgb1.fit(x_train.iloc[a],y_train.iloc[a])
        y_pred = xgb1.predict(x_train.iloc[b])
        auc_list.append(explained_variance_score(y_pred,y_train.iloc[b]))
    scores = sum(auc_list)/len(auc_list)
    return(scores)  

paramada={'learning_rate':(0.01,0.3),
          'n_estimators':(100,1000),
          'max_depth':(2,6),
          'subsample':(0.5,1),
          'colsample_bytree':(0.3,0.8),
          'gamma':(0,10)}
XGBbo=BayesianOptimization(xgb_c,paramada)
XGBbo.maximize(n_iter=30)
XGB_res=XGBbo.res['max']['max_val']
XGB_params=XGBbo.res['max']['max_params']


xgb1 = xgb.XGBRegressor(colsample_bytree=0.5571,gamma=0.091,learning_rate=0.1789, max_depth=5,  n_estimators=439 ,subsample=0.8785)
xgb1.fit(x_train,y_train)
y_pred_red = xgb1.predict(x_test)
print(explained_variance_score(y_pred_red,y_test))

# Fit Green 0.975737595129

x = gray
y = color['green']
x_train, x_test, y_train, y_test = train_test_split(x, y ,test_size=0.3,random_state = 233)

kf = KFold(len(x_train), n_folds=5,random_state = 233)

def xgb_c(n_estimators,max_depth,subsample,learning_rate,gamma,colsample_bytree):
    xgb_param ={'learning_rate' : learning_rate,'n_estimators' : int(n_estimators),'max_depth' : int(max_depth),'subsample' : subsample,
    'colsample_bytree' : colsample_bytree,'gamma': gamma}
    auc_list = []
    xgb1 = xgb.XGBRegressor(**xgb_param)
    for a,b in kf:
        xgb1.fit(x_train.iloc[a],y_train.iloc[a])
        y_pred = xgb1.predict(x_train.iloc[b])
        auc_list.append(explained_variance_score(y_pred,y_train.iloc[b]))
    scores = sum(auc_list)/len(auc_list)
    return(scores)  

paramada={'learning_rate':(0.01,0.3),
          'n_estimators':(100,1000),
          'max_depth':(2,6),
          'subsample':(0.5,1),
          'colsample_bytree':(0.3,0.8),
          'gamma':(0,10)}
XGBbo=BayesianOptimization(xgb_c,paramada)
XGBbo.maximize(n_iter=20)
XGB_res=XGBbo.res['max']['max_val']
XGB_params=XGBbo.res['max']['max_params']


xgb1 = xgb.XGBRegressor(colsample_bytree=0.7899,gamma=0.1321,learning_rate=0.0729, max_depth=5,  n_estimators=443 ,subsample=0.6004)
xgb1.fit(x_train,y_train)
y_pred_green = xgb1.predict(x_test)
print(explained_variance_score(y_pred_green,y_test))


# Fit Blue
x = gray
y = color['blue']
x_train, x_test, y_train, y_test = train_test_split(x, y ,test_size=0.3,random_state = 233)

paramada={'learning_rate':(0.01,0.3),
          'n_estimators':(100,1000),
          'max_depth':(2,6),
          'subsample':(0.5,1),
          'colsample_bytree':(0.3,0.8),
          'gamma':(0,10)}
XGBbo=BayesianOptimization(xgb_c,paramada)
XGBbo.maximize(n_iter=20)
XGB_res=XGBbo.res['max']['max_val']
XGB_params=XGBbo.res['max']['max_params']


xgb1 = xgb.XGBRegressor(colsample_bytree=0.5571,gamma=0.091,learning_rate=0.1789, max_depth=5,  n_estimators=439 ,subsample=0.8785)
xgb1.fit(x_train,y_train)
y_pred_blue = xgb1.predict(x_test)
print(explained_variance_score(y_pred_blue,y_test))




red = color['red'].tolist()
green = color['green'].tolist()
blue = color['blue'].tolist()

all_px = []
for i in range(174):
    all_px.append([])
    for j in range(281):
        pixel = []
        pixel.append(red[j+i*281]/float(255))
        pixel.append(green[j+i*281]/float(255))
        pixel.append(blue[j+i*281]/float(255))
        all_px[i].append(pixel)

import numpy as np
import matplotlib.pyplot as plt      
plt.imshow(np.array(all_px))

# retrieve
y = color['red']
x_train, x_test, y_train, y_test = train_test_split(x, y ,test_size=0.3,random_state = 233)

for i in range(14669):
    y_test.iloc[i,] = y_pred_red[i]
new_red = pd.concat([y_train,y_test])
new_red = new_red.sort_index()

y = color['green']
x_train, x_test, y_train, y_test = train_test_split(x, y ,test_size=0.3,random_state = 233)
for i in range(14669):
    y_test.iloc[i,] = y_pred_green[i]
new_green = pd.concat([y_train,y_test])
new_green = new_green.sort_index()

new_green_list = new_green.tolist()
new_red_list = new_red.tolist()
central = x['X5'].tolist()
new_blue = []
for i in range(48894):
    new_blue.append((central[i]-0.58262288*new_green_list[i]-0.30060342*new_red_list[i])/0.11418448)
    
all_px1 = []
for i in range(174):
    all_px1.append([])
    for j in range(281):
        pixel = []
        pixel.append(new_red_list[j+i*281]/float(255))
        pixel.append(new_green_list[j+i*281]/float(255))
        pixel.append(new_blue[j+i*281]/float(255))
        all_px1[i].append(pixel)

import numpy as np
import matplotlib.pyplot as plt      
plt.imshow(np.array(all_px1))
plt.imshow(np.array(all_px))  


# Fit linear


import sklearn.linear_model

lm_x_train, lm_x_test, lm_y_train, lm_y_test = train_test_split(color, gray['X5'] ,test_size=0.3,random_state = 233)
lm = sklearn.linear_model.LinearRegression(fit_intercept=False)
lm.fit(lm_x_train,lm_y_train)

