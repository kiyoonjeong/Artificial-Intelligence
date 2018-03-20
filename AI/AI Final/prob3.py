# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 22:12:37 2017

@author: Kiyoon Jeong
"""

import numpy as np
data1 = np.array([[1,	1,	0,	0,	0],
[0,	1,	1,	0,	0],
[0,	1,	0,	0,	0],
[1,	0,	0,	0,	0],
[0,	0,	0,	0,	0]])

data2 = np.array([[1,	1,	0,	0,	0],
[0,	0,	1,	0,	0],
[1,	0,	0,	0,	0],
[0,	0,	1,	0,	0],
[0,	0,	0,	0,	0]])

data3 = np.array([[0,1,0,0,0],
         [1,1,0,0,0],
         [1,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0]])

data4 = np.array([[1,1,0,0,0],
         [1,0,1,0,0],
         [0,1,0,1,0],
         [0,0,0,0,0],
         [0,0,1,0,0]])
         
data5 = np.array([[0,1,1,0,0],
         [1,1,0,0,1],
         [0,1,0,0,0],
         [0,0,1,0,0],
         [0,0,0,0,0]])

classA = np.concatenate((data1,data2,data3,data4,data5), axis = 0)
classA.resize(5,25)

data11 = np.array([[0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,1,0],
         [0,0,0,0,1],
         [0,0,1,1,1]])

data12 = np.array([[0,0,0,0,0],
         [1,0,0,0,0],
         [0,0,0,0,0],
         [1,0,1,0,1],
         [0,1,0,1,1]])


data13 = np.array([[0,1,0,0,0],
         [0,0,0,1,0],
         [1,0,0,1,0],
         [0,0,0,1,1],
         [0,0,0,1,0]])

data14 = np.array([[1,0,0,0,0],
         [0,0,0,0,0],
         [0,0,1,0,0],
         [0,0,0,1,1],
         [0,1,0,1,0]])
         
data15 = np.array([[0,0,0,0,0],
         [0,1,0,0,1],
         [0,0,0,0,1],
         [0,0,0,0,0],
         [0,0,1,1,1]])

classB = np.concatenate((data11,data12,data13,data14,data15), axis = 0)
classB.resize(5,25)


data21 = np.array([[1,0,0,0,0],
         [1,0,1,0,1],
         [0,0,0,0,0],
         [0,0,1,1,1],
         [0,0,0,1,0]])

data22 = np.array([[1,1,1,0,0],
         [1,1,1,0,0],
         [0,0,1,1,0],
         [0,0,0,0,0],
         [1,0,1,0,0]])


data23 = np.array([[0,0,0,0,1],
         [0,0,0,0,1],
         [0,0,0,1,1],
         [0,0,0,0,1],
         [0,1,0,1,1]])

data24 = np.array([[0,1,1,0,0],
         [1,1,0,0,0],
         [0,1,1,0,0],
         [0,1,0,0,1],
         [0,0,0,0,0]])
         
data25 = np.array([[1,0,0,0,1],
         [0,0,0,0,0],
         [0,0,1,0,0],
         [0,0,0,0,0],
         [1,0,0,0,1]])

M = np.concatenate((data21,data22,data23,data24,data25), axis = 0)
M.resize(5,25)


X = np.concatenate((classA,classB), axis = 0)
Y = np.array([1,1,1,1,1,0,0,0,0,0])

def predict(X,beta):
    ax = np.dot(X, beta) #sigmoid
    return 1/(1+np.exp(ax))

def logistic(X,Y,nsteps, learn_rate):
    beta = np.zeros(25)
    for step in range(nsteps):
        prediction = predict(X,beta)
        gradient = np.dot(X.T, (Y - prediction))
        beta -= learn_rate * gradient
    
    return beta


beta = logistic(X,Y,10000, 0.3)

print(beta)


test_predict = np.where(predict(X, beta)>0.5, 1, 0)
test2_predict = np.where(predict(M, beta)>0.5, 1, 0)

print(test_predict)
print(test2_predict)


#3-(C) Bernoulli Bayes

p1 = ((data1+data2+data3+data4+data5)/5)
p1.resize(1,25)
p1 = p1[0]

for i in range(25):
    if p1[i] == 0:
        p1[i] = 0.001
    if p1[i] == 1:
        p1[i] = 0.999

p0 = ((data11+data12+data13+data14+data15)/5)
p0.resize(1,25)
p0 = p0[0]

for i in range(25):
    if p0[i] == 0:
        p0[i] = 0.001
    if p0[i] == 1:
        p0[i] = 0.999

lhood1 = 1
lhood0 = 1
y = [5,5,5,5,5,5,5,5,5,5]
for k in range(10):
    x = X[k]
    lhood1 = 1
    lhood0 = 1
    for i in range(25):
        lhood1 *= (p1[i]**x[i]) * ((1-p1[i])**(1-x[i]))
        lhood0 *= (p0[i]**x[i]) * ((1-p0[i])**(1-x[i]))
        # probability of each class is 0.5    
    Likehood = [0.5*lhood0, 0.5*lhood1]
    print(Likehood)
    y[k] = Likehood.index(max(Likehood))
print(y)

y = [5,5,5,5,5]
for k in range(5):
    x = M[k]
    lhood1 = 1
    lhood0 = 1
    for i in range(25):
        lhood1 *= (p1[i]**x[i]) * ((1-p1[i])**(1-x[i]))
        lhood0 *= (p0[i]**x[i]) * ((1-p0[i])**(1-x[i]))
        # probability of each class is 0.5    
    Likehood = [0.5*lhood0, 0.5*lhood1]   
    print(Likehood)     
    y[k] = Likehood.index(max(Likehood))
print(y)
