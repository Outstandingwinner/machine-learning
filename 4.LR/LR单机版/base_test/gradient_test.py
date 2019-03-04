# -*- coding: utf-8 -*-
import os
import sys
import operator
from numpy import *
from common_libs import *
import matplotlib.pyplot as plt

Input = file2matrix("testSet.txt","\t")
target = Input[:,-1]
[m,n] = shape(Input)

drawScatterbyLabel(plt,Input)

dataMat = buildMat(Input)
alpha = 0.001
steps = 500

weights = ones((n,1))

for k in xrange(steps):
    gradient = dataMat*mat(weights) # 100x3 * 3x1
    output = logistic(gradient)
    errors = target - output
    weights = weights + alpha*dataMat.T*errors # 梯度的负数
print weights

X =np.linspace(-5,5,100)
Y = -(double(weights[0]) + X*(double(weights[1])))/double(weights[2])

plt.plot(X,Y)
plt.show()

