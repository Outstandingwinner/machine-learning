# -*- coding: UTF-8 -*-
import os,sys
import copy, numpy as np
from dllib import *
np.random.seed(0)

reload(sys) # 设置 UTF-8输出环境
sys.setdefaultencoding('utf-8')

binary_dim = 8
largest_number = pow(2,binary_dim)
dataset = int2binary(binary_dim,largest_number)

alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1
maxiter = 10000

# 初始化 LSTM 神经网络权重 synapse是神经元突触的意思
U = 2*np.random.random((input_dim,hidden_dim)) - 1 # 连接了输入层与隐含层的权值矩阵
V = 2*np.random.random((hidden_dim,output_dim)) - 1 # 连接了隐含层与输出层的权值矩阵
W = 2*np.random.random((hidden_dim,hidden_dim)) - 1 # 连接了隐含层与隐含层的权值矩阵
# 权值更新缓存：用于存储更新后的权值。np.zeros_like：返回全零的与参数同类型、同维度的数组
U_update = np.zeros_like(U) #
V_update = np.zeros_like(V) #
W_update = np.zeros_like(W) #

for j in xrange(maxiter):
    a,a_int,b,b_int,c,c_int = gensample(dataset,largest_number)
    d = np.zeros_like(c)
    overallError = 0

    layer_Ot_deltas = list()
    layer_Ht_values = list()
    layer_Ht_values.append(np.zeros(hidden_dim))

    for position in xrange(binary_dim):
        indx = binary_dim - position - 1
        X = np.array([[a[indx],b[indx]]])
        y = np.array([[c[indx]]]).T

        layer_Ht = sigmoid(np.dot(X,U) + np.dot(layer_Ht_values[-1],W))
        layer_Ot = sigmoid(np.dot(layer_Ht,V))

        layer_Ot_error = y - layer_Ot
        layer_Ot_deltas.append((layer_Ot_error)*dlogit(layer_Ot))
        overallError += np.abs(layer_Ot_error[0])

        d[indx] = np.round(layer_Ot[0][0])
        layer_Ht_values.append(copy.deepcopy(layer_Ht))

    future_layer_Ht_delta = np.zeros(hidden_dim)

    for position in xrange(binary_dim):
        X = np.array([[a[position],b[position]]])
        layer_Ht = layer_Ht_values[-position-1]
        pre_layer_Ht = layer_Ht_values[-position-2]

        layer_Ot_delta = layer_Ot_deltas[-position-1]
        layer_Ht_delta = (future_layer_Ht_delta.dot(W.T) + layer_Ot_delta.dot(V.T))*dlogit(layer_Ht)
        V_update += np.atleast_2d(layer_Ht).T.dot(layer_Ot_delta)
        W_update += np.atleast_2d(pre_layer_Ht).T.dot(layer_Ht_delta)
        U_update += X.T.dot(layer_Ht_delta)
        future_layer_Ht_delta = layer_Ht_delta

        U += U_update * alpha
        V += V_update * alpha
        W += W_update * alpha
        # 所有权值更新项归零
        U_update *= 0;
        V_update *= 0;
        W_update *= 0

        showresult(j, overallError, d, c, a_int, b_int)




















