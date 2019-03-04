# -*- coding: UTF-8 -*-
import os,sys
import copy, numpy as np
from dllib import *
np.random.seed(0)

reload(sys) # 设置 UTF-8输出环境
sys.setdefaultencoding('utf-8')

#1. 产生的训练集样本
binary_dim=8 # 生成的二进制bitset的宽度
largest_number = pow(2,binary_dim) # 最大数2^8=256
dataset = int2binary(binary_dim,largest_number) # 产生数据集

#2. 初始化网络参数
alpha = 0.1 # 学习速率
input_dim = 2 # 输入神经元个数
hidden_dim = 16 # 隐藏层神经元个数
output_dim = 1 # 输出神经元个数
maxiter = 10000 # # 最大迭代次数

# 初始化 LSTM 神经网络权重 synapse是神经元突触的意思
U = 2*np.random.random((input_dim,hidden_dim)) - 1 # 连接了输入层与隐含层的权值矩阵
V = 2*np.random.random((hidden_dim,output_dim)) - 1 # 连接了隐含层与输出层的权值矩阵
W = 2*np.random.random((hidden_dim,hidden_dim)) - 1 # 连接了隐含层与隐含层的权值矩阵
# 权值更新缓存：用于存储更新后的权值。np.zeros_like：返回全零的与参数同类型、同维度的数组
U_update = np.zeros_like(U) # 
V_update = np.zeros_like(V) # 
W_update = np.zeros_like(W) # 

#3. 主程序--训练过程：
for j in xrange(maxiter): 
	# 在实际应用中，可以从训练集中查询到一个样本: 生成形如[a] [b]--> [c]这样的样本：
	a,a_int,b,b_int,c,c_int = gensample(dataset,largest_number)
	
	# 初始化一个空的二进制数组，用来存储神经网络的预测值
	d = np.zeros_like(c)
	
	overallError = 0 # 重置全局误差

	layer_Ot_deltas = list(); # 记录layer 2的导数值
	layer_Ht_values = list(); # 与layer 1的值。
	layer_Ht_values.append(np.zeros(hidden_dim)) # 初始化时无值，存储一个全零的向量
	
	# 正向传播过程：逐个bit位(0,1)的遍历二进制数字。
	for position in xrange(binary_dim): 
		indx = binary_dim - position - 1 # 数组索引7,6,5,...,0
		# X 是样本集的记录，来自a[i]b[i]; y是样本集对应的标签,来自c[i]
		X = np.array([[a[indx],b[indx]]])
		y = np.array([[c[indx]]]).T
		
		# 隐含层 (input ~+ prev_hidden)
		# 1. np.dot(X,U)：从输入层传播到隐含层：输入层的数据*（输入层-隐含层的权值）
		# 2. np.dot(layer_Ht_values[-1],W)：从上一次的隐含层[-1]到当前的隐含层：上一次的隐含层权值*当前隐含层的权值
		# 3. sigmoid(input + prev_hidden)
		layer_Ht = sigmoid(np.dot(X,U) +np.dot(layer_Ht_values[-1],W))
		 
		# 输出层 (new binary representation)
		# np.dot(layer_Ht,V)：它从隐含层传播到输出层，即输出一个预测值。
		layer_Ot = sigmoid(np.dot(layer_Ht,V))
		 
		# 计算预测误差
		layer_Ot_error = y - layer_Ot
		layer_Ot_deltas.append((layer_Ot_error)*dlogit(layer_Ot)) # 保留输出层每个时刻的导数值
		overallError += np.abs(layer_Ot_error[0]) # 计算二进制位的误差绝对值的总和，标量
		 		
		d[indx] = np.round(layer_Ot[0][0]) # 存储预测的结果--显示使用
		
		layer_Ht_values.append(copy.deepcopy(layer_Ht)) # 存储隐含层的权值，以便在下次时间迭代中能使用
		
	future_layer_Ht_delta = np.zeros(hidden_dim) # 初始化下一隐含层的误差
	# 反向传播：从最后一个时间点开始，反向一直到第一个： position索引0,1,2,...,7
	for position in xrange(binary_dim): 		 
		X = np.array([[a[position],b[position]]]) 
		
		layer_Ht = layer_Ht_values[-position-1] # 从列表中取出当前的隐含层。从最后一层开始，-1，-2，-3
		prev_layer_Ht = layer_Ht_values[-position-2] # 从列表中取出当前层的前一隐含层。
				
		layer_Ot_delta = layer_Ot_deltas[-position-1] # 取出当前输出层的误差
		# 计算当前隐含层的误差:
		# future_layer_Ht_delta.dot(W.T): 下一隐含层误差*隐含层权重
		# layer_Ot_delta.dot(V.T):当前输出层误差*输出层权重
		# dlogit(layer_Ht)：当前隐含层的导数
		layer_Ht_delta = (future_layer_Ht_delta.dot(W.T) +layer_Ot_delta.dot(V.T)) *dlogit(layer_Ht)
		 
		# 反向更新权重: 更新顺序输出层-->隐含层-->输入层
		# np.atleast_2d：输入层reshape为2d的数组
		V_update +=np.atleast_2d(layer_Ht).T.dot(layer_Ot_delta)
		W_update +=np.atleast_2d(prev_layer_Ht).T.dot(layer_Ht_delta)
		U_update += X.T.dot(layer_Ht_delta)
	
		future_layer_Ht_delta = layer_Ht_delta # 下一隐含层的误差
	# 更新三个权值 
	U += U_update * alpha
	V += V_update * alpha
	W += W_update * alpha   
	# 所有权值更新项归零
	U_update *= 0;	V_update *= 0;	W_update *= 0
	
	# 逐次打印输出
	showresult(j,overallError,d,c,a_int,b_int)

