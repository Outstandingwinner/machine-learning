# -*- coding: UTF-8 -*-
import os,sys
import numpy as np
import random
import math

# 网络激活函数
def sigmoid(x): 
	return 1. / (1 + np.exp(-x))
# 创建一个范围在[a,b)的维度为args的随机矩阵
def rand_arr(a, b, *args): 
	np.random.seed(0)
	return np.random.rand(*args) * (b - a) + a
# 损失函数类：
class LossLayer:
	@classmethod # 计算平方损失
	def loss(self, pred, label):
		return (pred[0] - label) ** 2
	@classmethod # 实际输出和期望输出之间的差
	def bottom_diff(self, pred, label):
		diff = np.zeros_like(pred)
		diff[0] = 2 * (pred[0] - label)
		return diff
# lstm参数类
class LstmParam:
	def __init__(self, mem_cell_ct, x_dim):
		self.mem_cell_ct = mem_cell_ct
		self.x_dim = x_dim
		concat_len = x_dim + mem_cell_ct
		# 初始化各个门的权重矩阵
		self.wg = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
		self.wi = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len) 
		self.wf = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
		self.wo = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
		# 初始化各个门的偏置矩阵
		self.bg = rand_arr(-0.1, 0.1, mem_cell_ct) 
		self.bi = rand_arr(-0.1, 0.1, mem_cell_ct) 
		self.bf = rand_arr(-0.1, 0.1, mem_cell_ct) 
		self.bo = rand_arr(-0.1, 0.1, mem_cell_ct) 
		# 损失函数的导数：权重、偏置
		self.wg_diff = np.zeros((mem_cell_ct, concat_len)) 
		self.wi_diff = np.zeros((mem_cell_ct, concat_len)) 
		self.wf_diff = np.zeros((mem_cell_ct, concat_len)) 
		self.wo_diff = np.zeros((mem_cell_ct, concat_len)) 
		self.bg_diff = np.zeros(mem_cell_ct) 
		self.bi_diff = np.zeros(mem_cell_ct) 
		self.bf_diff = np.zeros(mem_cell_ct) 
		self.bo_diff = np.zeros(mem_cell_ct) 
	# 更新参数
	def apply_diff(self, lr = 1):
		self.wg -= lr * self.wg_diff
		self.wi -= lr * self.wi_diff
		self.wf -= lr * self.wf_diff
		self.wo -= lr * self.wo_diff
		self.bg -= lr * self.bg_diff
		self.bi -= lr * self.bi_diff
		self.bf -= lr * self.bf_diff
		self.bo -= lr * self.bo_diff
		# 重置权重和偏置的差异为0
		self.wg_diff = np.zeros_like(self.wg)
		self.wi_diff = np.zeros_like(self.wi) 
		self.wf_diff = np.zeros_like(self.wf) 
		self.wo_diff = np.zeros_like(self.wo) 
		self.bg_diff = np.zeros_like(self.bg)
		self.bi_diff = np.zeros_like(self.bi) 
		self.bf_diff = np.zeros_like(self.bf) 
		self.bo_diff = np.zeros_like(self.bo) 

# lstm状态类
class LstmState:
	def __init__(self, mem_cell_ct, x_dim):
		self.g = np.zeros(mem_cell_ct) # 隐藏输入
		self.i = np.zeros(mem_cell_ct) # 输入门 
		self.f = np.zeros(mem_cell_ct) # 忘记门
		self.o = np.zeros(mem_cell_ct) # 输出门
		self.s = np.zeros(mem_cell_ct) # 内部状态
		self.h = np.zeros(mem_cell_ct) # 实际输出
		self.bottom_diff_h = np.zeros_like(self.h) 
		self.bottom_diff_s = np.zeros_like(self.s) 
		self.bottom_diff_x = np.zeros(x_dim) 
	
class LstmNode:
	def __init__(self, lstm_param, lstm_state):		
		self.state = lstm_state # store reference to parameters and to activations
		self.param = lstm_param		
		self.x = None # 输入层（非循环层）节点		
		self.xc = None # 非循环(non-recurrent)层输入+ 循环(recurrent)层输入
	# 
	def bottom_data_is(self, x, s_prev = None, h_prev = None):
		# if this is the first lstm node in the network
		if s_prev is None: s_prev = np.zeros_like(self.state.s) 
		if h_prev is None: h_prev = np.zeros_like(self.state.h)
		# 存储数据用于反向传播
		self.s_prev = s_prev
		self.h_prev = h_prev
		
		xc = np.hstack((x,  h_prev)) # xc(t)=[x(t),h(t−1)]
		self.state.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg)  # 隐层输入
		self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)  # 输入门
		self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)  # 忘记门
		self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)  # 输出门
		self.state.s = self.state.g * self.state.i + s_prev * self.state.f # 内部状态
		self.state.h = self.state.s * self.state.o # 实际输出
		self.x = x
		self.xc = xc
	
	# 注意top_diff_s沿固定误差传递
	def top_diff_is(self, top_diff_h, top_diff_s):		
		ds = top_diff_s + self.state.o * top_diff_h 
		do = self.state.s * top_diff_h
		di = self.state.g * ds
		dg = self.state.i * ds
		df = self.s_prev * ds
		
		# diffs w.r.t. vector inside sigma / tanh function
		di_input = (1. - self.state.i) * self.state.i * di 
		df_input = (1. - self.state.f) * self.state.f * df 
		do_input = (1. - self.state.o) * self.state.o * do 
		dg_input = (1. - self.state.g ** 2) * dg
		
		# 输入层的误差
		self.param.wi_diff += np.outer(di_input, self.xc)
		self.param.wf_diff += np.outer(df_input, self.xc)
		self.param.wo_diff += np.outer(do_input, self.xc)
		self.param.wg_diff += np.outer(dg_input, self.xc)
		self.param.bi_diff += di_input
		self.param.bf_diff += df_input       
		self.param.bo_diff += do_input
		self.param.bg_diff += dg_input       
		
		# 计算底层（bottom）误差
		dxc = np.zeros_like(self.xc)
		dxc += np.dot(self.param.wi.T, di_input)
		dxc += np.dot(self.param.wf.T, df_input)
		dxc += np.dot(self.param.wo.T, do_input)
		dxc += np.dot(self.param.wg.T, dg_input)
		
		# 存储底层（bottom）误差
		self.state.bottom_diff_s = ds * self.state.f
		self.state.bottom_diff_x = dxc[:self.param.x_dim]
		self.state.bottom_diff_h = dxc[self.param.x_dim:]

class LstmNetwork():
	def __init__(self, lstm_param):
		self.lstm_param = lstm_param
		self.lstm_node_list = []
		self.x_list = [] # 输入序列
	
	def y_list_is(self, y_list, loss_layer):
		"""
		Updates diffs by setting target sequence with corresponding loss layer. 
		Will *NOT* update parameters.  To update parameters,call self.lstm_param.apply_diff()
		"""
		assert len(y_list) == len(self.x_list)
		idx = len(self.x_list) - 1
		# first node only gets diffs from label ...
		loss = loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
		diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])
		# here s is not affecting loss due to h(t+1), hence we set equal to zero
		diff_s = np.zeros(self.lstm_param.mem_cell_ct)
		self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
		idx -= 1
		
		### ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
		### we also propagate error along constant error carousel using diff_s
		while idx >= 0:
			loss += loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
			diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])
			diff_h += self.lstm_node_list[idx + 1].state.bottom_diff_h
			diff_s = self.lstm_node_list[idx + 1].state.bottom_diff_s
			self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
			idx -= 1
		return loss
	
	def x_list_clear(self):
		self.x_list = []
	
	def x_list_add(self, x):
		self.x_list.append(x)
		if len(self.x_list) > len(self.lstm_node_list):
			# need to add new lstm node, create new state mem
			lstm_state = LstmState(self.lstm_param.mem_cell_ct, self.lstm_param.x_dim)
			self.lstm_node_list.append(LstmNode(self.lstm_param, lstm_state))
		
		# get index of most recent x input
		idx = len(self.x_list) - 1
		if idx == 0:
			# no recurrent inputs yet
			self.lstm_node_list[idx].bottom_data_is(x)
		else:
			s_prev = self.lstm_node_list[idx - 1].state.s
			h_prev = self.lstm_node_list[idx - 1].state.h
			self.lstm_node_list[idx].bottom_data_is(x, s_prev, h_prev)

