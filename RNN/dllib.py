# -*- coding: UTF-8 -*-
import os,sys
import copy, numpy as np
np.random.seed(0)

reload(sys) # 设置 UTF-8输出环境
sys.setdefaultencoding('utf-8')

# sigmoid 函数
def sigmoid(x):
	return 1/(1+np.exp(-x))

# sigmoid 导函数
def dlogit(output): # dlogit
	return output*(1-output)

# 十进制转二进制数组
def int2binary(bindim,largest_number):
	int2bindic = {}
	binary = np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
	for i in range(largest_number):
		int2bindic[i] = binary[i]
	return int2bindic 
# 样本发生器:实现一个简单的(a + b = c)的加法
def gensample(dataset,largest_number):
		# 实现一个简单的(a + b = c)的加法
		a_int = np.random.randint(largest_number/2) # 十进制
		a = dataset[a_int] # 二进制 		
		b_int = np.random.randint(largest_number/2) # 十进制
		b = dataset[b_int] # 二进制 				
		c_int = a_int + b_int # 十进制的结果
		c = dataset[c_int] # 十进制转二进制的结果
		return a,a_int,b,b_int,c,c_int
	
def showresult(j,overallError,d,c,a_int,b_int):
	if(j % 1000 == 0):
		print "Error:" + str(overallError)
		print "Pred:" + str(d)
		print "True:" + str(c)
		out = 0
		for index,x in enumerate(reversed(d)):
			out += x*pow(2,index)
		print str(a_int) + " + " + str(b_int) + " = " + str(out)
		print "------------"	