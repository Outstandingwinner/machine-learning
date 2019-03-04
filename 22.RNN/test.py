# -*- coding: UTF-8 -*-
import os,sys
import copy, numpy as np

def int2binary(bindim,largest_number):
	int2bindic = {}
	binary = np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
	print("binary",binary)
	for i in range(largest_number):
		int2bindic[i] = binary[i]
		print  "int2bindic[%d]" %i
		print(int2bindic[i])
	return int2bindic

def gensample(dataset,largest_number):
		# 实现一个简单的(a + b = c)的加法
		a_int = np.random.randint(largest_number/2) # 十进制
		print("a_int",a_int)
		a = dataset[a_int] # 二进制
		print("a", a)
		b_int = np.random.randint(largest_number/2) # 十进制
		print("b_int", b_int)
		b = dataset[b_int] # 二进制
		print("b", b)
		c_int = a_int + b_int # 十进制的结果
		print("c_int", c_int)
		c = dataset[c_int] # 十进制转二进制的结果
		print("c", c)
		return a,a_int,b,b_int,c,c_int


dataset = int2binary(8,256)
a,a_int,b,b_int,c,c_int = gensample(dataset,256)
print(dataset)