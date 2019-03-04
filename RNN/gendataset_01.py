# -*- coding: UTF-8 -*-
import os,sys
import copy, numpy as np
np.random.seed(0)

reload(sys) # 设置 UTF-8输出环境
sys.setdefaultencoding('utf-8')

# 产生的训练集
int2binary = {}
binary_dim = 8

largest_number = pow(2,binary_dim)
binary = np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
print binary
'''
for i in range(largest_number):
	int2binary[i] = binary[i]
	print i, binary[i]
'''