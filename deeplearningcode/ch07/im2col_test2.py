# coding: utf-8
import numpy as np

def im2col(input_data,filtet_h,filter_w,stride=1,pad=0):
    N,C,H,W = input_data.shape
    out_h = (H+2*pad-filtet_h)//stride + 1
    out_w = (W + 2*pad -filter_w)//stride + 1

    img = np.pad(input_data,[(0,0),(0,0),(pad,pad),(pad,pad)],'constant')
    col = np.zeros((N,C,filtet_h,filter_w,out_h,out_w))

    for y in range(filtet_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:,:,y,x,:,:] = img[:,:,y:y_max:stride,x:x_max:stride]
    col = col.transpose(0,4,5,1,2,3).reshape(N*out_h,out_w,-1)
    return col








