import sys, os
import numpy as np
sys.path.append(os.pardir)



def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    N, C, H, W = input_data.shape
    print("input_data.shape=", input_data.shape)
    out_h = (H + 2*pad - filter_h)//stride + 1
    print("out_h=",out_h)
    out_w = (W + 2*pad - filter_w)//stride + 1
    print("out_w=", out_w)
    # print("++++++++++++++++++++++++++++++++++++++++++++++++")
    # print("np.shape(input_data)",np.shape(input_data))
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    # print("np.shape(img)", np.shape(img))
    print("img=",img)

    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            print("x",x)
            print("stride",stride)
            print("out_w",out_w)
            print("x_max",x_max)
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    print("****************")
    print("col", col)
    print("col.shape=",col.shape)  # col.shape= (1, 1, 2, 2, 3, 2)      N, C, H, W,out_h,out_w
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1) # N, C, H, W,out_h,out_w ->  N,out_h,out_w,C, H, W
    colx = col.reshape(N * out_h * out_w, -1)
    print("colx", colx)
    print("****************")
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 输入数据的形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]











x1 = np.random.rand(1, 1, 4, 3)*int(10)/int(1.0)
print(x1)
col1 = im2col(x1, 2, 2, stride=1, pad=0)
print(col1.shape) #
print("col1=",col1)


col12 = col2im(col1, (1,1,4,3), 2, 2, stride=1, pad=0)
print(col12.shape) #
print("col12=",col12)