import numpy as np
import time
import sys
import random
# from tt import TwoLayerNet

def conv2(X, k):
    # as a demo code, here we ignore the shape check
    x_row, x_col = X.shape
    k_row, k_col = k.shape
    ret_row, ret_col = x_row - k_row + 1, x_col - k_col + 1
    ret = np.empty((ret_row, ret_col))
    for y in range(ret_row):
        for x in range(ret_col):
            sub = X[y: y + k_row, x: x + k_col]
            ret[y, x] = np.sum(sub * k)
    return ret


def rot180(in_data):
    ret = in_data.copy()
    yEnd = ret.shape[0] - 1
    xEnd = ret.shape[1] - 1
    for y in range(int(ret.shape[0] / 2)):
        for x in range(ret.shape[1]):
            ret[yEnd - y][x] = ret[y][x]
    for y in range(ret.shape[0]):
        for x in range(int(ret.shape[1] / 2)):
            ret[y][xEnd - x] = ret[y][x]
    return ret


def padding(in_data, size):
    cur_r, cur_w = in_data.shape[0], in_data.shape[1]
    new_r = cur_r + size * 2
    new_w = cur_w + size * 2
    ret = np.zeros((new_r, new_w))
    ret[size:cur_r + size, size:cur_w + size] = in_data
    return ret


def discreterize(in_data, size):
    num = in_data.shape[0]
    ret = np.zeros((num, size))
    for i, idx in enumerate(in_data):
        ret[i, idx] = 1
    return ret


class ConvLayer(object):
    def __init__(self, in_channel, out_channel, kernel_size, lr=0.01, stride = 1, pad = 1, momentum=0.9, reg = 0.75, name='Conv'):
        self.w = np.random.randn(out_channel,in_channel, kernel_size, kernel_size)
        self.b = np.random.randn(out_channel)
        w_shape = (out_channel,in_channel, kernel_size, kernel_size)
        self.layer_name = name
        self.lr = lr
        self.momentum = momentum
        self.stride = stride
        self.pad = pad
        self.reg = reg

        self.prev_gradient_w = np.zeros_like(self.w)
        self.prev_gradient_b = np.zeros_like(self.b)

    def forward(self, in_data):
        self.out = None
        N, C, H, W = in_data.shape
        F, _, HH, WW = self.w.shape
        stride, pad = self.stride, self.pad
        H_out = int(1 + (H + 2 * pad - HH) / stride)
        W_out = int(1 + (W + 2 * pad - WW) / stride)
        self.out = np.zeros((N , F , H_out, W_out))

        in_data_pad = np.pad(in_data, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
        for i in range(H_out):
            for j in range(W_out):
                in_data_pad_masked = in_data_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
                for k in range(F):
                    self.out[:, k , i, j] = np.sum(in_data_pad_masked * self.w[k, :, :, :], axis=(1,2,3))

        self.bottom_val = in_data
        return self.out

    def backward(self, residual):
        N, C, H, W = self.bottom_val.shape
        F, _, HH, WW = self.w.shape
        stride, pad = self.stride, self.pad
        H_out = int(1 + (H + 2 * pad - HH) / stride)
        W_out = int(1 + (W + 2 * pad - WW) / stride)

        x_pad = np.pad(self.bottom_val, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
        dx = np.zeros_like(self.bottom_val)
        dx_pad = np.zeros_like(x_pad)
        dw = np.zeros_like(self.w)
        # db = np.zeros_like(self.b)

        db = np.sum(residual, axis=(0, 2, 3))

        x_pad = np.pad(self.bottom_val, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
        for i in range(H_out):
            for j in range(W_out):
                x_pad_masked = x_pad[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
                for k in range(F):  # compute dw
                    dw[k, :, :, :] += np.sum(x_pad_masked * (residual[:, k, i, j])[:, None, None, None], axis=0)
                for n in range(N):  # compute dx_pad
                    temp_w = np.rot90(self.w,2,(2,3))
                    dx_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW] += np.sum((self.w[:, :, :, :] *
                                                                                                    (residual[n, :, i,
                                                                                                     j])[
                                                                                                    :, None, None,
                                                                                                    None]), axis=0)
        dx[:,:,:,:] = dx_pad[:, :, pad:-pad, pad:-pad]
        self.w -= self.lr * (dw + self.prev_gradient_w * self.reg)
        self.b -= self.lr * db
        self.prev_gradient_w = self.w
        return dx


class FCLayer:
    def __init__(self, in_num, out_num, lr=0.01, momentum=0.9, std = 1e-4, reg = 0.75):
        self._in_num = in_num
        self._out_num = out_num
        self.w =std * np.random.randn(in_num, out_num)
        self.b =std * np.zeros(out_num)
        self.lr = lr
        self.momentum = momentum
        self.prev_grad_w = np.zeros_like(self.w)
        self.prev_grad_b = np.zeros_like(self.b)
        self.reg = reg

    def forward(self, in_data):
        self.bottom_val = in_data
        self.top_val = in_data.dot(self.w) + self.b
        return self.top_val

    def backward(self, loss):
        residual_x = loss.dot(self.w.T)
        self.w -= self.lr * (self.bottom_val.T.dot(loss) + self.prev_grad_w * self.reg)
        self.b -= self.lr * (np.sum(loss, axis=0))
        self.prev_grad_w = self.w
        self.prev_grad_b = self.b
        return residual_x


class ReLULayer:
    def __init__(self, name='ReLU'):
        pass

    def forward(self, in_data):

        in_data[in_data<0] = 0
        self.top_val = in_data
        return in_data

    def backward(self, residual):
        return (self.top_val > 0) * residual


class MaxPoolingLayer:
    def __init__(self, kernel_size, stride = 1, name='MaxPool'):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, in_data):
        self.bottom_val = in_data

        N, C, H, W = in_data.shape
        HH, WW, stride = self.kernel_size, self.kernel_size, self.stride
        H_out = int((H - HH) / stride + 1)
        W_out = int((W - WW) / stride + 1)
        out = np.zeros((N, C, H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                x_masked = in_data[:, :, i * stride: i * stride + HH, j * stride: j * stride + WW]
                out[:, :, i, j] = np.max(x_masked, axis=(2, 3))
        return out

    def backward(self, residual):
        N, C, H, W = self.bottom_val.shape
        HH, WW, stride = self.kernel_size, self.kernel_size, self.stride
        H_out = int((H - HH) / stride + 1)
        W_out = int((W - WW) / stride + 1)
        dx = np.zeros_like(self.bottom_val)

        for i in range(H_out):
            for j in range(W_out):
                x_masked = self.bottom_val[:, :, i * stride: i * stride + HH, j * stride: j * stride + WW]
                max_x_masked = np.max(x_masked, axis=(2, 3))
                temp_binary_mask = (x_masked == (max_x_masked)[:, :, None, None])
                dx[:, :, i * stride: i * stride + HH, j * stride: j * stride + WW] += temp_binary_mask * (residual[:, :, i,
                                                                                                          j])[:, :,
                                                                                                         None, None]
        return dx


class FlattenLayer:
    def __init__(self, name='Flatten'):
        pass

    def forward(self, in_data):
        self.in_batch, self.in_channel, self.r, self.c = in_data.shape
        return in_data.reshape(self.in_batch, self.in_channel * self.r * self.c)

    def backward(self, residual):
        return residual.reshape(self.in_batch, self.in_channel, self.r, self.c)


class SoftmaxLayer:
    def __init__(self, name='Softmax'):
        pass

    def forward(self, in_data):
        shift_scores = in_data - np.max(in_data, axis=1).reshape(-1, 1)
        self.top_val = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis=1).reshape(-1, 1)
        return self.top_val

    def backward(self, residual):
        N = residual.shape[0]
        dscores = self.top_val.copy()
        dscores[range(N), list(residual)] -= 1
        dscores /= N
        return dscores


class Net:
    def __init__(self):
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def train(self, trainData, trainLabel, validData, validLabel, batch_size, iteration):
        train_num = trainData.shape[0]
        strainData = trainData
        strainLabel = trainLabel
        for iter in range(iteration):
            index = np.random.choice([ i for i in range(train_num)], train_num)
            # index = [i for i in range(train_num)]
            trainData = strainData[index]
            trainLabel = strainLabel[index]

            if iter > 100:
                lay_num = len(self.layers)
                for i in range(lay_num):
                   self.layers[i].lr *= (0.001 ** ( iter - 100 ) / 100)

            print(str(time.clock()) + '  iter=' + str(iter))
            for batch_iter in range(0, train_num, batch_size):
                if batch_iter + batch_size < train_num:
                    loss = self.train_inner(trainData[batch_iter: batch_iter + batch_size],
                                     trainLabel[batch_iter: batch_iter + batch_size])
                else:
                    loss = self.train_inner(trainData[batch_iter: train_num],
                                     trainLabel[batch_iter: train_num])
                print(str(batch_iter) + '/' + str(train_num) + '   loss : ' + str(loss))
            print(str(time.clock()) + "  eval=" + str(self.eval(trainData, trainLabel)))
            print(str(time.clock()) + "  eval=" + str(self.eval(validData, validLabel)))

    def train_inner(self, data, label):
        lay_num = len(self.layers)
        in_data = data
        for i in range(lay_num):
            out_data = self.layers[i].forward(in_data)
            in_data = out_data
        N = out_data.shape[0]
        loss = -np.sum(np.log(out_data[range(N), list(label)]))
        loss /= N
        residual_in = label
        for i in range(0, lay_num):
            residual_out = self.layers[lay_num - i - 1].backward(residual_in)
            residual_in = residual_out
        return loss

    def eval(self, data, label):
        lay_num = len(self.layers)
        in_data = data
        for i in range(lay_num):
            out_data = self.layers[i].forward(in_data)
            in_data = out_data
        out_idx = np.argmax(in_data, axis=1)
        # label_idx = np.argmax(label, axis=1)
        label_idx = label
        return np.sum(out_idx == label_idx) / float(out_idx.shape[0])


import struct
from array import array


def loadImageSet(filename):
    print("load image set", filename)
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0)
    print("head,", head)

    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    # [60000]*28*28
    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B'  # like '>47040000B'

    imgs = struct.unpack_from(bitsString, buffers, offset)

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, -1])
    print("load imgs finished")
    return imgs


def loadLabelSet(filename):
    print("load label set", filename)
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0)
    print("head,", head)
    imgNum = head[1]

    offset = struct.calcsize('>II')
    numString = '>' + str(imgNum) + "B"
    labels = struct.unpack_from(numString, buffers, offset)
    binfile.close()
    labels = np.reshape(labels, [imgNum, 1])

    # ret = np.zeros((imgNum, 10), dtype=np.float64)
    # for i in range(imgNum):
    #     ret[i][labels[i]] = 1.0
    print('load label finished')
    ret = labels
    return ret


# train_feature = loadImageSet("data//MNIST_data//train-images.idx3-ubyte")
# train_feature[train_feature > 0] = 1
# train_label = loadLabelSet("data//MNIST_data//train-labels.idx1-ubyte")
# valid_feature = loadImageSet("data//MNIST_data//t10k-images.idx3-ubyte")
# valid_feature[valid_feature > 0] = 1
# valid_label = loadLabelSet("data//MNIST_data//t10k-labels.idx1-ubyte")
# train_feature = loadImageSet("data\\MNIST_data\\train-images.idx3-ubyte")
# train_label = loadLabelSet("data\\MNIST_data\\train-labels.idx1-ubyte")
# valid_feature = loadImageSet("data\\MNIST_data\\t10k-images.idx3-ubyte")
# valid_label = loadLabelSet("data\\MNIST_data\\t10k-labels.idx1-ubyte")


rate = 1e-5
net = Net()
net.addLayer(ConvLayer(3, 32, 5, rate))
net.addLayer(MaxPoolingLayer(3,2))
net.addLayer(ReLULayer())

net.addLayer(ConvLayer(32, 16, 3, rate))
net.addLayer(MaxPoolingLayer(3,2))
net.addLayer(ReLULayer())

net.addLayer(FlattenLayer())
net.addLayer(FCLayer(6 * 6 * 16, 100, rate))
net.addLayer(ReLULayer())
net.addLayer(FCLayer(100, 10, rate))
net.addLayer(SoftmaxLayer())
# print('net build ok')
from data_utils import load_CIFAR10


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    # cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    cifar10_dir = 'C:/Users/Administrator/Desktop/ml_in_action_note/CNN/cifar-10-batches-py/'

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    # X_train = X_train.reshape(num_training, -1)
    X_train = np.transpose(X_train,[0,3,1,2])
    X_val = np.transpose(X_val, [0,3,1,2])
    # X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
N = 1000
M = 100
net.train(X_train[0:N], y_train[0:N], X_val[0:M], y_val[0:M],100,200)
# net.train(train_feature, train_label, valid_feature[0:100], valid_label[0:100], 10 ,10)
