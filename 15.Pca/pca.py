# -*- coding: utf-8 -*-

'''
Created on Jun 1, 2011

@author: Peter Harrington
'''
from numpy import *

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return mat(datArr)

def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    print "eigVals= ",eigVals
    print "eigVects=",eigVects
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    print "eigValInd=",eigValInd
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    print "eigValInd =",eigValInd
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    print "redEigVects=",redEigVects
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
    print "meanRemoved.shape=",meanRemoved.shape
    print "redEigVects.shape=",redEigVects.shape
    print "lowDDataMat.shape=",lowDDataMat.shape
    print "meanVals.shape=", meanVals.shape
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

def replaceNanWithMean(): 
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat


if __name__ == "__main__":
    print "--------------------"
    dataMat = loadDataSet('testSet.txt')
    lowDMat,reconMat = pca(dataMat,1)
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 三角形表示原始数据点
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
    # 圆形点表示第一主成分点，点颜色为红色
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker = 'o', s = 50, c = 'red')
    plt.show()

    dataMat = replaceNanWithMean()
    meanVals = mean(dataMat,axis=0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved,rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    print "eigVals=",eigVals

    eigValInd = argsort(eigVals)  # sort, sort goes smallest to largest
    eigValInd = eigValInd[::-1]  # reverse
    sortedEigVals = eigVals[eigValInd]
    total = sum(sortedEigVals)
    varPercentage = sortedEigVals / total * 100

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, 21), varPercentage[:20], marker='^')
    plt.xlabel('Principal Component Number')
    plt.ylabel('Percentage of Variance')
    plt.show()