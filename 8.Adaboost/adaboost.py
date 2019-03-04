# -*- coding: utf-8 -*-
'''
Created on Nov 28, 2010
Adaboost is short for Adaptive Boosting
@author: Peter
'''
from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    """
    单层决策树分类函数
    Parameters:
        dataMatrix - 数据矩阵
        dimen - 第dimen列，也就是第几个特征
        threshVal - 阈值
        threshIneq - 标志
    Returns:
        retArray - 分类结果
    """
    retArray = ones((shape(dataMatrix)[0],1))                    #初始化retArray为1
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0        #如果小于阈值,则赋值为-1
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0         #如果大于阈值,则赋值为-1
    return retArray
    

def buildStump(dataArr,classLabels,D):
    """
    找到数据集上最佳的单层决策树
    Parameters:
        dataArr - 数据矩阵
        classLabels - 数据标签
        D - 样本权重
    Returns:
        bestStump - 最佳单层决策树信息
        minError - 最小误差
        bestClasEst - 最佳的分类结果
    """
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf #init error sum, to +infinity                 #最小误差初始化为正无穷大
    for i in range(n):#loop over all dimensions                  #遍历所有特征
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();          #找到特征中最小的值和最大值
        stepSize = (rangeMax-rangeMin)/numSteps                                       #计算步长
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            for inequal in ['lt', 'gt']: #go over less than and greater than          #大于和小于的情况，均遍历
                threshVal = (rangeMin + float(j) * stepSize)                          #计算阈值
                # print "rangeMin:\n", rangeMin
                # print "j:\n", j
                # print "stepSize:\n", stepSize
                # print "threshVal:\n", threshVal
                # 计算分类结果
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                # print "predictedVals:\n", predictedVals
                # 初始化误差矩阵
                errArr = mat(ones((m,1)))
                # 分类正确的,赋值为0
                errArr[predictedVals == labelMat] = 0
                # 计算误差
                weightedError = D.T*errArr  #calc total error multiplied by D
                # print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                # 找到误差最小的分类方式
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst


def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    # 初始化权重
    D = mat(ones((m,1))/m)   #init D to all equal
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        # 构建单层决策树
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
        #print "D:",D.T
        # 计算弱学习算法权重alpha,使error不等于0,因为分母不能为0
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
        # 存储弱学习算法权重
        bestStump['alpha'] = alpha
        # 存储单层决策树
        weakClassArr.append(bestStump)                  #store Stump Params in Array
        # print "classEst: ",classEst.T
        # 计算e的指数项
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy
        D = multiply(D,exp(expon))                              #Calc New D for next iteration
        # 根据样本权重公式，更新样本权重
        D = D/D.sum()
        # 计算AdaBoost误差，当误差为0的时候，退出循环
        aggClassEst += alpha*classEst
        # print "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"
        # print "alpha: "     ,alpha
        # print "classEst: "  , classEst
        # print "aggClassEst: ",aggClassEst.T
        # 计算误差
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print "total error: ",errorRate
        # 误差为0，退出循环
        if errorRate == 0.0: break
    # print "==============================="
    # print "i= ",i
    # print "weakClassArr= ",weakClassArr
    # return weakClassArr
    return weakClassArr, aggClassEst

def adaClassify(datToClass,classifierArr):
    """
    AdaBoost分类函数
    Parameters:
        datToClass - 待分类样例
        classifierArr - 训练好的分类器
    Returns:
        分类结果
    """
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    # 遍历所有分类器，进行分类
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
        print aggClassEst
    return sign(aggClassEst)




def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor                                                     #绘制光标的位置
    ySum = 0.0 #variable to calculate AUC                                       #用于计算AUC
    numPosClas = sum(array(classLabels)==1.0)                                   #统计正类的数量
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)   #y x轴步长
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse     #预测强度排序
    print "sortedIndicies:",sortedIndicies
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            # 高度累加
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        # 绘制ROC
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        # 更新绘制光标的位置
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    # 计算AUC
    print "the Area Under the Curve is: ",ySum*xStep

if __name__ == '__main__':
    # dataArr,classLabels = loadSimpData()
    # D = mat(ones((5, 1)) / 5)
    # bestStump,minError,bestClasEst = buildStump(dataArr,classLabels,D)
    # print('bestStump:\n', bestStump)
    # print('minError:\n', minError)
    # print('bestClasEst:\n', bestClasEst)
    # weakClassArr= adaBoostTrainDS(dataArr, classLabels)
    # print(weakClassArr)
    # print adaClassify([[5,5],[0,0]],weakClassArr)

    # 训练
    # dataArr,labelArr = loadDataSet('horseColicTraining2.txt')
    # classfierArray = adaBoostTrainDS(dataArr, classLabels,10)
    # # 测试
    # testArr, testylabelArr = loadDataSet('horseColicTest2.txt')
    # prediction10 = adaClassify(testArr,classfierArray)
    # errArr = mat(ones(67,1))
    # print "error rate:",errArr[prediction10 != mat(testylabelArr).T].sum/67

    #
    # dataArr, LabelArr = loadDataSet('horseColicTraining2.txt')
    # weakClassArr = adaBoostTrainDS(dataArr, LabelArr)
    # testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    # print(weakClassArr)
    # predictions = adaClassify(dataArr, weakClassArr)
    # errArr = mat(ones((len(dataArr), 1)))
    # print('训练集的错误率:%.3f%%' % float(errArr[predictions != mat(LabelArr).T].sum() / len(dataArr) * 100))
    # predictions = adaClassify(testArr, weakClassArr)
    # errArr = mat(ones((len(testArr), 1)))
    # print('测试集的错误率:%.3f%%' % float(errArr[predictions != mat(testLabelArr).T].sum() / len(testArr) * 100))

   # auc
    dataArr, LabelArr = loadDataSet('horseColicTraining2.txt')
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, LabelArr)
    plotROC(aggClassEst.T, LabelArr)

