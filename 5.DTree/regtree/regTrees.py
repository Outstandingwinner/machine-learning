# -*- coding: utf-8 -*-

'''
Created on Feb 4, 2011
Tree-Based Regression Methods
@author: Peter Harrington
'''
from numpy import *

#解析文本数据
def loadDataSet(filename):
    dataMat=[]
    fr=open(filename)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        #将每行数据映射为浮点数
        fltLine=map(float,curLine)
        dataMat.append(fltLine)
    return dataMat

#拆分数据集函数，二元拆分法
#@dataSet：待拆分的数据集
#@feature：作为拆分点的特征索引
#@value：特征的某一取值作为分割值
def binSplitDataSet(dataSet,feature,value):
    #采用条件过滤的方法获取数据集每个样本目标特征的取值大于
    #value的样本存入mat0
    #左子集列表的第一行
    #mat0=dataSet[nonzero(dataSet[:,feature]>value)[0],:][0]
    #左子集列表
    mat0=dataSet[nonzero(dataSet[:,feature]>value)[0],:]
    #同上，样本目标特征取值不大于value的样本存入mat1
    mat1=dataSet[nonzero(dataSet[:,feature]<=value)[0],:]
    #返回获得的两个列表
    return mat0,mat1


# 负责生成叶节点，当chooseBestSplit()函数确定不再对数据进行切分时，
# 将调用该regLeaf()函数来得到叶节点的模型，在回归树中，该模型其实就是目标变量的均值
def regLeaf(dataSet):#returns the value used for each leaf
    return mean(dataSet[:,-1])


# 误差估计函数，该函数在给定的数据上计算目标变量的平方误差，这里直接调用均方差函数var
# 因为这里需要返回的是总方差，所以要用均方差乘以数据集中样本的个数
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

#创建树函数
#@dataSet：数据集
#@leafType：生成叶节点的类型 1 回归树：叶节点为常数值 2 模型树：叶节点为线性模型
#@errType：计算误差的类型 1 回归错误类型：总方差=均方差*样本数
#                         2 模型错误类型：预测误差(y-yHat)平方的累加和
#@ops：用户指定的参数，包含tolS：容忍误差的降低程度 tolN：切分的最少样本数
def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    #选取最佳分割特征和特征值
    feat,val=chooseBestSplit(dataSet,leafType,errType,ops)
    #如果特征为none，直接返回叶节点值
    if feat == None:return val
    #树的类型是字典类型
    retTree={}
    #树字典的一个元素是切分的最佳特征
    retTree['spInd']=feat
    #第二个元素是最佳特征对应的最佳切分特征值
    retTree['spval']=val
    #根据特征索引及特征值对数据集进行二元拆分，并返回拆分的两个数据子集
    lSet,rSet=binSplitDataSet(dataSet,feat,val)
    #第三个元素是树的左分支，通过lSet子集递归生成左子树
    retTree['left']=createTree(lSet,leafType,errType,ops)
    #第四个元素是树的右分支，通过rSet子集递归生成右子树
    retTree['right']=createTree(rSet,leafType,errType,ops)
    #返回生成的数字典
    return retTree




#模型树叶节点生成函数
def linearSolve(dataSet):
    #获取数据行与列数
    m,n=shape(dataSet)
    #构建大小为(m,n)和(m,1)的矩阵
    X=mat(ones((m,n)));Y=mat(ones((m,1)))
    #数据集矩阵的第一列初始化为1，偏置项；每个样本目标变量值存入Y
    X[:,1:n]=dataSet[:,0:n-1];Y=dataSet[:,-1]
    #对数据集矩阵求内积
    xTx=X.T*X
    #计算行列式值是否为0，即判断是否可逆
    if linalg.det(xTx)==0.0:
        #不可逆，打印信息
        print('This matrix is singular,cannot do inverse,\n\
                try increasing the second value if ops')
    #可逆，计算回归系数
    ws=(xTx).I*(X.T*Y)
    #返回回顾系数;数据集矩阵;目标变量值矩阵
    return ws,X,Y

#模型树的叶节点模型
def modelLeaf(dataSet):
    #调用线性回归函数生成叶节点模型
    ws,X,Y=linearSolve(dataSet)
    #返回该叶节点线性方程的回顾系数
    return ws

#模型树的误差计算函数
def modelErr(dataSet):
    #构建模型树叶节点的线性方程，返回参数
    ws,X,Y=linearSolve(dataSet)
    #利用线性方程对数据集进行预测
    yHat=X*ws
    #返回误差的平方和，平方损失
    return sum(power(Y-yHat,2))


# 回归树的切分函数，构建回归树的核心函数。目的：找出数据的最佳二元切分方式。如果找不到
# 一个“好”的二元切分，该函数返回None并同时调用createTree()方法来产生叶节点，叶节点的
# 值也将返回None。
# 如果找到一个“好”的切分方式，则返回特征编号和切分特征值。
# 最佳切分就是使得切分后能达到最低误差的切分。
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    # tolS是容许的误差下降值
    # tolN是切分的最小样本数
    tolS = ops[0]; tolN = ops[1]
    #if all the target variables are the same value: quit and return value
    '''把 np 类型矩阵的一列转换成行，然后转换成 list 类型，取这一行有多少元素'''
    # 1 如果数据集切分之前，该数据集样本所有的目标变量值相同，那么不需要切分数据集，而直接将目标变量值作为叶节点返回
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        return None, leafType(dataSet)
    # 当前数据集的大小
    m,n = shape(dataSet)
    #the choice of the best feature is driven by Reduction in RSS error from mean
    # 当前数据集的误差
    # 计算数据集最后一列的特征总方差。
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        # for splitVal in set(dataSet[:,featIndex]):
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS

    # 2 当切分数据集后，误差的减小程度不够大（小于tolS）,就不需要切分，而是直接求取数据集目标变量的均值作为叶节点值返回
    #if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS: 
        return None, leafType(dataSet) #exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 3 当数据集切分后如果某个子集的样本个数小于tolN，也不需要切分，而直接生成叶节点
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #exit cond 3
        return None, leafType(dataSet)
    return bestIndex,bestValue#returns the best feature to split on
                              #and the value used for that split

# dataSet: 数据集合
# leafType: 给出建立叶节点的函数
# errType: 误差计算函数
# ops: 包含树构建所需其他参数的元组
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    # 将数据集分成两个部分，若满足停止条件，chooseBestSplit将返回None和某类模型的值
    # 若构建的是回归树，该模型是个常数。若是模型树，其模型是一个线性方程。
    # 若不满足停止条件，chooseBestSplit()将创建一个新的Python字典，并将数据集分成两份，
    # 在这两份数据集上将分别继续递归调用createTree()函数
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#choose the best split
    if feat == None: return val #if the splitting hit a stop condition return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree  

#该函数用于判断当前处理的是否是叶节点
def isTree(obj):
    return (type(obj).__name__=='dict')

#从上往下遍历树，寻找叶节点，并进行塌陷处理（用两个孩子节点的平均值代替父节点的值）
def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

# 检查是否适合合并分枝
def prune(tree, testData):
    '''
       Desc:
           从上而下找到叶节点，用测试数据集来判断将这些叶节点合并是否能降低测试误差
       Args:
           tree -- 待剪枝的树
           testData -- 剪枝所需要的测试数据 testData
       Returns:
           tree -- 剪枝完成的树
    '''
    # 判断是否测试数据集没有数据，如果没有，就直接返回tree本身的均值
    if shape(testData)[0] == 0:
        return getMean(tree) #if we have no test data collapse the tree
    # 如果测试集非空，按照保存的回归树对测试集进行切分
    # 如果树枝不是树，试着修剪它们。
    # 如果回归树的左右两边是树
    if (isTree(tree['right']) or isTree(tree['left'])):#if the branches are not trees try to prune them
        # 对测试数据 进行切分
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    # 对左树进行剪枝
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    # 对右树进行剪枝
    if isTree(tree['right']):
        tree['right'] =  prune(tree['right'], rSet)
    # 如果左右两边同时都不是dict字典，也就是左右两边都是叶节点，而不是子树了，那么分割测试数据集。
    # 1. 如果正确
    #   * 那么计算一下总方差 和 该结果集的本身不分枝的总方差比较
    #   * 如果 合并的总方差 < 不合并的总方差，那么就进行合并
    # 注意返回的结果： 如果可以合并，原来的dict就变为了 数值

    # 如果它们现在都是叶子，看看我们是否能合并它们。
    # 两边都是叶子

    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        # 对测试数据 进行切分
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # power(x, y)表示x的y次方
        # 没有合并的误差
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))
        # 求合并后的误差
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        # 如果 合并的总方差 < 不合并的总方差，那么就进行合并
        if errorMerge < errorNoMerge: 
            print "merging"
            return treeMean
        else:
            return tree
    else:
        return tree


def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        print "###############################"
        print "tree=",tree
        print "tree['spInd']=",tree['spInd']
        print "inData[tree['spInd']]=",inData[tree['spInd']]
        print "tree['spVal']=", tree['spVal']
        print "inData=",inData

        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat

if __name__ == '__main__':
    # testMat = mat(eye(4))
    # print testMat
    # mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    # print '$$$$$$$$$$$$$$$$$$$$'
    # print mat0
    # print mat1

    # myDat = loadDataSet('ex00.txt')
    # myMat = mat(myDat)
    # myTree = createTree(myMat)
    # print "myTree\n",myTree
    #
    #
    # myDat1 = loadDataSet('ex0.txt')
    # myMat1 = mat(myDat1)
    # myTree1 = createTree(myMat1)
    # print "myTree1\n", myTree1
    #
    # myDat2 = loadDataSet('ex2.txt')
    # myMat2 = mat(myDat1)
    # myTree2 = createTree(myMat2)
    # # myTree2 = createTree(myMat2, ops=(70, 4))
    # print "myTree2\n", myTree2
    #
    # myDatTest = loadDataSet('ex2test.txt')
    # myMat2Test = mat(myDatTest)
    # myTree2Test = createTree(myMat2,ops = (0,1))
    # print "myTree2\n", myTree2Test
    # print "myTree2Test减枝后：\n", prune(myTree2Test,myMat2Test)
    #
    # # 模型树
    # myMat3 = mat(loadDataSet('exp2.txt'))
    # myTree3 = createTree(myMat3,leafType=modelLeaf,errType=modelErr,ops=(1,10))
    # # myTree2 = createTree(myMat2, ops=(70, 4))
    # print "myTree3\n", myTree3

   # 预测
    trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat  = mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    myTree   = createTree(trainMat,ops = (1,20))
    yHat     =  createForeCast(myTree,testMat[:,0])
    cr = corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]
    print cr
    #
    # myTree = createTree(trainMat,leafType=modelLeaf,errType=modelErr,ops=(1,10))
    # yHat     =  createForeCast(myTree,testMat[:,0], modelEval=modelTreeEval)
    # cr = corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]
    # print cr
    #
    # ws,X,Y = linearSolve(trainMat)
    # for i in range(shape(testMat)[0]):
    #     yHat[i] = testMat[i,0]*ws[1,0] + ws[0,0]
    # cr = corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1]
    # print cr




