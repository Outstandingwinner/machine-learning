
from numpy import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB),2))

def randCent(dataSet,k):
    n = shape(dataSet)[1]
    centorids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centorids[:,j] = mat(minJ + rangeJ*random.rand(k,1))
    return centorids

def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataSet,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf; minIndex= -1
            for j in range(k):
                distJI = distMeas(centroids[j:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
                if clusterAssment[i,0] != minIndex:
                    clusterChanged = True
                    clusterAssment[i, :] = minIndex,minDist
        for cent in range(k):
            pstInClus = dataSet[nonzero(clusterAssment[:, 0].A==cent)[0],:]
            centroids[cent,:] = mean(pstInClus,axis=0)
    return centroids,clusterAssment

def bikmeans(dataSet,k,disMeas = distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros(m,2))
    centroid0 = mean(dataSet,axis=0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j,1] = disMeas(mat(centroid0),dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            pstInCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat,splitClustAss = kMeans(pstInCluster,2,disMeas)
            sseSplit = sum(splitClustAss[:,1])
            sseNoSplit  = sum(clusterAssment[nonzero(clusterAssment[:,0].A !=i)[0],1])
            if (sseSplit + sseNoSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseNoSplit + sseSplit
        bestClustAss[nonzero(bestClustAss[:,0].A ==1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A ==0)[0],0] = bestCentToSplit
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
        centList.append(bestNewCents[1,:].tolist())[0]
        clusterAssment[nonzero(clusterAssment[:,0].A==bestCentToSplit)[0],:] = bestClustAss
    return mat(centList),clusterAssment








