#! /usr/bin/python

import sys
import os
import random
import math

DocList = []
ClassCenterList = []
ClassSizeList = []
ClassList = []
DocNameList = []
K = 5
Threshold = 1.0
inpath = "ClusterData"
outFile = "clusterResult.txt"

#number of docs to load for each class
NumList = [200,200,200,200,200]
readNumList = [0,0,0,0,0]
MaxDocNum = 1000

def Normalize(vector):
    sqrt = 0.0
    for (wid, freq) in vector.items():
        sqrt += freq*freq
    sqrt = math.sqrt(sqrt)
    for (wid, freq) in vector.items():
        vector[wid] = freq/(float)(sqrt)

def LoadData():
    i = 0
    for filename in os.listdir(inpath):
        if filename.find("business") != -1:
            readNumList[0] += 1
            if readNumList[0] > NumList[0]:
                continue
        elif filename.find("auto") != -1:
            readNumList[1] += 1
            if readNumList[1] > NumList[1]:
                continue
        elif filename.find("sport") != -1:
            readNumList[2] += 1
            if readNumList[2] > NumList[2]:
                continue
        elif filename.find("it") != -1:
            readNumList[3] += 1
            if readNumList[3] > NumList[3]:
                continue
        elif filename.find("yule") != -1:
            readNumList[4] += 1
            if readNumList[4] > NumList[4]:
                continue
        i += 1
        if i > MaxDocNum:
            break
        infile = file(inpath+'/'+filename, 'r')
        DocNameList.append(filename)
        content = infile.read().strip()
        content = content.decode("utf-8")
        words = content.replace('\n', ' ').split(' ')
        newdic = {}
        for word in words:
            if len(word.strip()) < 1:
                continue
            if word not in newdic:
                newdic[word] = 1.0
            else:
              newdic[word] += 1.0
        Normalize(newdic)
        DocList.append(newdic)
        ClassList.append(-1)
    print len(DocList), "files loaded!"

def AddIDF(DocList):
    wordDic = {}
    for doc in DocList:
        for word in doc.keys():
            if word in wordDic:
                wordDic[word] += 1
            else:
                wordDic[word] = 1
    N = len(DocList)
    for doc in DocList:
        for word in doc.keys():
            #doc[word] *= math.log(N+1/(float)(wordDic[word]))
			doc[word] *= math.log(N/(1+(float)(wordDic[word])))
        Normalize(doc)

def Init():
    templist = random.sample(DocList, K)
    for i in range(K):
        ClassSizeList.append(0)
        ClassCenterList.append(templist[i])

def ComputeDis(doc1, doc2):
    sum = 0.0
    for (wid, freq) in doc1.items():
        if wid in doc2:
            d = freq - doc2[wid]
            sum += d*d
        else:
            sum += freq * freq
    for (wid, freq) in doc2.items():
        if wid not in doc1:
            sum += freq * freq
    #sum = math.sqrt(sum)
    return sum


def AddDoc(centroid, doc):
    for (wid, freq) in doc.items():
        if wid in centroid:
            centroid[wid] += freq
        else:
            centroid[wid] = freq

def Average(i):
    center = ClassCenterList[i]
    #print "center=", center
    newCenter = {}
    for (wid, freq) in center.items():
        newCenter[wid] = freq/(float)(ClassSizeList[i])
    return newCenter

#Reassign Classes
def ReAssignClass():
    did = 0
    totalDis = 0.0
    for doc in DocList:
        min = ComputeDis(doc, ClassCenterList[0])
        minIndex = 0
        for i in range(1, K):
            dis = ComputeDis(doc, ClassCenterList[i])
            if dis < min:
                min = dis
                minIndex = i
        ClassList[did] = minIndex
        did += 1
        totalDis += min
    return totalDis

def ReComputeCentroids():
    for i in range(K):
        ClassSizeList[i] = 0
        ClassCenterList[i] = {}
    for i in range(len(DocList)):
        classid = ClassList[i]
        ClassSizeList[classid] += 1
        AddDoc(ClassCenterList[classid], DocList[i])
    for i in range(K):
        ClassCenterList[i] = Average(i)


#main framework
LoadData()
#Use tf*idf as feature weight
AddIDF(DocList)
Init()
WCSS = 0.0
oldWCSS = WCSS
WCSS = ReAssignClass()

i = 1
while math.fabs(oldWCSS - WCSS) > Threshold:
    oldWCSS = WCSS
    print "Iteration", i, "WCSS:", WCSS
    ReComputeCentroids()
    WCSS = ReAssignClass()
    i += 1
ReComputeCentroids()
print "Final iteration WCSS:", WCSS

#write the clustering result into outFile
outfile = file(outFile,'w')
i = 0
while i < len(DocList):
    outfile.write(DocNameList[i])
    outfile.write(":")
    outfile.write(str(ClassList[i]))
    outfile.write('\n')
    i += 1
outfile.close()

def GetMax(ilist):
    max = 0
    for i in ilist:
        if max < i:
            max = i
    return max

#compute accuracy
CorrectNumList = []
TK = 5
for i in range(K):
    CorrectNumList.append([])
    for j in range(TK):
        CorrectNumList[i].append(0)

for i in range(len(DocList)):
    if DocNameList[i].find("business") != -1:
        CorrectNumList[ClassList[i]][0] += 1
    elif DocNameList[i].find("auto") != -1:
        CorrectNumList[ClassList[i]][1] += 1
    elif DocNameList[i].find("sport") != -1:
        CorrectNumList[ClassList[i]][2] += 1
    elif DocNameList[i].find("it") != -1:
        CorrectNumList[ClassList[i]][3] += 1
    elif DocNameList[i].find("yule") != -1:
        CorrectNumList[ClassList[i]][4] += 1
print "CorrectNumList= ",CorrectNumList
corNum = 0.0
for i in range(K):
    max = GetMax(CorrectNumList[i])
    pre = (float)(max)/(float)(ClassSizeList[i])
    print "Cluster",i,"Precision:",pre
    corNum += max

print corNum, "documents correctly clustered!"
acc = corNum / (float)(len(DocList))
print "Accuracy:",acc
