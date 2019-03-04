#Usage:
#Training: NB.py 1 TrainingDataFile ModelFile
#Testing: NB.py 0 TestDataFile ModelFile OutFile

import sys
import os
import math


DefaultFreq = 0.1
TrainingDataFile = "nb_data.train"
ModelFile = "nb_data.model"
TestDataFile = "nb_data.test"
TestOutFile = "nb_data.out"
ClassFeaDic = {}
ClassFreq = {}
WordDic = {}
ClassFeaProb = {}
ClassDefaultProb = {}
ClassProb = {}

def Dedup(items):
    tempDic = {}
    for item in items:
        if item not in tempDic:
            tempDic[item] = True
    return tempDic.keys()

def LoadData():
    i =0
    infile = file(TrainingDataFile, 'r')
    sline = infile.readline().strip()
    while len(sline) > 0:
        pos = sline.find("#")
        if pos > 0:
            sline = sline[:pos].strip()
        words = sline.split(' ')
        if len(words) < 1:
            print "Format error!"
            break
        classid = int(words[0])
        if classid not in ClassFeaDic:
            ClassFeaDic[classid] = {}
            ClassFeaProb[classid] = {}
            ClassFreq[classid]  = 0
        ClassFreq[classid] += 1
        words = words[1:]
        #remove duplicate words, binary distribution
        #words = Dedup(words)
        for word in words:
            if len(word) < 1:
                continue
            wid = int(word)
            if wid not in WordDic:
                WordDic[wid] = 1
            if wid not in ClassFeaDic[classid]:
                ClassFeaDic[classid][wid] = 1
            else:
                ClassFeaDic[classid][wid] += 1
        i += 1
        sline = infile.readline().strip()
    infile.close()
    print i, "instances loaded!"
    print len(ClassFreq), "classes!", len(WordDic), "words!"


def ComputeModel():
    sum = 0.0
    for freq in ClassFreq.values():
        sum += freq
    for classid in ClassFreq.keys():
        ClassProb[classid] = (float)(ClassFreq[classid])/(float)(sum)
    for classid in ClassFeaDic.keys():
        #Multinomial Distribution
        sum = 0.0
        for wid in ClassFeaDic[classid].keys():
            sum += ClassFeaDic[classid][wid]
        newsum = (float)(sum+len(WordDic)*DefaultFreq)
        #Binary Distribution
        #newsum = (float)(ClassFreq[classid]+2*DefaultFreq)
        for wid in ClassFeaDic[classid].keys():
            ClassFeaProb[classid][wid] = (float)(ClassFeaDic[classid][wid]+DefaultFreq)/newsum
        ClassDefaultProb[classid] = (float)(DefaultFreq) / newsum
    return


def SaveModel():
    outfile = file(ModelFile, 'w')
    for classid in ClassFreq.keys():
        outfile.write(str(classid))
        outfile.write(' ')
        outfile.write(str(ClassProb[classid]))
        outfile.write(' ')
        outfile.write(str(ClassDefaultProb[classid]))
        outfile.write(' ' )
    outfile.write('\n')
    for classid in ClassFeaDic.keys():
        for wid in ClassFeaDic[classid].keys():
            outfile.write(str(wid)+' '+str(ClassFeaProb[classid][wid]))
            outfile.write(' ')
        outfile.write('\n')
    outfile.close()


def LoadModel():
    global WordDic
    WordDic = {}
    global ClassFeaProb
    ClassFeaProb = {}
    global ClassDefaultProb
    ClassDefaultProb = {}
    global ClassProb
    ClassProb = {}
    infile = file(ModelFile, 'r')
    sline = infile.readline().strip()
    items = sline.split(' ')
    if len(items) < 6:
        print "Model format error!"
        return
    i = 0
    while i < len(items):
        classid = int(items[i])
        ClassFeaProb[classid] = {}
        i += 1
        if i >= len(items):
            print "Model format error!" 
            return
        ClassProb[classid] = float(items[i])
        i += 1
        if i >= len(items):
            print "Model format error!" 
            return
        ClassDefaultProb[classid] = float(items[i])
        i += 1
    for classid in ClassProb.keys():
        sline = infile.readline().strip()
        items = sline.split(' ')
        i = 0
        while i < len(items):
            wid  = int(items[i])
            if wid not in WordDic:
                WordDic[wid] = 1
            i += 1
            if i >= len(items):
                print "Model format error!"
                return
            ClassFeaProb[classid][wid] = float(items[i])
            i += 1
    infile.close()
    print len(ClassProb), "classes!", len(WordDic), "words!"

def Predict():
    global WordDic
    global ClassFeaProb
    global ClassDefaultProb
    global ClassProb

    TrueLabelList = []
    PredLabelList = []
    i =0
    infile = file(TestDataFile, 'r')
    outfile = file(TestOutFile, 'w')
    sline = infile.readline().strip()
    scoreDic = {}
    iline = 0
    while len(sline) > 0:
        iline += 1
        if iline % 10 == 0:
            print iline," lines finished!\r",
        pos = sline.find("#")
        if pos > 0:
            sline = sline[:pos].strip()
        words = sline.split(' ')
        if len(words) < 1:
            print "Format error!"
            break
        classid = int(words[0])
        TrueLabelList.append(classid)
        words = words[1:]
        #remove duplicate words, binary distribution
        #words = Dedup(words)
        for classid in ClassProb.keys():
            scoreDic[classid] = math.log(ClassProb[classid])
        for word in words:
            if len(word) < 1:
                continue
            wid = int(word)
            if wid not in WordDic:
                #print "OOV word:",wid
                continue
            for classid in ClassProb.keys():
                if wid not in ClassFeaProb[classid]:
                    scoreDic[classid] += math.log(ClassDefaultProb[classid])
                else:
                    scoreDic[classid] += math.log(ClassFeaProb[classid][wid])
        #binary distribution
        #wid = 1
        #while wid < len(WordDic)+1:
        #   if str(wid) in words:
        #       wid += 1
        #       continue
        #   for classid in ClassProb.keys():
        #       if wid not in ClassFeaProb[classid]:
        #           scoreDic[classid] += math.log(1-ClassDefaultProb[classid])
        #       else:
        #           scoreDic[classid] += math.log(1-ClassFeaProb[classid][wid])
        #   wid += 1
        i += 1
        maxProb = max(scoreDic.values())
        for classid in scoreDic.keys():
            if scoreDic[classid] == maxProb:
                PredLabelList.append(classid)
        sline = infile.readline().strip()
    infile.close()
    outfile.close()
    print len(PredLabelList),len(TrueLabelList)
    return TrueLabelList,PredLabelList

def Evaluate(TrueList, PredList):
    accuracy = 0
    i = 0
    while i < len(TrueList):
        if TrueList[i] == PredList[i]:
            accuracy += 1
        i += 1
    accuracy = (float)(accuracy)/(float)(len(TrueList))
    print "Accuracy:",accuracy

def CalPreRec(TrueList,PredList,classid):
    correctNum = 0
    allNum = 0
    predNum = 0
    i = 0
    while i < len(TrueList):
        if TrueList[i] == classid:
            allNum += 1
            if PredList[i] == TrueList[i]:
                correctNum += 1
        if PredList[i] == classid:
            predNum += 1
        i += 1
    return (float)(correctNum)/(float)(predNum),(float)(correctNum)/(float)(allNum)

#main framework
if len(sys.argv) < 4:
    print "Usage incorrect!"
elif sys.argv[1] == '1':
    print "start training:"
    TrainingDataFile = sys.argv[2]
    ModelFile = sys.argv[3]
    LoadData()
    ComputeModel()
    SaveModel()
elif sys.argv[1] == '0':
    print "start testing:"
    TestDataFile = sys.argv[2]
    ModelFile = sys.argv[3]
    TestOutFile = sys.argv[4]
    LoadModel()
    TList,PList = Predict()
    i = 0
    outfile = file(TestOutFile, 'w')
    while i < len(TList):
        outfile.write(str(TList[i]))
        outfile.write(' ')
        outfile.write(str(PList[i]))
        outfile.write('\n')
        i += 1
    outfile.close()
    Evaluate(TList,PList)
    for classid in ClassProb.keys():
        pre,rec = CalPreRec(TList, PList,classid)
        print "Precision and recall for Class",classid,":",pre,rec
else:
    print "Usage incorrect!"
