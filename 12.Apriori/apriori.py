# -*- coding: utf-8 -*-
'''
Created on Mar 24, 2011
Ch 11 code
@author: Peter
'''
from numpy import *

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
                
    C1.sort()
    return map(frozenset, C1)


def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.has_key(can): ssCnt[can]=1
                else: ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    # print "=============================="
    # print "ssCnt = ",ssCnt
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData


def aprioriGen(Lk, k):                                         # 用于生成下一层的频繁项集（即项数是当前一次的项数+1狼，即k）
    # print "********************************"
    # print "Lk=",Lk
    # print "k=",k

    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):                                     # O(n^2)组合生成新项集
        for j in range(i+1, lenLk): 
            L1 = list(Lk[i])[:k-2];
            L2 = list(Lk[j])[:k-2]     #去除两者的前k-2个项
            # print "&%%%%%%%%%%%%%%%%%%%%%%%%%%%"
            # print "L1=",L1
            # print "L2=",L2
            L1.sort();
            L2.sort()
            if L1==L2:                                         # 如果前k-2个项相等，那么将Lk[i]和Lk[j]并在一起，就形成了k+1个项的项集
                retList.append(Lk[i] | Lk[j]) #set union
            # print "retList=",retList
    return retList

def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)                                 #set形式数据集（即去除重复的数据）
    D = map(set, dataSet)                                  #单位项集
    L1, supportData = scanD(D, C1, minSupport)             #单位频繁项集
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):                              #如果当层的频繁项集的个数大于1，那么就可以根据当层的频繁项集生成下一层的频繁项
        Ck = aprioriGen(L[k-2], k)                        #生成下一层的项集
        # print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
        # print "L=",L
        # print "k-2=",k-2
        # print "L[k-2]=",L[k-2]
        # print "D=",D
        # print "CK=",Ck
        # print "minSupport=",minSupport
        Lk, supK = scanD(D, Ck, minSupport)               #生成下一层的频繁项集，同时得到项集的支持度
        # print "Lk=",Lk
        supportData.update(supK)                          #更新支持度库

        L.append(Lk)                                      #把下一层的频繁项集加入到“层叠”里面
        k += 1                                            #将下一层作为当层
    # print "L=",L
    # print "k=",k
    return L, supportData


def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    bigRuleList = []
    # 假设 L = [[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])], [frozenset([1, 3]), frozenset([2, 5]), frozenset([2, 3]), frozenset([3, 5])], [frozenset([2, 3, 5])]]
    for i in range(1, len(L)):#only get the sets with two or more items
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                print "______________________________"
                print "freqSet=",freqSet
                print "H1",H1
                print "supportData",supportData
                print "bigRuleList",bigRuleList
                print "minConf",minConf
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    print "calcConf...."
    print "freqSet=",freqSet
    print "H=",H
    print "supportData=",supportData
    print "brl=",brl
    prunedH = [] #create new list to return
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence
        print ")))))))))))))))))))))))"
        print freqSet - conseq, '-->', conseq, 'conf:', conf
        if conf >= minConf: 
            print freqSet-conseq,'-->',conseq,'conf:',conf
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    print "rulesFromConseq..."
    print "H[0]=",H[0]
    m = len(H[0])
    print "m=",m
    if (len(freqSet) > (m + 1)): #try further merging
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates
        # print "H=",H
        # print "m+1=",m+1
        # print "Hmp1=",Hmp1
        # print "freqSet=",freqSet
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        print "(((((((((((((((((((((((((((((((((((("
        print "Hmp1=", Hmp1
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
            
def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print itemMeaning[item]
        print "           -------->"
        for item in ruleTup[1]:
            print itemMeaning[item]
        print "confidence: %f" % ruleTup[2]
        print       #print a blank line
        
            
from time import sleep
from votesmart import votesmart
votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
def getActionIds():
    actionIdList = []; billTitleList = []
    fr = open('recent20bills.txt') 
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum) #api call
            for action in billDetail.actions:
                if action.level == 'House' and \
                (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print 'bill: %d has actionId: %d' % (billNum, actionId)
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print "problem getting bill %d" % billNum
        sleep(1)                                      #delay to be polite
    return actionIdList, billTitleList
        
def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
    itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
    for billTitle in billTitleList:#fill up itemMeaning list
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}#list of items in each transaction (politician) 
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print 'getting votes for actionId: %d' % actionId
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName): 
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except: 
            print "problem getting actionId: %d" % actionId
        voteCount += 2
    return transDict, itemMeaning

if __name__ == "__main__":
    dataSet = loadDataSet()
    # print dataSet
    # C1 = createC1(dataSet)
    # print "C1=",C1
    # D = map(set,dataSet)
    # print "D=",D
    # L1,suppData = scanD(D,C1,0.5)
    # print "L1=",L1
    # print "suppData=",suppData
    # L,suppData=apriori(dataSet)
    # print "L=",L
    # print "suppData=",suppData
    # print "aprioriGen(L(0),2)=",aprioriGen(L(0),2)

    # L,suppData=apriori(dataSet,minSupport=0.7)
    # print "L=",L
    # print "suppData=",suppData

    # # 导入数据集
    # myDat = loadDataSet()
    # # 选择频繁项集
    # L, suppData = apriori(myDat, 0.5)
    # print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    # print "L=",L
    # print "suppData=",suppData
    # print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    # rules = generateRules(L, suppData, minConf=0.7)
    # print 'rules:\n', rules

    mushDatSet = [line.split() for line in open('mushroom.dat').readlines()]
    L,suppData = apriori(mushDatSet,minSupport=0.3)
    # print "L=",L
    # print "suppData=",suppData
    for item in L[1]:
        if item.intersection('2'):
            print item

