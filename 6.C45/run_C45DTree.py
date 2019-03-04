# -*- coding: utf-8 -*-

from numpy import *
from math import log
from C45DTree import *
import treePlotter2

dtree = C45DTree()
dtree.loadDataSet("dataset.dat",["age","revenue","student","credit"])
dtree.train()
print dtree.tree

# predict
labels = ["age","revenue","student","credit"]
vector = ['0','1','0','0'] # ['0','1','0','0','no']
print "真实输出 ","no","->","决策树输出",dtree.predict(dtree.tree,labels,vector)

treePlotter2.createPlot(dtree.tree)
