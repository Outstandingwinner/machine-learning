# -*- coding: utf-8 -*-
import os
import sys
import numpy as np 
import operator
from numpy import *
from common_libs import *

weights = mat([[ 4.12414349],[ 0.48007329],[-0.6168482 ]])
testdata = mat([-2.46015, -6.866805])

m,n = shape(testdata)
testmat = zeros((m,n+1))
testmat[:,0] =1
testmat[:,1:] = testdata

print classifier(testmat,weights)
print regression_calc(testmat,weights)
