# -*- coding: utf-8 -*-

from math import log2
from copy import copy
from ..utils.utils import list_split

class Node(object):
    def __init__(self,prob=None):
        self.prob = prob

        self.left = None
        self.right = None
        self.feature = None
        self.split = None

class DecisionTree(object):
    def __init__(self):
        self.root = Node()
        self.depth = 1
        self._rules = None

    def _get_split_effect(self,X,y,idx,feature,split):
        n = len(idx)
        pos_cnt = [0,0]
        cnt = [0,0]

        for i in idx:
            xi,yi = X[i][feature],y[i]
            if xi < split:
                cnt[0] +=1
                pos_cnt[0] += yi
            else:
                cnt[1] +=1
                pos_cnt[1] += yi
        prob = [pos_cnt[0]/cnt[0],pos_cnt[1]/cnt[1]]
        rate = [cnt[0]/n,cnt[1]/n]

        return prob,rate

    def _get_entropy(self,p):
        if p ==1 or p == 0:
            return 0
        else:
            q = 1-p
            return -(p*log2(p) + q*log2(q))

    def _get_info(self,y,idx):
        p = sum(y[i] for i in idx)/len(idx)
        return self._get_entropy(p)

    def _get_cond_info(self,prob,rate):
        info_left =self._get_entropy(prob[0])
        info_right = self._get_entropy(prob[1])

        return rate[0]*info_left + rate[1]*info_right

    def _choose_split(self,X,y,idxs,feature):
        unique = set([X[i][feature] for i in idxs])
        if len(unique) == 1:
            return None
        unique.remove(min(unique))

        def f(split):
            info = self._get_info(y,idxs)
            prob,rate = self._get_split_effect(X,y,idxs,feature,split)
            cond_info = self._get_cond_info(prob,rate)
            gain = info - cond_info
            return gain,split,prob
        gain,split,prob = max((f(split) for split in unique),key= lambda x:x[0])
        return gain,feature,split,prob

    def _choose_feature(self,X,y,idxs):
        m = len(X[0])
        split_rets = map(lambda j:self._choose_split(X,y,idxs,j),range(m))
        split_rets = filter(lambda x:x is not None,split_rets)
        return max(split_rets,default=None,key=lambda x:x[0])

    def _expr2literal(self,expr):
        feature,op,split= expr
        op = ">=" if op == 1 else "<"
        return "Feature%d %s %.4f" %(feature,op,split)

    def _get_rules(self):
        que = [[self.root,[]]]
        self._rules = []
        while que:
            nd,exprs = que.pop(0)
        if not(nd.left or nd.right):
            literals = list(map(self._expr2literal,exprs))
            self._rules.append([literals,nd.prob])
        if nd.left:
            rule_left = copy(exprs)
            rule_left.append([nd.feature,-1,nd.split])
            que.append([nd.left,rule_left])
        if nd.right:
            rule_right = copy(exprs)
            rule_right.append([nd.feature,1,nd.split])
            que.append([nd.right,rule_right])

    def fit(self,X,y,max_depth=4,min_samples_split=2):
        idxs = list(range(len(y)))
        que = [(self.depth+1,self.root,idxs)]
        while que:
            depth ,nd , idxs = que.pop(0)
            if depth > max_depth:
                depth -= 1
                break
            if len(idxs) < min_samples_split or nd.prob ==1 or nd.prob == 0:
                continue
            split_ret = self._choose_feature(X,y,idxs)
            if split_ret is None:
                continue
            _,feature,split,prob=split_ret
            nd.feature = feature
            nd.split = split
            nd.left  = Node(prob[0])
            nd.right = None(prob[1])
            idxs_split = list_split(X,idxs,feature,split)
            que.append((depth+1,nd.left,idxs_split[0]))
            que.append((depth + 1, nd.right, idxs_split[1]))
        self.depth = depth
        self._get_rules()

    @property
    def rules(self):
        for i,rule in enumerate(self._rules):
            literals,prob = rule
            print("Rule %d: "%i, ' | '.join(literals) + ' => y_hat %.4f'%prob)
        print()


    def _predict(self,Xi):
        nd = self.root
        while nd.left and nd.right:
            if Xi[nd.feature] < nd.split:
                nd = nd.left
            else:
                nd = nd.right
        return nd.prob

    def predict(self,X,threshhold=0.5):
        return [int(self._predict(Xi) >=threshhold) for Xi in X]

















