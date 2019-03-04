# -*- coding: utf-8 -*-


from ..tree.regression_tree import RegressionTree
from random import choices

class GradientBoostingBase(object):
    def __init__(self):
        self.trees = None
        self.lr    = None
        self.init_val = None
        self.fn   = lambda x:NotImplemented

    def _get_init_val(self,y):
        return NotImplemented

    def _match_node(self,row,tree):
        nd = tree.root
        while nd.left and nd.right:
            if row[nd.feature] < nd.split:
                nd = nd.left
            else:
                nd = nd.right
        return  nd

    def _get_leaves(self,tree):
        nodes = []
        que = [tree.root]
        while que:
            node = que.pop(0)
            if node.left is None or node.right is None:
                nodes.append(node)
                continue
            left_node = node.left
            right_node = node.right
            que.append(left_node)
            que.append(right_node)
        return nodes

    def _divide_regions(self,tree,nodes,X):
        regions = {node:[] for node in nodes}
        for i,row in enumerate(X):
            node = self._match_node(row,tree)
            regions[node].append(i)
        return regions

    def _get_score(self,idxs,y_hat,residuals):

        return NotImplemented

    def _update_score(self,tree,X,y_hat,residuals):
        nodes =self._get_leaves(tree)

        regions =self._divide_regions(tree,nodes,X)
        for node,idxs in regions.items():
            node.score = self._get_score(idxs,y_hat,residuals)
        tree._get_rules()

    def _get_residuals(self,y,y_hat):
        return [yi- self.fn(y_hat_i) for yi,y_hat_i in zip(y,y_hat)]

    def fit(self,X,y,n_estimators,lr,max_depth,min_samples_split,subsample=None):
        self.init_val = self._get_init_val(y)
        n = len(y)
        y_hat = [self.init_val]*n
        residuals = self._get_residuals(y,y_hat)
        self.trees=[]
        self.lr= lr
        for _ in range(n_estimators):
            idx = range(n)
            if subsample is not None:
                k = int(subsample *n)
                idx = choices(population=idx,k=k)
            X_sub = [X[i] for i in idx]
            residuals_sub = [residuals[i] for i in idx]
            y_hat_sub = [y_hat[i] for i in idx]
            tree = RegressionTree()
            tree.fit(X_sub,residuals_sub,max_depth,min_samples_split)

            self._update_score(tree,X_sub,y_hat_sub,residuals_sub)
            y_hat = [y_hat_i + lr*res_hat_i for y_hat_i, res_hat_i in zip(y_hat,tree.predict(X))]
            residuals = self._get_residuals(y,y_hat)
            self.trees.append(tree)

    def _predict(self,Xi):
        ret = self.init_val + sum(self.lr*tree._predict(Xi) for tree in self.trees)
        return self.fn(ret)

    def predict(self,X):
        return NotImplemented





