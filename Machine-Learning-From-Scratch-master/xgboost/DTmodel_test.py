# -*- coding: utf-8 -*-

from __future__ import division,print_function
import numpy as np

from utils import divide_on_feature, train_test_split, standardize, mean_squared_error
from utils import calculate_entropy, accuracy_score, calculate_variance


class DecisionNode():
    def __init__(self,feature_i=None,threshold=None,
                 value=None,true_branch=None,false_branch=None):
        self.feature_i = feature_i
        self.threshold = threshold
        self.value     = value
        self.true_branch = true_branch
        self.false_branch = false_branch

class DecisionTree(object):
    def __init__(self,min_samples_split=2,min_impurity=1e-7,
                 max_depth=float("inf"),loss=None):
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_impurity      = min_impurity
        self.max_depth = max_depth
        self._impurity_calculation = None
        self.one_dim = None
        self.loss = loss

    def fit(self,X,y,loss=None):
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(X,y)
        self.loss = None

    def _build_tree(self,X,y,current_depth = 0):
        largest_impurity = 0
        best_criteria = None
        best_sets = None

        if len(np.shape(y)) == 1:
            y = np.expand_dims(y,axis=1)

        Xy = np.concatenate((X,y),axis=1)

        n_samples,n_features= np.shape(X)

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:,feature_i],axis=1)
                unique_values  = np.unique(feature_values)
                for threshold in unique_values:
                    Xy1,Xy2 = divide_on_feature(Xy,feature_i,threshold)
                    if len(Xy1) > 0 and len(Xy2) > 0:
                        y1 = Xy1[:,n_features:]
                        y2 = Xy2[:,n_features:]

                        impurity = self._impurity_calculation(y,y1,y2)

                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i":feature_i,"threshold":threshold}
                            best_sets = {
                                "leftX":Xy1[:,:n_features],
                                "lefty":Xy1[:,n_features:],
                                "rightX":Xy2[:,:n_features],
                                "righty":Xy2[:,n_features:]
                            }

        if largest_impurity > self.min_impurity:
            true_branch = self._build_tree(best_sets["leftX"],best_sets["lefty"],current_depth+1)
            false_branch = self._build_tree(best_sets["rightX"],best_sets["righty"],current_depth+1)
            return DecisionNode(feature_i=best_criteria["feature_i"],threshold=best_criteria["threshold"],
                                true_branch=true_branch,false_branch=false_branch)
        leaf_value = self._leaf_value_calculation(y)
        return DecisionNode(value=leaf_value)


    def predict_value(self,x,tree=None):
        if tree is None:
            tree = self.root

        if tree.value is not None:
            return tree.value

        feature_value = x[tree.feature_i]

        branch = tree.false_branch

        if isinstance(feature_value,int) or isinstance(feature_value,float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
            elif feature_value == tree.threshold:
                branch = tree.true_branch

        return self.predict_value(x,branch)

    def predict(self,X):
        y_pred = []
        for x in X:
            y_pred.append(self.predict_value(x))
        return y_pred

    def print_tree(self,tree=None,indent=" "):
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)
        else:
            print("%s:%s" %(tree.feature_i,tree.threshold))
            print("%sT->" %(indent),end="")
            self.print_tree(tree.true_branch,indent+indent)
            print("%sF->" % (indent), end="")
            self.print_tree(tree.false_branch, indent + indent)

class ClassificationTree(DecisionTree):
    def _calculate_information_gain(self,y,y1,y2):
        p = len(y1)/len(y)
        entropy = calculate_entropy(y)
        info_gain = entropy - p*calculate_entropy(y1) - (1-p)*calculate_entropy(y2)
        return info_gain

    def _majority_vote(self,y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            count = len(y[y==label])
            if count > max_count:
                most_common = label
                max_count   = count
        return most_common

    def fit(self,X,y):
        self._impurity_calculation = self._calculate_information_gain
        self._leaf_value_calculation = self._majority_vote
        super(ClassificationTree,self).fit(X,y)


class RegressionTree(DecisionTree):
    def _calculate_variance_reduction(self,y,y1,y2):
        var_tot = calculate_variance(y)
        var_1 = calculate_variance(y1)
        var_2 = calculate_variance(y2)
        frac_1 = len(y1)/len(y)
        frac_2 = len(y2)/len(y)

        variance_reduction = var_tot -(frac_1*var_1 + frac_2*var_2)

        return sum(variance_reduction)

    def _mean_of_y(self,y):
        value = np.mean(y,axis=0)
        return value if len(value) >1 else value[0]

    def fit(self,X,y):
        self._impurity_calculation = self._calculate_variance_reduction
        self._leaf_value_calculation = self._mean_of_y
        super(RegressionTree,self).fit(X,y)

class XGBoostRegressionTree(DecisionTree):
    def _split(self,y):
        col = int(np.shape(y)[1]/2)
        y,y_pred = y[:,:col],y[:,col:]
        return y,y_pred

    def _gain(self,y,y_pred):
        nominator = np.power((self.loss.gradient(y,y_pred)).sum(),2)
        denominator = self.loss.hess(y,y_pred).sum()
        return 0.5*(nominator/denominator)

    def _gain_by_taylor(self,y,y1,y2):
        y,y_pred = self._split(y)
        y1,y1_pred = self._split(y1)
        y2,y2_pred = self._split(y2)

        true_gain = self._gain(y1,y1_pred)
        false_gain = self._gain(y2,y2_pred)
        gain = self._gain(y,y_pred)
        return true_gain + false_gain - gain

    def _approximate_update(self,y):
        y,y_pred = self._split(y)
        gradient = np.sum(self.loss.gradient(y,y_pred),axis=0)
        hessian  = np.sum(self.loss.hess(y,y_pred),axis=0)
        update_approximation = gradient/hessian
        return update_approximation

    def fit(self,X,y):
        self._impurity_calculation =self._gain_by_taylor
        self._leaf_value_calculation = self._approximate_update
        super(XGBoostRegressionTree,self).fit(X,y)






















