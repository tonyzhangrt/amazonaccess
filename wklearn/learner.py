from __future__ import absolute_import
from __future__ import division

from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import itertools
import numpy as np
SEED = 1234

# regressor
class MyRegressor(object):
    def __init__(self, params):
        raise NotImplementedError
    def update_params(self, updates):
        raise NotImplementedError
    def fit(self, Xtrain, ytrain):
        raise NotImplementedError
    def predict(self, Xtest, option):
        raise NotImplementedError

# k-nearest neighbor
class MyKnnReg(MyRegressor):
    def __init__(self, params=dict()):
        self._params = params
        self._knn = KNeighborsRegressor(**(self._params))

    def update_params(self, updates):
        self._params.update(updates)
        self._knn = KNeighborsRegressor(**(self._params))

    def fit(self, Xtrain, ytrain):
        self._knn.fit(Xtrain, ytrain)

    def predict(self, Xtest, option = None):
      return self._knn.predict(Xtest)

# extremelyrandomforest
class MyExtraTreeReg(MyRegressor):
    def __init__(self, params=dict()):
        self._params = params
        self._extree = ExtraTreesRegressor(**(self._params))

    def update_params(self, updates):
        self._params.update(updates)
        self._extree = ExtraTreesRegressor(**(self._params))

    def fit(self, Xtrain, ytrain):
        self._extree.fit(Xtrain, ytrain)

    def predict(self, Xtest, option = None):
      return self._extree.predict(Xtest)

    def plt_feature_importance(self, fname_list, f_range = list()):
        importances = self._extree.feature_importances_

        std = np.std([tree.feature_importances_ for tree in self._extree.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        fname_array = np.array(fname_list)

        if not f_range:
            f_range = range(indices.shape[0])

        n_f = len(f_range)

        plt.figure()
        plt.title("Extra Tree Feature importances")
        plt.barh(range(n_f), importances[indices[f_range]],
               color="b", xerr=std[indices[f_range]], ecolor='k',align="center")
        plt.yticks(range(n_f), fname_array[indices[f_range]])
        plt.ylim([-1, n_f])
        plt.show()


    def list_feature_importance(self, fname_list, f_range = list(), return_list = False):
        importances = self._extree.feature_importances_
        indices = np.argsort(importances)[::-1]

        print 'Extra tree feature ranking:'

        if not f_range :
            f_range = range(indices.shape[0])

        n_f = len(f_range)

        for i in range(n_f):
            f = f_range[i]
            print '{0:d}. feature[{1:d}]  {2:s}  ({3:f})'.format(f + 1, indices[f], fname_list[indices[f]], importances[indices[f]])

        if return_list:
            return [indices[f_range[i]] for i in range(n_f)]

class MyRandomForestReg(MyRegressor):
    def __init__(self, params=dict()):
        self._params = params
        self._rf = RandomForestRegressor(**(self._params))

    def update_params(self, updates):
        self._params.update(updates)
        self._rf = RandomForestRegressor(**(self._params))

    def fit(self, Xtrain, ytrain):
        self._rf.fit(Xtrain, ytrain)

    def predict(self, Xtest, option = None):
      return self._extree.predict(Xtest)

    def plt_feature_importance(self, fname_list, f_range = list()):
        importances = self._rf.feature_importances_

        std = np.std([tree.feature_importances_ for tree in self._rf.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        fname_array = np.array(fname_list)

        if not f_range:
            f_range = range(indices.shape[0])

        n_f = len(f_range)

        plt.figure()
        plt.title("Random Forest Feature importances")
        plt.barh(range(n_f), importances[indices[f_range]],
               color="b", xerr=std[indices[f_range]], ecolor='k',align="center")
        plt.yticks(range(n_f), fname_array[indices[f_range]])
        plt.ylim([-1, n_f])
        plt.show()


    def list_feature_importance(self, fname_list, f_range = list(), return_list = False):
        importances = self._rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        print 'Random forest feature ranking:'

        if not f_range :
            f_range = range(indices.shape[0])

        n_f = len(f_range)

        for i in range(n_f):
            f = f_range[i]
            print '{0:d}. feature[{1:d}]  {2:s}  ({3:f})'.format(f + 1, indices[f], fname_list[indices[f]], importances[indices[f]])

        if return_list:
            return [indices[f_range[i]] for i in range(n_f)]

# sklearn gradient boost
class MyGradientBoostReg(MyRegressor):
    def __init__(self, params=dict()):
        self._params = params
        self._gb = GradientBoostingRegressor(**(self._params))

    def update_params(self, updates):
        self._params.update(updates)
        self._gb = GradientBoostingRegressor(**(self._params))

    def fit(self, Xtrain, ytrain):
        self._gb.fit(Xtrain, ytrain)

    def predict(self, Xtest, option = None):
      return self._gb.predict(Xtest)

    def plt_feature_importance(self, fname_list, f_range = list()):
        importances = self._gb.feature_importances_

        std = np.std([tree[0].feature_importances_ for tree in self._gb.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        fname_array = np.array(fname_list)

        if not f_range:
            f_range = range(indices.shape[0])

        n_f = len(f_range)

        plt.figure()
        plt.title("Gradient Boost Feature importances")
        plt.barh(range(n_f), importances[indices[f_range]],
               color="b", xerr=std[indices[f_range]], ecolor='k',align="center")
        plt.yticks(range(n_f), fname_array[indices[f_range]])
        plt.ylim([-1, n_f])
        plt.show()    

    def list_feature_importance(self, fname_list, f_range = list(), return_list = False):
        importances = self._gb.feature_importances_
        indices = np.argsort(importances)[::-1]

        print 'Gradient Boost feature ranking:'

        if not f_range :
            f_range = range(indices.shape[0])

        n_f = len(f_range)

        for i in range(n_f):
            f = f_range[i]
            print '{0:d}. feature[{1:d}]  {2:s}  ({3:f})'.format(f + 1, indices[f], fname_list[indices[f]], importances[indices[f]])

        if return_list:
            return [indices[f_range[i]] for i in range(n_f)]

# xgboost
class MyXGBoostReg(MyRegressor):
    def __init__(self, params=dict()):
        self._params = params
        if 'num_round' in params:
            self._num_round = params['num_round']
            del self._params['num_round']
        else:
             self._num_round = None
        if 'verbose_eval' in params:
            self._verbose_eval = params['verbose_eval']
            del self._params['verbose_eval']
        else:
            self._verbose_eval = True

        self._xgb = None

    def update_params(self, updates):
        self._params.update(updates)
        if 'num_round' in updates:
            self._num_round = updates['num_round']
            del self._params['num_round']
        if 'verbose_eval' in params:
            self._verbose_eval = params['verbose_eval']
            del self._params['verbose_eval']

    def fit(self, Xtrain, ytrain):
        dtrain = xgb.DMatrix( Xtrain, label=ytrain)
        if self._num_round:
            self._xgb = xgb.train(self._params, dtrain, self._num_round, verbose_eval = self._verbose_eval)
        else:
            self._xgb = xgb.train(self._params, dtrain, verbose_eval = self._verbose_eval )

    def predict(self, Xtest, option = dict()):
        dtest = xgb.DMatrix(Xtest)
        if 'ntree_limit' not in option:
            return self._xgb.predict(dtest)
        else:
            return self._xgb.predict(dtest, ntree_limit=option['ntree_limit'])

    def plt_feature_importance(self, fname_list, f_range = list()):
        importances = np.array(self._xgb.get_fscore().values())
        features = np.array([ int(re.search(r'f(\d+)', f).group(1))  for f in self._xgb.get_fscore().keys()])

        tmp_indices =np.argsort(importances)[::-1]

        indices = features[tmp_indices]

        fname_array = np.array(fname_list)

        importances = importances[tmp_indices]

        importances = importances / np.sum(importances)

        if not f_range :
            f_range = range(indices.shape[0])

        n_f = len(f_range)

        plt.figure()
        plt.title("Xgboost Feature importances")
        plt.barh(range(n_f), importances[f_range],
               color="b", align="center")
        plt.yticks(range(n_f), fname_array[indices[f_range]])
        plt.ylim([-1, n_f])
        plt.show()

    def list_feature_importance(self, fname_list, f_range = list(), return_list = False):
        importances = np.array(self._xgb.get_fscore().values())
        features = np.array([ int(re.search(r'f(\d+)', f).group(1))  for f in self._xgb.get_fscore().keys()])

        tmp_indices =np.argsort(importances)[::-1]

        indices = features[tmp_indices]

        importances = importances[tmp_indices]

        importances = importances / np.sum(importances)

        if not f_range :
            f_range = range(indices.shape[0])

        n_f = len(f_range)

        print 'Xgboost feature ranking:'

        for i in range(n_f):
            f = f_range[i]
            print '{0:d}. feature[{1:d}]  {2:s}  ({3:f})'.format(f + 1, indices[f], fname_list[indices[f]], importances[f])

        if return_list:
            return [indices[f_range[i]] for i in range(n_f)]

def reg_cv_predict(myregressor, Xtrain, ytrain, nfolds=5, randstate=SEED, score_func=mean_absolute_error):
        y = np.zeros(ytrain.shape, float)

        k_cv_score = np.zeros(nfolds, float)
        k = 0
        kfold = KFold(n = ytrain.shape[0], n_folds = nfolds, shuffle = True, random_state = randstate)
        for train_index, test_index in kfold:
            k_Xtrain, k_Xtest = Xtrain[train_index], Xtrain[test_index]
            k_ytrain, k_ytest = ytrain[train_index], ytrain[test_index]
            myregressor.fit(k_Xtrain, k_ytrain)
            k_ypred = myregressor.predict(k_Xtest)
            y[test_index] = k_ypred
            k_cv_score[k] = score_func(k_ytest, k_ypred)
            k += 1

        return y, k_cv_score


def reg_cv_score(myregressor, Xtrain, ytrain, nfolds=5, randstate=SEED, score_func=mean_absolute_error):
    k_cv_score = np.zeros(nfolds, float)

    k = 0
    kfold = KFold(n = ytrain.shape[0], n_folds=nfolds, shuffle = True, random_state=randstate)
    for train_index, test_index in kfold:
        k_Xtrain, k_Xtest = Xtrain[train_index], Xtrain[test_index]
        k_ytrain, k_ytest = ytrain[train_index], ytrain[test_index]
        myregressor.fit(k_Xtrain, k_ytrain)
        k_ypred = myregressor.predict(k_Xtest)
        k_cv_score[k] = score_func(k_ytest, k_ypred)
        k += 1

    return k_cv_score

# here param_grid just need to contain the parameters required to be updated
def reg_cv_grid_search(myregressor, param_grid, Xtrain, ytrain, nfolds=5, randstate=SEED, score_func=mean_absolute_error, criterion = 'min'):
    # can be adapted to use sklearn CVGridSearch
    # org_params = myclassifier._params
    # org_param_grid = dict{}
    # for key, value in org_params:
    #   org_param_grid[key] = [value]
    # org_param_grid.update(param_grid)
    param_names = param_grid.keys()
    param_pools = param_grid.values()
    num_param = len(param_names)
    param_set_list = []
    mean_score_list = []
    best_param_set = None
    best_mean_score = None
    for param_valuelist in itertools.product(*param_pools):
        param_set = dict()
        for k in xrange(num_param):
            param_set[param_names[k]] = param_valuelist[k]

        param_set_list.append(param_set)

        print param_set
        myregressor.update_params(param_set)
        cur_scores = cv_score(myregressor, Xtrain, ytrain, nfolds, randstate, score_func)  
        cur_mean_score = np.mean(cur_scores)
        print cur_mean_score, np.std(cur_scores)

        mean_score_list.append(cur_mean_score)

        if not best_param_set:
            best_param_set = param_set
            best_mean_score = cur_mean_score
        elif criterion == 'max' and cur_mean_score > best_mean_score:
            best_param_set = param_set
            best_mean_score = cur_mean_score
        elif criterion == 'min' and cur_mean_score < best_mean_score:
            best_param_set = param_set
            best_mean_score = cur_mean_score

    return best_param_set, best_mean_score, param_set_list, mean_score_list


# template class mostly just list methods to be implemented
class MyClassifier(object):
    def __init__(self, params):
        raise NotImplementedError
    def update_params(self, updates):
        raise NotImplementedError
    def fit(self, Xtrain, ytrain):
        raise NotImplementedError
    # def predict(self, Xtest, option):
    #   raise NotImplementedError
    def predict_proba(self, Xtest, option):
        raise NotImplementedError

    # def predict_proba_multi(self, Xtest, option):
    #     raise NotImplementedError

# logistic regression
class MyLogisticReg(MyClassifier):
    def __init__(self, params=dict()):
        self._params = params
        self._lr = LogisticRegression(**(self._params))

    def update_params(self, updates):
        self._params.update(updates)
        self._lr = LogisticRegression(**(self._params))

    def fit(self, Xtrain, ytrain):
        self._lr.fit(Xtrain, ytrain)

    # def predict(self, Xtest, option = None):
    #   return self._lr.predict(Xtest)

    def predict_proba(self, Xtest, option = None):
        return self._lr.predict_proba(Xtest)[:, 1]

    def predict_proba_multi(self, Xtest, option = None):
        return self._lr.predict_proba(Xtest)


# k-nearest neighbor
class MyKnn(MyClassifier):
    def __init__(self, params=dict()):
        self._params = params
        self._knn = KNeighborsClassifier(**(self._params))

    def update_params(self, updates):
        self._params.update(updates)
        self._knn = KNeighborsClassifier(**(self._params))

    def fit(self, Xtrain, ytrain):
        self._knn.fit(Xtrain, ytrain)

    # def predict(self, Xtest, option = None):
    #   return self._knn.predict(Xtest)

    def predict_proba(self, Xtest, option = None):
        return self._knn.predict_proba(Xtest)[:, 1]

    def predict_proba_multi(self, Xtest, option = None):
        return self._knn.predict_proba(Xtest)

# extremelyrandomforest
class MyExtraTree(MyClassifier):
    def __init__(self, params=dict()):
        self._params = params
        self._extree = ExtraTreesClassifier(**(self._params))

    def update_params(self, updates):
        self._params.update(updates)
        self._extree = ExtraTreesClassifier(**(self._params))

    def fit(self, Xtrain, ytrain):
        self._extree.fit(Xtrain, ytrain)

    # def predict(self, Xtest, option = None):
    #   return self._extree.predict(Xtest)

    def predict_proba(self, Xtest, option = None):
        return self._extree.predict_proba(Xtest)[:, 1]

    def predict_proba_multi(self, Xtest, option = None):
        return self._extree.predict_proba(Xtest)

    def plt_feature_importance(self, fname_list, f_range = list()):
        importances = self._extree.feature_importances_

        std = np.std([tree.feature_importances_ for tree in self._extree.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        fname_array = np.array(fname_list)

        if not f_range:
            f_range = range(indices.shape[0])

        n_f = len(f_range)

        plt.figure()
        plt.title("Extra Tree Feature importances")
        plt.barh(range(n_f), importances[indices[f_range]],
               color="b", xerr=std[indices[f_range]], ecolor='k',align="center")
        plt.yticks(range(n_f), fname_array[indices[f_range]])
        plt.ylim([-1, n_f])
        plt.show()


    def list_feature_importance(self, fname_list, f_range = list(), return_list = False):
        importances = self._extree.feature_importances_
        indices = np.argsort(importances)[::-1]

        print 'Extra tree feature ranking:'

        if not f_range :
            f_range = range(indices.shape[0])

        n_f = len(f_range)

        for i in range(n_f):
            f = f_range[i]
            print '{0:d}. feature[{1:d}]  {2:s}  ({3:f})'.format(f + 1, indices[f], fname_list[indices[f]], importances[indices[f]])

        if return_list:
            return [indices[f_range[i]] for i in range(n_f)]

class MyRandomForest(MyClassifier):
    def __init__(self, params=dict()):
        self._params = params
        self._rf = RandomForestClassifier(**(self._params))

    def update_params(self, updates):
        self._params.update(updates)
        self._rf = RandomForestClassifier(**(self._params))

    def fit(self, Xtrain, ytrain):
        self._rf.fit(Xtrain, ytrain)

    # def predict(self, Xtest, option = None):
    #   return self._extree.predict(Xtest)

    def predict_proba(self, Xtest, option = None):
        return self._rf.predict_proba(Xtest)[:, 1]

    def predict_proba_multi(self, Xtest, option = None):
        return self._rf.predict_proba(Xtest)

    def plt_feature_importance(self, fname_list, f_range = list()):
        importances = self._rf.feature_importances_

        std = np.std([tree.feature_importances_ for tree in self._rf.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        fname_array = np.array(fname_list)

        if not f_range:
            f_range = range(indices.shape[0])

        n_f = len(f_range)

        plt.figure()
        plt.title("Random Forest Feature importances")
        plt.barh(range(n_f), importances[indices[f_range]],
               color="b", xerr=std[indices[f_range]], ecolor='k',align="center")
        plt.yticks(range(n_f), fname_array[indices[f_range]])
        plt.ylim([-1, n_f])
        plt.show()


    def list_feature_importance(self, fname_list, f_range = list(), return_list = False):
        importances = self._rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        print 'Random forest feature ranking:'

        if not f_range :
            f_range = range(indices.shape[0])

        n_f = len(f_range)

        for i in range(n_f):
            f = f_range[i]
            print '{0:d}. feature[{1:d}]  {2:s}  ({3:f})'.format(f + 1, indices[f], fname_list[indices[f]], importances[indices[f]])

        if return_list:
            return [indices[f_range[i]] for i in range(n_f)]

# sklearn gradient boost
class MyGradientBoost(MyClassifier):
    def __init__(self, params=dict()):
        self._params = params
        self._gb = GradientBoostingClassifier(**(self._params))

    def update_params(self, updates):
        self._params.update(updates)
        self._gb = GradientBoostingClassifier(**(self._params))

    def fit(self, Xtrain, ytrain):
        self._gb.fit(Xtrain, ytrain)

    # def predict(self, Xtest, option = None):
    #   return self._gb.predict(Xtest)

    def predict_proba(self, Xtest, option = None):
        return self._gb.predict_proba(Xtest)[:, 1]

    def predict_proba_multi(self, Xtest, option = None):
        return self._gb.predict_proba(Xtest)

    def plt_feature_importance(self, fname_list, f_range = list()):
        importances = self._gb.feature_importances_

        std = np.std([tree[0].feature_importances_ for tree in self._gb.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        fname_array = np.array(fname_list)

        if not f_range:
            f_range = range(indices.shape[0])

        n_f = len(f_range)

        plt.figure()
        plt.title("Gradient Boost Feature importances")
        plt.barh(range(n_f), importances[indices[f_range]],
               color="b", xerr=std[indices[f_range]], ecolor='k',align="center")
        plt.yticks(range(n_f), fname_array[indices[f_range]])
        plt.ylim([-1, n_f])
        plt.show()    

    def list_feature_importance(self, fname_list, f_range = list(), return_list = False):
        importances = self._gb.feature_importances_
        indices = np.argsort(importances)[::-1]

        print 'Gradient Boost feature ranking:'

        if not f_range :
            f_range = range(indices.shape[0])

        n_f = len(f_range)

        for i in range(n_f):
            f = f_range[i]
            print '{0:d}. feature[{1:d}]  {2:s}  ({3:f})'.format(f + 1, indices[f], fname_list[indices[f]], importances[indices[f]])

        if return_list:
            return [indices[f_range[i]] for i in range(n_f)]

# xgboost
class MyXGBoost(MyClassifier):
    def __init__(self, params=dict()):
        self._params = params
        if 'num_round' in params:
            self._num_round = params['num_round']
            del self._params['num_round']
        else:
             self._num_round = None
        if 'verbose_eval' in params:
            self._verbose_eval = params['verbose_eval']
            del self._params['verbose_eval']
        else:
            self._verbose_eval = True
        self._xgb = None

    def update_params(self, updates):
        self._params.update(updates)
        if 'num_round' in updates:
            self._num_round = updates['num_round']
            del self._params['num_round']
        if 'verbose_eval' in updates:
            self._verbose_eval = updates['verbose_eval']
            del self._params['verbose_eval']

    def fit(self, Xtrain, ytrain):
        dtrain = xgb.DMatrix( Xtrain, label=ytrain)
        if self._num_round:
            self._xgb = xgb.train(self._params, dtrain, self._num_round, verbose_eval = self._verbose_eval)
        else:
            self._xgb = xgb.train(self._params, dtrain, verbose_eval = self._verbose_eval )

    def predict_proba(self, Xtest, option = dict()):
        dtest = xgb.DMatrix(Xtest)
        if 'ntree_limit' not in option:
            return self._xgb.predict(dtest)
        else:
            return self._xgb.predict(dtest, ntree_limit=option['ntree_limit'])

    def predict_proba_multi(self, Xtest, option = dict()):
        dtest = xgb.DMatrix(Xtest)
        if 'ntree_limit' not in option:
            return self._xgb.predict(dtest).reshape((Xtest.shape[0], self._params['num_class']))
        else:
            return self._xgb.predict(dtest, ntree_limit=option['ntree_limit']).reshape((Xtest.shape[0], self._params['num_class']))

    def plt_feature_importance(self, fname_list, f_range = list()):
        importances = np.array(self._xgb.get_fscore().values())
        features = np.array([ int(re.search(r'f(\d+)', f).group(1))  for f in self._xgb.get_fscore().keys()])

        tmp_indices =np.argsort(importances)[::-1]

        indices = features[tmp_indices]

        fname_array = np.array(fname_list)

        importances = importances[tmp_indices]

        importances = importances / np.sum(importances)

        if not f_range :
            f_range = range(indices.shape[0])

        n_f = len(f_range)

        plt.figure()
        plt.title("Xgboost Feature importances")
        plt.barh(range(n_f), importances[f_range],
               color="b", align="center")
        plt.yticks(range(n_f), fname_array[indices[f_range]])
        plt.ylim([-1, n_f])
        plt.show()

    def list_feature_importance(self, fname_list, f_range = list(), return_list = False):
        importances = np.array(self._xgb.get_fscore().values())
        features = np.array([ int(re.search(r'f(\d+)', f).group(1))  for f in self._xgb.get_fscore().keys()])

        tmp_indices =np.argsort(importances)[::-1]

        indices = features[tmp_indices]

        importances = importances[tmp_indices]

        importances = importances / np.sum(importances)

        if not f_range :
            f_range = range(indices.shape[0])

        n_f = len(f_range)

        print 'Xgboost feature ranking:'

        for i in range(n_f):
            f = f_range[i]
            print '{0:d}. feature[{1:d}]  {2:s}  ({3:f})'.format(f + 1, indices[f], fname_list[indices[f]], importances[f])

        if return_list:
            return [indices[f_range[i]] for i in range(n_f)]

# xgboost
class MyKerasModel(MyClassifier):
    # when initialize we need a compiled keras model, how to fit for cv grid search ? currently update model in params, need to prepare a bunch of models
    def __init__(self, params=dict()):
        self._params = params
        self._kmodel = None
        self._kweights = './keras_default_weight.h5'
        self._seed = 1234
        if 'seed' in params:
            self._seed = params['seed']
            del self._params['seed']
        if 'keras_model' in params:
            self._kmodel = params['keras_model']
            del self._params['keras_model']
        if 'keras_weight' in params:
            self._kweights = params['keras_weight']
            del self._params['keras_weight']
        self._kmodel.save_weights(self._kweights, overwrite = True)

    def update_params(self, updates):
        self._params.update(updates)
        if 'seed' in updates:
            self._seed = updates['seed']
            del self._params['seed']
        if 'keras_model' in updates:
            self._kmodel = updates['keras_model']
            del self._params['keras_model']

    def fit(self, Xtrain, ytrain):
        np.random.seed(self._seed)
        self._kmodel.load_weights(self._kweights)
        self._kmodel.fit(Xtrain, ytrain, **self._params)

    def predict_proba(self, Xtest, option = dict()):
        return np.squeeze(self._kmodel.predict_proba(Xtest))


    # def predict_proba_multi(self, Xtest, option = dict()):
    #     dtest = xgb.DMatrix(Xtest)
    #     if 'ntree_limit' not in option:
    #         return self._xgb.predict(dtest).reshape((Xtest.shape[0], self._params['num_class']))
    #     else:
    #         return self._xgb.predict(dtest, ntree_limit=option['ntree_limit']).reshape((Xtest.shape[0], self._params['num_class']))

# cv_score related functions

def strat_cv_predict_proba(myclassifier, Xtrain, ytrain, nfolds=5, randstate=SEED, score_func=roc_auc_score):
        y = np.zeros(ytrain.shape, float)

        sk_cv_score = np.zeros(nfolds, float)
        k = 0
        skfold = StratifiedKFold(ytrain, n_folds = nfolds, shuffle = True, random_state = randstate)
        for train_index, test_index in skfold:
            sk_Xtrain, sk_Xtest = Xtrain[train_index], Xtrain[test_index]
            sk_ytrain, sk_ytest = ytrain[train_index], ytrain[test_index]
            myclassifier.fit(sk_Xtrain, sk_ytrain)
            sk_ypred = myclassifier.predict_proba(sk_Xtest)
            y[test_index] = sk_ypred
            sk_cv_score[k] = score_func(sk_ytest, sk_ypred)
            k += 1

        return y, sk_cv_score

def cv_predict_proba(myclassifier, Xtrain, ytrain, nfolds=5, randstate=SEED, score_func=roc_auc_score):
        y = np.zeros(ytrain.shape, float)

        k_cv_score = np.zeros(nfolds, float)
        k = 0
        kfold = KFold(n = ytrain.shape[0], n_folds = nfolds, shuffle = True, random_state = randstate)
        for train_index, test_index in kfold:
            k_Xtrain, k_Xtest = Xtrain[train_index], Xtrain[test_index]
            k_ytrain, k_ytest = ytrain[train_index], ytrain[test_index]
            myclassifier.fit(k_Xtrain, k_ytrain)
            k_ypred = myclassifier.predict_proba(k_Xtest)
            y[test_index] = k_ypred
            k_cv_score[k] = score_func(k_ytest, k_ypred)
            k += 1

        return y, k_cv_score


def cv_score(myclassifier, Xtrain, ytrain, nfolds=5, randstate=SEED, score_func=roc_auc_score):
    k_cv_score = np.zeros(nfolds, float)

    k = 0
    kfold = KFold(n = ytrain.shape[0], n_folds=nfolds, shuffle = True, random_state=randstate)
    for train_index, test_index in kfold:
        k_Xtrain, k_Xtest = Xtrain[train_index], Xtrain[test_index]
        k_ytrain, k_ytest = ytrain[train_index], ytrain[test_index]
        myclassifier.fit(k_Xtrain, k_ytrain)
        k_ypred = myclassifier.predict_proba(k_Xtest)
        k_cv_score[k] = score_func(k_ytest, k_ypred)
        k += 1

    return k_cv_score

def strat_cv_score(myclassifier, Xtrain, ytrain, nfolds=5, randstate=SEED, score_func=roc_auc_score):
        sk_cv_score = np.zeros(nfolds, float)

        k = 0
        skfold = StratifiedKFold(ytrain, n_folds=nfolds, shuffle = True, random_state=randstate)
        for train_index, test_index in skfold:
            sk_Xtrain, sk_Xtest = Xtrain[train_index], Xtrain[test_index]
            sk_ytrain, sk_ytest = ytrain[train_index], ytrain[test_index]
            myclassifier.fit(sk_Xtrain, sk_ytrain)
            sk_ypred = myclassifier.predict_proba(sk_Xtest)
            sk_cv_score[k] = score_func(sk_ytest, sk_ypred)
            k += 1

        return sk_cv_score

# here param_grid just need to contain the parameters required to be updated
def cv_grid_search(myclassifier, param_grid, Xtrain, ytrain, nfolds=5, randstate=SEED, score_func=roc_auc_score, criterion = 'max'):
    # can be adapted to use sklearn CVGridSearch
    # org_params = myclassifier._params
    # org_param_grid = dict{}
    # for key, value in org_params:
    #   org_param_grid[key] = [value]
    # org_param_grid.update(param_grid)
    param_names = param_grid.keys()
    param_pools = param_grid.values()
    num_param = len(param_names)
    param_set_list = []
    mean_score_list = []
    best_param_set = None
    best_mean_score = None
    for param_valuelist in itertools.product(*param_pools):
        param_set = dict()
        for k in xrange(num_param):
            param_set[param_names[k]] = param_valuelist[k]

        param_set_list.append(param_set)

        print param_set
        myclassifier.update_params(param_set)
        cur_scores = cv_score(myclassifier, Xtrain, ytrain, nfolds, randstate, score_func)  
        cur_mean_score = np.mean(cur_scores)
        print cur_mean_score, np.std(cur_scores)

        mean_score_list.append(cur_mean_score)

        if not best_param_set:
            best_param_set = param_set
            best_mean_score = cur_mean_score
        elif criterion == 'max' and cur_mean_score > best_mean_score:
            best_param_set = param_set
            best_mean_score = cur_mean_score
        elif criterion == 'min' and cur_mean_score < best_mean_score:
            best_param_set = param_set
            best_mean_score = cur_mean_score

    return best_param_set, best_mean_score, param_set_list, mean_score_list

def strat_cv_grid_search(myclassifier, param_grid, Xtrain, ytrain, nfolds=5, randstate=SEED, score_func=roc_auc_score, criterion = 'max'):
    # can be adapted to use sklearn CVGridSearch
    # org_params = myclassifier._params
    # org_param_grid = dict{}
    # for key, value in org_params:
    #   org_param_grid[key] = [value]
    # org_param_grid.update(param_grid)
    param_names = param_grid.keys()
    param_pools = param_grid.values()
    num_param = len(param_names)
    param_set_list = []
    mean_score_list = []
    best_param_set = None
    best_mean_score = None
    for param_valuelist in itertools.product(*param_pools):
        param_set = dict()
        for k in xrange(num_param):
            param_set[param_names[k]] = param_valuelist[k]

        param_set_list.append(param_set)

        print param_set
        myclassifier.update_params(param_set)
        cur_scores = strat_cv_score(myclassifier, Xtrain, ytrain, nfolds, randstate, score_func)  
        cur_mean_score = np.mean(cur_scores)
        print cur_mean_score, np.std(cur_scores)

        mean_score_list.append(cur_mean_score)

        if not best_param_set:
            best_param_set = param_set
            best_mean_score = cur_mean_score
        elif criterion == 'max' and cur_mean_score > best_mean_score:
            best_param_set = param_set
            best_mean_score = cur_mean_score
        elif criterion == 'min' and cur_mean_score < best_mean_score:
            best_param_set = param_set
            best_mean_score = cur_mean_score

    return best_param_set, best_mean_score, param_set_list, mean_score_list

#### multi class

def strat_cv_predict_proba_multi(myclassifier, Xtrain, ytrain, nfolds=5, randstate=SEED, score_func=log_loss):
        num_class = np.unique(ytrain).shape[0]

        y = np.zeros((ytrain.shape[0], num_class) , float)

        sk_cv_score = np.zeros(nfolds, float)
        k = 0
        skfold = StratifiedKFold(ytrain, n_folds = nfolds, shuffle = True, random_state = randstate)
        for train_index, test_index in skfold:
            sk_Xtrain, sk_Xtest = Xtrain[train_index], Xtrain[test_index]
            sk_ytrain, sk_ytest = ytrain[train_index], ytrain[test_index]
            myclassifier.fit(sk_Xtrain, sk_ytrain)
            sk_ypred = myclassifier.predict_proba_multi(sk_Xtest)
            y[test_index, :] = sk_ypred
            sk_cv_score[k] = score_func(sk_ytest, sk_ypred)
            k += 1

        return y, sk_cv_score

def cv_predict_proba_multi(myclassifier, Xtrain, ytrain, nfolds=5, randstate=SEED, score_func=roc_auc_score):
        num_class = np.unique(ytrain).shape[0]

        y = np.zeros((ytrain.shape[0], num_class), float)

        k_cv_score = np.zeros(nfolds, float)
        k = 0
        kfold = KFold(n = ytrain.shape[0], n_folds = nfolds, shuffle = True, random_state = randstate)
        for train_index, test_index in kfold:
            k_Xtrain, k_Xtest = Xtrain[train_index], Xtrain[test_index]
            k_ytrain, k_ytest = ytrain[train_index], ytrain[test_index]
            myclassifier.fit(k_Xtrain, k_ytrain)
            k_ypred = myclassifier.predict_proba_multi(k_Xtest)
            y[test_index, :] = k_ypred
            k_cv_score[k] = score_func(k_ytest, k_ypred)
            k += 1

        return y, k_cv_score

def cv_score_multi(myclassifier, Xtrain, ytrain, nfolds=5, randstate=SEED, score_func=log_loss):
    k_cv_score = np.zeros(nfolds, float)

    k = 0
    kfold = KFold(n = ytrain.shape[0], n_folds=nfolds, shuffle = True, random_state=randstate)
    for train_index, test_index in kfold:
        k_Xtrain, k_Xtest = Xtrain[train_index], Xtrain[test_index]
        k_ytrain, k_ytest = ytrain[train_index], ytrain[test_index]
        myclassifier.fit(k_Xtrain, k_ytrain)
        k_ypred = myclassifier.predict_proba_multi(k_Xtest)
        k_cv_score[k] = score_func(k_ytest, k_ypred)
        k += 1

    return k_cv_score

def strat_cv_score_multi(myclassifier, Xtrain, ytrain, nfolds=5, randstate=SEED, score_func=log_loss):
        sk_cv_score = np.zeros(nfolds, float)

        k = 0
        skfold = StratifiedKFold(ytrain, n_folds=nfolds, shuffle = True, random_state=randstate)
        for train_index, test_index in skfold:
            sk_Xtrain, sk_Xtest = Xtrain[train_index], Xtrain[test_index]
            sk_ytrain, sk_ytest = ytrain[train_index], ytrain[test_index]
            myclassifier.fit(sk_Xtrain, sk_ytrain)
            sk_ypred = myclassifier.predict_proba_multi(sk_Xtest)
            sk_cv_score[k] = score_func(sk_ytest, sk_ypred)
            k += 1

        return sk_cv_score

# here param_grid just need to contain the parameters required to be updated
def cv_grid_search_multi(myclassifier, param_grid, Xtrain, ytrain, nfolds=5, randstate=SEED, score_func=log_loss, criterion = 'min'):
    # can be adapted to use sklearn CVGridSearch
    # org_params = myclassifier._params
    # org_param_grid = dict{}
    # for key, value in org_params:
    #   org_param_grid[key] = [value]
    # org_param_grid.update(param_grid)
    param_names = param_grid.keys()
    param_pools = param_grid.values()
    num_param = len(param_names)
    param_set_list = []
    mean_score_list = []
    best_param_set = None
    best_mean_score = None
    for param_valuelist in itertools.product(*param_pools):
        param_set = dict()
        for k in xrange(num_param):
            param_set[param_names[k]] = param_valuelist[k]

        param_set_list.append(param_set)

        print param_set
        myclassifier.update_params(param_set)
        cur_scores = cv_score_multi(myclassifier, Xtrain, ytrain, nfolds, randstate, score_func)  
        cur_mean_score = np.mean(cur_scores)
        print cur_mean_score, np.std(cur_scores)

        mean_score_list.append(cur_mean_score)

        if not best_param_set:
            best_param_set = param_set
            best_mean_score = cur_mean_score
        elif criterion == 'max' and cur_mean_score > best_mean_score:
            best_param_set = param_set
            best_mean_score = cur_mean_score
        elif criterion == 'min' and cur_mean_score < best_mean_score:
            best_param_set = param_set
            best_mean_score = cur_mean_score

    return best_param_set, best_mean_score, param_set_list, mean_score_list

def strat_cv_grid_search_multi(myclassifier, param_grid, Xtrain, ytrain, nfolds=5, randstate=SEED, score_func=log_loss, criterion = 'min'):
    # can be adapted to use sklearn CVGridSearch
    # org_params = myclassifier._params
    # org_param_grid = dict{}
    # for key, value in org_params:
    #   org_param_grid[key] = [value]
    # org_param_grid.update(param_grid)
    param_names = param_grid.keys()
    param_pools = param_grid.values()
    num_param = len(param_names)
    param_set_list = []
    mean_score_list = []
    best_param_set = None
    best_mean_score = None
    for param_valuelist in itertools.product(*param_pools):
        param_set = dict()
        for k in xrange(num_param):
            param_set[param_names[k]] = param_valuelist[k]

        param_set_list.append(param_set)

        print param_set
        myclassifier.update_params(param_set)
        cur_scores = strat_cv_score_multi(myclassifier, Xtrain, ytrain, nfolds, randstate, score_func)  
        cur_mean_score = np.mean(cur_scores)
        print cur_mean_score, np.std(cur_scores)

        mean_score_list.append(cur_mean_score)

        if not best_param_set:
            best_param_set = param_set
            best_mean_score = cur_mean_score
        elif criterion == 'max' and cur_mean_score > best_mean_score:
            best_param_set = param_set
            best_mean_score = cur_mean_score
        elif criterion == 'min' and cur_mean_score < best_mean_score:
            best_param_set = param_set
            best_mean_score = cur_mean_score

    return best_param_set, best_mean_score, param_set_list, mean_score_list