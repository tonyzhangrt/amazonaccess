from __future__ import division

import pandas as pd
import numpy as np
from pandas.core.series import Series
import os
from sklearn import preprocessing
import scipy
import re

SEED = 1234

# show statistics of dataframe columns
def show_basic_statics(df, col_list = list()):
    print 'Showing Statistics\n'

    if not col_list :
        col_list = list(df.columns.values)
    for colname in col_list:
        col = df[colname].values
        num_unique = np.unique(col).shape[0]
        min_elem = np.amin(col)
        max_elem = np.amax(col)
        mean = np.mean(col)
        std = np.std(col)
        mode = scipy.stats.mstats.mode(col)
        print 'Col:{0:s}, dtype:{1:s}'.format(colname, col.dtype)
        print 'unique:', num_unique, 'min:', min_elem, 'max:', max_elem, 'mean:', mean, 'std:', std, 'mode:', mode
        print

def remove_constant_col(df, col_list = list()):
    if not col_list:
        col_list = list(df.columns.values)

    drop_list = []
    for colname in col_list:
        col = df[colname].values
        if np.std(col) == 0.0:
            drop_list.append(colname)

    df = df.drop(drop_list, axis =1)

    return df

def relabel_integer_col(df, col_list = list()):
    if not col_list:
        col_list = list(df.columns.values)

    for col_name in col_list:
        col_data = df[col_name].values
        if issubclass(col_data.dtype.type, np.integer):
                le = preprocessing.LabelEncoder()
                le.fit(col_data)
                col_data = le.transform(col_data)
                df[col_name] = Series(col_data, index = df.index)

def relabel_float_col(df, col_list = list(), rthresh = 0.1, atrhesh = 30):
    if not col_list:
        col_list = list(df.columns.values)

    for col_name in col_list:
        col_data = df[col_name].values
        if issubclass(col_data.dtype.type, np.float):
            nnz = np.nonzero(col_data)[0].shape[0]
            n_unique = np.unique(col_data).shape[0]
            rate = float(n_unique)/nnz
            if rate < rthresh or n_unique < atrhesh:
                le = preprocessing.LabelEncoder()
                le.fit(col_data)
                col_data = le.transform(col_data)
                df[col_name] = Series(col_data, index = df.index)


def remove_identical_columns(df, col_list = list()):
    if not col_list:
        col_list = list(df.columns.values)

    n_col = len(col_list)

    df_data = df[col_list].values

    drop_list = []
    for i in xrange(n_col):
        col_i_name = col_list[i]
        col_i_data = df_data[:, i]
        for j in xrange(i+1, n_col):
            col_j_name = col_list[j]
            col_j_data = df_data[:, j]    
            if np.array_equal(col_i_data, col_j_data) and (col_j_name not in drop_list):
                drop_list.append(col_j_name)

    df = df.drop(drop_list, axis = 1)

    return df


# check the combination of different columns
def check_two_columns(df, col1, col2):
    print 'Checking {0:s} and {1:s}'.format(col1, col2)

    # c1_c2_list = df[[col1, col2]].values.tolist()
    # c1_c2_tuple = [tuple(c1_c2) for c1_c2 in c1_c2_list]
    c1_c2_tuple = zip(df[col1].values.astype(float), df[col2].values.astype(float))

    num_unique_c1_c2 = len(set(c1_c2_tuple))
    num_unique_c1 = np.unique(df[col1].values).shape[0]
    num_unique_c2 = np.unique(df[col2].values).shape[0]

    print '{0:s}:'.format(col1), num_unique_c1, '{0:s}:'.format(col2), num_unique_c2, 'comb:', num_unique_c1_c2,'\n'

    return float(num_unique_c1_c2)/ (num_unique_c1 * num_unique_c2)

def merge_two_cat_columns(df, col1, col2, col_new=None, remove='none', hasher=None):
    if not col_new:
        col_new=col1+'_COMB_'+col2
    print 'Combining {0:s} and {1:s} into {2:s}'.format(col1, col2, col_new)

    if col_new in list(df.columns.values):
        print 'Overwriting exisiting {0:s}'.format(col_new)
        # df.drop([col_new])

    c1_data = df[col1].values
    c2_data = df[col2].values
    c1_c2_tuple = zip(c1_data, c2_data)
    c1_c2_set = set(c1_c2_tuple)

    c1_c2_tuple_dict = dict()
    i = 0
    for c1_c2 in c1_c2_set:
        c1_c2_tuple_dict[c1_c2] = i
        i+=1

    col_new_data = np.zeros(df[col1].shape, np.int)
    for i in xrange(col_new_data.shape[0]):
        col_new_data[i] = c1_c2_tuple_dict[c1_c2_tuple[i]]

    if hasher:
        col_new_data = hasher(col_new_data)

    if remove == 'both':
        df = df.drop[col1, col2]
    elif remove == 'col1':
        df = df.drop[col1]
    elif remove == 'col2':
        df = df.drop[col2]

    df[col_new] = Series(col_new_data, index = df.index)

# put all rare event into one group
def combine_rare(df, col_list = list(), new_name_list = list(), rare_line=1):
    if not col_list :
        col_list = list(df.columns.values)
    if not new_name_list :
        new_name_list = ['CR_'+col for col in col_list]

    for col, new_name in zip(col_list, new_name_list):
        col_data = df[col].values
        if issubclass(col_data.dtype.type, np.integer):
            le = preprocessing.LabelEncoder()
            le.fit(col_data)
            col_data = le.transform(col_data)
            max_label = np.amax(col_data)
            counts = np.bincount(col_data)
            rare_cats = np.argwhere(counts <= rare_line)
            rare_cats = rare_cats.reshape(rare_cats.shape[0])
            rare_positions = [np.argwhere(col_data == rare_cat)[0,0] for rare_cat in rare_cats]
            col_data[rare_positions] = max_label+1
            df[new_name] = Series(col_data, index = df.index)
        else:
            print 'col:{0:s} not integer'.format(col)


import cPickle as pickle

# generating countable

class MyCountTable(object):
    def __init__(self):
        self.one_count_table = dict()
        self.two_count_table = dict()
        self._file_path = None

    def save_count_tables(self, file_path = None):
        if file_path:
            self._file_path = file_path
        elif self._file_path:
            file_path = self._file_path
        else:
            print 'Saving count table file failed: path not specified'
            return

        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_path, 'wb') as f:
            pickle.dump([self.one_count_table, self.two_count_table], f, pickle.HIGHEST_PROTOCOL)
        

    def load_count_tables(self, file_path = None):
        if file_path:
            self._file_path = file_path
        elif self._file_path:
            file_path = self._file_path
        else:
            print 'Loading count table file failed: path not specified'
            return

        try:
            with open( file_path, 'rb') as f:
                self.one_count_table, self.two_count_table = pickle.load(f)
        except IOError:
            print 'Loading count table file failed: file not found.'
        

    # count table operation related with data frame
    def df_generate_count_tables(self, df, task = 'both', col_list=list(), file_path = None):
        if not col_list:
            col_list = list(df.columns.values)

        n_col = len(col_list)

        if task == 'both' or task == 'one':
            for i in xrange(n_col):
                col_i = col_list[i]
                col_i_data = df[col_i].values
                col_i_set = np.unique(col_i_data)

                if col_i in self.one_count_table:
                    continue
                self.one_count_table[col_i] = dict()
                for i_elem in col_i_set:
                    sub_i_ind = np.argwhere(col_i_data == i_elem)
                    self.one_count_table[col_i][i_elem] = sub_i_ind.shape[0]

        if task == 'both' or task == 'two':
            for i in xrange(n_col):
                col_i = col_list[i]
                col_i_data = df[col_i].values
                col_i_set = np.unique(col_i_data)

                for j in xrange(n_col):
                    if j == i:
                        continue
                    col_j = col_list[j]
                    col_j_data = df[col_j].values

                    tuple_col_ij = (col_i, col_j)
                    if tuple_col_ij in self.two_count_table:
                        continue
                    self.two_count_table[tuple_col_ij] = dict()
                    for i_elem in col_i_set:
                        sub_i_ind = np.argwhere(col_i_data == i_elem)
                        sub_i_ind = sub_i_ind.reshape(sub_i_ind.shape[0])
                        sub_j_data = col_j_data[sub_i_ind]
                        sub_j_set = np.unique(sub_j_data)

                        self.two_count_table[tuple_col_ij][i_elem] = {'unique': len(sub_j_set)}
                        for j_elem in sub_j_set:
                            sub_j_ind = np.argwhere(sub_j_data == j_elem)
                            self.two_count_table[tuple_col_ij][i_elem][j_elem] = sub_j_ind.shape[0]

        self.save_count_tables(file_path)

    # only called when directed from df_two_degree_counts or df_one_degree_counts
    # helper function to load and generate count table when necessary
    def _df_get_count_tables(self, df, task = 'both', col_list = list(), file_path = None):
        if not col_list:
            col_list = list(df.columns.values)

        if not file_path:
            file_path = self._file_path

        # first try local count tables
        flag = 0
        if task == 'both' or task == 'one':
            for col in col_list:
                if col not in self.one_count_table:
                    flag = 1
                    break
        if task == 'both' or task == 'two':
            for col1 in col_list:
                for col2 in col_list:
                    if col1 == col2:
                        continue
                    if (col1, col2) not in self.two_count_table:
                        flag = 1
                        break

        if flag == 0:
            return

        # if not good try dumped count tables
        if file_path:
            self.load_count_tables(file_path)
            flag = 0
            if task == 'both' or task == 'one':
                for col in col_list:
                    if col not in self.one_count_table:
                        flag = 1
                        break
            if task == 'both' or task == 'two':
                for col1 in col_list:
                    for col2 in col_list:
                        if col1 == col2:
                            continue
                        if (col1, col2) not in self.two_count_table:
                            flag = 1
                            break

        # generate countables if necessary
        if flag == 1:
                self.df_generate_count_tables(df, task, col_list, file_path)


    def df_two_degree_counts(self, df, col_i, col_j, operation, file_path = None):
        # if two columns are the same just return one_degree_count
        if col_i == col_j:
            return self.df_one_degree_counts(df, col_i, file_path)

        if operation == 'per':
            task = 'both'
        elif operation == 'num':
            task = 'two'
        else:
            print 'unknown operation'
            return
                
        self._df_get_count_tables(df, task, [col_i, col_j], file_path)

        i_table = one_count_table[col_i]
        ij_table = two_count_table[(col_i, col_j)]

        col_i_data = df[col_i].values
        col_j_data = df[col_j].values
        if operation == 'per':  # 'per': percentage of (elem_i, elem_j) in all (elem_i, col_j)  
            vfunc = np.vectorize(lambda x,y: float(ij_table[x][y])/i_table[x])
            col_new = vfunc(col_i_data, col_j_data)
        elif operation == 'num':    # 'num': number of different kinds of (elem_i, col_j) 
            vfunc = np.vectorize(lambda x: ij_table[x]['unique'])
            col_new = vfunc(col_i_data)

        return col_new

    def df_one_degree_counts(df, col_i, file_path = None):
        self._df_get_count_tables(df, 'one', [col_i], file_path)

        i_table = one_count_table[col_i]

        col_i_data = df[col_i].values
        vfunc = np.vectorize(lambda x: i_table[x])
        col_new = vfunc(col_i_data)

        return col_new

    # fset  version of countable
    def fset_generate_count_tables(self, myfset, task = 'both' ,col_list=list(), file_path = None):
        if not col_list:
            col_list = range(len(myfset.fname_list))

        if not file_path:
            file_path = self._file_path 

        # n_col = len(col_list)

        if task == 'both' or task == 'one':
            for col_i in col_list:
                col_i_name = myfset.fname_list[col_i]
                col_i_ind = myfset.find_list[col_i]
                col_i_data_train = myfset.Xtrain[:, col_i_ind]
                col_i_data_test = myfset.Xtest[:, col_i_ind]
                col_i_data = np.hstack((col_i_data_train, col_i_data_test))
                col_i_set = np.unique(col_i_data)

                if col_i_name in self.one_count_table:
                    continue
                self.one_count_table[col_i_name] = dict()
                for i_elem in col_i_set:
                    sub_i_ind = np.argwhere(col_i_data == i_elem)
                    self.one_count_table[col_i_name][i_elem] = sub_i_ind.shape[0]

        if task == 'both' or task == 'two':
            for col_i in col_list:
                col_i_name = myfset.fname_list[col_i]
                col_i_ind = myfset.find_list[col_i]
                col_i_data_train = myfset.Xtrain[:, col_i_ind]
                col_i_data_test = myfset.Xtest[:, col_i_ind]
                col_i_data = np.hstack((col_i_data_train, col_i_data_test))
                col_i_set = np.unique(col_i_data)

                for col_j in col_list:
                    if col_j == col_i:
                        continue
                    col_j_name = myfset.fname_list[col_j]
                    col_j_ind = myfset.find_list[col_j]
                    col_j_data_train = myfset.Xtrain[:, col_j_ind]
                    col_j_data_test = myfset.Xtest[:, col_j_ind]
                    col_j_data = np.hstack((col_j_data_train, col_j_data_test))

                    tuple_col_ij = (col_i_name, col_j_name)
                    if tuple_col_ij in self.two_count_table:
                        continue
                    self.two_count_table[tuple_col_ij] = dict()
                    for i_elem in col_i_set:
                        sub_i_ind = np.argwhere(col_i_data == i_elem)
                        sub_i_ind = sub_i_ind.reshape(sub_i_ind.shape[0])
                        sub_j_data = col_j_data[sub_i_ind]
                        sub_j_set = np.unique(sub_j_data)

                        self.two_count_table[tuple_col_ij][i_elem] = {'unique': len(sub_j_set)}
                        for j_elem in sub_j_set:
                            sub_j_ind = np.argwhere(sub_j_data == j_elem)
                            self.two_count_table[tuple_col_ij][i_elem][j_elem] = sub_j_ind.shape[0]

        self.save_count_tables(file_path)    

    def _fset_get_count_tables(self, myfset, task = 'both', col_list=list(), file_path = None):
        if not col_list:
            col_list = range(len(myfset.fname_list))

        if not file_path:
            file_path = self._file_path

        # first try local count tables
        flag = 0
        if task == 'both' or task == 'one':
            for col in col_list:
                col_name = myfset.fname_list[col]
                if col_name not in self.one_count_table:
                    flag = 1
                    break

        if task == 'both' or task == 'two':
            for col1 in col_list:
                col1_name = myfset.fname_list[col1]
                for col2 in col_list:
                    if col1 == col2:
                        continue
                    col2_name = myfset.fname_list[col2]
                    if (col1_name, col2_name) not in self.two_count_table:
                        flag = 1
                        break

        if flag == 0:
            return

        # if not good try dumped count tables
        if file_path:
            self.load_count_tables(file_path)
            flag = 0
            if task == 'both' or task == 'one':
                for col in col_list:
                    col_name = myfset.fname_list[col]
                    if col_name not in self.one_count_table:
                        flag = 1
                        break

            if task == 'both' or task == 'two':
                for col1 in col_list:
                    col1_name = myfset.fname_list[col1]
                    for col2 in col_list:
                        if col1 == col2:
                            continue
                        col2_name = myfset.fname_list[col2]
                        if (col1_name, col2_name) not in self.two_count_table:
                            flag = 1
                            break

        # generate countables if necessary
        if flag == 1:
                self.fset_generate_count_tables(myfset, task, col_list, file_path)

    def fset_two_degree_counts(self, myfset, col_i, col_j, operation, file_path = None):
        # if two columns are the same just return one_degree_count
        if col_i == col_j:
            return self.fset_one_degree_counts(myfset, col_i, file_path)

        if operation == 'per':
            task = 'both'
        if operation == 'num':
            task = 'two'

        self._fset_get_count_tables(myfset, task, [col_i, col_j], file_path)

        col_i_name = myfset.fname_list[col_i]
        col_j_name = myfset.fname_list[col_j]
        col_i_ind = myfset.find_list[col_i]
        col_j_ind = myfset.find_list[col_j]


        i_table = self.one_count_table[col_i_name]
        ij_table = self.two_count_table[(col_i_name, col_j_name)]

        col_i_data_train = myfset.Xtrain[:, col_i_ind]
        col_i_data_test = myfset.Xtest[:, col_i_ind]
        col_j_data_train = myfset.Xtrain[:, col_j_ind]
        col_j_data_test = myfset.Xtest[:, col_j_ind]
        if operation == 'per':  # 'per': percentage of (elem_i, elem_j) in all (elem_i, col_j)  
            vfunc = np.vectorize(lambda x,y: float(ij_table[x][y])/i_table[x])
            col_new_train = vfunc(col_i_data_train, col_j_data_train)
            col_new_test = vfunc(col_i_data_test, col_j_data_test)
        elif operation == 'num':    # 'num': number of different kinds of (elem_i, col_j) 
            vfunc = np.vectorize(lambda x: ij_table[x]['unique'])
            col_new_train = vfunc(col_i_data_train)
            col_new_test = vfunc(col_i_data_test)

        return col_new_train, col_new_test

    def fset_one_degree_counts(self, myfset, col_i, file_path = None):
        self._fset_get_count_tables(myfset, 'one', [col_i], file_path)

        col_i_name = myfset.fname_list[col_i]
        col_i_ind = myfset.find_list[col_i]

        i_table = self.one_count_table[col_i_name]

        col_i_data_train = myfset.Xtrain[:, col_i_ind]
        col_i_data_test = myfset.Xtest[:, col_i_ind]
        vfunc = np.vectorize(lambda x: i_table[x])
        col_new_train = vfunc(col_i_data_train)
        col_new_test = vfunc(col_i_data_test)

        return col_new_train, col_new_test

# data preprocessing and generate different set of features

# perform in place combination of rare events
def np_combine_rare(Xtrain, Xtest, col_list = list(), rare_line=1):
    if Xtrain.shape[1] != Xtest.shape[1]:
        print 'Xtrain, Xtest shape not match.'
        return

    if not col_list :
        col_list = range(Xtrain.shape[1])
        check_int = True
    else:
        check_int = False

    n_train = Xtrain.shape[0]
    for col in col_list:
        col_data_train = Xtrain[:, col]
        col_data_test = Xtest[:, col]
        col_data = np.hstack((col_data_train, col_data_test))
        # print col_data[0]
        if issubclass(col_data.dtype.type, np.integer) or (not check_int):
            le = preprocessing.LabelEncoder()
            le.fit(col_data)
            col_data = le.transform(col_data)
            max_label = np.amax(col_data)
            counts = np.bincount(col_data)
            rare_cats = np.argwhere(counts <= rare_line)
            rare_cats = rare_cats.reshape(rare_cats.shape[0])
            rare_positions = [np.argwhere(col_data == rare_cat)[0,0] for rare_cat in rare_cats]
            # print len(rare_positions)
            col_data[rare_positions] = max_label+1
            Xtrain[:, col] = col_data[:n_train]
            Xtest[:, col] = col_data[n_train:]
        else:
            print 'col:{0:d} not integer'.format(col)

# perform in place numerical transform on selected col, here X is numpy matrix, currently do not support sparse matrix
def np_numeric_transform(Xtrain, Xtest, col_list = list(), operation='log', standardize=False):
    if Xtrain.shape[1] != Xtest.shape[1]:
        print 'Xtrain, Xtest shape not match.'
        return

    if not col_list:
        col_list = range(Xtrain.shape[1])

    if operation == 'log':
        vfunc = np.vectorize(lambda x: np.log(x))
    elif operation == 'log1p':
        vfunc = np.vectorize(lambda x: np.log1p(x))
    elif operation == 'exp':
        vfunc = np.vectorize(lambda x: np.exp(x))
    elif operation == 'expm1':
        vfunc = np.vectorize(lambda x: np.expm1(x))
    elif operation == 'square':
        vfunc = np.vectorize(lambda x: x**2)
    elif operation == 'none':
        vfunc = None
    else:
        vfunc = None
        print 'Unkown operation not performed'

    for col in col_list:
        if vfunc:
            Xtrain[:, col] = vfunc(Xtrain[:, col])
            Xtest[:, col] = vfunc(Xtest[:, col])
        if standardize:
            col_data_train = Xtrain[:, col]
            col_data_test = Xtest[:, col]
            col_data = np.hstack((col_data_train, col_data_test))
            col_mean = np.mean(col_data)
            col_std = np.std(col_data)
            # print col_mean, col_std
            Xtrain[:, col] = 1./col_std * (Xtrain[:, col] - col_mean)
            Xtest[:, col] = 1./col_std * (Xtest[:, col] - col_mean)




# we consider model as a combination of feature and classifier
# classifier need to implement several methods
# many parts of the implementation only suitable for binary classification
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
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

    def plt_feature_importance(self, fname_list, f_range = list()):
        importances = self._extree.feature_importances_

        std = np.std([tree.feature_importances_ for tree in self._extree.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        if not f_range:
            f_range = range(indices.shape[0])

        n_f = len(f_range)

        plt.figure()
        plt.title("Extra Tree Feature importances")
        plt.barh(range(n_f), importances[indices[f_range]],
               color="b", xerr=std[indices[f_range]], ecolor='k',align="center")
        plt.yticks(range(n_f), fname_list[indices[f_range]])
        plt.ylim([-1, n_f])
        plt.show()


    def list_feature_importance(self, fname_list, f_range = list()):
        importances = self._extree.feature_importances_
        indices = np.argsort(importances)[::-1]

        print 'Extra tree feature ranking:'

        if not f_range :
            f_range = range(indices.shape[0])

        n_f = len(f_range)

        for i in range(n_f):
            f = f_range[i]
            print '{0:d}. feature[{1:d}]  {2:s}  ({3:f})'.format(f + 1, indices[f], fname_list[indices[f]], importances[indices[f]])

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

    def plt_feature_importance(self, fname_list, f_range = list()):
        importances = self._rf.feature_importances_

        std = np.std([tree.feature_importances_ for tree in self._rf.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        if not f_range:
            f_range = range(indices.shape[0])

        n_f = len(f_range)

        plt.figure()
        plt.title("Random Forest Feature importances")
        plt.barh(range(n_f), importances[indices[f_range]],
               color="b", xerr=std[indices[f_range]], ecolor='k',align="center")
        plt.yticks(range(n_f), fname_list[indices[f_range]])
        plt.ylim([-1, n_f])
        plt.show()


    def list_feature_importance(self, fname_list, f_range = list()):
        importances = self._rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        print 'Random forest feature ranking:'

        if not f_range :
            f_range = range(indices.shape[0])

        n_f = len(f_range)

        for i in range(n_f):
            f = f_range[i]
            print '{0:d}. feature[{1:d}]  {2:s}  ({3:f})'.format(f + 1, indices[f], fname_list[indices[f]], importances[indices[f]])

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
    #   return self._extree.predict(Xtest)

    def predict_proba(self, Xtest, option = None):
        return self._gb.predict_proba(Xtest)[:, 1]

    def plt_feature_importance(self, fname_list, f_range = list()):
        importances = self._gb.feature_importances_

        std = np.std([tree[0].feature_importances_ for tree in self._gb.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        if not f_range:
            f_range = range(indices.shape[0])

        n_f = len(f_range)

        plt.figure()
        plt.title("Gradient Boost Feature importances")
        plt.barh(range(n_f), importances[indices[f_range]],
               color="b", xerr=std[indices[f_range]], ecolor='k',align="center")
        plt.yticks(range(n_f), fname_list[indices[f_range]])
        plt.ylim([-1, n_f])
        plt.show()    

    def list_feature_importance(self, fname_list, f_range = list()):
        importances = self._gb.feature_importances_
        indices = np.argsort(importances)[::-1]

        print 'Gradient Boost feature ranking:'

        if not f_range :
            f_range = range(indices.shape[0])

        n_f = len(f_range)

        for i in range(n_f):
            f = f_range[i]
            print '{0:d}. feature[{1:d}]  {2:s}  ({3:f})'.format(f + 1, indices[f], fname_list[indices[f]], importances[indices[f]])

# xgboost
class MyXGBoost(MyClassifier):
    def __init__(self, params=dict()):
        self._params = params
        if 'num_round' in params:
            self._num_round = params['num_round']
            del self._params['num_round']
        else:
             self._num_round = None
        self._xgb = None

    def update_params(self, updates):
        self._params.update(updates)
        if 'num_round' in updates:
            self._num_round = updates['num_round']
            del self._params['num_round']

    def fit(self, Xtrain, ytrain):
        dtrain = xgb.DMatrix( Xtrain, label=ytrain)
        if self._num_round:
            self._xgb = xgb.train(self._params, dtrain, self._num_round)
        else:
            self._xgb = xgb.train(self._params, dtrain)

    def predict_proba(self, Xtest, option = dict()):
        dtest= xgb.DMatrix(Xtest)
        if 'ntree_limit' not in option:
            return self._xgb.predict(dtest)
        else:
            return self._xgb.predict(dtest, ntree_limit=option['ntree_limit'])

    def plt_feature_importance(self, fname_list, f_range = list()):
        importances = np.array(self._xgb.get_fscore().values())
        features = np.array([ int(re.search(r'f(\d+)', f).group(1))  for f in self._xgb.get_fscore().keys()])

        tmp_indices =np.argsort(importances)[::-1]

        indices = features[tmp_indices]

        importances = importances[tmp_indices]

        if not f_range :
            f_range = range(indices.shape[0])

        n_f = len(f_range)

        plt.figure()
        plt.title("Xgboost Feature importances")
        plt.barh(range(n_f), importances[f_range],
               color="b", align="center")
        plt.yticks(range(n_f), fname_list[indices[f_range]])
        plt.ylim([-1, n_f])
        plt.show()

    def list_feature_importance(self, fname_list, f_range = list()):
        importances = np.array(self._xgb.get_fscore().values())
        features = np.array([ int(re.search(r'f(\d+)', f).group(1))  for f in self._xgb.get_fscore().keys()])

        tmp_indices =np.argsort(importances)[::-1]

        indices = features[tmp_indices]

        importances = importances[tmp_indices]

        if not f_range :
            f_range = range(indices.shape[0])

        n_f = len(f_range)

        print 'Xgboost feature ranking:'

        for i in range(n_f):
            f = f_range[i]
            print '{0:d}. feature[{1:d}]  {2:s}  ({3:f})'.format(f + 1, indices[f], fname_list[indices[f]], importances[f])

# cv_score related functions

def strat_cv_predict_proba(myclassifier, Xtrain, ytrain, nfolds=5, randstate=SEED, score_func=roc_auc_score):
        y = np.zeros(ytrain.shape, float)

        sk_cv_score = np.zeros(nfolds, float)
        k = 0
        skfold = StratifiedKFold(ytrain, n_folds=nfolds, random_state=randstate)
        for train_index, test_index in skfold:
            sk_Xtrain, sk_Xtest = Xtrain[train_index], Xtrain[test_index]
            sk_ytrain, sk_ytest = ytrain[train_index], ytrain[test_index]
            myclassifier.fit(sk_Xtrain, sk_ytrain)
            sk_ypred = myclassifier.predict_proba(sk_Xtest)
            y[test_index] = sk_ypred
            sk_cv_score[k] = score_func(sk_ytest, sk_ypred)
            k += 1

        return y, sk_cv_score

def cv_score(myclassifier, Xtrain, ytrain, nfolds=5, randstate=SEED, score_func=roc_auc_score):
    k_cv_score = np.zeros(nfolds, float)

    k = 0
    kfold = KFold(n = ytrain.shape[0], n_folds=nfolds, random_state=randstate)
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
        skfold = StratifiedKFold(ytrain, n_folds=nfolds, random_state=randstate)
        for train_index, test_index in skfold:
            sk_Xtrain, sk_Xtest = Xtrain[train_index], Xtrain[test_index]
            sk_ytrain, sk_ytest = ytrain[train_index], ytrain[test_index]
            myclassifier.fit(sk_Xtrain, sk_ytrain)
            sk_ypred = myclassifier.predict_proba(sk_Xtest)
            sk_cv_score[k] = score_func(sk_ytest, sk_ypred)
            k += 1

        return sk_cv_score

import itertools
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

# define model as a combination of feature and classifier
# class MyMdoel(object):

# class MyFeature(object):
#   self.feature_name = None
#   self.feature_dmatrix = None

#   def __init__(self, name = ):


class MyFeatureSet(object):
    # self.set_name = None
    def __init__(self):
        self.fname_list = list()
        self.find_list = list()
        self.Xtrain = None
        self.Xtest = None
        self._file_path = None


    def generate_feature_set(self, file_path):
        raise NotImplementedError

    def fetch_feature_set(self, file_path = None):
        if (not self.Xtrain) or (not self.Xtrain):
            if self._file_path:
                self.load_feature_set(self._file_path)
            elif file_path:
                self.load_feature_set(file_path) 
            else:
                # print 'Feature set not available.'
                self.generate_feature_set(file_path)

        return self.Xtrain, self.Xtest

    def load_feature_set(self, file_path = None):
        if not file_path:
            file_path = self._file_path
        if file_path:
            try:
                with open( file_path, 'rb') as f:
                    self.fname_list, self.find_list, self.Xtrain, self.Xtest = pickle.load(f)
            except IOError:
                print 'Loading feature set file failed: file not found.'
                self.generate_feature_set(file_path)
        else:
            print 'Loading featue set file failed: file not saved.'

    def save_feature_set(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_path, 'wb') as f:
            pickle.dump([self.fname_list, self.find_list, self.Xtrain, self.Xtest], f, pickle.HIGHEST_PROTOCOL)
        self._file_path = file_path


# concatenate feature set in feature set list, sparsify if one of the feature set is sparse
def concat_feature_set(myfset_list, sparsify = False):  
    if not sparsify:
        for myfset in myfset_list:
            if scipy.sparse.issparse(myfset.Xtrain):
                sparsify = True
                break

    if sparsify:
        hstack_func = scipy.sparse.hstack
    else:
        hstack_func = np.hstack

    newfname_list = None
    newfind_list = None
    newXtrain = None
    newXtest = None

    n_fset = len(myfset_list)
    for i in xrange(n_fset):
        myfset = myfset_list[i]

        if i == 0:
            newfname_list = myfset.fname_list
            
            newfind_list = myfset.find_list
            
            newXtrain = myfset.Xtrain
            
            newXtest = myfset.Xtest

        else:
            newfname_list = newfname_list + myfset.fname_list

            prev_total = newfind_list[-1]
            curfind_list = [prev_total + find for find in myfset.find_list[1:]]
            newfind_list = newfind_list + curfind_list

            newXtrain = hstack_func((newXtrain, myfset.Xtrain))

            newXtest = hstack_func((newXtest, myfset.Xtest))
 

    if sparsify:
        newXtrain = newXtrain.tocsr()
        newXtest = newXtest.tocsr()

    return newfname_list, newfind_list, newXtrain, newXtest

# selecting extra feature 
def random_select_feature(myclassifier, bfset, myfset, ytrain, nfolds=5, randstate=SEED, score_func=roc_auc_score):
    n_feature = len(myfset.fname_list)

    ftrain_list = [myfset.Xtrain[:, myfset.find_list[i]:myfset.find_list[i+1]] for i in xrange(n_feature)]

    if scipy.sparse.issparse(myfset.Xtrain):
        hstack_func = scipy.sparse.hstack
    else:
        hstack_func = np.hstack

    np.random.seed(randstate)
    permute = list(np.random.permutation(n_feature))

    flag = 0
    prev_score = 0.
    goodf_list = list()
    good_Xtrain = bfset.Xtrain
    while len(goodf_list) < n_feature and flag == 0:
        print 'current score :', prev_score
        if prev_score == 0.:
            test_cv_scores = cv_score(myclassifier, good_Xtrain, ytrain, nfolds, randstate, score_func)
            cur_score = np.mean(test_cv_scores)
        else:
            for curf in permute:
                cur_Xtrain = hstack_func((good_Xtrain, ftrain_list[curf]))
                test_cv_scores = cv_score(myclassifier, cur_Xtrain, ytrain, nfolds, randstate, score_func)
                cur_score = np.mean(test_cv_scores)
                if cur_score > prev_score:
                    good_Xtrain = cur_Xtrain
                    goodf_list.append(curf)
                    permute.remove(curf)
                    print '{0:s} selected'.format(myfset.fname_list[curf])
                    break
        if cur_score > prev_score:
            prev_score = cur_score
        else:
            flag = 1

    goodf_list.sort()
    n_goodf = len(goodf_list)
    newfname_list = bfset.fname_list
    newfind_list = bfset.find_list
    newXtrain = bfset.Xtrain
    newXtest = bfset.Xtest

    for i in xrange(n_goodf):
        goodf = goodf_list[i]
        prev_total = newfind_list[-1]
        ind_low = myfset.find_list[goodf]
        ind_up = myfset.find_list[goodf+1]
        newfname_list.append(myfset.fname_list[goodf])
        newfind_list.append(prev_total + ind_up - ind_low)
        newXtrain = hstack_func((newXtrain, myfset.Xtrain[:, ind_low:ind_up]))
        newXtest = hstack_func((newXtest, myfset.Xtest[:, ind_low:ind_up]))

    return newfname_list, newfind_list, newXtrain, newXtest 

# merge two feature in feature set, now only support catergorical feature
def fset_check_two_columns(myfset, col1, col2):
    c1_ind = myfset.find_list[col1]
    c2_ind = myfset.find_list[col2]

    c1_name = myfset.fname_list[col1]
    c2_name = myfset.fname_list[col2]

    print 'Checking {0:s} and {1:s}'.format(c1_name, c2_name)

    c1_data_train = myfset.Xtrain[:, c1_ind]
    c1_data_test = myfset.Xtest[:, c1_ind]
    c2_data_train = myfset.Xtrain[:, c2_ind]
    c2_data_test = myfset.Xtest[:, c2_ind]

    c1_data = np.hstack((c1_data_train, c1_data_test))
    c2_data = np.hstack((c2_data_train, c2_data_test))

    c1_c2_tuple = zip(c1_data, c2_data)

    num_unique_c1_c2 = len(set(c1_c2_tuple))
    num_unique_c1 = np.unique(c1_data).shape[0]
    num_unique_c2 = np.unique(c2_data).shape[0]

    print '{0:s}:'.format(c1_name), num_unique_c1, '{0:s}:'.format(c2_name), num_unique_c2, 'comb:', num_unique_c1_c2


def fset_merge_two_cat_columns(myfset, col_tuple, hasher=None):
    col1 = col_tuple[0]
    col2 = col_tuple[1]

    c1_ind = myfset.find_list[col1]
    c2_ind = myfset.find_list[col2]

    c1_data_train = myfset.Xtrain[:, c1_ind]
    c1_data_test = myfset.Xtest[:, c1_ind]
    c2_data_train = myfset.Xtrain[:, c2_ind]
    c2_data_test = myfset.Xtest[:, c2_ind]

    c1_data = np.hstack((c1_data_train, c1_data_test))
    c2_data = np.hstack((c2_data_train, c2_data_test))
    c1_c2_tuple = zip(c1_data, c2_data)
    c1_c2_set = set(c1_c2_tuple)

    c1_c2_tuple_dict = dict()
    i = 0
    for c1_c2 in c1_c2_set:
        c1_c2_tuple_dict[c1_c2] = i
        i+=1

    col_new_data = np.zeros(c1_data.shape[0], np.int)
    for i in xrange(col_new_data.shape[0]):
        col_new_data[i] = c1_c2_tuple_dict[c1_c2_tuple[i]]

    if hasher:
        col_new_data = hasher(col_new_data)

    n_train = myfset.Xtrain.shape[0]
    col_new_data_train = col_new_data[:n_train]
    col_new_data_test = col_new_data[n_train:]

    return col_new_data_train, col_new_data_test

def fset_merge_three_cat_columns(myfset, col_triple, hasher=None):
    col1 = col_triple[0]
    col2 = col_triple[1]
    col3 = col_triple[2]

    c12_data_train, c12_data_test = fset_merge_two_cat_columns(myfset, (col1, col2), hasher)
    c12_data = np.hstack((c12_data_train, c12_data_test))

    c3_ind = myfset.find_list[col3]

    c3_data_train = myfset.Xtrain[:, c3_ind]
    c3_data_test = myfset.Xtest[:, c3_ind]

    c3_data = np.hstack((c3_data_train, c3_data_test))

    c12_c3_tuple = zip(c12_data, c3_data)
    c12_c3_set = set(c12_c3_tuple)

    c12_c3_tuple_dict = dict()
    i = 0
    for c12_c3 in c12_c3_set:
        c12_c3_tuple_dict[c12_c3] = i
        i+=1

    col_new_data = np.zeros(c12_data.shape[0], np.int)
    for i in xrange(col_new_data.shape[0]):
        col_new_data[i] = c12_c3_tuple_dict[c12_c3_tuple[i]]

    if hasher:
        col_new_data = hasher(col_new_data)

    n_train = myfset.Xtrain.shape[0]
    col_new_data_train = col_new_data[:n_train]
    col_new_data_test = col_new_data[n_train:]

    return col_new_data_train, col_new_data_test

def fset_merge_multiple_cat_columns(myfset, col_multiple, hasher=None):
    n_col = len(col_multiple)
    if n_col <= 1:
        return

    col_list = list(col_multiple)
    col_ind = list(np.array(myfset.find_list)[col_list])

    X_merge = np.vstack((myfset.Xtrain[:,col_ind], myfset.Xtest[:, col_ind]))

    cm_data = X_merge[:, 0]

    for col_t in xrange(1, n_col):
        ct_data = X_merge[:, col_t]

        cm_ct_tuple = zip(cm_data, ct_data)
        cm_ct_set = set(cm_ct_tuple)

        cm_ct_tuple_dict = dict()
        i = 0
        for cm_ct in cm_ct_set:
            cm_ct_tuple_dict[cm_ct] = i
            i+=1

        cm_data = np.zeros(cm_data.shape[0], np.int)
        for i in xrange(cm_data.shape[0]):
            cm_data[i] = cm_ct_tuple_dict[cm_ct_tuple[i]]

    if hasher:
        cm_data = hasher(cm_data)

    n_train = myfset.Xtrain.shape[0]
    col_new_data_train = cm_data[:n_train]
    col_new_data_test = cm_data[n_train:]

    return col_new_data_train, col_new_data_test


# def fset_relabel_integer_col(df, col_list = list()):
#     if not col_list:
#         col_list = list(df.columns.values)

#     for col_name in col_list:
#         col_data = df[col_name].values
#         if issubclass(col_data.dtype.type, np.integer):
#                 le = preprocessing.LabelEncoder()
#                 le.fit(col_data)
#                 col_data = le.transform(col_data)
#                 df[col_name] = Series(col_data, index = df.index)

# def fset_relabel_float_col(df, col_list = list(), rthresh = 0.1, atrhesh = 30):
#     if not col_list:
#         col_list = list(df.columns.values)

#     for col_name in col_list:
#         col_data = df[col_name].values
#         if issubclass(col_data.dtype.type, np.float):
#             nnz = np.nonzero(col_data)[0].shape[0]
#             n_unique = np.unique(col_data).shape[0]
#             rate = float(n_unique)/nnz
#             if rate < rthresh or n_unique < atrhesh:
#                 le = preprocessing.LabelEncoder()
#                 le.fit(col_data)
#                 col_data = le.transform(col_data)
#                 df[col_name] = Series(col_data, index = df.index)


def main():
    pass

if __name__ == "__main__":
    main()
#   def remove(myfeature)

# from sklearn import ensemble
# from sklearn import linear_model



# dfAmTrain = pd.read_csv('./train.csv')
# dfAmTest = pd.read_csv('./test.csv')