from __future__ import absolute_import
from __future__ import division

import pandas as pd
import numpy as np
from pandas.core.series import Series
from sklearn import preprocessing

SEED = 1234

# show statistics of dataframe columns
def show_basic_statics(df, col_list = list()):
    print 'Showing Statistics\n'

    if not col_list :
        col_list = list(df.columns.values)
    for colname in col_list:
        col = df[colname].values
        min_elem = np.amin(col)
        max_elem = np.amax(col)
        mean = np.mean(col)
        std = np.std(col)
        values, counts = np.unique(col, return_counts=True)
        ind = np.argmax(counts)
        num_unique = values.shape[0]
        mode = values[ind]
        count = counts[ind]
        print 'Col:{0:s}, dtype:{1:s}'.format(colname, col.dtype)
        print 'unique:', num_unique, 'min:', min_elem, 'max:', max_elem, 'mean:', mean, 'std:', std, '(mode, count):({0:f}, {1:d})'.format(mode, count)
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

def relabel_float_col(df, col_list = list(), rthresh = 0.1, athresh = 30):
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