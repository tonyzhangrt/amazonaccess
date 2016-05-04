from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
import scipy
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
import cPickle as pickle
from collections import deque
from .learner import strat_cv_score

SEED = 1234
SHUFFLE_BOUND = 2147483647  

class MyFeatureSet(object):
    # self.set_name = None
    def __init__(self):
        self.fname_list = list()
        self.ftype_list = list()
        self.find_list = list()
        self.Xtrain = None
        self.Xtest = None
        self._file_path = None

    # support using fname or index to fetch item
    def __getitem__(self, input_f_list):
        if isinstance(input_f_list, slice):
            start = input_f_list.start
            stop = input_f_list.stop
            step = input_f_list.step
            input_f_list = range(start, stop, step)

        if not hasattr(input_f_list, '__iter__'):
            input_f_list = [input_f_list]

        n_feature = len(self.fname_list)
        if isinstance(input_f_list[0], str):
            f_list = [self.fname_list.index(f) for f in input_f_list if f in self.fname_list ]
        elif isinstance(input_f_list[0], int):
            f_list = [ind for ind in input_f_list if ind in range(n_feature)]
        else:
            print 'Unsupported indexing'
            return None

        if scipy.sparse.issparse(self.Xtrain):
            sparsify = True
        else:
            sparsify = False

        if sparsify:
            hstack_func = scipy.sparse.hstack
        else:
            hstack_func = np.hstack

        n_item = len(f_list)

        new_fset = MyFeatureSet()
        new_fset.fname_list = [self.fname_list[f] for f in f_list]
        new_fset.ftype_list = [self.ftype_list[f] for f in f_list]

        new_fset.find_list.append(0)
        for i in xrange(n_item):
            f = f_list[i]
            find_last = new_fset.find_list[-1]
            find_low = self.find_list[f] 
            find_up = self.find_list[f+1]
            new_fset.find_list.append(find_last + find_up - find_low)
            if i == 0 :
                new_fset.Xtrain = self.Xtrain[:, find_low:find_up ]
                new_fset.Xtest = self.Xtest[:, find_low:find_up ]
            else:
                new_fset.Xtrain = hstack_func((new_fset.Xtrain, self.Xtrain[:, find_low:find_up ]))
                new_fset.Xtest = hstack_func((new_fset.Xtest, self.Xtest[:, find_low:find_up ]))

        if sparsify:
            new_fset.Xtrain.tocsr()
            new_fset.Xtest.tocsr()

        return new_fset

    # only support item by item setting, can use index or feature name to access
    def __setitem__(self, input_f, X_tuple):
        if len(X_tuple) != 2:
            print 'X input not equal two'
            return

        f_train, f_test = X_tuple
        if len(f_train.shape) == 1:
            f_train = f_train.reshape((f_train.shape[0], 1))
        if len(f_test.shape) == 1:
            f_test = f_test.reshape((f_test.shape[0], 1))

        if f_train.shape[1] != f_test.shape[1]:
            print 'train and test size not matching'
            return

        # if this is an empty feature set, if input fname is not str convert to str
        if not self.fname_list:
            self.Xtrain = f_train
            self.Xtest = f_test
            self.fname_list.append(str(input_f))
            self.find_list.append(0)
            self.find_list.append(f_train.shape[1])
            self.ftype_list.append(f_train.dtype.type)
            return

        # if not empty need to check if it is compatible with the stored data
        if f_train.shape[0] != self.Xtrain.shape[0]:
            print 'train size not mathcing'
            return
        if f_test.shape[0] != self.Xtest.shape[0]:
            print 'test size not matching'
            return

        n_feature = len(self.fname_list)

        if isinstance(input_f, str):
            if input_f in self.fname_list:
                f_id = self.fname_list.index(input_f)
                f_name = input_f
            else:
                f_id = -1
                f_name = input_f
        elif isinstance(input_f, int):
            if input_f in range(n_feature):
                f_id = input_f
                f_name = self.fname_list[f_id]
            elif (input_f + n_feature) in range(n_feature):
                f_id = input_f + n_feature
                f_name = self.fname_list[f_id]
            else:
                print 'index out of range'
                return
        else:
            print 'unsupported indexing'
            return

        if f_id in range(n_feature):
            find_low = self.find_list[f_id]
            find_up = self.find_list[f_id + 1]
            self.Xtrain[:, find_low:find_up ] = f_train
            self.Xtest[:, find_low:find_up ] = f_test
        elif f_id == -1:
            if scipy.sparse.issparse(self.Xtrain) or scipy.sparse.issparse(f_train):
                sparsify = True
            else:
                sparsify = False

            if sparsify:
                hstack_func = scipy.sparse.hstack
            else:
                hstack_func = np.hstack

            self.Xtrain = hstack_func((self.Xtrain, f_train))
            self.Xtest = hstack_func((self.Xtest, f_test))

            if sparsify:
                self.Xtrain.tocsr()
                self.Xtest.tocsr()

            find_last = self.find_list[-1]

            self.fname_list.append(f_name)
            self.ftype_list.append(f_train.dtype.type)
            self.find_list.append(find_last + f_train.shape[1])


    def __add__(self, other):
        if not isinstance(other, MyFeatureSet):
            print 'not MyFeatureSet instance'
            return 

        return concat_fsets([self, other])

    # def generate_feature_set(self, file_path):
    #     raise NotImplementedError

    def load_feature_set(self, file_path = None):
        if not file_path:
            file_path = self._file_path
        if file_path:
            try:
                with open( file_path, 'rb') as f:
                    self.fname_list, self.find_list, self.ftype_list, self.Xtrain, self.Xtest = pickle.load(f)
            except IOError:
                print 'Loading feature set file failed: file not found.'
                if hasattr(self, 'generate_feature_set'):
                    self.generate_feature_set(file_path)
        else:
            print 'Loading featue set file failed: file not saved.'

    def save_feature_set(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_path, 'wb') as f:
            pickle.dump([self.fname_list, self.find_list, self.ftype_list, self.Xtrain, self.Xtest], f, pickle.HIGHEST_PROTOCOL)
        self._file_path = file_path

    def feature_names(self):
        return self.fname_list

    def values(self):
        return self.Xtrain, self.Xtest

    def copy(self):
        clone = MyFeatureSet();
        clone.fname_list = self.fname_list[:]
        clone.find_list = self.find_list[:]
        clone.ftype_list = self.ftype_list[:]
        clone.Xtrain = self.Xtrain.copy()
        clone.Xtest = self.Xtest.copy()
        return clone

    def inplace_sparsify(self):
        self.Xtrain = self.Xtrain.tocsr()
        self.Xtest = self.Xtest.tocsr()

    def sparsify(self):
        new_fset = self.copy()
        new_fset.Xtrain = new_fset.Xtrain.tocsr()
        new_fset.Xtest = new_fset.Xtest.tocsr()
        return new_fsets

    # return a MyFeatureSet object with features in dorp_list dropped
    # either use name or use order/index, cannot mix
    def drop(self, drop_list):
        if not hasattr(drop_list, '__iter__'):
            drop_list = [drop_list]

        n_feature = len(self.fname_list)
        if isinstance(drop_list[0], str):
            keep_list = [self.fname_list.index(f) for f in self.fname_list if f not in drop_list]
        elif isinstance(drop_list[0], int):
            keep_list = [ind for ind in range(n_feature) if ind not in drop_list]
        else:
            print 'Unsupported drop list'
            return None

        return self.__getitem__(keep_list)

    def inplace_combine_rare(self, input_col_list = list(), prefix = 'cr_', rare_line = 1):
        fset_combine_rare(self, input_col_list, prefix, rare_line)

    def combine_rare(self, input_col_list = list(), prefix = 'cr_', rare_line = 1):
        new_fset = self.copy()
        fset_combine_rare(new_fset, input_col_list, prefix, rare_line)
        return new_fset

    def onehot_encode(self, input_col_list = list(), prefix = 'oh_'):
        return fset_onehot_encode(self, input_col_list, prefix)

    def inplace_label_encode(self, input_col_list = list(), prefix = 'le_'):
        fset_label_encode(self, input_col_list, prefix)

    def label_encode(self, input_col_list = list(), prefix = 'le_'):
        new_fset = self.copy()
        fset_label_encode(new_fset, input_col_list, prefix)
        return new_fset

    def inplace_numeric_transform(self, input_col_list = list(), operation='log', standardize=False):
        fset_numeric_transform(self, input_col_list, operation, standardize)

    def numeric_transform(self, input_col_list = list(), operation='log', standardize=False):
        new_fset = self.copy()
        fset_numeric_transform(new_fset, input_col_list, operation, standardize)
        return new_fset

    def inplace_random_shuffle(self, input_col_list = list(), random_state = SEED, prefix = 'rs_'):        
        fset_random_shuffle(self, input_col_list, random_state, prefix)

    def random_shuffle(self, input_col_list = list(), random_state = SEED, prefix = 'rs_'):        
        new_fset = self.copy()
        fset_random_shuffle(new_fset, input_col_list, random_state, prefix)
        return new_fset

    def one_degree_count(self, input_col_list = list(), prefix = 'cnt_'):
        return fset_one_degree_count(self, input_col_list, prefix)

    def resplit_train_test(self, input_target_col, cr_func = np.isnan):
        return fset_resplit_train_test(self, input_target_col, cr_func)

    def merge_multiple_cat_columns(self, input_col_multiple, hasher=None):
        return fset_merge_multiple_cat_columns(self, input_col_multiple, hasher)

    def binary_operation(self, input_col1, input_col2, operation = 'add'):
        return fset_binary_operation(self, input_col1, input_col2, operation)

    def stats_transform_bygroup(self, input_col = None, input_target_col = None, operation='percentile', helper = None):
        return fset_stats_transform_bygroup(self, input_col, input_target_col, operation, helper)
# basic functions
    def show_basic_statics(self, col_list = list()):
        fset_show_basic_statics(self, col_list)

    def test_lead_digit(self, input_col_list = list(), return_entropy = False):
        return fset_test_lead_digit(self, input_col_list, return_entropy)

    def remove_constant_col(self, input_col_list = list(), return_drop_list = False):
        return fset_remove_constant_col(self, input_col_list, return_drop_list)

    def remove_identical_col(self, input_col_list = list(), return_drop_list = False):
        return fset_remove_identical_col(self, input_col_list, return_drop_list)

    def check_two_columns(self, input_col1, input_col2):
        fset_check_two_columns(self, input_col1, input_col2)

    def mutual_info(self, input_col1, input_col2):
        return fset_mutual_info(self, input_col1, input_col2)


# generate MyFeatureSet object from a tuple of dataframe, support dropping certain feature
def df_to_fset(df_tuple, drop_list = list()):
    if len(df_tuple) != 2:
        print 'require two dataframe'
        return

    df_train, df_test = df_tuple

    col_train = df_train.columns.values
    col_test = df_test.columns.values

    col_all = [col for col in col_train if col in col_test]
    col_keep = [col for col in col_all if col not in drop_list]

    new_fset = MyFeatureSet()
    new_fset.fname_list = col_keep
    new_fset.find_list = range(len(col_keep)+1)
    new_fset.Xtrain = df_train[col_keep].values
    new_fset.Xtest = df_test[col_keep].values
    for col in col_keep:
        train_type = df_train[col].values.dtype.type
        test_type = df_test[col].values.dtype.type
        if issubclass(train_type, np.integer) and issubclass(test_type, np.integer):
            new_fset.ftype_list.append(train_type)
        elif not issubclass(train_type, np.integer):
            new_fset.ftype_list.append(train_type)
        else:
            new_fset.ftype_list.append(test_type)

    return new_fset
        

# new version, returns a fset
def concat_fsets(myfset_list, sparsify = False):  
    if not sparsify:
        for myfset in myfset_list:
            if scipy.sparse.issparse(myfset.Xtrain):
                sparsify = True
                break

    if sparsify:
        hstack_func = scipy.sparse.hstack
    else:
        hstack_func = np.hstack

    newfname_list = []
    newftype_list = []
    newfind_list = []
    newXtrain = None
    newXtest = None

    n_fset = len(myfset_list)
    for i in xrange(n_fset):
        myfset = myfset_list[i]

        if i == 0:
            newfname_list = myfset.fname_list
            newftype_list = myfset.ftype_list
            newfind_list = myfset.find_list
            
            newXtrain = myfset.Xtrain.copy()
            newXtest = myfset.Xtest.copy()

        else:
            newfname_list = newfname_list + myfset.fname_list
            newftype_list = newftype_list + myfset.ftype_list

            prev_total = newfind_list[-1]
            curfind_list = [prev_total + find for find in myfset.find_list[1:]]
            newfind_list = newfind_list + curfind_list

            newXtrain = hstack_func((newXtrain, myfset.Xtrain))

            newXtest = hstack_func((newXtest, myfset.Xtest))
 

    if sparsify:
        newXtrain = newXtrain.tocsr()
        newXtest = newXtest.tocsr()

    new_fset = MyFeatureSet()
    new_fset.fname_list = newfname_list
    new_fset.ftype_list = newftype_list
    new_fset.find_list = newfind_list
    new_fset.Xtrain = newXtrain
    new_fset.Xtest = newXtest

    return new_fset

# old version
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
            
            newXtrain = myfset.Xtrain.copy
            newXtest = myfset.Xtest.copy

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

def fset_onehot_encode(myfset, input_col_list = list(), prefix = 'oh_'):
    if isinstance(input_col_list, slice):
        start = input_col_list.start
        stop = input_col_list.stop
        step = input_col_list.step
        input_col_list = range(start, stop, step)

    if not hasattr(input_col_list, '__iter__'):
        input_col_list = [input_col_list]

    n_feature = len(myfset.fname_list)
    if not input_col_list :
        input_col_list = range(n_feature)
        check_int = True
    else:
        check_int = False

    # modification if we have string
    if isinstance(input_col_list[0], int):
        col_list = input_col_list
    elif isinstance(input_col_list[0], str):
        col_list = [myfset.fname_list.index(col) for col in input_col_list if col in myfset.fname_list]
    else:
        print 'unsupported indexing'
        return    
    col_list.sort()

    new_fset = myfset.copy()

    if scipy.sparse.issparse(new_fset.Xtrain):
        sparsify = True
    else:
        sparsify = False

    n_train = myfset.Xtrain.shape[0]
    n_test = myfset.Xtest.shape[0]

    encode_col = list()
    encode_ind = list()
    unencode_col = list()
    unencode_ind = list()

    # first perfrom label encoding
    lb_encoder = preprocessing.LabelEncoder()
    for col in col_list:
        if (new_fset.find_list[col+1] - new_fset.find_list[col]) > 1:
            unencode_col.append(col)
            unencode_ind = unencode_ind + range(new_fset.find_list[col], new_fset.find_list[col+1])
            continue
        else:
            ind = new_fset.find_list[col]
            if not issubclass(new_fset.ftype_list[col], np.integer):
                if check_int:
                    unencode_col.append(col)
                    unencode_ind.append(ind)
                    continue
                else:
                    new_fset.ftype_list[col] = np.integer

            encode_col.append(col)
            encode_ind.append(ind)

        train_data = new_fset.Xtrain[:, ind]
        test_data = new_fset.Xtest[:, ind]

        if sparsify:
            train_data = np.squeeze(train_data.toarray())
            test_data = np.squeeze(test_data.toarray())

        lb_encoder.fit(np.hstack((train_data, test_data)))

        new_train_data = lb_encoder.transform(train_data)
        new_test_data = lb_encoder.transform(test_data)

        if sparsify:
            new_train_data = new_train_data.reshape(n_train, 1)
            new_test_data = new_test_data.reshape(n_test, 1)

        new_fset.Xtrain[:, ind] = new_train_data
        new_fset.Xtest[:, ind] = new_test_data

    #one hot encoding
    encode_Xtrain = new_fset.Xtrain[:, encode_ind]
    encode_Xtest = new_fset.Xtest[:, encode_ind]

    if sparsify:
        encode_Xtrain = encode_Xtrain.toarray()
        encode_Xtest = encode_Xtest.toarray()

    oh_encoder = preprocessing.OneHotEncoder()
    oh_encoder.fit(np.vstack((encode_Xtrain, encode_Xtest)))
    new_fset.Xtrain = oh_encoder.transform(encode_Xtrain).tocsr()  
    new_fset.Xtest = oh_encoder.transform(encode_Xtest).tocsr()
    

    if len(unencode_ind) > 0:
        unencode_Xtrain = new_fset.Xtrain[:, unencode_ind]
        unencode_Xtest = new_fset.Xtest[:, unencode_ind]
        new_fset.Xtrain = scipy.sparse.hstack((new_fset.Xtrain, unencode_Xtrain)).tocsr()
        new_fset.Xtest = scipy.sparse.hstack((new_fset.Xtest, unencode_Xtest)).tocsr()

    encode_fname_list = [prefix + new_fset.fname_list[col] for col in encode_col]
    unencode_fname_list = [new_fset.fname_list[col] for col in unencode_col]

    encode_ftype_list = [new_fset.ftype_list[col] for col in encode_col]
    unencode_ftype_list = [new_fset.ftype_list[col] for col in unencode_col]

    new_fset.fname_list = encode_fname_list + unencode_fname_list
    new_fset.ftype_list = encode_ftype_list + unencode_ftype_list

    new_find_list = list(oh_encoder.feature_indices_)
    for col in unencode_col:
        find_last = newfind_list[-1]
        find_low = new_fset.find_list[col]
        find_up = new_fset.find_list[col+1]
        new_find_list.append(find_last + find_up - find_low)

    new_fset.find_list = new_find_list

    return new_fset

def fset_label_encode(myfset, input_col_list = list(), prefix = 'le_'):
    if isinstance(input_col_list, slice):
        start = input_col_list.start
        stop = input_col_list.stop
        step = input_col_list.step
        input_col_list = range(start, stop, step)    

    if not hasattr(input_col_list, '__iter__'):
        input_col_list = [input_col_list]

    n_feature = len(myfset.fname_list)
    if not input_col_list :
        input_col_list = range(n_feature)
        check_int = True
    else:
        check_int = False

    # modification if we have string
    if isinstance(input_col_list[0], int):
        col_list = input_col_list
    elif isinstance(input_col_list[0], str):
        col_list = [myfset.fname_list.index(col) for col in input_col_list if col in myfset.fname_list]
    else:
        print 'unsupported indexing'
        return

    if scipy.sparse.issparse(myfset.Xtrain):
        sparsify = True
    else:
        sparsify = False

    n_train = myfset.Xtrain.shape[0]
    n_test = myfset.Xtest.shape[0]
    for col in col_list:
        if (myfset.find_list[col+1] - myfset.find_list[col]) > 1:
            continue

        ind = myfset.find_list[col]
        col_type = myfset.ftype_list[col]
        # print col_data[0]
        if issubclass(col_type, np.integer) or (not check_int):
            col_data_train = myfset.Xtrain[:, ind]
            col_data_test = myfset.Xtest[:, ind]
            
            if sparsify:
                col_data_train = np.squeeze(col_data_train.toarray())
                col_data_test = np.squeeze(col_data_test.toarray())

            col_data = np.hstack((col_data_train, col_data_test))
            le = preprocessing.LabelEncoder()
            le.fit(col_data)
            col_data = le.transform(col_data)

            new_col_data_train = col_data[:n_train]
            new_col_data_test = col_data[n_train:]

            if sparsify:
                new_col_data_train = new_col_data_train.reshape(n_train, 1)
                new_col_data_test = new_col_data_test.reshape(n_test, 1)

            myfset.Xtrain[:, ind] = new_col_data_train
            myfset.Xtest[:, ind] = new_col_data_test
            if not issubclass(col_type, np.integer):
                myfset.ftype_list[col] = np.integer
            myfset.fname_list[col] = prefix + myfset.fname_list[col]
        else:
            print 'col:{0:s} not integer, include in list if insist'.format(myfset.fname_list[col])    



def fset_combine_rare(myfset, input_col_list = list(), prefix = 'cr_', rare_line = 1):
    if isinstance(input_col_list, slice):
        start = input_col_list.start
        stop = input_col_list.stop
        step = input_col_list.step
        input_col_list = range(start, stop, step)

    if not hasattr(input_col_list, '__iter__'):
        input_col_list = [input_col_list]

    n_feature = len(myfset.fname_list)
    if not input_col_list :
        input_col_list = range(n_feature)
        check_int = True
    else:
        check_int = False

    # modification if we have string
    if isinstance(input_col_list[0], int):
        col_list = input_col_list
    elif isinstance(input_col_list[0], str):
        col_list = [myfset.fname_list.index(col) for col in input_col_list if col in myfset.fname_list]
    else:
        print 'unsupported indexing'
        return

    if scipy.sparse.issparse(myfset.Xtrain):
        sparsify = True
    else:
        sparsify = False

    n_train = myfset.Xtrain.shape[0]
    n_test = myfset.Xtest.shape[0]
    for col in col_list:
        if (myfset.find_list[col+1] - myfset.find_list[col] > 1):
            continue

        ind = myfset.find_list[col]
        col_type = myfset.ftype_list[col]
        # print col_data[0]
        if issubclass(col_type, np.integer) or (not check_int):
            col_data_train = myfset.Xtrain[:, ind]
            col_data_test = myfset.Xtest[:, ind]
            
            if sparsify:
                col_data_train = np.squeeze(col_data_train.toarray())
                col_data_test = np.squeeze(col_data_test.toarray())

            col_data = np.hstack((col_data_train, col_data_test))
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
            new_col_data_train = col_data[:n_train]
            new_col_data_test = col_data[n_train:]

            if sparsify:
                new_col_data_train = new_col_data_train.reshape(n_train, 1)
                new_col_data_test = new_col_data_test.reshape(n_test, 1)

            myfset.Xtrain[:, ind] = new_col_data_train
            myfset.Xtest[:, ind] = new_col_data_test
            if not issubclass(col_type, np.integer):
                myfset.ftype_list[col] = np.integer
            myfset.fname_list[col] = prefix + myfset.fname_list[col] 
        else:
            print 'col:{0:s} not integer, include in list if insist'.format(myfset.fname_list[col])


def fset_numeric_transform(myfset, input_col_list = list(), operation='log', standardize=False):
    if isinstance(input_col_list, slice):
        start = input_col_list.start
        stop = input_col_list.stop
        step = input_col_list.step
        input_col_list = range(start, stop, step)

    if not hasattr(input_col_list, '__iter__'):
        input_col_list = [input_col_list]

    n_feature = len(myfset.fname_list)
    if not input_col_list :
        input_col_list = range(n_feature)
        check_int = True
    else:
        check_int = False

    # modification if we have string
    if isinstance(input_col_list[0], int):
        col_list = input_col_list
    elif isinstance(input_col_list[0], str):
        col_list = [myfset.fname_list.index(col) for col in input_col_list if col in myfset.fname_list]
    else:
        print 'unsupported indexing'
        return

    if scipy.sparse.issparse(myfset.Xtrain):
        sparsify = True
    else:
        sparsify = False

    n_train = myfset.Xtrain.shape[0]
    n_test = myfset.Xtest.shape[0]

    if operation == 'log':
        vfunc = lambda x: np.log(x)
        suffix = '_l'
    elif operation == 'log1p':
        vfunc = lambda x: np.log1p(x)
        suffix = '_l1p'
    elif operation == 'log1pabs':
        vfunc = lambda x: np.multiply(np.sign(x), np.log1p(np.fabs(x)))
        suffix = '_l1pa'
    elif operation == 'exp':
        vfunc = lambda x: np.exp(x)
        suffix = '_e'
    elif operation == 'expm1':
        vfunc = lambda x: np.expm1(x)
        suffix = '_em1'
    elif operation == 'square':
        vfunc = np.vectorize(lambda x: x**2)
        suffix ='_sq'
    elif operation == 'none':
        vfunc = None
        suffix = ''
    elif re.search(r'^round_(\d+)', operation):
        dec = int(re.search(r'^round_(\d+)', operation).group(1))
        vfunc = lambda x : np.around(x , decimals = dec)
        suffix = '_rnd'+str(dec)
    else:
        vfunc = None
        suffix = ''
        print 'Unkown operation not performed'

    if standardize:
        if not suffix:
            suffix = '_s'
        else:
            suffix = suffix + 's'

    for col in col_list:
        if (myfset.find_list[col+1] - myfset.find_list[col] > 1):
            continue

        ind = myfset.find_list[col]

        col_data_train = myfset.Xtrain[:, ind]
        col_data_test = myfset.Xtest[:, ind]
        if sparsify:
            col_data_train = np.squeeze(col_data_train.toarray())
            col_data_test = np.squeeze(col_data_test.toarray())

        if vfunc:
            col_data_train = vfunc(col_data_train)
            col_data_test = vfunc(col_data_test)
        if standardize:
            col_data = np.hstack((col_data_train, col_data_test))
            col_mean = np.mean(col_data)
            col_std = np.std(col_data)
            # print col_mean, col_std
            col_data_train = 1./col_std * (col_data_train - col_mean)
            col_data_test = 1./col_std * (col_data_test - col_mean)

        if sparsify:
            col_data_train = col_data_train.reshape(n_train, 1)
            col_data_test = col_data_test.reshape(n_test, 1)

        myfset.Xtrain[:, ind] = col_data_train
        myfset.Xtest[:, ind] = col_data_test

        myfset.fname_list[col] = myfset.fname_list[col] + suffix


def np_rank(array):
    unique, counts = np.unique(array, return_counts = True)
    num_unique = unique.shape[0]
    counts_cumsum = np.zeros(num_unique+1)
    counts_cumsum[1:] = np.cumsum(counts)
    rank = (counts_cumsum[1:] + counts_cumsum[:-1]-1)/2
    rank_dict = dict(zip(unique, rank))
    rank_map = np.vectorize(lambda x: rank_dict[x])
    array_rank = rank_map(array)
    return array_rank


def fset_stats_transform_bygroup(myfset, input_col = None, input_target_col = None, operation='percentile', helper = None):
    if not input_col:
        print 'col for operation not selected'
        return

    if isinstance(input_col, int):
        col = input_col
    elif isinstance(input_col, str):
        if input_col not in myfset.fname_list:
            print 'col for operation not found'
            return
        col = myfset.fname_list.index(input_col)
    else:
        print 'unsupported indexing'
        return

    # modification if we have string
    if input_target_col:
        if isinstance(input_target_col, int):
            target_col = input_target_col
        elif isinstance(input_target_col, str):
            if input_target_col not in myfset.fname_list:
                print 'tareget not found'
                return
            else:
                target_col = myfset.fname_list.index(input_target_col)
        else:
            print 'unsupported indexing'
            return
    else:
        target_col = None

    # remove target_col from col_list 
    if target_col == col:
        print 'taget collide'
        return

    if scipy.sparse.issparse(myfset.Xtrain):
        sparsify = True
    else:
        sparsify = False

    n_train = myfset.Xtrain.shape[0]
    n_test = myfset.Xtest.shape[0]

    if operation == 'rank':
        vfunc = lambda x: np_rank(x)
        # suffix = '_rnk'
    elif operation == 'percentile':
        vfunc = lambda x: np_rank(x)/(x.shape[0]-1)
        # suffix = '_pct'
    elif operation == 'digitize':
        bins = helper
        vfunc = lambda x: np.digitize(x, bins, right=False)
        # suffix = '_dg'    
    else:
        vfunc = None
        suffix = ''
        print 'Unkown operation not performed'
        return

    n_train = myfset.Xtrain.shape[0]
    n_test = myfset.Xtest.shape[0]
    # first get the indices of different groups in the target_col
    if target_col:
        if (myfset.find_list[target_col+1] - myfset.find_list[target_col] > 1):
            print '{0:s} expands more than one column: assuming one hot encoded'.format(myfset.fname_list[target_col])
            arg_group_list = []
            for ind in xrange(myfset.find_list[target_col], myfset.find_list[target_col+1]):
                target_data_train = myfset.Xtrain[:, ind]
                target_data_test = myfset.Xtest[:, ind]
                if sparsify:
                    target_data_train = np.squeeze(target_data_train.toarray())
                    target_data_test = np.squeeze(target_data_test.toarray())
                target_data = np.hstack((target_data_train, target_data_test))
                arg_group = np.squeeze(np.argwhere(target_data == 1))
                arg_group_list.append(arg_group)
        else:
            ind = myfset.find_list[target_col]
            arg_group_list = []
            target_data_train = myfset.Xtrain[:, ind]
            target_data_test = myfset.Xtest[:, ind]
            if sparsify:
                target_data_train = np.squeeze(target_data_train.toarray())
                target_data_test = np.squeeze(target_data_test.toarray())
            target_data = np.hstack((target_data_train, target_data_test))
            target_unique = np.unique(target_data)
            for target in target_unique:
                arg_group = np.squeeze(np.argwhere(target_data == target))
                arg_group_list.append(arg_group)
    else:
        arg_group_list = []


    # do operation
    if (myfset.find_list[col+1] - myfset.find_list[col] > 1):
        print 'col to operate is one hot encoded, not suitable for operation'
        return
        
    ind = myfset.find_list[col]

    col_data_train = myfset.Xtrain[:, ind]
    col_data_test = myfset.Xtest[:, ind]
    if sparsify:
        col_data_train = np.squeeze(col_data_train.toarray())
        col_data_test = np.squeeze(col_data_test.toarray())

    col_data = np.hstack((col_data_train, col_data_test))
    new_col_data = np.zeros(n_train + n_test)
    
    if arg_group_list:
        for arg_group in arg_group_list:
            print arg_group.shape
            new_col_data[arg_group] = vfunc(col_data[arg_group])
            
    else:
        new_col_data = vfunc(col_data)

    new_col_data_train = new_col_data[:n_train]
    new_col_data_test = new_col_data[n_train:]


    return new_col_data_train, new_col_data_test


def np_divide_w_zero(c1_data, c2_data, infty = 1e10-1):
    new_data = np.divide(c1_data, c2_data)
    arg_c1_zero = np.squeeze(np.argwhere(c1_data == 0))
    arg_c1_pos = np.squeeze(np.argwhere(c1_data > 0))
    arg_c1_neg = np.squeeze(np.argwhere(c1_data < 0))
    arg_c2_zero = np.squeeze(np.argwhere(c2_data == 0))
    arg_div_posinf = np.intersect1d(arg_c1_pos, arg_c2_zero)
    arg_div_neginf = np.intersect1d(arg_c1_neg, arg_c2_zero)
    if arg_c1_zero.shape[0]:
        new_data[arg_c1_zero] = np.zeros(arg_c1_zero.shape)
    if arg_div_posinf.shape[0]:
        new_data[arg_div_posinf] = infty * np.ones(arg_div_posinf.shape)
    if arg_div_neginf.shape[0]:
        new_data[arg_div_neginf] = -infty * np.ones(arg_div_neginf.shape)
    return new_data

def np_change_rate(c1_data, c2_data, infty = 1e10-1, option = None):
    if option == 'abs':
        new_data = np.divide((c1_data - c2_data), np.fabs(c2_data))
    else:
        new_data = np.divide((c1_data - c2_data), c2_data)
    arg_diff_zero = np.squeeze(np.argwhere((c1_data - c2_data) == 0))
    arg_diff_pos = np.squeeze(np.argwhere((c1_data - c2_data) > 0))
    arg_diff_neg = np.squeeze(np.argwhere((c1_data - c2_data) < 0))
    arg_c2_zero = np.squeeze(np.argwhere(c2_data == 0))
    arg_diff_posinf = np.intersect1d(arg_diff_pos, arg_c2_zero)
    arg_diff_neginf = np.intersect1d(arg_diff_neg, arg_c2_zero)
    if arg_diff_zero.shape[0]:
        new_data[arg_diff_zero] = np.zeros(arg_diff_zero.shape)
    if arg_diff_posinf.shape[0]:
        new_data[arg_diff_posinf] = infty * np.ones(arg_diff_posinf.shape)
    if arg_diff_neginf.shape[0]:
        new_data[arg_diff_neginf] = -infty * np.ones(arg_diff_neginf.shape)
    return new_data

def fset_binary_operation(myfset, input_col1, input_col2, operation='add', infty = 1e10-1):
    if isinstance(input_col1, int):
        col1 = input_col1
        c1_ind = myfset.find_list[col1]
        c1_name = myfset.fname_list[col1]
    elif isinstance(input_col1, str):
        col1 = myfset.fname_list.index(input_col1)
        c1_ind = myfset.find_list[col1]
        c1_name = myfset.fname_list[col1]

    if isinstance(input_col2, int):
        col2 = input_col2
        c2_ind = myfset.find_list[col2]
        c2_name = myfset.fname_list[col2]
    elif isinstance(input_col2, str):
        col2 = myfset.fname_list.index(input_col2)
        c2_ind = myfset.find_list[col2]
        c2_name = myfset.fname_list[col2]

    if scipy.sparse.issparse(myfset.Xtrain):
        sparsify = True
    else:
        sparsify = False

    if operation == 'add':
        bin_op = np.add
    elif operation == 'sub':
        bin_op = np.subtract
    elif operation == 'mul':
        bin_op = np.multiply
    elif operation == 'div':
        bin_op = np.divide
    elif operation == 'chg_rate':
        bin_op = lambda x,y: np_change_rate(x, y, infty)
    elif operation == 'abs_chg_rate':
        bin_op = lambda x,y: np_change_rate(x, y, infty, option = 'abs')
    elif operation == 'div_w_zero':
        bin_op = lambda x,y: np_divide_w_zero(x, y, infty)
    else:
        print 'unsupported oepration'
        return

    c1_data_train = myfset.Xtrain[:, c1_ind]
    c1_data_test = myfset.Xtest[:, c1_ind]
    c2_data_train = myfset.Xtrain[:, c2_ind]
    c2_data_test = myfset.Xtest[:, c2_ind]

    if sparsify:
        c1_data_train = np.squeeze(c1_data_train.toarray())
        c1_data_test = np.squeeze(c1_data_test.toarray())
        c2_data_train = np.squeeze(c2_data_train.toarray())
        c2_data_test = np.squeeze(c2_data_test.toarray())

    c1_data = np.hstack((c1_data_train, c1_data_test))
    c2_data = np.hstack((c2_data_train, c2_data_test))

    new_data = bin_op(c1_data, c2_data)

    n_train = c1_data_train.shape[0]
    new_data_train = new_data[:n_train]
    new_data_test = new_data[n_train:]

    return new_data_train, new_data_test

# randomly shuffle some features
def fset_random_shuffle(myfset, input_col_list = list(), random_state = SEED, prefix = 'rs_'):
    if isinstance(input_col_list, slice):
        start = input_col_list.start
        stop = input_col_list.stop
        step = input_col_list.step
        input_col_list = range(start, stop, step)    

    if not hasattr(input_col_list, '__iter__'):
        input_col_list = [input_col_list]

    n_feature = len(myfset.fname_list)
    if not input_col_list :
        input_col_list = range(n_feature)
        check_int = True
    else:
        check_int = False

    # modification if we have string
    if isinstance(input_col_list[0], int):
        col_list = input_col_list
    elif isinstance(input_col_list[0], str):
        col_list = [myfset.fname_list.index(col) for col in input_col_list if col in myfset.fname_list]
    else:
        print 'unsupported indexing'
        return

    if scipy.sparse.issparse(myfset.Xtrain):
        sparsify = True
    else:
        sparsify = False

    np.random.seed(SEED)

    n_train = myfset.Xtrain.shape[0]
    n_test = myfset.Xtest.shape[0]

    shuffler = np.arange(n_train + n_test)
    np.random.shuffle(shuffler)
    for col in col_list:
        if (myfset.find_list[col+1] - myfset.find_list[col]) > 1:
            find_low = myfset.find_list[col]
            find_high = myfset.find_list[col+1]
            col_data_train = myfset.Xtrain[:, find_low:find_high].toarray()
            col_data_train = myfset.Xtrain[:, find_low:find_high].toarray()

            col_data = np.vstack((col_data_train, col_data_test))
            
            col_data[:, :] = col_data[shuffler, :]

            new_col_data_train = col_data[:n_train, :]
            new_col_data_test = col_data[n_train:, :]
            
            myfset.Xtrain[:, find_low:find_high] = new_col_data_train
            myfset.Xtest[:, find_low:find_high] = new_col_data_test

        else:
            ind = myfset.find_list[col]
            # print col_data[0]
            
            col_data_train = myfset.Xtrain[:, ind]
            col_data_test = myfset.Xtest[:, ind]
            
            if sparsify:
                col_data_train = np.squeeze(col_data_train.toarray())
                col_data_test = np.squeeze(col_data_test.toarray())

            col_data = np.hstack((col_data_train, col_data_test))
            col_data = col_data[shuffler]

            new_col_data_train = col_data[:n_train]
            new_col_data_test = col_data[n_train:]

            if sparsify:
                new_col_data_train = new_col_data_train.reshape(n_train, 1)
                new_col_data_test = new_col_data_test.reshape(n_test, 1)

            myfset.Xtrain[:, ind] = new_col_data_train
            myfset.Xtest[:, ind] = new_col_data_test


        myfset.fname_list[col] = prefix + myfset.fname_list[col]
        

# create a new feature set with counts
def fset_one_degree_count(myfset, input_col_list = list(), prefix = 'cnt_'):
    if isinstance(input_col_list, slice):
        start = input_col_list.start
        stop = input_col_list.stop
        step = input_col_list.step
        input_col_list = range(start, stop, step)

    if not hasattr(input_col_list, '__iter__'):
        input_col_list = [input_col_list]

    n_feature = len(myfset.fname_list)
    if not input_col_list :
        input_col_list = range(n_feature)
        check_int = True
    else:
        check_int = False

    # modification if we have string
    if isinstance(input_col_list[0], int):
        col_list = input_col_list
    elif isinstance(input_col_list[0], str):
        col_list = [myfset.fname_list.index(col) for col in input_col_list if col in myfset.fname_list]
    else:
        print 'unsupported indexing'
        return

    if scipy.sparse.issparse(myfset.Xtrain):
        sparsify = True
    else:
        sparsify = False

    n_train = myfset.Xtrain.shape[0]
    n_test = myfset.Xtest.shape[0]

    new_fset = MyFeatureSet()
    for col in col_list:
        if (myfset.find_list[col+1] - myfset.find_list[col] > 1):
            continue

        ind = myfset.find_list[col]
        col_type = myfset.ftype_list[col]
        # print col_data[0]
        if issubclass(col_type, np.integer) or (not check_int):
            col_data_train = myfset.Xtrain[:, ind]
            col_data_test = myfset.Xtest[:, ind]
            
            if sparsify:
                col_data_train = np.squeeze(col_data_train.toarray())
                col_data_test = np.squeeze(col_data_test.toarray())

            col_data = np.hstack((col_data_train, col_data_test))
            values, counts = np.unique(col_data, return_counts = True)
            value_to_count = dict(zip(values, counts))
            vfunc = np.vectorize(lambda x : value_to_count[x])
            cnt_col_data = vfunc(col_data)
            cnt_col_data_train = col_data[:n_train].reshape(n_train, 1).astype(float)
            cnt_col_data_test = col_data[n_train:].reshape(n_test, 1).astype(float)

            if not new_fset.fname_list:
                new_fset.Xtrain = cnt_col_data_train
                new_fset.Xtest = cnt_col_data_test
                new_fset.find_list.append(0)
            else:
                new_fset.Xtrain = np.hstack((new_fset.Xtrain, cnt_col_data_train))
                new_fset.Xtest = np.hstack((new_fset.Xtest, cnt_col_data_test))

            find_last = new_fset.find_list[-1]
            new_fset.find_list.append(find_last + 1)
            new_fset.ftype_list.append(np.float)            
            new_fset.fname_list.append(prefix + myfset.fname_list[col])
        else:
            print 'col:{0:s} not integer, include in list if insist'.format(myfset.fname_list[col])

    return new_fset


def fset_resplit_train_test(myfset, input_target_col, cr_func = np.isnan):
    if isinstance(input_target_col, int):
        target_col = input_target_col
    elif isinstance(input_target_col, str):
        target_col = myfset.fname_list.index(input_target_col)
    else:
        print 'unsupported indexing'
        return
    
    if myfset.find_list[target_col + 1] - myfset.find_list[target_col] > 1:
        print 'unspported: target col onehot encoded'
        return
    
    if scipy.sparse.issparse(myfset.Xtrain):
        sparsify = True
    else:
        sparsify = False

    target_find = myfset.find_list[target_col]
    
    target_col_data_train = myfset.Xtrain[:, target_find]
    target_col_data_test = myfset.Xtest[:, target_find]
    
    if sparsify:
        target_col_data_train = np.squeeze(target_col_data_train.toarray())
        target_col_data_test = np.squeeze(target_col_data_test.toarray())
    
    target_col_data = np.hstack((target_col_data_train, target_col_data_test))
    
    cr_vfunc = np.vectorize(cr_func)
    
    cr_target = cr_vfunc(target_col_data)
    
    new_row_train = np.squeeze(np.argwhere(cr_target == False))
    new_row_test = np.squeeze(np.argwhere(cr_target == True))
    new_row_all = np.hstack((new_row_train, new_row_test))
    
    new_ytrain = target_col_data[new_row_train]
    
    if new_row_test.shape[0] == 0:
        print 'no row satisfies criterion'
        return
    
    new_fset = myfset.drop(target_col)
    
    if sparsify:
        vstack_func = scipy.sparse.vstack
    else:
        vstack_func = np.vstack

    X_all = vstack_func((new_fset.Xtrain, new_fset.Xtest))
    new_fset.Xtrain = X_all[new_row_train, :]
    new_fset.Xtest = X_all[new_row_test, :]
    
    return new_fset, new_ytrain, new_row_all

  
# new version of random feature selection, return
def fset_forward_random_select(mylearner, myfset, ytrain, input_fini_list = list(), input_fselect_list = list(), n_feature_final = None, \
        nfolds=5, randstate_list = SEED, score_func=roc_auc_score, cv_func = strat_cv_score, reshuffle_round = 0, max_trial = 3):
   # parsing input_fini_list
    if isinstance(input_fini_list, slice):
        start = input_fini_list.start
        stop = input_fini_list.stop
        step = input_fini_list.step
        input_fini_list = range(start, stop, step)

    if not hasattr(input_fini_list, '__iter__'):
        input_fini_list = [input_fini_list]

    # modification if we have string
    if isinstance(input_fini_list[0], int):
        fini_list = input_fini_list
    elif isinstance(input_fini_list[0], str):
        fini_list = [myfset.fname_list.index(col) for col in input_fini_list if col in myfset.fname_list]
    else:
        print 'unsupported indexing'
        return
    # parsing input_fselect_list
    if isinstance(input_fselect_list, slice):
        start = input_fselect_list.start
        stop = input_fselect_list.stop
        step = input_fselect_list.step
        input_fselect_list = range(start, stop, step)

    if not hasattr(input_fselect_list, '__iter__'):
        input_fselect_list = [input_fselect_list]

    # modification if we have string
    if input_fselect_list:
        if isinstance(input_fselect_list[0], int):
            fselect_list = input_fselect_list
        elif isinstance(input_fselect_list[0], str):
            fselect_list = [myfset.fname_list.index(col) for col in input_fselect_list if col in myfset.fname_list]
        else:
            print 'unsupported indexing'
            return
    else:
        fselect_list = list()

    if not hasattr(randstate_list, '__iter__'):
        randstate_list = [randstate_list]

    n_feature = len(myfset.fname_list)

    # use all remaining features if fselect_list not provided and permute
    if not fselect_list:
        fselect_list = [i for i in range(n_feature) if i not in fini_list]
        np.random.seed(randstate_list[0])
        np.random.shuffle(fselect_list)
    fselect_deque = deque(fselect_list)

    # if do not specify upper bound on num feature, try to use as many as possible
    if not n_feature_final:
        n_feature_final = len(fselect_list)
    
    fselect_Xtrain_list = [myfset.Xtrain[:, myfset.find_list[i]:myfset.find_list[i+1]] for i in fselect_list]
    fselect_Xtrain_dict = dict(zip(fselect_list, fselect_Xtrain_list))

    if scipy.sparse.issparse(myfset.Xtrain):
        hstack_func = scipy.sparse.hstack
    else:
        hstack_func = np.hstack

    flag = 0
    good_list = fini_list
    good_Xtrain = myfset[fini_list].Xtrain

    num_randstate = len(randstate_list)
    cur_randstate_list = randstate_list

    test_cv_scores = np.zeros((nfolds, num_randstate), float)

    for i in xrange(num_randstate):
        randstate = cur_randstate_list[i]
        test_cv_scores[:, i] = cv_func(mylearner, good_Xtrain, ytrain, nfolds, randstate, score_func)
    
    cur_mean_cv_score = np.mean(test_cv_scores)

    mean_cv_scores = list()

    mean_cv_scores.append(cur_mean_cv_score)
    mean_good_score = cur_mean_cv_score

    print 'initial score:', cur_mean_cv_score

    trial = 0
    iteration = 0
    while len(good_list) < n_feature_final and flag == 0:
        iteration += 1
        if reshuffle_round and iteration == reshuffle_round:
            for i in xrange(num_randstate):
                cur_randstate_list[i] = (cur_randstate_list[i] + randstate_list[i]) % SHUFFLE_BOUND

            for i in xrange(num_randstate):
                randstate = cur_randstate_list[i]
                test_cv_scores[:, i] = cv_func(mylearner, good_Xtrain, ytrain, nfolds, randstate, score_func)

            mean_good_score = np.mean(test_cv_scores)

            print 'current score: {0:f} (cv randstate changed)'.format(mean_good_score)

            iteration = 0

        flag = 1
        for fid in xrange(len(fselect_deque)):
            f = fselect_deque[fid]
            # print myfset.fname_list[f]
            cur_Xtrain = hstack_func((good_Xtrain, fselect_Xtrain_dict[f]))

            for i in xrange(num_randstate):
                randstate = cur_randstate_list[i]
                test_cv_scores[:, i] = cv_func(mylearner, cur_Xtrain, ytrain, nfolds, randstate, score_func)
            
            cur_mean_cv_score = np.mean(test_cv_scores)

            if cur_mean_cv_score > mean_good_score:
                good_Xtrain = cur_Xtrain
                good_list.append(f)
                fselect_deque.rotate(-fid)
                fselect_deque.remove(f)
                print 'current score: {0:f}, {1:s} selected'.format(cur_mean_cv_score, myfset.fname_list[f])
                mean_cv_scores.append(cur_mean_cv_score)
                mean_good_score = cur_mean_cv_score
                flag = 0
                break

        # if trial number less than max_trial reset randomstate and give another trial
        if flag == 1 and trial < max_trial:
            trial += 1
            flag = 0
            iteration = reshuffle_round - 1

    return myfset[good_list], mean_cv_scores

# random backward selection
def fset_backward_random_select(mylearner, myfset, ytrain, input_fkeep_list = list(), input_fselect_list = list(), n_feature_final = 0, \
        nfolds=5, randstate_list = SEED, score_func=roc_auc_score, cv_func = strat_cv_score, reshuffle_round = 0, max_trial = 1):
    if isinstance(input_fkeep_list, slice):
        start = input_fkeep_list.start
        stop = input_fkeep_list.stop
        step = input_fkeep_list.step
        input_fkeep_list = range(start, stop, step)

    if not hasattr(input_fkeep_list, '__iter__'):
        input_fkeep_list = [input_fkeep_list]

    # modification if we have string
    if input_fkeep_list:
        if isinstance(input_fkeep_list[0], int):
            fkeep_list = input_fkeep_list
        elif isinstance(input_fkeep_list[0], str):
            fkeep_list = [myfset.fname_list.index(col) for col in input_fkeep_list if col in myfset.fname_list]
        else:
            print 'unsupported indexing'
            return
    else:
        fkeep_list = input_fkeep_list

    n_feature = len(myfset.fname_list)

    if not hasattr(randstate_list, '__iter__'):
        randstate_list = [randstate_list]

    # create fselect_list
    if isinstance(input_fselect_list, slice):
        start = input_fselect_list.start
        stop = input_fselect_list.stop
        step = input_fselect_list.step
        input_fselect_list = range(start, stop, step)

    if not hasattr(input_fselect_list, '__iter__'):
        input_fselect_list = [input_fselect_list]

    if not input_fselect_list:
        input_fselect_list = [i for i in range(n_feature) if i not in fkeep_list]
        np.random.seed(randstate_list[0])
        np.random.shuffle(input_fselect_list)

    if isinstance(input_fselect_list[0], int):
        fselect_list = input_fselect_list
    elif isinstance(input_fselect_list[0], str):
        fselect_list = [myfset.fname_list.index(col) for col in input_fselect_list if col in myfset.fname_list]
    else:
        print 'unsupported indexing'
        return

    fselect_Xtrain_list = [myfset.Xtrain[:, myfset.find_list[i]:myfset.find_list[i+1]] for i in fselect_list]
    fselect_Xtrain_dict = dict(zip(fselect_list, fselect_Xtrain_list))

    fselect_deque = deque(fselect_list)

    if scipy.sparse.issparse(myfset.Xtrain):
        hstack_func = scipy.sparse.hstack
    else:
        hstack_func = np.hstack

    flag = 0
    if fkeep_list:
        fkeep_Xtrain = myfset[fkeep_list].Xtrain
        n_feature_final = max(n_feature_final, len(fkeep_list))
    else:
        fkeep_Xtrain = None

    num_randstate = len(randstate_list)
    cur_randstate_list = randstate_list

    test_cv_scores = np.zeros((nfolds, num_randstate), float)

    if fkeep_list:
        prev_Xtrain = hstack_func((fkeep_Xtrain, hstack_func((fselect_Xtrain_dict[sf] for sf in fselect_deque)) ))
    else:
        prev_Xtrain = hstack_func((fselect_Xtrain_dict[sf] for sf in fselect_deque))    


    for i in xrange(num_randstate):
        randstate = cur_randstate_list[i]
        test_cv_scores[:, i] = cv_func(mylearner, prev_Xtrain, ytrain, nfolds, randstate, score_func)
    
    prev_mean_cv_score = np.mean(test_cv_scores)

    mean_cv_scores = list()

    mean_cv_scores.append(prev_mean_cv_score)

    print 'initial score: {0:f}'.format(prev_mean_cv_score), prev_Xtrain.shape

    trial = 0
    iteration = 0
    while (len(fkeep_list) + len(fselect_deque)) > n_feature_final and flag == 0:
        iteration += 1
        if reshuffle_round and iteration == reshuffle_round:
            for i in xrange(num_randstate):
                cur_randstate_list[i] = (cur_randstate_list[i] + randstate_list[i]) % SHUFFLE_BOUND

            for i in xrange(num_randstate):
                randstate = cur_randstate_list[i]
                test_cv_scores[:, i] = cv_func(mylearner, prev_Xtrain, ytrain, nfolds, randstate, score_func)

            prev_mean_cv_score = np.mean(test_cv_scores)

            print 'current score: {0:f} (cv randstate changed)'.format(prev_mean_cv_score)

            iteration = 0

        flag = 1
        for fid in xrange(len(fselect_deque)):
            f = fselect_deque[fid]
            # print myfset.fname_list[f]
            if fkeep_list:
                cur_Xtrain = hstack_func((fkeep_Xtrain, hstack_func((fselect_Xtrain_dict[sf] for sf in fselect_deque if sf != f)) ))
            else:
                cur_Xtrain = hstack_func((fselect_Xtrain_dict[sf] for sf in fselect_deque if sf != f))

            for i in xrange(num_randstate):
                randstate = cur_randstate_list[i]
                test_cv_scores[:, i] = cv_func(mylearner, cur_Xtrain, ytrain, nfolds, randstate, score_func)
            
            cur_mean_cv_score = np.mean(test_cv_scores)

            print 'current score: {0:f}, {1:s}'.format(cur_mean_cv_score, myfset.fname_list[f])

            if cur_mean_cv_score > prev_mean_cv_score:
                fselect_deque.rotate(-fid)
                fselect_deque.remove(f)
                print 'current score: {0:f}, {1:s} dropped'.format(cur_mean_cv_score, myfset.fname_list[f]), cur_Xtrain.shape, 
                mean_cv_scores.append(cur_mean_cv_score)
                prev_mean_cv_score = cur_mean_cv_score
                prev_Xtrain = cur_Xtrain
                flag = 0
                break

        # if trial number less than max_trial reset randomstate and give another trial
        if flag == 1 and trial < max_trial:
            trial += 1
            flag = 0
            iteration = reshuffle_round - 1


    ffinal_list = fkeep_list + list(fselect_deque)

    return myfset[ffinal_list], mean_cv_scores

def fset_greedy_merge_replace(mylearner, myfset, ytrain, input_ftuple_list = list(), input_new_fname_list = list(), \
        nfolds=5, randstate_list = SEED, score_func=roc_auc_score, cv_func = strat_cv_score, reshuffle_round = 1, max_trial = 1):
    if not hasattr(input_ftuple_list, '__iter__'):
        input_ftuple_list = [input_ftuple_list]

    # check input_ftuple_list convert to feature_name to make sure consistency
    if not input_ftuple_list:
        print 'no feature tuple to merge specified'
        return

    n_feature = len(myfset.fname_list)

    ftuple_list = []
    new_fname_list = []
    for i_tuple in xrange(len(input_ftuple_list)):
        input_ftuple = input_ftuple_list[i_tuple]
        new_fname = input_new_fname_list[i_tuple]
        if isinstance(input_ftuple[0], int):
            ftuple = tuple(myfset.fname_list[i] for i in input_ftuple if i in range(n_feature))
            if len(ftuple) == len(input_ftuple):
                ftuple_list.append(ftuple)
                new_fname_list.append(new_fname)
            else:
                print input_ftuple, 'not all found'
        elif isinstance(input_ftuple[0], str):
            ftuple = tuple(col for col in input_ftuple if col in myfset.fname_list)
            if len(ftuple) == len(input_ftuple):
                ftuple_list.append(ftuple)
                new_fname_list.append(new_fname)
            else:
                print input_ftuple, 'not all found'
        else:
            print 'unsupported indexing'
            return

    new_fname_dict = dict(zip(ftuple_list, new_fname_list))

    if not hasattr(randstate_list, '__iter__'):
        randstate_list = [randstate_list]

    fselect_list = list(myfset.fname_list)

    f_Xtrain_list = [myfset.Xtrain[:, myfset.find_list[col]:myfset.find_list[col+1]] for col in range(n_feature)]
    f_Xtrain_dict = dict(zip(fselect_list, f_Xtrain_list))

    ftuple_merge_Xtrain_list = [np.expand_dims(myfset.merge_multiple_cat_columns(ftuple)[0], axis = 1) for ftuple in ftuple_list]
    ftuple_merge_Xtrain_dict = dict(zip(ftuple_list, ftuple_merge_Xtrain_list))

    # fselect_deque = deque(fselect_list)

    if scipy.sparse.issparse(myfset.Xtrain):
        hstack_func = scipy.sparse.hstack
    else:
        hstack_func = np.hstack

    num_randstate = len(randstate_list)
    cur_randstate_list = randstate_list

    test_cv_scores = np.zeros((nfolds, num_randstate), float)


    prev_Xtrain = hstack_func((f_Xtrain_dict[f] for f in fselect_list)) 

    for i in xrange(num_randstate):
        randstate = cur_randstate_list[i]
        test_cv_scores[:, i] = cv_func(mylearner, prev_Xtrain, ytrain, nfolds, randstate, score_func)
    
    prev_mean_cv_score = np.mean(test_cv_scores)

    mean_cv_scores = list()

    mean_cv_scores.append(prev_mean_cv_score)

    print 'initial score: {0:f}'.format(prev_mean_cv_score)
    
    fdrop_list = list()
    selected_ftuple_list = list()
    flag = 0
    trial = 0
    iteration = -1
    while ftuple_list and flag == 0:
        iteration += 1
        if reshuffle_round and iteration == reshuffle_round:
            for i in xrange(num_randstate):
                cur_randstate_list[i] = (cur_randstate_list[i] + randstate_list[i]) % SHUFFLE_BOUND

            for i in xrange(num_randstate):
                randstate = cur_randstate_list[i]
                test_cv_scores[:, i] = cv_func(mylearner, prev_Xtrain, ytrain, nfolds, randstate, score_func)

            prev_mean_cv_score = np.mean(test_cv_scores)

            print 'current score: {0:f} (cv randstate changed)'.format(prev_mean_cv_score)

            iteration = 0

        flag = 1
        max_mean_cv_score = prev_mean_cv_score
        for ftuple in ftuple_list:
            # print myfset.fname_list[f]
            cur_Xtrain = hstack_func((hstack_func((f_Xtrain_dict[col] for col in fselect_list if col not in ftuple )), ftuple_merge_Xtrain_dict[ftuple]))

            for i in xrange(num_randstate):
                randstate = cur_randstate_list[i]
                test_cv_scores[:, i] = cv_func(mylearner, cur_Xtrain, ytrain, nfolds, randstate, score_func)
            
            cur_mean_cv_score = np.mean(test_cv_scores)

            print 'current score: {0:f}, merge:'.format(cur_mean_cv_score), ftuple

            if cur_mean_cv_score > max_mean_cv_score:
                max_mean_cv_score = cur_mean_cv_score
                cur_ftuple = ftuple
                prev_mean_cv_score = cur_mean_cv_score
                prev_Xtrain = cur_Xtrain
                flag = 0

        # if there is a good tuple for merging, delete tuple from fselect_list add tuple to the fselect_list and put it in final merge_list
        if flag == 0:
            fselect_list = [col for col in fselect_list if col not in cur_ftuple]
            fselect_list = fselect_list + [new_fname_dict[cur_ftuple]]
            f_Xtrain_dict[new_fname_dict[cur_ftuple]] = ftuple_merge_Xtrain_dict[cur_ftuple]
            ftuple_list.remove(cur_ftuple)
            selected_ftuple_list.append(cur_ftuple)
            fdrop_list += [col for col in cur_ftuple]
            mean_cv_scores.append(max_mean_cv_score)
            print 'best score: {0:f}, merge:'.format(max_mean_cv_score), cur_ftuple
        elif flag == 1 and trial < max_trial:
            print 'no good tuple to merge found'
            trial += 1
            flag = 0
            iteration = reshuffle_round - 1
        else:
            print 'no good tuple to merge found'


    new_fset = myfset.copy()
    for ftuple in selected_ftuple_list:
        print new_fname_dict[ftuple], ftuple
        new_fset[new_fname_dict[ftuple]] = new_fset.merge_multiple_cat_columns(ftuple)

    print fdrop_list
    if fdrop_list:
        new_fset = new_fset.drop(fdrop_list)

    return new_fset, mean_cv_scores

# test interaction of different features from a pool of features by using such features as the train set
def fset_random_comb_select(mylearner, myfset, ytrain, input_fpool_list = list(), input_fprob_list = list(), n_feature_comb = 5, max_tests = 400,\
        nfolds=5, randstate_list = SEED, score_func=roc_auc_score, cv_func = strat_cv_score, max_trial = 20, return_fname = False):
    if isinstance(input_fpool_list, slice):
        start = input_fpool_list.start
        stop = input_fpool_list.stop
        step = input_fpool_list.step
        input_fpool_list = range(start, stop, step)

    if not hasattr(input_fpool_list, '__iter__'):
        input_fpool_list = [input_fpool_list]

    n_feature = len(myfset.fname_list)
    
    if not input_fpool_list:
        input_fpool_list = range(n_feature)

    if isinstance(input_fpool_list[0], int):
        fpool_list = input_fpool_list
    elif isinstance(input_fpool_list[0], str):
        fpool_list = [myfset.fname_list.index(col) for col in input_fpool_list if col in myfset.fname_list]
    else:
        print 'unsupported indexing'
        return
    
    if not hasattr(randstate_list, '__iter__'):
        randstate_list = [randstate_list]
    
    if not fpool_list:
        fpool_list = range(n_feature)

    if len(input_fprob_list) != len(fpool_list):
        fprob_list = None
    else:
        fprob_list = np.array(input_fprob_list) / np.sum(np.array(input_fprob_list))

    fpool_Xtrain_list = [myfset.Xtrain[:, myfset.find_list[i]:myfset.find_list[i+1]] for i in fpool_list]
    fpool_Xtrain_dict = dict(zip(fpool_list, fpool_Xtrain_list))

    if scipy.sparse.issparse(myfset.Xtrain):
        hstack_func = scipy.sparse.hstack
    else:
        hstack_func = np.hstack

    num_randstate = len(randstate_list)

    test_cv_scores = np.zeros((nfolds, num_randstate), float)

    mean_cv_scores = list()
    f_comb_list = list()

    # dealing with the case where we want to check the importance of each single feature
    if n_feature_comb == 1:
        max_tests = len(fpool_list)

    iteration = 0
    trial = 0
    while  len(mean_cv_scores) < max_tests and trial < max_trial:
        if n_feature_comb == 1:
            cur_f_comb = [fpool_list[iteration]]
            iteration += 1
        else:
            cur_f_comb = list(np.sort(np.random.choice(fpool_list, n_feature_comb, replace=False, p = fprob_list)))

        if cur_f_comb in f_comb_list:
            trial += 1
            continue

        cur_Xtrain = hstack_func((fpool_Xtrain_dict[f] for f in cur_f_comb))

        for i in xrange(num_randstate):
            randstate = randstate_list[i]
            test_cv_scores[:, i] = cv_func(mylearner, cur_Xtrain, ytrain, nfolds, randstate, score_func)
        
        cur_mean_cv_score = np.mean(test_cv_scores)

        print [myfset.fname_list[f] for f in cur_f_comb], 'cv score: {0:f}'.format(cur_mean_cv_score)

        f_comb_list.append(cur_f_comb)
        mean_cv_scores.append(cur_mean_cv_score)

    if return_fname:
        fname_comb_list = [[myfset.fname_list[f] for f in f_comb] for f_comb in f_comb_list]
        result_list = zip(fname_comb_list, mean_cv_scores)
    else:
        result_list = zip(f_comb_list, mean_cv_scores)

    result_list.sort(key = lambda x: - x[1])

    return result_list

# test interaction of different features from a pool of features by shuffling these features
def fset_random_comb_shuffle(mylearner, myfset, ytrain, input_fpool_list = list(), input_fprob_list = list(), n_feature_comb = 5, max_tests = 400,\
        nfolds=5, randstate_list = SEED, score_func=roc_auc_score, cv_func = strat_cv_score, max_trial = 20, return_fname = False):
    if isinstance(input_fpool_list, slice):
        start = input_fpool_list.start
        stop = input_fpool_list.stop
        step = input_fpool_list.step
        input_fpool_list = range(start, stop, step)

    if not hasattr(input_fpool_list, '__iter__'):
        input_fpool_list = [input_fpool_list]
    
    n_feature = len(myfset.fname_list)

    if not input_fpool_list:
        input_fpool_list = range(n_feature)

    if isinstance(input_fpool_list[0], int):
        fpool_list = input_fpool_list
    elif isinstance(input_fpool_list[0], str):
        fpool_list = [myfset.fname_list.index(col) for col in input_fpool_list if col in myfset.fname_list]
    else:
        print 'unsupported indexing'
        return
    
    if not hasattr(randstate_list, '__iter__'):
        randstate_list = [randstate_list]
    

    if scipy.sparse.issparse(myfset.Xtrain):
        sparsify = True
    else:
        sparsify = False

    if len(input_fprob_list) != len(fpool_list):
        fprob_list = None
    else:
        fprob_list = np.array(input_fprob_list) / np.sum(np.array(input_fprob_list))

    n_train = myfset.Xtrain.shape[0]

    np.random.seed(randstate_list[0])

    num_randstate = len(randstate_list)

    test_cv_scores = np.zeros((nfolds, num_randstate), float)

    # run a benchmark on the original featureset
    for i in xrange(num_randstate):
        randstate = randstate_list[i]
        test_cv_scores[:, i] = cv_func(mylearner, myfset.Xtrain, ytrain, nfolds, randstate, score_func)    

    bench_mean_cv_score = np.mean(test_cv_scores)

    print 'benchmark score:{0:f}'.format(bench_mean_cv_score)

    mean_cv_drop_scores = list()
    f_comb_list = list()

    # dealing with the case where we want to check the importance of each single feature
    if n_feature_comb == 1:
        max_tests = len(fpool_list)

    iteration = 0
    trial = 0
    while  len(mean_cv_drop_scores) < max_tests and trial < max_trial:
        if n_feature_comb == 1:
            cur_f_comb = [fpool_list[iteration]]
            iteration += 1
        else:
            cur_f_comb = list(np.sort(np.random.choice(fpool_list, n_feature_comb, replace=False, p = fprob_list)))

        if cur_f_comb in f_comb_list:
            trial += 1
            continue

        for i in xrange(num_randstate):
            cur_Xtrain = myfset.Xtrain.copy()
            # shuffle the data in selected columns keep the
            shuffler = np.arange(n_train)
            np.random.shuffle(shuffler)

            for col in cur_f_comb:
                find_low = myfset.find_list[col]
                find_high = myfset.find_list[col+1]
                cur_Xtrain[:, find_low:find_high] = cur_Xtrain[shuffler, find_low:find_high]            

            randstate = randstate_list[i]
            test_cv_scores[:, i] = cv_func(mylearner, cur_Xtrain, ytrain, nfolds, randstate, score_func)
        
        cur_mean_cv_score = np.mean(test_cv_scores)
        cur_mean_cv_drop = bench_mean_cv_score - cur_mean_cv_score

        f_comb_list.append(cur_f_comb)
        mean_cv_drop_scores.append(cur_mean_cv_drop)

        print [myfset.fname_list[f] for f in cur_f_comb],', cur score: {0:f}, cv drop: {1:f}'.format(cur_mean_cv_score, cur_mean_cv_drop)

    if return_fname:
        fname_comb_list = [[myfset.fname_list[f] for f in f_comb] for f_comb in f_comb_list]
        result_list = zip(fname_comb_list, mean_cv_drop_scores)
    else:
        result_list = zip(f_comb_list, mean_cv_drop_scores)

    result_list.sort(key = lambda x: -x[1])

    return result_list


# selecting extra feature 
def random_select_feature(mylearner, bfset, myfset, ytrain, nfolds=5, randstate=SEED, score_func=roc_auc_score):
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
            test_cv_scores = cv_score(mylearner, good_Xtrain, ytrain, nfolds, randstate, score_func)
            cur_score = np.mean(test_cv_scores)
        else:
            for curf in permute:
                cur_Xtrain = hstack_func((good_Xtrain, ftrain_list[curf]))
                test_cv_scores = cv_score(mylearner, cur_Xtrain, ytrain, nfolds, randstate, score_func)
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


## fset version of basic operations, can access using index or name
def fset_show_basic_statics(myfset, input_col_list = list()):
    if isinstance(input_col_list, slice):
        start = input_col_list.start
        stop = input_col_list.stop
        step = input_col_list.step
        input_col_list = range(start, stop, step)    

    if not hasattr(input_col_list, '__iter__'):
        input_col_list = [input_col_list]

    n_feature = len(myfset.fname_list)
    if not input_col_list:
        input_col_list = range(n_feature)

    if isinstance(input_col_list[0], str):
        col_list = [myfset.fname_list.index(col) for col in input_col_list if col in myfset.fname_list]
    elif isinstance(input_col_list[0], int):
        col_list = [col for col in input_col_list if col in range(n_feature)]
    else:
        print 'indexing not supported'
        return

    if scipy.sparse.issparse(myfset.Xtrain):
        sparsify = True
    else:
        sparsify = False

    n_train = myfset.Xtrain.shape[0]
    n_test = myfset.Xtest.shape[0]
    entropy_vfunc = np.vectorize(lambda x : x * np.log(x))
    print 'Showing Statistics\n'
    for col in col_list:
        col_name = myfset.fname_list[col]
        col_type = myfset.ftype_list[col]
        find_low = myfset.find_list[col]
        find_up = myfset.find_list[col+1]
        if (find_up - find_low) > 1:
            oh_encoded = True
        else:
            oh_encoded = False

        train_data = myfset.Xtrain[:, find_low]
        test_data = myfset.Xtest[:, find_low]

        if sparsify:
            train_data = np.squeeze(train_data.toarray())
            test_data = np.squeeze(test_data.toarray())

        print 'Col:', col_name, 'type:', col_type
        if oh_encoded:
            print '(oh_encoded) unique:', find_up-find_low, 'skipping details'
            print
            continue

        num_unique = np.unique(train_data).shape[0]
        min_elem = np.amin(train_data)
        max_elem = np.amax(train_data)
        mean = np.mean(train_data)
        std = np.std(train_data)
        values, counts = np.unique(train_data, return_counts=True)
        ind = np.argmax(counts)
        prob = counts / n_train
        entropy = -np.sum(entropy_vfunc(prob))
        print 'train:'
        print 'unique:', num_unique, 'min:', min_elem, 'max:', max_elem, 'mean:', mean, 'std:', std
        print '(mode, count):({0:f}, {1:d})'.format(values[ind], counts[ind]), 'entropy:', entropy


        num_unique = np.unique(test_data).shape[0]
        min_elem = np.amin(test_data)
        max_elem = np.amax(test_data)
        mean = np.mean(test_data)
        std = np.std(test_data)
        values, counts = np.unique(test_data, return_counts=True)
        ind = np.argmax(counts)
        prob = counts / n_test
        entropy = -np.sum(entropy_vfunc(prob))
        print 'test:'
        print 'unique:', num_unique, 'min:', min_elem, 'max:', max_elem, 'mean:', mean, 'std:', std
        print '(mode, count):({0:f}, {1:d})'.format(values[ind], counts[ind]), 'entropy:', entropy

        all_data = np.hstack((train_data, test_data))
        num_unique = np.unique(all_data).shape[0]
        min_elem = np.amin(all_data)
        max_elem = np.amax(all_data)
        mean = np.mean(all_data)
        values, counts = np.unique(all_data, return_counts=True)
        ind = np.argmax(counts)
        prob = counts / (n_train + n_test)
        entropy = -np.sum(entropy_vfunc(prob))
        print 'all:'
        print 'unique:', num_unique, 'min:', min_elem, 'max:', max_elem, 'mean:', mean, 'std:', std
        print '(mode, count):({0:f}, {1:d})'.format(values[ind], counts[ind]), 'entropy:', entropy
        print

## fset version of basic operations, can access using index or name
def fset_test_lead_digit(myfset, input_col_list = list(), return_entropy = False):
    if isinstance(input_col_list, slice):
        start = input_col_list.start
        stop = input_col_list.stop
        step = input_col_list.step
        input_col_list = range(start, stop, step)    

    if not hasattr(input_col_list, '__iter__'):
        input_col_list = [input_col_list]

    n_feature = len(myfset.fname_list)
    if not input_col_list:
        input_col_list = range(n_feature)

    if isinstance(input_col_list[0], str):
        col_list = [myfset.fname_list.index(col) for col in input_col_list if col in myfset.fname_list]
    elif isinstance(input_col_list[0], int):
        col_list = [col for col in input_col_list if col in range(n_feature)]
    else:
        print 'indexing not supported'
        return

    if scipy.sparse.issparse(myfset.Xtrain):
        sparsify = True
    else:
        sparsify = False

    n_train = myfset.Xtrain.shape[0]
    n_test = myfset.Xtest.shape[0]
    lead_digit_vfunc = np.vectorize(lambda x : int(repr(abs(x)).rstrip('0')[0]))
    benford_prob =  np.log10(1.0 + (1.0/np.arange(1, 10)))

    # entropy_dict = np.ones((len(col_list)), float)
    entropy_info_list = []  
    print 'Testing leading digit\n'
    for icol in xrange(len(col_list)):
        col = col_list[icol]
        col_name = myfset.fname_list[col]
        col_type = myfset.ftype_list[col]
        find_low = myfset.find_list[col]
        find_up = myfset.find_list[col+1]
        if (find_up - find_low) > 1:
            oh_encoded = True
        else:
            oh_encoded = False

        train_data = myfset.Xtrain[:, find_low]
        test_data = myfset.Xtest[:, find_low]

        if sparsify:
            train_data = np.squeeze(train_data.toarray())
            test_data = np.squeeze(test_data.toarray())

        print 'Col:', col_name, 'type:', col_type
        if oh_encoded:
            print '(oh_encoded) unique:', find_up-find_low, 'skipping'
            print
            continue

        all_data = np.hstack((train_data, test_data))

        lead_digit_train = lead_digit_vfunc(train_data)
        lead_digit_test = lead_digit_vfunc(test_data)
        lead_digit_all = lead_digit_vfunc(all_data)

        lead_counts_train = np.zeros(10)
        lead_counts_test = np.zeros(10)
        lead_counts_all = np.zeros(10) 

        unique_train, counts_train = np.unique(lead_digit_train, return_counts = True)
        unique_test, counts_test = np.unique(lead_digit_test, return_counts = True)
        unique_all, counts_all = np.unique(lead_digit_all, return_counts = True)

        for i in xrange(len(unique_train)):
            digit = unique_train[i]
            lead_counts_train[digit] = counts_train[i]

        for i in xrange(len(unique_test)):
            digit = unique_test[i]
            lead_counts_test[digit] = counts_test[i]

        for i in xrange(len(unique_all)):
            digit = unique_all[i]
            lead_counts_all[digit] = counts_all[i]

        lead_digit_prob_train = lead_counts_train[1:10] / np.sum(lead_counts_train[1:10])
        lead_digit_prob_test = lead_counts_test[1:10] / np.sum(lead_counts_test[1:10])
        lead_digit_prob_all = lead_counts_all[1:10] / np.sum(lead_counts_all[1:10])

        entropy_train = -np.dot(benford_prob, np.log(lead_digit_prob_train))
        entropy_test = -np.dot(benford_prob, np.log(lead_digit_prob_test))
        entropy_all = -np.dot(benford_prob, np.log(lead_digit_prob_all))

        min_entropy = -np.dot(benford_prob, np.log(benford_prob))

        dist_train = entropy_train - min_entropy
        dist_test = entropy_test - min_entropy
        dist_all = entropy_all - min_entropy

        p_zero_train = lead_counts_train[0] / n_train
        p_zero_test = lead_counts_test[0] / n_test
        p_zero_all = lead_counts_all[0] / (n_train + n_test)

        print 'train: 0-percent: {0:f}, cross entropy: {1:f}, dist: {2:f}'.format(p_zero_train, entropy_train, dist_train)
        print 'test: 0-percent: {0:f}, cross entropy: {1:f}, dist: {2:f}'.format(p_zero_test, entropy_test, dist_test)
        print 'all: 0-percent: {0:f}, cross entropy: {1:f}, dist: {2:f}'.format(p_zero_all, entropy_all, dist_all)
        print 
        # print zip(unique_all, counts_all)
        # print lead_counts_all
        # print lead_digit_prob_test


        entropy_info_list.append((col_name, dist_all))

    if return_entropy:
        return entropy_info_list


# merge two feature in feature set, now only support catergorical feature
def fset_check_two_columns(myfset, input_col1, input_col2):
    if isinstance(input_col1, int):
        col1 = input_col1
        c1_ind = myfset.find_list[col1]
        c1_name = myfset.fname_list[col1]
    elif isinstance(input_col1, str):
        col1 = myfset.fname_list.index(input_col1)
        c1_ind = myfset.find_list[col1]
        c1_name = myfset.fname_list[col1]

    if isinstance(input_col2, int):
        col2 = input_col2
        c2_ind = myfset.find_list[col2]
        c2_name = myfset.fname_list[col2]
    elif isinstance(input_col2, str):
        col2 = myfset.fname_list.index(input_col2)
        c2_ind = myfset.find_list[col2]
        c2_name = myfset.fname_list[col2]

    if scipy.sparse.issparse(myfset.Xtrain):
        sparsify = True
    else:
        sparsify = False

    print 'Checking {0:s} and {1:s}'.format(c1_name, c2_name)

    c1_data_train = myfset.Xtrain[:, c1_ind]
    c1_data_test = myfset.Xtest[:, c1_ind]
    c2_data_train = myfset.Xtrain[:, c2_ind]
    c2_data_test = myfset.Xtest[:, c2_ind]

    if sparsify:
        c1_data_train = np.squeeze(c1_data_train.toarray())
        c1_data_test = np.squeeze(c1_data_test.toarray())
        c2_data_train = np.squeeze(c2_data_train.toarray())
        c2_data_test = np.squeeze(c2_data_test.toarray())

    c1_data = np.hstack((c1_data_train, c1_data_test))
    c2_data = np.hstack((c2_data_train, c2_data_test))

    c1_c2_tuple = zip(c1_data, c2_data)

    num_unique_c1_c2 = len(set(c1_c2_tuple))
    num_unique_c1 = np.unique(c1_data).shape[0]
    num_unique_c2 = np.unique(c2_data).shape[0]

    print '{0:s}:'.format(c1_name), num_unique_c1, '{0:s}:'.format(c2_name), num_unique_c2, 'comb:', num_unique_c1_c2

# merge two feature in feature set, now only support catergorical feature
def fset_mutual_info(myfset, input_col1, input_col2):
    if isinstance(input_col1, int):
        col1 = input_col1
        c1_ind = myfset.find_list[col1]
        c1_name = myfset.fname_list[col1]
    elif isinstance(input_col1, str):
        col1 = myfset.fname_list.index(input_col1)
        c1_ind = myfset.find_list[col1]
        c1_name = myfset.fname_list[col1]

    if isinstance(input_col2, int):
        col2 = input_col2
        c2_ind = myfset.find_list[col2]
        c2_name = myfset.fname_list[col2]
    elif isinstance(input_col2, str):
        col2 = myfset.fname_list.index(input_col2)
        c2_ind = myfset.find_list[col2]
        c2_name = myfset.fname_list[col2]

    if scipy.sparse.issparse(myfset.Xtrain):
        sparsify = True
    else:
        sparsify = False

    print 'Checking {0:s} and {1:s}'.format(c1_name, c2_name)

    n_train = myfset.Xtrain.shape[0]
    n_test = myfset.Xtest.shape[0]

    c1_data_train = myfset.Xtrain[:, c1_ind]
    c1_data_test = myfset.Xtest[:, c1_ind]
    c2_data_train = myfset.Xtrain[:, c2_ind]
    c2_data_test = myfset.Xtest[:, c2_ind]

    if sparsify:
        c1_data_train = np.squeeze(c1_data_train.toarray())
        c1_data_test = np.squeeze(c1_data_test.toarray())
        c2_data_train = np.squeeze(c2_data_train.toarray())
        c2_data_test = np.squeeze(c2_data_test.toarray())

    c1_data = np.hstack((c1_data_train, c1_data_test))
    c2_data = np.hstack((c2_data_train, c2_data_test))

    c1_count = np.unique(c1_data, return_counts = True)[1]
    c2_count = np.unique(c2_data, return_counts = True)[1]

    c1_prob = c1_count / (n_train + n_test)
    c2_prob = c2_count / (n_train + n_test)

    c1_c2_tuple = zip(c1_data, c2_data)
    c1_c2_set = set(c1_c2_tuple)

    c1_c2_tuple_dict = dict()
    i = 0
    for c1_c2 in c1_c2_set:
        c1_c2_tuple_dict[c1_c2] = i
        i+=1

    c1_c2_data = np.zeros(n_train + n_test, np.int)
    for i in xrange(n_train + n_test):
        c1_c2_data[i] = c1_c2_tuple_dict[c1_c2_tuple[i]]

    c1_c2_count = np.unique(c1_c2_data, return_counts = True)[1]
    c1_c2_prob = c1_c2_count / (n_train + n_test)

    entropy_vfunc = np.vectorize(lambda x : x * np.log(x))

    c1_entropy = -np.sum(entropy_vfunc(c1_prob))
    c2_entropy = -np.sum(entropy_vfunc(c2_prob))
    c1_c2_entropy = -np.sum(entropy_vfunc(c1_c2_prob))

    m_info = c1_entropy + c2_entropy - c1_c2_entropy

    print '{0:s}:'.format(c1_name), c1_entropy, '{0:s}:'.format(c2_name), c2_entropy
    print 'comb:', c1_c2_entropy, 'mutual_info:', m_info, 'uncertainty:', 2 * m_info / (c1_entropy + c2_entropy)
    print 

    return 2 * m_info / (c1_entropy + c2_entropy)

def fset_remove_constant_col(myfset, input_col_list = list(), return_drop_list = False):
    if isinstance(input_col_list, slice):
        start = input_col_list.start
        stop = input_col_list.stop
        step = input_col_list.step
        input_col_list = range(start, stop, step)    

    if not hasattr(input_col_list, '__iter__'):
        input_col_list = [input_col_list]

    n_feature = len(myfset.fname_list)
    if not input_col_list:
        input_col_list = range(n_feature)

    if isinstance(input_col_list[0], str):
        col_list = [myfset.fname_list.index(col) for col in input_col_list if col in myfset.fname_list]
    elif isinstance(input_col_list[0], int):
        col_list = [col for col in input_col_list if col in range(n_feature)]
    else:
        print 'indexing not supported'
        return

    if scipy.sparse.issparse(myfset.Xtrain):
        sparsify = True
    else:
        sparsify = False

    n_train = myfset.Xtrain.shape[0]
    n_test = myfset.Xtest.shape[0]

    drop_list = list()
    for col in col_list:
        # col_name = myfset.fname_list[col]
        # col_type = myfset.ftype_list[col]
        find_low = myfset.find_list[col]
        find_up = myfset.find_list[col+1]
        if (find_up - find_low) > 1:
            continue

        train_data = myfset.Xtrain[:, find_low]
        test_data = myfset.Xtest[:, find_low]
    
        if sparsify:
            train_data = np.squeeze(train_data.toarray())
            test_data = np.squeeze(test_data.toarray())

        all_data = np.hstack((train_data, test_data))

        if np.std(all_data) == 0.0:
            drop_list.append(col)

    print [myfset.fname_list[i] for i in drop_list]

    new_fset = myfset.drop(drop_list)

    if not return_drop_list:
        return new_fset
    else:
        return new_fset, drop_list

def fset_remove_identical_col(myfset, input_col_list = list(), return_drop_list = False):
    if isinstance(input_col_list, slice):
        start = input_col_list.start
        stop = input_col_list.stop
        step = input_col_list.step
        input_col_list = range(start, stop, step)    

    if not hasattr(input_col_list, '__iter__'):
        input_col_list = [input_col_list]

    n_feature = len(myfset.fname_list)
    if not input_col_list:
        input_col_list = range(n_feature)

    if isinstance(input_col_list[0], str):
        col_list = [myfset.fname_list.index(col) for col in input_col_list if col in myfset.fname_list]
    elif isinstance(input_col_list[0], int):
        col_list = [col for col in input_col_list if col in range(n_feature)]
    else:
        print 'indexing not supported'
        return

    if scipy.sparse.issparse(myfset.Xtrain):
        sparsify = True
    else:
        sparsify = False

    n_train = myfset.Xtrain.shape[0]
    n_test = myfset.Xtest.shape[0]

    n_col = len(col_list)

    drop_list = list()
    for i in xrange(n_col):
        col_i = col_list[i]
        # col_name = myfset.fname_list[col]
        # col_type = myfset.ftype_list[col]
        col_i_find_low = myfset.find_list[col_i]
        col_i_find_up = myfset.find_list[col_i+1]
        if (col_i_find_up - col_i_find_low) > 1:
            continue

        col_i_train_data = myfset.Xtrain[:, col_i_find_low]
        col_i_test_data = myfset.Xtest[:, col_i_find_low]
    
        if sparsify:
            col_i_train_data = np.squeeze(col_i_train_data.toarray())
            col_i_test_data = np.squeeze(col_i_test_data.toarray())

        col_i_all_data = np.hstack((col_i_train_data, col_i_test_data))
        for j in xrange(i + 1, n_col):
            col_j = col_list[j]
            # col_name = myfset.fname_list[col]
            # col_type = myfset.ftype_list[col]
            col_j_find_low = myfset.find_list[col_j]
            col_j_find_up = myfset.find_list[col_j+1]
            if (col_j_find_up - col_j_find_low) > 1:
                continue

            col_j_train_data = myfset.Xtrain[:, col_j_find_low]
            col_j_test_data = myfset.Xtest[:, col_j_find_low]
        
            if sparsify:
                col_j_train_data = np.squeeze(col_j_train_data.toarray())
                col_j_test_data = np.squeeze(col_j_test_data.toarray())

            col_j_all_data = np.hstack((col_j_train_data, col_j_test_data))

            if np.array_equal(col_i_all_data, col_j_all_data) and (col_j not in drop_list):
                print [myfset.fname_list[col_i], myfset.fname_list[col_j]]
                drop_list.append(col_j)

    new_fset = myfset.drop(drop_list)

    if not return_drop_list:
        return new_fset
    else:
        return new_fset, drop_list


def fset_merge_multiple_cat_columns(myfset, input_col_multiple, hasher=None):
    if isinstance(input_col_multiple, slice):
        start = input_col_multiple.start
        stop = input_col_multiple.stop
        step = input_col_multiple.step
        input_col_multiple = range(start, stop, step)

    if not hasattr(input_col_multiple, '__iter__'):
        input_col_multiple = [input_col_multiple]    

    n_col = len(input_col_multiple)
    if n_col <= 1:
        return

    if isinstance(input_col_multiple[0], str):
        col_multiple = [myfset.fname_list.index(col) for col in input_col_multiple if col in myfset.fname_list]
    elif isinstance(input_col_multiple[0], int):
        n_feature = len(myfset.fname_list)
        col_multiple = [col for col in input_col_multiple if col in range(n_feature)]
    else:
        print 'indexing not supported'
        return

    if len(col_multiple) < len(input_col_multiple):
        print 'some feature not found'
        return

    if scipy.sparse.issparse(myfset.Xtrain):
        sparsify = True
    else:
        sparsify = False

    if sparsify:
        vstack_func = scipy.sparse.vstack
    else:
        vstack_func = np.vstack

    pre_col_list = list(col_multiple)
    col_list = [col for col in pre_col_list if (myfset.find_list[col+1] - myfset.find_list[col]) == 1]
    col_ind = list(np.array(myfset.find_list)[col_list])

    X_merge = vstack_func((myfset.Xtrain[:,col_ind], myfset.Xtest[:, col_ind]))

    if sparsify:
        X_merge = X_merge.toarray()

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