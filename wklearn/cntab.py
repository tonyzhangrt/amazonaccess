from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
import cPickle as pickle


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

    def df_one_degree_counts(self, df, col_i, file_path = None):
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