from __future__ import absolute_import

import os
import cPickle as pickle

class MyPickleHelper(object):
    def __init__(self):
        self._content = None
        self._file_path = None

    def dump(self, content):
        self._content = content

    def fetch(self, file_path = None):
        if not file_path:
            file_path = self._file_path
        if file_path:
            self.load(file_path)
        return self._content


    def load(self, file_path = None):
        if not file_path:
            file_path = self._file_path
        if file_path:
            try:
                with open( file_path, 'rb') as f:
                    self._content = pickle.load(f)
            except IOError:
                print 'Loading feature set file failed: file not found.'
        else:
            print 'Loading featue set file failed: file not saved.'

    def save(self, file_path):
        if not self._content:
            print 'No content to be saved'
            return
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_path, 'wb') as f:
            pickle.dump(self._content, f, pickle.HIGHEST_PROTOCOL)
        self._file_path = file_path