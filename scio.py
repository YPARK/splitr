# -*- coding: utf-8 -*-

import sys
import os
import os.path
import scipy as sp
import numpy as np
import shlex
import subprocess
import mmutil as mm
import gzip
from scipy.sparse import csr_matrix

sys.path.insert(1, os.path.dirname(__file__))

from util import _log_msg

__all__ = ["read_mtx_file",
           "save_array",
           "save_list"]

def read_mtx_file(filename):
    return mm.read_triplets_numpy(filename)

def save_array(arr, fname):
    assert(isinstance(arr, np.ndarray))
    assert(len(arr.shape) == 2)
    _ = mm.write_numpy(arr, fname)
    return

def save_list(ll, fname):

    if fname.endswith('.gz'):
        fh = gzip.open(fname, 'wb')
    else:
        fh = open(fname, 'wb')

    _out = str.encode('\n'.join(map(lambda x: '%.4f'%x, ll)))
    fh.write(_out)
    _log_msg("Wrote %s"%fname)
    fh.close()

    return
