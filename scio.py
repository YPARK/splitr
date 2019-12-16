# -*- coding: utf-8 -*-

import sys
import os
import os.path
import scipy as sp
import numpy as np
import shlex
import subprocess
import mmutil as mm
from scipy.sparse import csr_matrix

sys.path.insert(1, os.path.dirname(__file__))

from util import _log_msg

__all__ = ["read_mtx_file",
           "save_array",
           "save_list"]

def read_mtx_file(filename):
    return mm.read_numpy(filename)

def save_array(arr, fname):

    assert(isinstance(arr, np.ndarray))

    do_zstd = False
    do_gzip = False

    if len(fname) > 4 and fname[-4:] == ".zst":
        fname = fname[:-4]
        do_zstd = True

    if len(fname) > 3 and fname[-4:] == ".gz":
        fname = fname[:-3]
        do_gzip = True

    fmt = "%d" if(arr.dtype == np.int) else "%.4f"

    np.savetxt(
        fname=fname,
        X=arr,
        fmt=fmt,
        delimiter="\t"
    )

    if do_zstd :
        cmd = "zstd -f %s"%(fname)
        out = fname + ".zst"
        proc = subprocess.call(cmd, shell=True)
        if os.path.exists(out):
            os.remove(fname)

    if do_gzip :
        cmd = "gzip %s"%(fname)
        out = fname + ".gz"
        proc = subprocess.call(cmd, shell=True)
        if os.path.exists(out):
            os.remove(fname)

    return

def save_list(ll, fname):
    with open(fname, 'w') as fh:
        _out = '\n'.join(map(lambda x: '%.4f'%x, ll))
        fh.write(_out)
        _log_msg("Wrote %s"%fname)

    return
