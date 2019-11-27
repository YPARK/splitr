import sys
import os
import os.path
import zstd
import scipy as sp
import numpy as np
import shlex
import subprocess
from scipy.sparse import csr_matrix

sys.path.insert(1, os.path.dirname(__file__))

from util import _log_msg

def read_mtx_zstd(mtxFile, **kwargs):
    """
Read Matrix Market triplets compressed by Zstandard
"""

    verbose = kwargs.get('verbose',True)

    fh = open(mtxFile, 'rb')
    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(fh)

    wrap = io.BufferedReader(reader)
    line = wrap.readline()

    _log_msg('Start reading the file: %s'%mtxFile)

    _check, _, _format, _, sym = [x.strip().decode('utf-8') for x in line.split()]

    while line.startswith(b'%'):
        line = wrap.readline()

    _nrow, _ncol, _nelem = [int(x.strip()) for x in line.split()]

    _log_msg('[%d x %d]'%(_nrow, _ncol))

    m = 0

    ii = np.zeros(_nelem, dtype=np.intc)
    jj = np.zeros(_nelem, dtype=np.intc)
    vv = np.zeros(_nelem, dtype=np.float32)

    while True:
        line = wrap.readline()

        if len(line) < 1:
            break

        larr = line.split()

        if len(larr) is not 3:
            _log_msg('corrupted in line %d'%(m+1))
            break

        ii[m] = int(larr[0])
        jj[m] = int(larr[1])
        vv[m] = float(larr[2])
        m += 1
        if verbose is True:
            if m%1000000 is 999999:
                _log_msg('%10d/%10d'%(m+1, _nelem))

    fh.close()
    assert(m == _nelem)

    _log_msg('Finished reading the file: %s'%mtxFile)

    ii -= 1
    jj -= 1

    ret = csr_matrix((vv, (ii, jj)),
                     shape=(_nrow, _ncol),
                     dtype=np.float32)

    return ret


def read_mtx_cmd(cmd, **kwargs):
    """
    Read a matrix market format by shell command
    """

    verbose = kwargs.get('verbose',True)

    cmd = shlex.split(cmd)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    _log_msg('Start reading the file: %s'%cmd)

    line = proc.stdout.readline()

    _check, _, _format, _, sym = [x.strip().decode('utf-8') for x in line.split()]

    if(not _check.startswith('%%MatrixMarket')):
        raise Exception("Unsupported file type")

    if(_format != 'coordinate'):
        raise Exception("Unsupported file format")

    # skip comment lines
    while line.startswith(b'%'):
        line = proc.stdout.readline()

    _nrow, _ncol, _nelem = [int(x.strip()) for x in line.split()]

    _log_msg('[%d x %d]'%(_nrow, _ncol))

    m = 0

    ii = np.zeros(_nelem, dtype=np.intc)
    jj = np.zeros(_nelem, dtype=np.intc)
    vv = np.zeros(_nelem, dtype=np.float32)

    while True:
        line = proc.stdout.readline()

        if line == '' and proc.poll() is not None:
            break

        if len(line) < 1: break

        larr = line.split()

        if len(larr) is not 3:
            _log_msg('corrupted in line %d'%(m+1))
            break

        ii[m] = int(larr[0])
        jj[m] = int(larr[1])
        vv[m] = float(larr[2])
        m += 1
        if verbose is True:
            if m%100000 is 99999:
                _log_msg('%10d/%10d'%(m+1, _nelem))

    if(m != _nelem):
        raise Exception("Found a different number of elements")

    _log_msg('Finished reading the file: %s'%(' '.join(cmd)))

    ii -= 1
    jj -= 1

    ret = csr_matrix((vv, (ii, jj)),
                     shape=(_nrow, _ncol),
                     dtype=np.float32)

    proc.terminate()
    return ret

def save_array(arr, fname, zstd=True):

    assert(isinstance(arr, np.ndarray))

    if len(fname) > 4 and fname[-4:] == ".zst":
        fname = fname[:-4]
        zstd = True

    np.savetxt(
        fname=fname,
        X=arr,
        fmt="%.4f",
        delimiter="\t"
    )

    if zstd :
        cmd = "zstd -f %s"%(fname)
        out = fname + ".zst"
        proc = subprocess.call(cmd, shell=True)
        if os.path.exists(out):
            os.remove(fname)

    return
