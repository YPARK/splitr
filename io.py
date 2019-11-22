import sys
import os
import zstd
from sp.sparse import csr_matrix

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
