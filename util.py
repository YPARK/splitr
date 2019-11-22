import io
import sys
import datetime

import numpy as np
import scipy as sp

from sklearn import preprocessing

def filter_zero_rows_cols(X, p_cells_cutoff = .99, p_genes_cutoff = .99, eps = 1e-8):
    """
    Filter cells (rows) and genes (columns) that contain too many zeros

    returns _X, _cells, _genes
    """

    ret = X.copy()

    observed_cells = np.mean(ret > eps, axis=1)
    _cells = np.where(observed_cells > (1.0 - p_cells_cutoff))[0]

    ret = ret[_cells, :]

    observed_genes = np.mean(ret.T > eps, axis=1)
    _genes = np.where(observed_genes > (1.0 - p_genes_cutoff))[0]

    _log_msg("Valid genes: %05d"%len(_genes))

    ret = ret[:, _genes]

    return ret, _cells, _genes

def normalize_X(_X, _target=None):
    """
    Normalization (following scanpy)
    """
    _per_cell = _X.sum(1)
    _target = (
        np.median(_per_cell[_per_cell > 0])
        if _target is None else _target
    )
    _per_cell += (_per_cell == 0)
    _per_cell /= _target.T

    ret = _X.copy()
    ret /= _per_cell

    return ret

def standardize_NB_X(xx, min_val : float = -4.0, max_val : float = 4.0):
    """
    Scale NB values
    """
    xx = np.log(1.0 + xx)
    ss = preprocessing.StandardScaler().fit(xx)
    xx = ss.transform(xx)
    xx[xx < min_val] = min_val
    xx[xx > max_val] = max_val
    return np.exp(xx)

#########
# misc. #
#########

def _log_msg(msg):
    """
    just print out message with time
    """
    tt = datetime.datetime.now()
    sys.stderr.write("[%s] %s\n"%(tt.strftime("%Y-%m-%d %H:%M:%S"), msg))
    sys.stderr.flush()
