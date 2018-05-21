"""
Created on Tue Jan 23 17:33:47 2018

@author: zc223
"""
from .sq_dist import sq_dist

from .util import tri_vec_shape
from .util import vec_to_tri
from .util import ceil_divide
from .util import log_cholesky_det
from .util import mul_sum
from .util import mat_square
from .util import matmul_br
from .util import cholesky_solve_br
from .util import broadcast
from . import plot
from .metrics import init_metrics
from .metrics import update_metrics
from .metrics import record_metrics
from .plot_classification import Classification
from .train_helper import construct_from_flags, post_training
