# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 18:56:08 2018

@author: zc223
"""

import numpy as np
import tensorflow as tf

from . import cov
from . import mean
from . import lik
from . import inf
from . import util


class GaussianProcess:

    def __init__(self,
                 cov_func,
                 mean_func,
                 inf_func,
                 lik_func):

        self.cov = cov_func
        self.mean = mean_func
        self.inf = inf_func
        self.lik = lik_func
