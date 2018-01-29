#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 12:41:54 2018

@author: zc223
"""

import numpy as np
import tensorflow as tf

class ExactInference:

    def __init__(self, x, y,
                 mean_func,
                 cov_func,
                 lik_func,):


