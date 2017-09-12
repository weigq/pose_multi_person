from __future__ import absolute_import

import os
import shutil
import torch 
import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    lr = lr * gamma ** (epoch / schedule)
    for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr