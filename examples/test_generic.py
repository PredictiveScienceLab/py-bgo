"""
A toy 2D function.

Author:
    Ilias Bilionis

Date:
    10/16/2014

"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pybgo
import numpy as np
import scipy
import math
import GPy
import matplotlib.pyplot as plt
import seaborn as sns
from design import latin_center


def f(x, sigma=0.):
    """
    A 1D function to look at.
    """
    return 4. * (1. - np.exp(-0.1 * x) * np.sin((x + 8. * np.exp(x - 7.)))) + \
           sigma * np.random.randn(*x.shape)

np.random.seed(3145252)

# The objective
objective = lambda(x): f(x, sigma=1.5)

# Just for plotting
X_design = np.linspace(0, 6., 100)[:, None]

# The initial points to start from
X_init = latin_center(5, 1) * 6.

Y_init = f(X_init)

optimizer = pybgo.GlobalOptimizer(X_init, X_design, objective,
                                  true_func=f)
optimizer.optimize()
