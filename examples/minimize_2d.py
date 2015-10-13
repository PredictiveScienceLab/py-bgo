"""
A toy 2D function.

Author:
    Ilias Bilionis

Date:
    10/16/2014
    10/13/2015
"""


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pybgo
import shutil
import numpy as np
import design
import math
import matplotlib.pyplot as plt
import seaborn as sns


def f(x):
    """
    A toy 2D function.
    """
    return np.log((x[1] - 5.1 / 4 / math.pi ** 2 * x[0] ** 2 + 5 / math.pi * x[0] - 6.) ** 2.
                  + 10. * (1. - 1. / 8. / math.pi) * np.cos(x[0]) + 10.)


# The domain of the problem
domain = np.array([[-5., 10.],
                   [0., 15.]])


# The number of design points over which we are looking for the minimum
num_design = 20 # Per input dimension

# The number design points used to start the algorithm
num_init = 20

# The folder on which we save the results
out_dir = 'minimize_2d_res'

# Fix the random seed to ensure reproducibility
seed = 3141569
np.random.seed(seed)

print '=' * 80
print 'Bayesian Global Optimization 1D Example.'.center(80)
print 'Parameters'
print '-' * 80
print '+ {0:20s}: {1:d}'.format('random seed', seed)
print '+ {0:20s}: {1:d}'.format('num. init', num_init)
print '+ {0:20s}: {1:d}'.format('num. design', num_design)
print '+ {0:20s}: {1:s}'.format('output directory', out_dir)
print '=' * 80
print '+ starting...'
if os.path.isdir(out_dir):
    print '+ removing existing', out_dir
    shutil.rmtree(out_dir)
print '+ making', out_dir
os.makedirs(out_dir)

# We are looking for the minimum over these points
# Use something like the following for a generic problem
#X_design = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * design.latin_center(num_design, 2) 
# For this one we use a regular grid only because we want to do some contour
# plots
x1 = np.linspace(domain[0, 0], domain[0, 1], num_design)
x2 = np.linspace(domain[1, 0], domain[1, 1], num_design)
X1, X2= np.meshgrid(x1, x2)
X_design = np.hstack([X1.flatten()[:, None], X2.flatten()[:, None]])

# The initial points to start from
X_init = np.random.rand(num_init)[:, None] * 6.
X_init = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * design.latin_center(num_init, 2) 

# Globally minimize f
x, y, ei, _ = pybgo.minimize(f, X_init, X_design, tol=1e-5,
                          callback=pybgo.plot_summary_2d,      # This plots the results
                                                            # at each iteration of
                                                            # the algorithm
                          prefix=os.path.join(out_dir, 'out'),
                          save_model=True)

# The best value at each iteration
bv = np.array([y[:i, 0].min() for i in xrange(1, y.shape[0])])

fig, ax = plt.subplots()
it = np.arange(1, bv.shape[0] + 1)
ax.plot(it, bv, linewidth=2)
ax.set_xlabel('Iteration', fontsize=16)
ax.set_ylabel('Best value', fontsize=16)
fig.savefig(os.path.join(out_dir, 'bv.png'))
plt.close(fig)

fig, ax = plt.subplots()
it = np.arange(1, len(ei) + 1)
ax.plot(it, ei, linewidth=2)
ax.set_xlabel('Iteration', fontsize=16)
ax.set_ylabel('Expected improvement', fontsize=16)
fig.savefig(os.path.join(out_dir, 'ei.png'))
plt.close(fig)
