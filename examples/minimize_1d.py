"""
This is the simplest possible, brute force test of the informative learning
scheme.

Author:
    Ilias Bilionis

Date:
    10/08/2014
    01/29/2015
    10/13/2015
"""


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pybgo
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def f(x):
    """
    A 1D function to be minimized
    """
    return 4. * (1. - np.sin((x + 8. * np.exp(x - 7.)) * 2.))


# The number of design points over which we are looking for the minimum
num_design = 100

# The number design points used to start the algorithm
num_init = 6

# The folder on which we save the results
out_dir = 'minimize_1d_res'

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
X_design = np.linspace(0, 6., num_design)[:, None]

# The initial points to start from
#X_init = np.linspace(0, 6., num_init)[:, None]
X_init = np.random.rand(num_init)[:, None] * 6.

# Globally minimize f
x, y, ei, _ = pybgo.minimize(f, X_init, X_design, tol=1e-5,
                          callback=pybgo.plot_summary,      # This plots the results
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
