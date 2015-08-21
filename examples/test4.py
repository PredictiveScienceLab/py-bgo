"""
A toy 2D function.

Author:
    Ilias Bilionis

Date:
    10/16/2014

"""


from core import *
import numpy as np
import scipy
import math



def f(x):
    """
    A toy 2D function.
    """
    return np.log((x[1] - 5.1 / 4 / math.pi ** 2 * x[0] ** 2 + 5 / math.pi * x[0] - 6.) ** 2.
                  + 10. * (1. - 1. / 8. / math.pi) * np.cos(x[0]) + 10.)


# The domain of the problem
domain = np.array([[-5., 10.],
                   [0., 15.]])



# Solve the problem with Sequential Least Squares Programming
f_minus = lambda(x): -f(x)
bounds = tuple((domain[i, 0], domain[i, 1]) for i in range(2))
res = scipy.optimize.minimize(f_minus, np.mean(domain, axis=0), method='SLSQP',
                              bounds=bounds)

# The design points
n_design = 64
tmp = (np.linspace(domain[i, 0], domain[i, 1], n_design) for i in range(2))
X1, X2 = np.meshgrid(*tmp)
X_design = np.hstack((t.flatten()[:, None] for t in (X1, X2)))
y_design = np.array([f(X_design[i,:]) for i in xrange(X_design.shape[0])])
Y_design = y_design.reshape(X1.shape)

# The initial points to start from
n_init = 8
tmp = (np.linspace(domain[i, 0], domain[i, 1], n_init) for i in range(2))
X1, X2 = np.meshgrid(*tmp)
X_init = np.hstack((t.flatten()[:, None] for t in (X1, X2)))
#X_init = np.random.rand(10, 2) * (domain[:, 1] - domain[:, 0]) + domain[:, 0]


# Globally minimize f
minimize(f, X_init, X_design, prefix='branin_function', max_it=100,
         callback=plot_summary_2d, tol=1e-4)
