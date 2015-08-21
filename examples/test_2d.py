"""
This is the simplest possible, brute force test of the informative learning
scheme.

Author:
    Ilias Bilionis

Date:
    10/08/2014
    01/29/2015

"""


from pydes import *
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    """
    A 1D function to look at.
    """
    return x[0] ** 2 + x[1]

#np.random.seed(3145252)

# Just for plotting
#X_design = np.linspace(0, 6., 100)[:, None]
X_design = np.random.rand(1000, 2)

# The initial points to start from
#X_init = np.random.rand(1)[:, None] * 6.
X_init = np.random.rand(10, 2)

# Globally minimize f
x_fb, y_fb = minimize(f, X_init, X_design, tol=1e-3, callback=plot_summary_2d,
         Gamma=fb_expected_improvement, Gamma_name='FB-EI', 
         prefix='examples/minimize_fb')
pbs_fb = np.array([y_fb[:i, 0].min() for i in xrange(1, y_fb.shape[0])])
x, y = minimize(f, X_init, X_design, tol=1e-3, callback=plot_summary,
         Gamma=expected_improvement, Gamma_name='EI', 
         prefix='examples/minimize_ei')
pbs = np.array([y[:i, 0].min() for i in xrange(1, y.shape[0])])
import matplotlib.pyplot as plt
s = np.arange(1, pbs.shape[0] + 1)
plt.plot(s, pbs, 'bo-', linewidth=2, markersize=10, markeredgewidth=2, label='EI')
s_fb = np.arange(1, pbs_fb.shape[0] + 1)
plt.plot(s_fb, pbs_fb, 'gx--', linewidth=2, markersize=10, markeredgewidth=2, label='FB-EI')
plt.legend()
plt.savefig('examples/minimize_pbs.png')
#a = input('press key')
