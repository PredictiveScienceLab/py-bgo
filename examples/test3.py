"""
Pressure Vessel Problem

Author:
    Ilias Bilionis

Date:
    10/16/2014

"""


from core import *
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt


def W(x):
    """
    The weight of the vessel.
    """
    # Density (lb / ft ** 3)
    rho = 500.
    # Thickness (in)
    T = 0.5
    # The radius (in)
    R = x[0]
    # The length (in)
    L = x[1]
    # The weight is
    W = math.pi * rho * (4. / 3. * (R + T) ** 3 + (R + T) ** 2 * L
                         - (4. / 3. * R ** 3 + R ** 2 * L))
    return W


def V(x):
    """
    The volume of the vessel.
    """
    # The radius (in)
    R = x[0]
    # The length (in)
    L = x[1]
    # The enclosed volume is
    V = math.pi * (4. / 3. * R ** 3 + R ** 2 * L)
    return V


def f(x):
    """
    The Pressure Vessel Problem objective function
    """
    # Density (lb / ft ** 3)
    rho = 500.
    # Thickness (in)
    T = 0.5
    # The radius (in)
    R = x[0]
    # The length (in)
    L = x[1]
    # The enclosed volume is
    V = math.pi * (4. / 3. * R ** 3 + R ** 2 * L)
    # The weight is
    W = math.pi * rho * (4. / 3. * (R + T) ** 3 + (R + T) ** 2 * L
                         - (4. / 3. * R ** 3 + R ** 2 * L))
    # Single-attribute utility functions:
    Wmax = 12100105.9047
    Vmax = 765442.766862
    UW = 1. - (W / Wmax) ** 2
    UV = (V / Vmax) ** 2
    return 0.52 * UW + 0.48 * UV


# The domain of the problem
domain = np.array([[0.1, 36.],
                   [0.1, 140.]])



# Solve the problem with Sequential Least Squares Programming
f_minus = lambda(x): -f(x)
bounds = tuple((domain[i, 0], domain[i, 1]) for i in range(2))
constraints = ({'type': 'ineq', 'fun': lambda x: x[0] - 5 * 0.5},
               {'type': 'ineq', 'fun': lambda x: 40. - x[0] - 0.5},
               {'type': 'ineq', 'fun': lambda x: 150. - x[1] - 2. * x[0] - 2. * 0.5})
res = scipy.optimize.minimize(f_minus, np.mean(domain, axis=0), method='SLSQP',
                              bounds=bounds,
                              constraints=constraints)

# The design points
tmp = (np.linspace(domain[i, 0], domain[i, 1], 16) for i in range(2))
X1, X2 = np.meshgrid(*tmp)
X_design = np.hstack((t.flatten()[:, None] for t in (X1, X2)))
y_design = np.array([f(X_design[i,:]) for i in xrange(X_design.shape[0])])
Y_design = y_design.reshape(X1.shape)
plt.contourf(X1, X2, Y_design)
plt.xlabel('R (in)', fontsize=16)
plt.ylabel('L (in)', fontsize=16)
plt.savefig('pressure_vessel_response.png')
plt.clf()
quit()

# The initial points to start from
X_init = np.vstack((np.random.rand(10) * (domain[i, 1] - domain[i, 0]) + domain[i, 0]
                    for i in range(2)))
#X_init = np.hstack([np.random.rand(10) * (36. - 1.) + 10

# Globally minimize f
#maximize(f, X_init, X_design, prefix='pressure_vessel')
