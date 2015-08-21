"""
A toy 2D function.

Author:
    Ilias Bilionis

Date:
    10/16/2014

"""

import pydes
import numpy as np
import scipy
import math
import GPy
import matplotlib.pyplot as plt
import seaborn as sns


def f(x):
    """
    A 1D function to look at.
    """
    return 4. * (1. - np.sin((x + 8. * np.exp(x - 7.)) * 2.)) + np.random.randn(x.shape[0], 1)

#np.random.seed(3145252)

# Just for plotting
X_design = np.linspace(0, 6., 100)[:, None]

# The initial points to start from
X_init = np.random.rand(100)[:, None] * 6.

Y_init = f(X_init)

#optimizer = pydes.GlobalOptimizer(X_init, X_design, f)
#optimizer.optimize(max_it=100, num_samples=10, burn=1, thin=1, num_restarts=10)

#quit()

# Train a GP model
k = GPy.kern.RBF(1, ARD=True)
model = GPy.models.GPRegression(X_init, Y_init, k)
model.rbf.variance.set_prior(pydes.JeffreysPrior())
model.rbf.lengthscale.set_prior(pydes.LogLogisticPrior())
model.likelihood.variance.set_prior(pydes.JeffreysPrior())

print str(model)

me = pydes.ModelEnsemble.train(model, num_samples=100, thin=1, burn=1, hmc_iters=20, num_restarts=10)
sns.kdeplot(me.param_list[:, 0])
sns.kdeplot(me.param_list[:, 1])
sns.kdeplot(me.param_list[:, 2])
plt.show()
a = raw_input('press_enter')

# Predict
#Y, V = me.raw_predict(X_design)
#Y = me.posterior_samples(X_design, size=1)
ei = me.expected_improvement(X_design)
#q = me.predict_quantiles(X_design, size=1000)

#ei = pydes.expected_improvement(X_design, model, mode='max')

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
#model.optimize()
#model.plot(ax=ax1, plot_limits=[0, 6])
eim = pydes.expected_improvement(X_design, model)[0]
ax1.plot(X_init, Y_init, 'kx', markersize=10, markeredgewidth=2)
ax1.plot(X_design, f(X_design), 'r', linewidth=2)
ax1.plot(X_design, me.posterior_mean_samples(X_design).T, 'g', linewidth=1)
#ax1.plot(X_design, q[0, :], 'g', linewidth=2)
#ax1.fill_between(X_design.flatten(), q[1, :], q[2, :], color='green', alpha=0.25)
#ax2.plot(X_design, ei, '--m', linewidth=2)
ax2.plot(X_design, eim, '-.b', linewidth=2)
plt.show()
a = raw_input('press enter')

#X_init = np.random.rand(10, 2) * (domain[:, 1] - domain[:, 0]) + domain[:, 0]


# Globally minimize f
#minimize(f, X_init, X_design, prefix='branin_function', max_it=100,
#         callback=plot_summary_2d, tol=1e-4)
