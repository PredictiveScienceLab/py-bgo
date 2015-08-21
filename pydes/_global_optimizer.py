"""
A myopic global optimizer class.

Author:
    Ilias Bilionis

Date:
    5/1/2015

"""


__all__ = ['GlobalOptimizer']


import numpy as np
from collections import Iterable
import math
import GPy
import matplotlib.pyplot as plt 
import seaborn
from . import expected_improvement
from . import ModelEnsemble


class GlobalOptimizer(object):

    """
    A global optimizer class.

    It is essentially a myopic, sequential, global optimizer.

    :param func: The function you wish to optimize.
    :arapm args: Additional arguments for the function we optimize.
    :param afunc: The acquisition function you wish to use.
    :param afunc_args: Extra arguments to the optimization function.

    """

    # The initial design
    _X_init = None

    # The initial observations
    _Y_init = None

    # The total design we have available
    _X_design = None

    # The indexes of the observations we have made so far (list of integers)
    _idx_X_obs = None

    # The objectives we have observed so far (list of whatever the observations are)
    _Y_obs = None

    # The function we wish to optimize
    _func = None

    # Extra arguments to func
    _args = None

    # The acquisition function we are going to use
    _acquisition_function = None

    # Extra arguments to the acquisition function
    _af_args = None

    @property 
    def X_init(self):
        """
        :getter: Get the initial design.
        """
        return self._X_init

    @X_init.setter
    def X_init(self, value):
        """
        :setter: Set the initial design.
        """
        assert isinstance(value, Iterable)
        self._X_init = value

    @property
    def Y_init(self):
        """
        :getter: Get the initial observations.
        """
        return self._Y_init

    @Y_init.setter
    def Y_init(self, value):
        """
        :setter: Set the initial observations.
        """
        if value is not None:
            assert isinstance(value, Iterable)
            value = np.array(value)
        self._Y_init = value

    @property 
    def X_design(self):
        """
        :getter: Get the design.
        """
        return self._X_design

    @X_design.setter
    def X_design(self, value):
        """
        :setter: Set the design.
        """
        assert isinstance(value, Iterable)
        self._X_design = value

    @property 
    def idx_X_obs(self):
        """
        :getter: The indexes of currently observed design points.
        """
        return self._idx_X_obs

    @property 
    def Y_obs(self):
        """
        :getter: The values of the currently observed design points.
        """
        return self._Y_obs

    @property 
    def func(self):
        """
        :getter: Get the function we are optimizing.
        """
        return self._func

    @func.setter
    def func(self, value):
        """
        :setter: Set the function we are optimizing.
        """
        assert hasattr(value, '__call__')
        self._func = value

    @property
    def args(self):
        """
        :getter: The extra arguments of func.
        """
        return self._args

    @property 
    def acquisition_function(self):
        """
        :getter: Get the acquisition function.
        """
        return self._acquisition_function

    @acquisition_function.setter
    def acquisition_function(self, value):
        """
        :setter: Set the acquisition function.
        """
        assert hasattr(value, '__call__')
        self._acquisition_function = value

    @property 
    def af_args(self):
        """
        :getter: The arguments of the acquisition function.
        """
        return self._af_args

    @property 
    def X(self):
        """
        :getter: Get all the currently observed points.
        """
        return np.vstack([self.X_init, self.X_design[self.idx_X_obs]])

    @property 
    def Y(self):
        """
        :getter: Get all the currently observed objectives.
        """
        if len(self.Y_obs) == 0:
            return np.array(self.Y_init)
        return np.vstack([self.Y_init, self.Y_obs])

    def __init__(self, X_init, X_design, func, args=(), Y_init=None,
                 af=expected_improvement, af_args=()):
        """
        Initialize the object.
        """
        self.X_init = X_init
        self.X_design = X_design
        self.Y_init = Y_init
        self.func = func
        self._args = args
        self.acquisition_function = af
        self._af_args = af_args
        self._idx_X_obs = []
        self._Y_obs = []

    def optimize(self, max_it=100, tol=1e-1, kernel=None, fixed_noise=None,
                 GPModelClass=GPy.models.GPRegression,
                 verbose=True,
                 add_at_least=10,
                 **kwargs):
        """
        Optimize the objective.
        """
        assert add_at_least >= 1
        if self.Y_init is None:
            self.Y_init = [self.func(x, *self.args) for x in self.X_init]
        if kernel is None:
            kernel = GPy.kern.RBF(self.X_init.shape[1], ARD=True)
            kernel.variance.set_prior(GPy.priors.Jeffreys())
            kernel.lengthscale.set_prior(GPy.priors.LogLogistic())
        # Acquisition function values at each iteration
        af_values = []
        for it in xrange(max_it):
            kernel = GPy.kern.RBF(self.X_init.shape[1], ARD=True)
            kernel.variance.set_prior(GPy.priors.LogGaussian(0., 1.))
            kernel.lengthscale.set_prior(GPy.priors.LogGaussian(math.log(1.), 1.))
            model = GPModelClass(self.X, self.Y, kernel)
            model._X_predict = self.X_design
            model.likelihood.variance.set_prior(GPy.priors.Jeffreys())
            if fixed_noise is not None:
                model.Gaussian_noise.variance.constrain_fixed(fixed_noise)
            model.pymc_trace_denoised_max()
            model.pymc_trace_denoised_argmax()
            model.pymc_trace_posterior_samples()
            model.pymc_trace_expected_improvement(denoised=True)
            model.pymc_mcmc.sample(10000, burn=1000, thin=100,
                                   tune_throughout=False)
            theta = model.pymc_mcmc.trace('hyperparameters')[:]
            ei_all = model.pymc_mcmc.trace('denoised_ei_min')[:]
            ei = ei_all.mean(axis=0)
            Y = model.pymc_mcmc.trace('denoised_posterior_samples')[:]
            Y = np.vstack(Y)
            i = np.argmax(ei)
            af_values.append(ei[i])
            if it >= add_at_least and ei[i] / af_values[0] < tol:
                if verbose:
                    print '*** Converged (af[i] / afmax0 = {0:1.7f})'.format(ei[i] / af_values[0])
                break
            if verbose:
                print '{0:8d} {1:1.7f}'.format(i, ei[i])
            self.idx_X_obs.append(i)
            self.Y_obs.append(self.func(self.X_design[i], *self.args))
            #q = me.predict_quantiles(self.X_design, size=100)
            fig, ax = plt.subplots()
            ax.plot(theta)
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.plot(self.X, self.Y, 'x', markersize=10, markeredgewidth=2)
            ax1.plot(self.X_design, Y.T, 'r', linewidth=0.1)
            #ax1.plot(self.X_design, q[0, :], 'g', linewidth=2)
            #ax1.fill_between(self.X_design.flatten(), q[1, :], q[2, :], color='green', alpha=0.25)
            ax2.plot(self.X_design, ei, '--m', linewidth=2)
            #fig = plt.figure()
            #ax = fig.add_subplot(111)
            #ax.hist(M_ns, normed=True)
            #fif, ax = plt.subplots()
            #ax.hist(X_ns, normed=True, bins=100)
            plt.show()
            a = raw_input('press enter')
