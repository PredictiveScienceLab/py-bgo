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
import sys
import warnings
warnings.simplefilter('ignore')
import GPy
from . import colorAlpha_to_rgb


class GlobalOptimizer(object):

    """
    A global optimizer class.

    It is essentially a myopic, sequential, global optimizer.

    :param func:    The function you wish to optimize.
    :arapm args:    Additional arguments for the function we optimize.
    :param kernel:  An initialized kernel that you might want to use. If ``None``,
                    then we will use an RBF with standard parameters.
    :param gp_type: The GPy model class you would like to use for the regression.
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

    # The kernel object to be used for the GP
    _kernel = None

    # The kernel type used for the GP by default
    _kernel_type = None

    # The GPy regression model that we are using
    _gp_type = None

    # The trained GP model representing the objective
    _model = None

    # The prior we use for the variance of the likelihood
    _model_like_variance_prior = None

    # The expected improvement of the current model
    _ei = None

    # The denoised posterior GP samples
    _denoised_posterior_samples = None

    # The indices of the best designs of each GP sample
    _best_idx = None

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

    @property
    def num_dim(self):
        """
        :getter:    The number of design dimensions.
        """
        return self.X_init.shape[1]

    @property
    def kernel(self):
        """
        :getter:    The kernel to be used for the GP.
        """
        if self._kernel is None:
            self._kernel = self._get_fresh_kernel()
        return self._kernel.copy()

    @property
    def kernel_type(self):
        """
        :getter:    The kernel type used for the GP by default.
        """
        return self._kernel_type

    @property
    def gp_type(self):
        """
        :getter:    The GPy model to be used for the regression.
        """
        return self._gp_type

    @property
    def model(self):
        """
        :getter:    Get the GP model of the objective.
        """
        if self._model is None:
            self._model = self._get_fresh_model()
        return self._model

    @property
    def model_like_variance_prior(self):
        """
        :getter:    Get the prior of the variance of the likelihood that we
                    are using.
        """
        return self._model_like_variance_prior

    @property
    def ei(self):
        """
        :getter:    The expected improvement on the design points.
        """
        if self._ei is None:
            ei_all = self.model.pymc_mcmc.trace('denoised_ei_min')[:]
            ei = ei_all.mean(axis=0)
            self._ei = ei
        return self._ei

    def __init__(self, X_init, X_design, func, args=(), Y_init=None,
                 kernel=None,
                 kernel_type=GPy.kern.RBF,
                 gp_type=GPy.models.GPRegression,
                 model_like_variance_prior=GPy.priors.Jeffreys(),
                 optimize_model_before_init_mcmc=False,
                 optimize_model_before_mcmc=False,
                 optimize_model_num_restarts=1,
                 num_predict=100,
                 fixed_noise=False,
                 max_it=100,
                 add_at_least=10,
                 rel_tol=1e-2,
                 num_mcmc_samples=1000,
                 num_mcmc_burn=100,
                 num_mcmc_thin=10,
                 mcmc_tune_throughout=False,
                 mcmc_progress_bar=True,
                 mcmc_start_from_scratch=False,
                 verbose=True,
                 make_plots=True,
                 plot_ext='png',
                 plot_prefix='optimizer',
                 true_func=None,
                 new_fig_func=None,
                 trace_posterior_samples=False):
        """
        Initialize the object.
        """
        self.X_init = X_init
        self.X_design = X_design
        self.Y_init = Y_init
        self.func = func
        self._args = args
        self._kernel = kernel
        self._kernel_type = kernel_type
        self._gp_type = gp_type
        self._model_like_variance_prior = model_like_variance_prior
        self._idx_X_obs = []
        self._Y_obs = []
        self.num_predict = 100
        self.optimize_model_before_init_mcmc = optimize_model_before_init_mcmc
        self.optimize_model_before_mcmc = optimize_model_before_mcmc
        self.optimize_model_num_restarts = optimize_model_num_restarts
        self.fixed_noise = fixed_noise
        self.max_it = max_it
        self.add_at_least = add_at_least
        self.rel_tol = rel_tol
        self.num_mcmc_samples = num_mcmc_samples
        self.num_mcmc_burn = num_mcmc_burn
        self.num_mcmc_thin = num_mcmc_thin
        self.mcmc_tune_throughout = mcmc_tune_throughout
        self.mcmc_progress_bar = mcmc_progress_bar
        if optimize_model_before_mcmc:
            mcmc_start_from_scratch = True
        self.mcmc_start_from_scratch = mcmc_start_from_scratch
        self.verbose = True
        self.make_plots = make_plots
        self.plot_prefix = plot_prefix
        self.plot_ext = plot_ext
        self.true_func = true_func
        if self.true_func is not None:
            # Assuming this is a test and that true_func is very cheap
            self.Y_true = self.true_func(self.X_design)
            i_best = np.argmin(self.Y_true)
            self.X_true_best = self.X_design[i_best, :]
            self.Y_true_best = self.Y_true[i_best, 0]
        if new_fig_func is None:
            def new_fig():
                import matplotlib.pyplot as plt
                return plt.subplots()
            new_fig_func = new_fig
        self.new_fig_func = new_fig_func
        self.trace_posterior_samples = trace_posterior_samples

    def _get_fresh_kernel(self):
        """
        :getter: Get a kernel that is a fresh copy of the kernel the user
                 provided us with.
        """
        kernel = self.kernel_type(self.num_dim, ARD=True)
        kernel.variance.unconstrain()
        kernel.variance.set_prior(GPy.priors.Jeffreys())
        kernel.lengthscale.unconstrain()
        kernel.lengthscale.set_prior(GPy.priors.LogLogistic())
        return kernel

    def _get_fresh_model(self):
        """
        :getter: Get a fresh Gaussian process model.
        """
        model = self.gp_type(self.X, self.Y, self.kernel)
        model.likelihood.variance.unconstrain()
        model._X_predict = self.X_design
        if self.fixed_noise:
            model.likelihood.variance.unconstrain()
            model.Gaussian_noise.variance.constrain_fixed(self.fixed_noise)
        else:
            model.likelihood.variance.set_prior(self.model_like_variance_prior)
        model._num_predict = self.num_predict
        model.pymc_trace_denoised_min()
        model.pymc_trace_denoised_argmin()
        model.pymc_trace_expected_improvement(denoised=True)
        if self.trace_posterior_samples:
            model.pymc_trace_posterior_samples()
        return model

    def initialize(self):
        """
        Initialize everything.

        Computes the initial output data if they are not provided already.
        """
        if self.Y_init is None:
            if self.verbose:
                print '\t> did not find observed objectives'
                sys.stdout.write('\t> computing the objectives now... ')
                sys.stdout.flush()
            self.Y_init = [self.func(x, *self.args) for x in self.X_init]
            if self.verbose:
                sys.stdout.write('done!\n')

    def optimize_step(self, it=0):
        """
        Perform a single optimization step.
        """
        # Train current model
        self._ei = None
        self._denoised_posterior_samples = None
        self._best_idx = None
        if self.mcmc_start_from_scratch:
            self._model = None
        if (it == 0 and self.optimize_model_before_init_mcmc) \
                or self.optimize_model_before_mcmc:
            if self.verbose:
                sys.stdout.write('\t> optimizing gp model...')
                sys.stdout.flush()
            if self.optimize_model_num_restarts == 1:
                self.model.optimize()
            else:
                self.model.optimize_restarts(
                        num_restarts=self.optimize_model_num_restarts)
            if self.verbose:
                sys.stdout.write(' done!\n')
        if self.verbose:
            print '\t> starting mcmc sampling'
        self.model.pymc_mcmc.sample(self.num_mcmc_samples,
                               burn=self.num_mcmc_burn,
                               thin=self.num_mcmc_thin,
                               tune_throughout=self.mcmc_tune_throughout,
                               progress_bar=self.mcmc_progress_bar)
        # Find best expected improvement
        ei = self.ei
        i = np.argmax(ei)
        # Do the simulation and add it
        self.idx_X_obs.append(i)
        self.Y_obs.append(self.func(self.X_design[i], *self.args))
        self.model.set_XY(self.X, self.Y)
        if self.verbose:
            print '\t> design point id to be added : {0:d}'.format(i)
            print '\t> maximum expected improvement: {0:1.3f}'.format(ei[i])
        return i, ei[i]

    def optimize(self):
        """
        Optimize the objective.
        """
        if self.verbose:
            print '> initializing algorithm'
        self.initialize()
        self.ei_values = []
        self.y_best_p500 = []
        self.y_best_p025 = []
        self.y_best_p975 = []
        self.x_best_p500 = []
        self.x_best_p025 = []
        self.x_best_p975 = []
        for it in xrange(self.max_it):
            i, ei_max = self.optimize_step(it)
            self.ei_values.append(ei_max)
            if self.make_plots:
                self.plot(it)
            if self.verbose:
                print '> checking convergence'
            if it >= self.add_at_least and ei_max / self.ei_values[0] < self.rel_tol:
                if self.verbose:
                    print '*** Converged (ei[i_max] / eimax0 = {0:1.7f})'.format(
                            ei_max / self.ei_values[0])
                break
            else:
                print '> rel. ei = {0:1.3f}'.format(ei_max / self.ei_values[0])

    def plot(self, it):
        """
        Plot the results of our analysis at the current step of the algorithm.
        """
        if self.verbose:
            print '> plotting intermediate results'
        try:
            self.plot_mcmc_diagnostics(it)
        except:
            pass
        self.plot_opt_status(it)
        self.plot_opt_dist(it)
        self.plot_opt_joint(it)

    def _get_nd(self):
        """
        Get the number of digits used for filenames.
        """
        return len(str(self.max_it))

    def _fig_name(self, name, it):
        """
        Get the figure name.
        """
        return self.plot_prefix + '_' + str(it).zfill(self._get_nd()) \
                + '_' + name + '.' + self.plot_ext

    def _hyper_id(self, data, i):
        return str(i + 1).zfill(len(str(data.shape[1])))

    def _hyper_name(self, data, name, i):
        return name + '_' + self._hyper_id(data, i)
    
    def _hyper_fig_name(self, figname, data, name, i, it):
        return self._fig_name(figname + '_' + self._hyper_name(data, name, i),
                              it)

    def _hyper_tex(self, data, name, i):
        return r'$\%s_{%s}$' % (name, self._hyper_id(data, i))

    def plot_autocorrelations(self, data, name, it):
        """
        Plot all autocorrelation plots.
        """
        if self.verbose:
            print '\t\t> plotting autocorrelations'
        import matplotlib.pyplot as plt
        for i in xrange(data.shape[1]):
            fig, ax = self.new_fig_func()
            ax.acorr(data[:, i], maxlags=data.shape[0] /  2)
            figname = self._hyper_fig_name('acorr', data, name, i, it)
            if self.verbose:
                print '\t\t> writing:', figname
            fig.savefig(figname)
            plt.close(fig)

    def plot_trace(self, data, name, it):
        if self.verbose:
            print '\t\t> ploting trace of', name
        import matplotlib.pyplot as plt
        fig, ax = self.new_fig_func()
        handles = ax.plot(data)
        if data.shape[1] <= 5:
            labels = [self._hyper_tex(data, name, i)
                      for i in xrange(data.shape[1])]
            fig.legend(handles, labels)
        figname = self._fig_name('trace_' + name, it)
        if self.verbose:
            print '\t\t> writing:', figname
        fig.savefig(figname)
        plt.close(fig)

    def plot_dist(self, data, name, it, labels=None):
        if self.verbose:
            print '\t\t> ploting distribution of', name
        import matplotlib.pyplot as plt
        import seaborn as sns
        for i in xrange(data.shape[1]):
            fig, ax = self.new_fig_func()
            sns.distplot(data[:, i])
            if labels is not None:
                ax.set_xlabel(labels[i])
            else:
                ax.set_xlabel(self._hyper_tex(data, name, i))
            figname = self._hyper_fig_name('dist', data, name, i, it)
            if self.verbose:
                print '\t\t> writing:', figname
            fig.savefig(figname)
            plt.close(fig)

    def plot_opt_status(self, it):
        self.plot_opt_status_gen(it)
        if self.num_dim == 1:
            self.plot_opt_status_1d(it)
        elif self.num_dim == 2:
            self.plot_opt_status_2d(it)

    def plot_opt_status_1d(self, it):
        if self.verbose:
            print '\t\t> plotting the optimization status'
        import matplotlib.pyplot as plt
        import seaborn as sns
        Y = self.model.pymc_mcmc.trace('posterior_samples')[:]
        Y = np.vstack(Y)
        Y_d = self.model.pymc_mcmc.trace('denoised_posterior_samples')[:]
        Y_d = np.vstack(Y_d)
        fig, ax1 = self.new_fig_func()
        ax2 = ax1.twinx()
        p_025 = np.percentile(Y_d, 2.5, axis=0)
        p_500 = np.percentile(Y_d, 50, axis=0)
        p_975 = np.percentile(Y_d, 97.5, axis=0)
        ax1.fill_between(self.X_design.flatten(), p_025, p_975,
                         color=colorAlpha_to_rgb(sns.color_palette()[0], 0.25),
                         label='95\% error')
        ax1.plot(self.X_design, p_500, color=sns.color_palette()[0],
                      label='Pred. mean')
        ax1.plot(self.X[:-1, :], self.Y[:-1, :],
                 'kx', markersize=10, markeredgewidth=2,
                 label='Observations')
        if self.true_func is not None:
            ax1.plot(self.X_design, self.Y_true,
                     ':', color=sns.color_palette()[2])
        ax1.plot(self.X[-1, 0], self.Y[-1, 0], 'o',
                 markersize=10, markeredgewidth=2,
                 color=sns.color_palette()[1])
        ax2.plot(self.X_design, self.ei / self.ei_values[0],
                 '--', color=sns.color_palette()[3],
                      label='Exp. improvement')
        ax2.set_ylim(0, 1.5)
        plt.setp(ax2.get_yticklabels(), color=sns.color_palette()[3])
        figname = self._fig_name('state', it)
        if self.verbose:
            print '\t\t> writing:', figname
        fig.savefig(figname)
        plt.close(fig)
        
    def plot_opt_status_gen(self, it):
        import matplotlib.pyplot as plt
        import seaborn as sns
        # Plot the expected improvement so far
        if self.verbose:
            print '\t\t> plotting the max expected improvement'
        fig, ax = self.new_fig_func()
        rel_ei = self.ei_values / self.ei_values[0]
        ax.plot(np.arange(1, it + 2), rel_ei)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Maximum expected improvement')
        figname = self._fig_name('ei', it)
        if self.verbose:
            print '\t\t> writing:', figname
        fig.savefig(figname)
        plt.close(fig)
        # Plot the range of the best objectives we have found so far
        if self.verbose:
            print '\t\t> plotting stat. about optimal objective value'
        y_best_p500 = np.median(self.Y_best)
        y_best_p025 = np.percentile(self.Y_best, 2.5)
        y_best_p975 = np.percentile(self.Y_best, 97.5)
        self.y_best_p500.append(y_best_p500)
        self.y_best_p025.append(y_best_p025)
        self.y_best_p975.append(y_best_p975)
        fig, ax = self.new_fig_func()
        ax.fill_between(np.arange(1, it + 2), self.y_best_p025,
                        self.y_best_p975,
                        color=colorAlpha_to_rgb(sns.color_palette()[0], 0.25))
        ax.plot(np.arange(1, it + 2), self.y_best_p500)
        if self.true_func is not None:
            ax.plot(np.arange(1, it + 2), [self.Y_true_best] * (it + 1),
                    '--', color=sns.color_palette()[2])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Optimal objective')
        figname = self._fig_name('objective', it)
        if self.verbose:
            print '\t\t> writing:', figname
        fig.savefig(figname)
        plt.close(fig)
        # Do the same for the design (only 1D)
        if self.num_dim == 1:
            if self.verbose:
                print '\t\t> plotting stat. about optimal design'
            x_best_p500 = np.median(self.X_best.flatten())
            x_best_p025 = np.percentile(self.X_best.flatten(), 2.5)
            x_best_p975 = np.percentile(self.X_best.flatten(), 97.5)
            self.x_best_p500.append(x_best_p500)
            self.x_best_p025.append(x_best_p025)
            self.x_best_p975.append(x_best_p975)
            fig, ax = self.new_fig_func()
            ax.fill_between(np.arange(1, it + 2), self.x_best_p025,
                            self.x_best_p975,
                            color=colorAlpha_to_rgb(sns.color_palette()[0], 0.25))
            ax.plot(np.arange(1, it + 2), self.x_best_p500)
            if self.true_func is not None:
                ax.plot(np.arange(1, it + 2), [self.X_true_best.flatten()] * (it + 1),
                        '--', color=sns.color_palette()[2])
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Optimal design')
            figname = self._fig_name('design', it)
            fig.savefig(figname)
            if self.verbose:
                print '\t\t> writing:', figname
            plt.close(fig)

    def plot_opt_joint(self, it):
        if self.num_dim == 1:
            self.plot_opt_joint_1d(it)

    def plot_opt_dist(self, it):
        self.plot_dist(self.Y_best[:, None], 'objective', it,
                       labels=['Optimal objective'])
        if self.num_dim == 1:
            self.plot_dist(self.X_best, 'design', it,
                           labels=['Optimal design'])

    @property
    def denoised_posterior_samples(self):
        """
        :getter:    Get the denoised posterior samples from the GP.
        """
        if self._denoised_posterior_samples is None:
            Y = self.model.pymc_mcmc.trace('denoised_posterior_samples')[:]
            Y = np.vstack(Y)
            self._denoised_posterior_samples = Y
        return self._denoised_posterior_samples

    @property
    def num_posterior_samples(self):
        """
        :getter:    The number of posterior samples of the GP.
        """
        return self.denoised_posterior_samples.shape[0]

    @property
    def best_idx(self):
        """
        :getter:    The indices of the best design of each GP sample.
        """
        if self._best_idx is None:
            self._best_idx = np.argmin(self.denoised_posterior_samples, axis=1)
        return self._best_idx

    @property
    def X_best(self):
        """
        :getter:    Get samples of the optimal designs.
        """
        return self.X_design[self.best_idx, :]

    @property
    def Y_best(self):
        """
        :getter:    Get the values of the optimal designs.
        """
        return np.array([self.denoised_posterior_samples[i, self.best_idx[i]]
                         for i in xrange(self.num_posterior_samples)])
    
    def plot_opt_joint_1d(self, it):
        import matplotlib.pyplot as plt
        import seaborn as sns
        g = sns.jointplot(self.X_best.flatten(), self.Y_best, kind='kde')
        g.set_axis_labels('Optimal design', 'Optimal objective')
        if self.true_func is not None:
            g.ax_joint.plot(self.X_true_best[0], self.Y_true_best, 'x',
                            color=sns.color_palette()[2],
                            markersize=10, markeredgewidth=2)
        plt.savefig(self._fig_name('opt', it))

    def plot_mcmc_diagnostics(self, it):
        """
        Plot diagnostics about the MCMC chain.
        """
        if self.verbose:
            print '\t> plotting mcmc diagnostics'
        theta = self.model.pymc_mcmc.trace('transformed_hyperparameters')[:]
        phi = self.model.pymc_mcmc.trace('hyperparameters')[:]
        self.plot_autocorrelations(theta, 'theta', it)
        self.plot_trace(theta, 'theta', it)
        self.plot_trace(phi, 'phi', it)
        self.plot_dist(theta, 'theta', it)
        self.plot_dist(phi, 'phi', it)

