"""
This is the simplest possible, brute force test of the informative learning
scheme.

Author:
    Ilias Bilionis

Date:
    10/08/2014
    01/29/2015

"""


import GPy
import numpy as np
np.random.seed(3141)
import scipy
import scipy.stats as stats
import matplotlib.pyplot as plt


def f(x):
    """
    A 1D function to look at.
    """
    return 4. * (1. - np.sin(x + 8. * np.exp(x - 7.))) + np.random.randn(x.shape[0])
#    return np.sin(x) / (x + 1e-6)


def m(x):
    """
    The mean we are using.
    """
    if isinstance(x, np.ndarray):
        return np.zeros(x.shape[0])[:, None]
    else:
        return 0.


def m_star(X1, X, y, L_K, m, k):
    """
    The updated mean.
    """
    k1 = k.K(X1, X)
    y1 = scipy.linalg.cho_solve(L_K, k1.T)
    y2 = scipy.linalg.cho_solve(L_K, y - m(X))
    return m(X1) + np.dot(y1.T, y2)


def k_star(X1, X2, X, y, L_K, k):
    """
    The updated covariance.
    """
    k12 = k.K(X1, X2)
    k1 = k.K(X, X1)
    k2 = k.K(X, X2)
    y1 = scipy.linalg.cho_solve(L_K, k1)
    y2 = scipy.linalg.cho_solve(L_K, k2)
    return k12 - np.dot(y1.T, y2) + np.ones(k12.shape) * 1e-4




def IG(xx, model, num_y=10, num_x_star=1000):
    """
    Compute the Expected Information gain criterion at ``xx``.
    """
    num_x = xx.shape[0]
    p = np.zeros((num_x, num_y, num_x))
    L = model.L
    Y = model.likelihood.Y
    X = model.X
    num_X = X.shape[0]
    v = model._get_params()[model.grep_param_names('noise_variance')][0]
    # The covariance on the points in xx:
    Kp = model.kern.K(xx, xx) + np.eye(num_x) * v
    KXxx = model.kern.K(X, xx)
    #mu_s, k_s = model.predict(xx, full_cov=True)[:2]
    # Loop over the xx's
    for i in xrange(num_x):
        print i
        # Pretend as if we have observed xx[i, :]
        # There is definitely a fast way to do this.
        # For now, we do it the stupid way
        X_new = np.vstack([X, xx[i:(i+1), :]])
        K_new = model.kern.K(X_new, X_new) + np.eye(num_X + 1) * v
        L_new = np.linalg.cholesky(K_new)
        tmp = np.linalg.solve(L_new, model.kern.K(X_new, xx))
        Kp = model.kern.K(xx, xx) + np.eye(num_x) * v - np.dot(tmp.T, tmp)
        Lp = np.linalg.cholesky(Kp)
        # Sample some y's from the predictive
        y_i = model.posterior_samples(xx[i, :][None, :], size=num_y)
        # Now condition the GP on each one of those and draw a
        # random sample.
        z = np.dot(l_s, np.random.randn(num_x, num_x_star))

        plt.plot(xx, z, 'b')
        plt.savefig('z_test.png')
        print z
        quit()

# Just for plotting
xx = np.linspace(0, 6., 100)[:, None]

# Loop over observations
x_obs = []
y_obs = []
for i in xrange(10):
    x_obs.append(6. * np.random.rand())
    y_obs.append(f(x_obs[-1]))
    X = np.array(x_obs)[:, None]
    y = np.array(y_obs)[:, None]
    if i <= 5:
        continue
    k = GPy.kern.rbf(1, lengthscale=0.5)
    model = GPy.models.GPRegression(X, y, k)
    #model.constrain_fixed('noise_variance', 1e-10)
    model.optimize(messages=True)
    print str(model)
    #K = k.K(X) + 1e-4 * np.eye(X.shape[0])
    #L_K = scipy.linalg.cho_factor(K)
    m_s, k_s, m_05, m_95 = model.predict(xx, full_cov=True)
    m_05 = np.diag(m_05)
    m_95 = np.diag(m_95)
    plt.plot(xx, f(xx), 'b', linewidth=2)
    plt.plot(X, y, 'go', linewidth=2, markersize=10, markeredgewidth=2)
    plt.plot(xx, m_s, 'r--', linewidth=2)
    plt.fill_between(xx.flatten(), m_05.flatten(), m_95.flatten(), color='grey',
                     alpha=0.5)
    prefix = 'test_1_' + str(i).zfill(2)
    plt.savefig(prefix + '.png')
    plt.clf()
    # Look at the KLE
    w, vi = scipy.linalg.eigh(k_s, eigvals=(90, 99))
    # Plot the eigenvalues
    plt.plot(w[::-1], 'g', linewidth=2)
    plt.savefig(prefix + '_eigvals.png')
    plt.clf()
    # Plot the eigenvectors
    plt.plot(xx, f(xx), 'b', linewidth=2)
    plt.plot(X, y, 'go', linewidth=2, markersize=10, markeredgewidth=2)
    plt.plot(xx, m_s, 'r--', linewidth=2)
    plt.plot(xx, vi[:, 5:], 'k')
    plt.legend(['True', 'Obs.', 'GP Mean', 'KLE basis'])
    plt.savefig(prefix + '_eigvecs.png')
    plt.clf()
    # Draw some random samples using the KLE
    plt.plot(xx, f(xx), 'b', linewidth=2)
    plt.plot(X, y, 'go', linewidth=2, markersize=10, markeredgewidth=2)
    plt.plot(xx, m_s, 'r--', linewidth=2)
    phi = vi[:, 5:] * w[5:]
    xi = np.random.randn(5, 10000)
    sa = m_s + np.dot(phi, xi)
    plt.plot(xx, sa[:, :10], 'k')
    plt.savefig(prefix + '_samples.png')
    plt.clf()
    # For each of the samples, find the probability density of the minimum
    x_star = xx.flatten()[np.argmin(sa, axis=0)]
    kde = stats.gaussian_kde(x_star[None, :])
    fig, ax1 = plt.subplots()
    ax1.plot(xx, f(xx), 'b', linewidth=2)
    ax1.plot(X, y, 'go', linewidth=2, markersize=10, markeredgewidth=2)
    ax1.plot(xx, m_s, 'r--', linewidth=2)
    ax1.fill_between(xx.flatten(), m_05.flatten(), m_95.flatten(), color='grey',
                     alpha=0.5)
    ax1.set_ylabel('$f(x)$', fontsize=16)
    ax2 = ax1.twinx()
    #ax2.hist(x_star, normed=True, color='green', alpha=0.3)
    ax2.plot(xx, kde(xx.T), 'g', linewidth=2)
    ax2.set_ylabel('$p(x^*|\mathcal{D})$', fontsize=16)
    plt.setp(ax2.get_yticklabels(), color='g')
    plt.savefig(prefix + '_x_star.png')
    plt.clf()
    # Plot the expected improvement criterion
    ei = EI(xx, model)
    fig, ax1 = plt.subplots()
    ax1.plot(xx, f(xx), 'b', linewidth=2)
    ax1.plot(X, y, 'go', linewidth=2, markersize=10, markeredgewidth=2)
    ax1.plot(xx, m_s, 'r--', linewidth=2)
    ax1.fill_between(xx.flatten(), m_05.flatten(), m_95.flatten(), color='grey',
                     alpha=0.5)
    ax1.set_ylabel('$f(x)$', fontsize=16)
    ax2 = ax1.twinx()
    #ax2.hist(x_star, normed=True, color='green', alpha=0.3)
    ax2.plot(xx, ei, 'g', linewidth=2)
    ax2.set_ylabel('$EI(r)$', fontsize=16, color='g')
    plt.setp(ax2.get_yticklabels(), color='g')
    plt.savefig(prefix + '_ei.png')
    plt.clf()
    # Plot the expected information gain criterion
    ig = IG(xx, model)
