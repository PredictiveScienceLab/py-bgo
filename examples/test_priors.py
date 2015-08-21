"""
Test the priors.

Author:
    Ilias Bilionis

Date:
    5/5/2015

"""


from pydes import LogLogisticPrior
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 


p = LogLogisticPrior()
log_x = np.linspace(-10, 2., 100.)
log_p = p.lnpdf(np.exp(log_x))
data = p.rvs(100)

plt.plot(np.exp(log_x), np.exp(log_p))
#plt.plot(np.exp(log_x), p.lnpdf_grad(np.exp(log_x)))
plt.hist(data, normed=True)
plt.show()
a = raw_input('press any key')