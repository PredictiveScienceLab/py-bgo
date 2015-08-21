#!/usr/bin/env python

from numpy.distutils.core import setup
from numpy.distutils.core import Extension


setup(name='py-bgo',
      version='0.1',
      descreption='Bayesian gloabl optimization',
      author='Ilias Bilionis',
      author_email='ibilion@purdue.edu',
      keywords=['Bayesian global optimization', 'Gaussian process regression'],
      packages=['pybgo']
      )
