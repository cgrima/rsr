"""
Various tools for extracting signal components from a fit of the amplitude
distribution
Author: Cyril Grima <cyril.grima@gmail.com>
"""

import pdf
import numpy as np
import time
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, report_fit
from scipy import optimize
from Classdef import Statfit



def hk_param0(sample, method='basic'):
    """Estimate initial parameters for HK fitting
    """
    if method is 'basic':
        a0 = mean(x)
        s0 = np.std(x)
        mu0 = 10.
    return a0, s0, mu0



def hk(sample, x=None, param0 = {'a0':.3, 's0':.04, 'mu0':5}, ftol=1e-4,
       xtol=1e-4, bins=20, range=(0,1), density=True):
    """HK fit with lmfit.minimize
    sample = if x not specified, sample should be amplitudes between 0 and 1
    x = a vector if 'sample' is already the y coordinates of a distribution
    param0 = Initial parameters
    ftol = 
    xtol = 
    bins = number of bins in the range
    density = If True, the result is the value of the probability density
              function at the bin, normalized such that the integral over the
              range is 1
    """
    
    start = time.time()

    if x is None: # Make the histogram
        y, edges = np.histogram(sample, bins=bins, range=range, density=density)
        x = edges[1:] - abs(edges[1]-edges[0])/2
    else:
        y = sample
    
    eps = y*.1 # Uncertainty
    p0 = Parameters()
    #     (Name,    Value,  Vary,   Min,    Max,    Expr)
    p0.add('a',     param0['a0'],     True,   0,      1,      None)
    p0.add('s',     param0['s0'],     True,   0.001,  1,      None)
    p0.add('mu',    param0['mu0'],    True,   0.1,    50,     None)
    
    p = minimize(pdf.hk, p0, args=(x, y), xtol=xtol, ftol=ftol) # Fit
    
    yfit = pdf.hk(p.values, x) #fitted y-vector
    
    elapse = time.time() - start

    return Statfit(sample, p.userfcn, p.kws, range, bins, p.values, p.params,
		           p.chisqr, p.redchi, elapse)
