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
    
    Arguments
    ---------
    sample : sequence
        amplitudes
        
    Keywords
    --------
    method : string
        methodto compute the initial parameters
    """
    if method is 'basic':
        a0 = np.average(sample)
        s0 = np.std(sample)
        mu0 = 1.
    return {'a0':a0, 's0':s0, 'mu0':mu0}


def hk(sample, x=None, param0 = {'a0':.3, 's0':.04, 'mu0':5}, bins=50,
       range=(0,1), density=True, method='lbfgsb', **pdfkws):
    """HK fit with lmfit.minimize
    
    Arguments
    ---------
    sample : sequence
        if x not specified, sample should be amplitudes between 0 and 1.
    
    Keywords
    --------
    x : sequence
        a vector if 'sample' is already the y coordinates of a distribution.
    param0 : dict
        Initial parameters.
    range : sequence
        x range considered for the fit.
    bins : int
        number of bins in the range.
    density : bool
        If True, the result is the value of the probability density function at
        the bin, normalized such that the integral over the range is 1.
    fitmethod : string
        name of the optimization method
    pdfkws : dict
        keywords to be passed to the HK function
    """
    start = time.time()

    sample = np.array(sample)
    sample = sample[~np.isnan(sample)] # Remove NaN

    if x is None: # Make the histogram
        y, edges = np.histogram(sample, bins=bins, range=range, density=density)
        x = edges[1:] - abs(edges[1]-edges[0])/2
    else:
        y = sample
   
    ind = rm3zeros(y) #remove 0 if more than three consecutives
    y = y[ind]
    x = x[ind]

    eps = y*.1 # Uncertainty

    p0 = Parameters()
    #     (Name,    Value,                 Vary,   Min,    Max,    Expr)
    p0.add('a',     param0['a0'],          True,   0,      1,      None)
    p0.add('s',     param0['s0'],          True,   0.001,  1,      None)
    p0.add('mu',    param0['mu0'],         True,   None,   100,     None)
    p0.add('pt',    np.average(sample)**2, None,   None,   None,   'a**2+2*s**2')
    
    p = minimize(pdf.hk, p0, args=(x, y), kws=pdfkws, method=method) # Fit
    
    elapsed = time.time() - start
    
    return Statfit(sample, p.userfcn, p.kws, range, bins, p.values, p.params,
                   p.chisqr, p.redchi, elapsed, p.nfev, p.message, p.success)



def rm3zeros(vec):
    """Return a index vector to locate where zeros are not consecutive
    
    Arguments
    ---------
    vec : sequence
    
    Example
    -------
    vec = [3,    0,    0,     0,    4,    5,    0,    0]
    out = [True, True, False, True, True, True, True, True]
    """
    out = ~np.empty(vec.size, bool)
    
    for i in np.arange(vec.size-2)+1:
        if vec[i-1] == 0 and vec[i+1] == 0:
            out[i]=False
    if vec[0] == 0 and vec[1] == 0: out[0] = False
    if vec[-1] == 0 and vec[-2] == 0: out[-1] = False
    return out 
