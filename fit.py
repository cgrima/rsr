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
from mpfit import mpfit
from scipy import optimize
from Classdef import Statfit


def param0(sample, method='basic'):
    """Estimate initial parameters for HK fitting
    
    Arguments
    ---------
    sample : sequence
        amplitudes
        
    Keywords
    --------
    method : string
        method to compute the initial parameters
    """
    if method is 'basic':
        a = np.nanmean(sample)
        s = np.nanstd(sample)
        mu = 1.
    return {'a':a, 's':s, 'mu':mu}



def lmfit(sample, myfunct='hk', x=None, p0 = None, bins=200, range=(0,1),
          density=True, algo='lmfit', xtol=1e-4, ftol=1e-4, **kws):
    """Lmfit
    
    Arguments
    ---------
    sample : sequence
        amplitudes between 0 and 1.
    
    Keywords
    --------
    myfunct : string
        name of the function (in pdf module) to use for the fit
    x : sequence
        a vector if 'sample' is already the y coordinates of a distribution.
    p0 : dict
        Initial parameters.
    range : sequence
        x range considered for the fit.
    bins : int
        number of bins in the range.
    density : bool
        If True, the result is the value of the probability density function at
        the bin, normalized such that the integral over the range is 1.
    algo: string
        fit algorithm to use, whether 'lmfit' or 'mpfit'.
    kws : dict
        keywords to be passed to myfunct.
    """
    start = time.time()

     # Rework sample
    sample = np.array(sample)
    sample = sample[~np.isnan(sample)]
    
    # Make the histogram
    if x is None: # Make the histogram
        y, edges = np.histogram(sample, bins=bins, range=range, density=density)
        x = edges[1:] - abs(edges[1]-edges[0])/2
    else:
        y = sample

    eps = y*.1 # Uncertainty on the y values

     # Remove where y=0 if more than three consecutives. Will speed the fit up
    ind = rm3zeros(y)
    y = y[ind]
    x = x[ind]

    # Initial Parameters
    if p0 is None:
        p0 = param0(sample) 

    prm0 = Parameters()
    #     (Name,    Value,                 Vary,   Min,    Max,    Expr)
    prm0.add('a',     p0['a'],              True,   0,      1,    None)
    prm0.add('s',     p0['s'],              True,   0.001,  1,    None)
    prm0.add('mu',    p0['mu'],             True,   0.1,    1000, None)
    prm0.add('pt',    np.average(sample)**2,None,   0,      1,    'a**2+2*s**2')

    # Fit
    pdf2use = getattr(pdf, myfunct)

    try: # use 'lbfgs' if error with 'leastsq'
        p = minimize(pdf2use, prm0, args=(x, y), method='leastsq',
            xtol=xtol, ftol=ftol, **kws)
    except KeyboardInterrupt:
        raise
    except:
        print('!! Error with LEASTSQ fit, use L-BFGS-B instead')
        p = minimize(pdf2use, prm0, args=(x, y), method='lbfgs', **kws)
    
    # End
    elapsed = time.time() - start
    
    return Statfit(sample, p.userfcn, p.kws, range, bins, p.values, p.params,
                   p.chisqr, p.redchi, elapsed, p.nfev, p.message, p.success,
                   p.residual, y)



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
    #if vec[0] == 0 and vec[1] == 0: out[0] = False
    #if vec[-1] == 0 and vec[-2] == 0: out[-1] = False
    return out 
