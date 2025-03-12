"""
Various tools for extracting signal components from a fit of the amplitude
distribution
"""

import time
#import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, report_fit

from . import pdf
from .Classdef import Statfit


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
    if method == 'basic':
        a = np.nanmean(sample)
        s = np.nanstd(sample)
        mu = 1.
    return {'a':a, 's':s, 'mu':mu}


def lmfit(sample, fit_model='hk', bins='auto', p0 = None,
          xtol=1e-4, ftol=1e-4):
    """Lmfit

    Arguments
    ---------
    sample : sequence
        amplitudes between 0 and 1.

    Keywords
    --------
    fit_model : string
        name of the function (in pdf module) to use for the fit
    bins : string
        method to compute the bin width (inherited from numpy.histogram)
    p0 : dict
        Initial parameters. If None, estimated automatically.
    xtol : float
        ??
    ftol : float
        ??

    Return
    ------
    A Statfit Class
    """
    start = time.time()
    #winsize = len(sample)
    bad = False

    #--------------------------------------------------------------------------
    # Clean sample
    #--------------------------------------------------------------------------
    sample = np.array(sample)
    sample = sample[np.isfinite(sample)]
    if len(sample) == 0:
        bad = True
        sample = np.zeros(10)+1

    #--------------------------------------------------------------------------
    # Make the histogram
    #--------------------------------------------------------------------------
#    n, edges, patches = hist(sample, bins=bins, normed=True)
    n, edges = np.histogram(sample, bins=bins, density=True)
#    plt.clf()

    x = ((np.roll(edges, -1) + edges)/2.)[0:-1]

    #--------------------------------------------------------------------------
    # Initial Parameters for the fit
    #--------------------------------------------------------------------------
    if p0 is None:
        p0 = param0(sample)

    prm0 = Parameters()
    #     (Name,    Value,                 Vary,   Min,    Max,    Expr)
    prm0.add('a',     p0['a'],              True,   0,  1,      None)
    prm0.add('s',     p0['s'],              True,   0,  1,      None)
    prm0.add('mu',    p0['mu'],             True,   .5, 10,    None)
    prm0.add('pt',    np.average(sample)**2,False,  0,  1,      'a**2+2*s**2*mu')

    #if fit_model == 'hk':
    #    # From [Dutt and Greenleaf. 1994, eq.14]
    #    prm0.add('a4',    np.average(sample)**4,False,  0,  1,
    #             '8*(1+1/mu)*s**4 + 8*s**2*s**2 + a**4')

    #--------------------------------------------------------------------------
    # Fit
    #--------------------------------------------------------------------------
    pdf2use = getattr(pdf, fit_model)

    # use 'lbfgs' fit if error with 'leastsq' fit
    try:
        p = minimize(pdf2use, prm0, args=(x, n), method='leastsq',
            xtol=xtol, ftol=ftol)
    except KeyboardInterrupt:
        raise
    except:
        # TODO: do we expect a specific exception?
        print('!! Error with LEASTSQ fit, use L-BFGS-B instead')
        p = minimize(pdf2use, prm0, args=(x, n), method='lbfgs')

    #--------------------------------------------------------------------------
    # Output
    #--------------------------------------------------------------------------
    elapsed = time.time() - start

    values = {}

    # Create values dict For lmfit >0.9.0 compatibility since it is no longer
    # in the minimize output
    for i in p.params.keys():
        values[i] = p.params[i].value

    # Results
    result = Statfit(sample, pdf2use, values, p.params,
             p.chisqr, p.redchi, elapsed, p.nfev, p.message, p.success,
             p.residual, x, n, edges, bins=bins)

    # Identify bad results
    # TODO: consider making this code part of Statfit.py?
    if bad:
        result.success = False
        result.values['a'] = 0
        result.values['s'] = 0
        result.values['mu'] = 0
        result.values['pt'] = 0
        result.chisqr = 0
        result.redchi = 0
        result.message = 'No valid data in the sample'
        result.residual = 0

    return result
