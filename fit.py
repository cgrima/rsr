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
from astroML.plotting import hist


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


def lmfit(sample, fit_model='hk', bins='knuth', p0 = None,
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
        method to compute the bin width (inherited from astroML.plotting.hist)
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

    #--------------------------------------------------------------------------
    # Clean sample
    #--------------------------------------------------------------------------
    sample = np.array(sample)
    sample = sample[~np.isnan(sample)]

    #--------------------------------------------------------------------------
    # Make the histogram
    #--------------------------------------------------------------------------
    n, edges, patches = hist(sample, bins=bins, normed=True)
    plt.close()

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
    prm0.add('mu',    p0['mu'],             True,   0,  1000,   None)
    prm0.add('pt',    np.average(sample)**2,None,   0,  1,      'a**2+2*s**2')

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
        print('!! Error with LEASTSQ fit, use L-BFGS-B instead')
        p = minimize(pdf2use, prm0, args=(x, n), method='lbfgs')

    #--------------------------------------------------------------------------
    # Output
    #--------------------------------------------------------------------------
    elapsed = time.time() - start

    #result = Statfit(sample, p.userfcn, p.kws, range, bins, p.values, p.params,
    #         p.chisqr, p.redchi, elapsed, p.nfev, p.message, p.success,
    #         p.residual, y)

    result = Statfit(sample, p.userfcn, p.kws, p.values, p.params,
             p.chisqr, p.redchi, elapsed, p.nfev, p.message, p.success,
             p.residual, x, n, edges, bins=bins)

    return result
