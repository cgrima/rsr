"""
Probability density functions compliant with the lmfit package
Author: Cyril Grima <cyril.grima@gmail.com>
"""

import math 
import numpy as np
from scipy import stats, integrate
from scipy.special import jv, kv, j0, digamma



def gamma(params, x, data=None, eps=None):
    """Gamma PDF from scipy with adequate variables
    """
    # Initialisation
    mu = params['mu']
    if hasattr(mu, 'value'): mu = mu.value #debug due to lmfit.minimize
    # Model function
    model = stats.gamma.pdf(x, mu, scale = 1)
    model = np.nan_to_num(model)

    if data is None:
        return model
    if eps is None:
        return (model - data) #residual
    return (model - data)/eps



def rayleigh(params, x, data=None, eps=None):
    """Rayleigh PDF from scipy with adequate variables
    """
    # Initialisation
    s = params['s']
    # Model function
    model = stats.rayleigh.pdf(x, scale = s)
    model = np.nan_to_num(model)

    if data is None:
        return model
    if eps is None:
        return (model - data) #residual
    return (model - data)/eps



def rice(params, x, data=None, eps=None):
    """Rice PDF from scipy with adequate variables
    """
    # Initialisation
    a = params['a']
    s = params['s']
    # Model function
    model = stats.rice.pdf(x, a/s, scale = s)
    model = np.nan_to_num(model)

    if data is None:
        return model
    if eps is None:
        return (model - data) #residual
    return (model - data)/eps



def k(params, x, data=None, eps=None):
    """K PDF
    """
    # Initialisation
    s = params['s']
    mu = params['mu']
    # Model function
    b = np.sqrt(2*mu)/s
    model = 2*(x/2.)**mu *b**(mu+1.) /math.gamma(mu) *kv(mu-1,b*x)
    model = np.nan_to_num(model)

    if data is None:
        return model
    if eps is None:
        return (model - data) #residual
    return (model - data)/eps



def hk(params, x, data=None, eps=None, method = 'analytic', verbose=False):
    """Homodyne K-distribution from various methods
    
    Arguments
    ---------
    params : dict
        params for the hk function {'a', 's', 'mu'}
    x : sequence
        x coordinates
    
    Keywords
    --------
    data : sequence
        data to compare the result with (for minimization)
    eps : sequence
        error on the data
    method : string
        'analytic' = from the common analytic form
        'compound' = from the compound representation [Destrempes and Cloutier,
                     2010, Ultrasound in Med. and Biol. 36(7): 1037-51, Eq. 16]
        NB: 'compound' is less unstable than 'analytic' but is ~10x longer!
    verbose : bool
        print fitting report
    """
    # Initialisation
    a = params['a']
    s = params['s']
    mu = params['mu']
    if hasattr(a, 'value'): a = a.value #debug due to lmfit.minimize
    if hasattr(s, 'value'): s = s.value #idem
    if hasattr(mu, 'value'): mu = mu.value #idem
    if verbose is True: print(a, s, mu)
    x = np.array([x]).flatten(0) # debug for iteration over 0-d element
    
    def integrand(w, x, a, s, mu, method=method):
        if method == 'analytic':
            return x*w*j0(w*a)*j0(w*x)*(1. +w**2*s**2/(2.*mu))**-mu
        if method == 'compound':
            return rice({'a':a,'s':s*np.sqrt(w/mu)}, x) * gamma({'mu':mu}, w)
            
    model = [integrate.quad(integrand, 0., np.inf, args=(i, a, s, mu, method))[0] 
             for i in x] # Integration
    model = np.array(model)
    
    if data is None:
        return model
    if eps is None:
        return (model - data) #residual
    return (model - data)/eps


def hk_mpfit(p, fjac=None, x=None, y=None, err=None, method='analytic'):
    """Homodyne K-distribution adapted for use of the mpfit.py package
    
    Arguments
    ---------
    p : sequence
        params for the HK function in this order (a, s, mu)
    
    Keywords
    --------
    fjac : If None then partial derivatives should not be computed. It will
        always be None if MPFIT is called with default flag.
    x : sequence
        x coordinates
    y : sequence
        data to compare the result with (for minimization)
    err : sequence
        error on the data
    """
    params = {'a':p[0],'s':p[1],'mu':p[2]}
    print('!! Not implemented yet')
    #return hk(params, x, data=y, eps=err, method=method)
