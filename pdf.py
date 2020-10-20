"""
Probability density functions compliant with the lmfit package
Author: Cyril Grima <cyril.grima@gmail.com>
"""

import math
import numpy as np
from scipy import stats, integrate
from scipy.special import jv, kv, j0, i0, digamma



def gamma(params, x, data=None, eps=None):
    """Gamma PDF from scipy with adequate variables
    """
    # Initialisation
    mu = params['mu']
    # Debug inputs
    if hasattr(mu, 'value'): mu = mu.value #idem
    x = np.array([x]).flatten('C') # debug for iteration over 0-d element
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
    # Debug inputs
    if hasattr(s, 'value'): s = s.value #idem
    x = np.array([x]).flatten('C') # debug for iteration over 0-d element
    # Model function
    model = stats.rayleigh.pdf(x, scale = s)
    model = np.nan_to_num(model)

    if data is None:
        return model
    if eps is None:
        return (model - data) #residual
    return (model - data)/eps



def rice(params, x, data=None, eps=None, method = None):
    """Rice PDF from scipy with adequate variables
    """
    # Initialisation
    a = params['a']
    s = params['s']
    # Debug inputs
    if hasattr(a, 'value'): a = a.value #debug due to lmfit.minimize
    if hasattr(s, 'value'): s = s.value #idem
    x = np.array([x]).flatten('C') # debug for iteration over 0-d element
    # Model function
    model = stats.rice.pdf(x, a/s, scale = s)
    #model = (x/s**2) * np.exp(-(x**2+a**2)/(2*s**2)) * np.i0(a*x/s**2)
    #model = np.nan_to_num(model)

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
    # Debug inputs
    if hasattr(s, 'value'): s = s.value #idem
    if hasattr(mu, 'value'): mu = mu.value #idem
    x = np.array([x]).flatten('C') # debug for iteration over 0-d element

    # Model function
    b = np.sqrt(2*mu)/s
    model = 2*(x/2.)**mu *b**(mu+1.) /math.gamma(mu) *kv(mu-1,b*x)
    model = np.nan_to_num(model)

    if data is None:
        return model
    if eps is None:
        return (model - data) #residual
    return (model - data)/eps



def hk(params, x, data=None, eps=None, method = 'analytic'):
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
    # Debug inputs
    if hasattr(a, 'value'): a = a.value #debug due to lmfit.minimize
    if hasattr(s, 'value'): s = s.value #idem
    if hasattr(mu, 'value'): mu = mu.value #idem
    x = np.array([x]).flatten('C') # debug for iteration over 0-d element

    def integrand_analytic(w, x, a, s, mu):
        return x*w*j0(w*a)*j0(w*x)*(1. +w*w*s*s/(2.*mu))**-mu
    def integrand_compound(w, x, a, s, mu):
        return rice({'a':a,'s':s*np.sqrt(w/mu)}, x) * gamma({'mu':mu}, w)

    integrands = {
        'analytic': integrand_analytic,
        'compound': integrand_compound,
    }
    integrand = integrands[method]

    model = [integrate.quad(integrand, 0., np.inf, args=(i, a, s, mu), full_output=1)[0]
            for i in x] # Integration
    model = np.array(model)

    #def integrand(w, x, a, s, mu, method=method):
    #    if method == 'analytic':
    #        return x*w*j0(w*a)*j0(w*x)*(1. +w*w*s*s/(2.*mu))**-mu
    #    if method == 'compound':
    #        return rice({'a':a,'s':s*np.sqrt(w/mu)}, x) * gamma({'mu':mu}, w)

    #model = [integrate.quad(integrand, 0., np.inf, args=(i, a, s, mu, method), full_output=1)[0]
    #         for i in x] # Integration
    #model = np.array(model)

    if data is None:
        return model
    if eps is None:
        return (model - data) #residual
    return (model - data)/eps
