"""
Probability density functions compliant with the lmfit package
Author: Cyril Grima <cyril.grima@gmail.com>
"""

import math
import mpmath
import numpy as np
from scipy import stats, integrate
from scipy.special import jv, kv, j0, i0, digamma, factorial, kn



def gamma(params, x, data=None, eps=None):
    """Gamma PDF from scipy with adequate variables
    """
    # Initialisation
    mu = params['mu']
    # Debug inputs
    if hasattr(mu, 'value'):
        mu = mu.value #idem
    x = np.array([x]).flatten('C') # debug for iteration over 0-d element
    # Model function
    model = stats.gamma.pdf(x, mu, scale = 1)
    model = np.nan_to_num(model, posinf=np.nan)

    if data is None:
        return model
    if eps is None:
        return model - data #residual
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
        return model - data #residual
    return (model - data)/eps



def rice(params, x, data=None, eps=None, method='scipy'):
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
    if method == 'scipy':
        model = stats.rice.pdf(x, a/s, scale = s)
    if method == 'analytic':
        model = (x/s**2) * np.exp(-(x**2+a**2)/(2*s**2)) * np.i0(a*x/s**2)
    #model = np.nan_to_num(model)

    if data is None:
        return model
    if eps is None:
        return model - data #residual
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
        return model - data #residual
    return (model - data)/eps


def hk_auto(params, x, **kwargs):
    """ !!! EXPERIMENTAL !!
    Selector and application of the best method for HK.
    the 'analytic' method is privileged over 'compound' when it is
    stable enough because it is faster. Valid when :
         (Pc + Pn) <= 1
            AND
    -40 dB < Pc/Pn > 25 dB
    """
    pt = 10*np.log10(params['a']**2 + 2*params['s']**2*params['mu'])
    if (params['mu'] > 1) and (pt > -20):
        kwargs['method'] = 'analytic'
    else:
        kwargs['method'] = 'compound'

    return hk(params, x, **kwargs)


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
        'auto' = Choose the best method from the range of input parameters.
        'analytic' = from the common analytic form Destrempes and Cloutier [2010].
                     Instable for mu < 1.
        'compound' = from the compound representation [Destrempes and Cloutier,
                     2010, Ultrasound in Med. and Biol. 36(7): 1037-51, Eq. 16].
                     Gives best results , especially for mu < 1 but is ~10x longer!
        'drumheller' = from Drumheller [2002, Eq. 34].
                       For testing only, not advised.
    verbose : bool
        print fitting report
    """
    if method == 'auto':
        return hk_auto(params, x, data=None, eps=None)

    # Initialisation
    a = params['a']
    s = params['s']
    mu = params['mu']

    # Debug inputs
    if hasattr(a, 'value'):
        a = a.value #debug due to lmfit.minimize
    if hasattr(s, 'value'):
        s = s.value #idem
    if hasattr(mu, 'value'):
        mu = mu.value #idem
    x = np.array([x]).flatten('C') # debug for iteration over 0-d element

    def integrand_analytic(w, x, a, s, mu):
        return x*w *j0(w*a) *j0(w*x) *(1. +w**2*s**2/2.)**-mu

    def integrand_compound(w, x, a, s, mu):
        r = rice({'a':a,'s':np.sqrt(s**2*w)}, x, method='scipy')
        g = gamma({'mu':mu}, w)
        return r*g

    def integrand_drumheller(w, x, a, s, mu):
        w = np.float64(w) #Convert mpmath to float64 type. Solve 'u_fonc not supported' error
        # Parameters correspondance
        r = x
        r0 = a
        v = mu-1
        b2 = 2/s**2
        b = np.sqrt(b2)
        # Function
        f1 = b2*r/math.gamma(v+1)
        f2 = (b2*r0*r)**(2*w) / 4**(2*w) / factorial(w)**2
        f3 = (b*np.sqrt(r0**2+r**2)/2.)**(v-2*w)
        f4 = kn(2*w-v, b*np.sqrt(r0**2+r**2))
        return f1*f2*f3*f4

    integrands = {
        'analytic': integrand_analytic,
        'compound': integrand_compound,
        'drumheller':integrand_drumheller,
    }
    integrand = integrands[method]
    # W0640: Cell variable i defined in loop (cell-var-from-loop)
    if method == 'drumheller':
        model = [mpmath.nsum(lambda w: integrand_drumheller(w, i, a, s, mu), [0, np.inf])
                for i in x] # Summation
    else:
        model = [integrate.quad(integrand, 0., np.inf, args=(i, a, s, mu), full_output=1)[0]
                 for i in x] # Integration

    model = np.array(model)

    # Set as nan where gamma distribution is not defined
    g = gamma({'mu':mu}, x)
    model[np.isfinite(g) == False] = np.nan

    if data is None:
        return model
    if eps is None:
        return model - data #residual
    return (model - data)/eps
