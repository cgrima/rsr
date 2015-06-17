"""
Probability density functions
Author: C. Grima (cyril.grima@gmail.com)
"""

import math 
import numpy as np
from scipy import stats, integrate
from scipy.special import jv, kv, digamma


def gamma(x, mu):
    """Gamma PDF from scipy with adequate variables
    """
    y = stats.gamma.pdf(x, mu, scale = 1)
    return np.nan_to_num(y)


def rayleigh(x, s):
    """Rayleigh PDF from scipy with adequate variables
    """
    return stats.rayleigh.pdf(x, scale = s)


def rice(x, a, s):
    """Rice PDF from scipy with adequate variables
    """
    y = stats.rice.pdf(x, a/s, scale = s)
    return np.nan_to_num(y)


def k(x, s, mu):
    """K PDF
    """
    b = np.sqrt(2*mu)/s
    y = 2*(x/2.)**mu *b**(mu+1.) /math.gamma(mu) *kv(mu-1,b*x)
    return np.nan_to_num(y)


def hk(x, a, s, mu, method='compound'):
    """Homodyne-K PDF. Choose the method:
    analytic: Analytic representation. 
    compound: Compound representation is from [Destrempes and Cloutier, 2010,
    Ultrasound in Med. and Biol. 36(7): 1037 to 1051, Equation 16]
    """
    #Function integrand if any
    def integrand(w, x, a, s, mu, method=method):
        if method == 'analytic':
            print('UNSTABLE: use compound instead')
            y = w *jv(0, w*a) *jv(0, w*x) *(1. +w**2*s**2/(2.*mu))**-mu
        if method == 'compound':
            y = rice(x, a, np.sqrt(s**2*w/mu)) * gamma(w, mu)
        return np.nan_to_num(y)
    #Function
    def func(x, a, s, mu, method=method):      
        if method == 'analytic' or method == 'compound':
            y = integrate.quad(integrand, 0., np.inf, args=(x, a, s, mu), 
                epsrel = -1, epsabs = 1)[0]
        return np.nan_to_num(y)
    #Check x size
    x = np.array(x)
    if x.size == 1: y = func(x, a, s, mu)
    else: y = [func(i, a, s, mu) for i in x]
    return np.nan_to_num(y)
