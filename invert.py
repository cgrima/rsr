"""Various tools for inverting terrain properties from signal components
Author: Cyril Grima <cyril.grima@gmail.com>
"""

import numpy as np
from scipy import constants as ct
from numpy import exp, sqrt


def spm(frq, pc, pn):
    """Use the Small Perturbation Model
    
    Arguments
    ---------
    frq : float
        radar frequency [Hz]
    pc : float
        calibrated coherent component [dB]
    pn : float
        calibrated incoherent component [dB]
    """
    
    wl = ct.c/frq
    k= 2*ct.pi/wl
    
    pc_lin = 10**(pc/10.)
    pn_lin = 10**(pn/10.)
    u = pc_lin/pn_lin
    sh = wl*exp(1/(2*u))/(4*ct.pi*sqrt(u))
    r = -sqrt( pc_lin/exp(2*k*sh)**2 )
    eps = (1-r)**2/(1+r)**2
    
    return {'eps':eps, 'sh':sh}
