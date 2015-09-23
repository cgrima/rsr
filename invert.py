"""Various tools for inverting terrain properties from signal components
Author: Cyril Grima <cyril.grima@gmail.com>
"""

import numpy as np
from scipy import constants as ct


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
    
    def u_equation(k, sh):
        a = 2*k*sh
        return np.exp(-a**2)/a**2

    sh = wl*.001
    while (u_equation(k, sh) > u):
        sh = sh+wl*.001
    
    r = -np.sqrt( pc_lin*np.exp((2*k*sh)**2) )
    eps = (1-r)**2/(1+r)**2
    
    return {'eps':eps, 'sh':sh}
