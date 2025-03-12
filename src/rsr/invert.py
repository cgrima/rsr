"""Various tools for inverting terrain properties from signal components
Author: Cyril Grima <cyril.grima@gmail.com>
"""

import numpy as np
import subradar as sr
from scipy import constants as ct


def spm(frq, pc, pn):
    """Use the Small Perturbation Model with large correlation length assumption

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
    while u_equation(k, sh) > u:
        sh = sh+wl*.001

    r = -np.sqrt( pc_lin*np.exp((2*k*sh)**2) )
    eps = (1-r)**2/(1+r)**2

    return {'eps':eps, 'sh':sh}


def srf_coeff(Psc=None, Psn=None, h0=None, wb=None):
    """Invert the received powers (coherent, incoherent) into surface coefficients
    """
    L = sr.utils.geo_loss

    Psc = 10**(Psc/10.)
    Psn = 10**(Psn/10.)

    # Footprints
    As = 2*np.pi*sr.utils.footprint_rad_pulse(h0, wb)**2

    # Surface coefficients
    Rsc = Psc/L(2*h0)
    Rsn = Psn/L(h0)**2/As

    return 10*np.log10(Rsc), 10*np.log10(Rsn)


def bed_coeff(Psc=None, Psn=None, Pbc=None, Pbn=None, n1=None, sh=None,
              h0=None, h1=None, Q1=None, wf=None, wb=None):
    """Use the surface properties and bed powers to get bed reflectivity and
    backscatter coefficients. Signal is assumed to be already calibrated.
    Arguments must be energy in dB
    """
    L = sr.utils.geo_loss

    wk = sr.utils.wf2wk(wf)

    Rsc, Rsn = srf_coeff(Psc=Psc, Psn=Psn, h0=h0, wb=wb)

    Psc = 10**(Psc/10.)
    Psn = 10**(Psn/10.)
    Pbc = 10**(Pbc/10.)
    Pbn = 10**(Pbn/10.)
    Rsc = 10**(Rsc/10.)
    Rsn = 10**(Rsn/10.)
    Q1 = 10**(Q1/10.)

    # Footprints
    As = 2*np.pi*sr.utils.footprint_rad_pulse(h0, wb)**2
    Ab = 2*np.pi*sr.utils.footprint_rad_pulse(h0+h1/n1, wb)**2

    Tsc = np.abs(4*n1/(1+n1)**2) * np.exp(-wk**2*sh**2*(1-n1)**2)
    Tsn = n1*Rsn

    #Coefficients
    b1 = L(2*h0+2*h1)            *Tsc**2

    Rbc = Pbc/b1/Q1**2

    b2 = L(h0+2*h1)*L(h0)        *Tsc*Tsn*As
    b3 = L(h0+h1)*L(h0+h1)       *Tsc**2*Ab
    b4 = L(h0+h1)*L(h0)*L(h1)    *Tsc*Tsn*As*Ab
    b5 = b2
    b6 = L(h0)*L(2*h1)*L(h0)     *Tsn**2*As**2
    b7 = b4
    b8 = L(h0)*L(h1)*L(h0)*L(h1) *Tsn**2*As**2*Ab

    Rbn = (Pbn/Q1**2-Rbc*(b2+b5+b6)) / (b3+b4+b7+b8)

    return 10*np.log10(Rbc), 10*np.log10(Rbn)
