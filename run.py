"""
Wrappers for running RSR processing
"""

from . import fit

import numpy as np


def timing(func):
    """Outputs the time a function takes to execute.
    """
    def func_wrapper(*args, **kwargs):
        t1 = time.time()
        func(*args, **kwargs)
        t2 = time.time()
        print("- Processed in %.1f s.\n" % (t2-t1))
    return func_wrapper


def scale(amp):
    """Provide a factor to scale a set of amplitudes between 0 and 1
    for correct rsr processing through fit.lmfit
    """
    y, x = np.histogram(np.abs(amp), bins='fd')
    pik = x[y.argmax()]
    out = 1/(pik*10)
    return out


def processor(amp, gain=0., bins='stone', fit_model='hk', scaling=True, **kwargs):
    """Apply RSR over a sample of amplitudes

    Arguments
    ---------
    amp: float array
        Linear amplitudes

    Keywords
    --------
    gain: float
        Gain (in dB power) to add to the amplitudes
    bins: string
        Method to compute the bin width (inherited from numpy.histogram)
    fit_model: string
        Name of the function (in pdf module) to use for the fit
    scaling: boolean
        Whether to scale the amplitudes before processing.
        That ensures a correct fit in case the amplitudes are << 1

    Return
    ------
    A Statfit class
    """
    # Gain and Scaling
    amp = amp * 10**(gain/20.)
    scale_amp = scale(amp) if scaling is True else 1
    amp = amp*scale_amp

    # Fit
    a = fit.lmfit( np.abs(amp), bins=bins, fit_model=fit_model)

    # Remove Scaling
    pc = 10*np.log10( a.values['a']**2  ) - 20*np.log10(scale_amp)
    pn = 10*np.log10( 2*a.values['s']**2  ) - 20*np.log10(scale_amp)
    a.sample = amp/scale_amp
    a.values['a'] = np.sqrt( 10**(pc/10.) )
    a.values['s'] = np.sqrt( 10**(pn/10.)/2. )

    return a


def frames(x ,winsize=1000., sampling=250, **kwargs):
    """
    Defines along-track frames coordinates for rsr application

    Arguments
    ---------
    x: float array
        vector index

    Keywords
    --------
    winsize: int
        Number of elements in a window
    sampling: int
        Sampling step
    """
    # Window first and last id
    xa = x[:np.int(x.size-winsize):np.int(sampling)]
    xb = xa + winsize-1

    # Cut last window in limb
    if xb[-1] > x[-1]: xb[-1] = x[-1]
    xo = [val+(xb[i]-val)/2. for i, val in enumerate(xa)]

    # Output
    out = [ np.array([xa[i], xb[i]]).astype('int64') for i in np.arange(xa.size)  ]

    return out

