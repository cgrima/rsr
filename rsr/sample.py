import numpy as np
import scipy
from . import pdf as  funcs
from . import run


def inverse_cdf(f, xN=1e3, xlim=(0,1), **kwargs):
    """Provides the inverse cumulative density function (CDF) of a probability
    density function

    ARGUMENTS
    ---------
    f : function
        The PDF to evaluate
    xN : integer
        The x-axis sampling of the PDF
    xlim : (integer, integer)
        The x-axis range overwhich the pdf is considered

    RETURN
    ------
    the inverse CDF
    """
    # Probability Density Function
    x = np.linspace(xlim[0],xlim[1],int(xN))
    #func = f(x)
    # Cumulative Distribution Function
    cdf = np.cumsum(f(x))
    # Normalisation to 1
    cdf = cdf/cdf.max()
    # Inverse CDF
    return scipy.interpolate.interp1d(cdf,x)


def sample(f, N=1e3, xN=1e3, xlim=(0,1), precision=None, **kwargs):
    """Provides a random trial X over a given PDF following the Inverse
    Transform Sampling technique so that X=F^{-1}(U), where F^{-1} is the
    inverse cumulative distribution function of the PDF and U is a random
    variable uniformly distributed between 0 and 1.

    ARGUMENTS
    ---------
    f : function
        The PDF to evaluate
    N : Integer
        The number of random evaluation to make
    xN : integer
        The x-axis sampling of the PDF
    xlim : (integer, integer)
        The x-axis range overwhich the pdf is considered
    precision : float
        precision on the measurement [dB]

    RETURN
    ------
    N random evaluation over the considered PDF
    """
    uniform_samples = np.random.random(int(N))
    required_samples = inverse_cdf(f, xN=xN, xlim=xlim)(uniform_samples)

    if precision:
        for i, rsample in enumerate(required_samples):
            linear_precision = np.abs(10**(precision/2/20) -
                                      10**(-precision/2/20))*rsample
            required_samples[i] = np.random.normal(rsample, linear_precision)

    return required_samples


def pdf(func, params, method=None, **kwargs):
    """Provides a random trial X over a PDF in rsr.pdf

    ARGUMENTS
    ---------
    func : string
        PDF name in rsr.pdf
    params : dict
        parameters to be passed to the PDF function
    kwargs : dict
        Any arguments used bu sample.sample

    RETURN
    ------
    N random evaluation over the considered PDF
    """
    if method is None:
        f = lambda x: getattr(funcs, func)(params, x,)
    else:
        f = lambda x: getattr(funcs, func)(params, x, method=method)
    return sample(f, **kwargs)


def rsr(func, params, method=None, **kwargs):
    """Apply the RSR over a set of generated random amplitudes

    ARGUMENTS
    ---------
    func : string
        PDF name in rsr.pdf
    params : dict
        parameters to be passed to the PDF function
    kwargs : dict
        Any arguments used bu sample.sample

    RETURN
    ------
    N random evaluation over the considered PDF
    """
    amp = pdf(func, params, method=method, **kwargs)
    p = run.processor(amp)
    return p


def power_to_params(params):
    """Converts Pc and Pn powers in dB to a params dictionary
    """
    pc_db, pn_db, mu = params['pc'], params['pn'], params['mu']
    pc, pn = 10**(pc_db/10), 10**(pn_db/10)
    params['a'] = np.sqrt(pc)
    params['s'] = np.sqrt(pn/2./mu)
    return params


def params_to_power(params, dB=True):
    """Converts a params dict into pc and pn powers in dB
    """
    mu = params['mu']
    pc = params['a']**2
    pn = 2*params['s']**2*mu
    if dB:
        pc = 10*np.log10(pc)
        pn = 10*np.log10(pn)

    params['pc'] = pc
    params['pn'] = pn
    return params


def effective_precision(func, params, Nsets=1, **kwargs):
    """Gets the effective precision applicable to the derivation of Pc and Pn
    from the rsr algorithm. The precision is the median of Nsets of histograms
    randomly obtained from params.

    ARGUMENTS
    ---------
    func : string
        PDF name in rsr.pdf
    params : dict
        parameters to be passed to the PDF function
    Nsets: integer
        Number of amplitude histogram to draw
    kwargs : dict
        Any arguments used bu sample.sample

    RETURN
    ------
    pc and pn median precisions

   """
    power = params_to_power(params, dB=True)
    ps = [rsr(func, params, **kwargs) for i in np.arange(Nsets)]
    d_pcs = [p.power()['pc']-power['pc'] for p in ps]
    d_pns = [p.power()['pn']-power['pn'] for p in ps]
    out = {
        'ps':ps, 'power':power, 'd_pcs':d_pcs, 'd_pns':d_pns,
        'd_pc':np.median(d_pcs), 'd_pn':np.median(d_pns)
    }
    return out
