import numpy as np
from . import pdf as  funcs
import scipy


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


def sample(f, N=1e3, xN=1e3, xlim=(0,1), **kwargs):
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
        
    RETURN
    ------
    N random evaluation over the considered PDF
    """
    uniform_samples = np.random.random(int(N))
    required_samples = inverse_cdf(f, xN=xN, xlim=xlim)(uniform_samples)
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
