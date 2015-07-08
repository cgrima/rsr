"""
Various tools for application of the rsr
Author: Cyril Grima <cyril.grima@gmail.com>
"""

import numpy as np
import pdf, fit
import time
from pandas import DataFrame



def inline_estim(vec, stat='hk', winsize=1000., sampling=100.,
                 save='.inline_estim_last', verbose=True, **kws):
    """Histogram statistical estimation over windows sliding along a vector
    
    Arguments
    ---------
    vec : sequence
        A vector of linear amplitude values
    
    Keywords
    --------
    stat : string
        stat to use to estimate the histogram statistics (in .fit)
    winsize : int
        number of amplitude values within a window
    sampling : int
        window repeat rate
    save : string
        file name (without extension) to save the results in an ascii file
    verbose : bool
        Display fit results informations
    """
    start = time.time()

    x = np.arange(vec.size) #vector index
    xa = x[:x.size-sampling/2.:sampling] #windows starting coordinate
    xb = xa + winsize-1 #window end coordinate
    if xb[-1] > x[-1]: xb[-1] = x[-1] #cut last window in limb
    xo = [val+(xb[i]-val)/2. for i, val in enumerate(xa)]

    columns = ['xa', 'xb', 'xo', 'pt', 'pc', 'pn', 'crl', 'mu', 'flag']
    index = np.arange(xa.size)
    table = DataFrame({'xa':xa, 'xb':xb, 'xo':xo},
                      index=index, columns=columns) # Table to hold the results

    for i, val in enumerate(xo): # Histogram estimation
        if verbose is True:
            print('ITER '+ str(i+1) + '/' + str(xa.size) +
            ' (observations ' + str(xa[i]) + ':' + str(xb[i]) + ')')
            
        sample = vec[xa[i]:xb[i]]
        param0 = getattr(fit, stat+'_param0')(sample)
        p = getattr(fit, stat)(sample, param0=param0, kws=kws)
        
        table['pt'][i] = p.power()['pt']
        table['pc'][i] = p.power()['pc']
        table['pn'][i] = p.power()['pn']
        table['crl'][i] = p.crl()
        table['mu'][i] = p.values['mu']
        table['flag'][i] = int(p.success*p.crl() > 0)
        
        if verbose is True:
            p.report()

    elapsed = time.time() - start
    print('DURATION: %4.1f min.' % (elapsed/60.))

    if save is not None:
        ext = '.'+stat
        table.to_csv(save+ext+'.txt', sep='\t', index=False, float_format='%.3f')

    return table



def hk_pdf_sample(params, x):
    """Generate a hk distribution with noise. Use it fit testing
    
    Arguments
    ---------
    params : dict
        parameters for the HK function
    x : sequence
        x coordinates
    """
    # Extract parameters
    a = params['s']
    s = params['s']
    mu = params['mu']
    # Generate noisy hk distribution
    x = np.linspace(0,1,100)
    y = pdf.hk(params, x)
    noise = np.random.normal(0,max(y)*.05,x.size)
    sample = (y-noise) *((y-noise) >= 0)
    return sample
