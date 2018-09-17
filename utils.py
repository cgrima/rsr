"""
Various tools for application of the rsr
Author: Cyril Grima <cyril.grima@gmail.com>
"""

import numpy as np
from . import pdf
from . import fit
from . import invert
import time
from pandas import DataFrame
import matplotlib.pyplot as plt


def inline_estim(vec, fit_model='hk', bins='knuth', inv='spm', winsize=1000.,
                 sampling=250., frq=60e6, save='.inline_estim_last',
                 verbose=True, **kwargs):
    """Histogram statistical estimation over windows sliding along a vector

    Arguments
    ---------
    vec : sequence
        A vector of linear amplitude values

    Keywords
    --------
    fit_model : string
        pdf to use to estimate the histogram statistics (in fit.)
    bins : string
        method to compute the bin width (inherited from astroML.plotting.hist)
    inv : string
        inversion method (in invert.)
    winsize : int
        number of amplitude values within a window
    sampling : int
        window repeat rate
    frq : float
        Radar center frequency [Hz]
    save : string
        file name (without extension) to save the results in an ascii file
    verbose : bool
        Display fit results informations
    """
    start = time.time()

    #--------------------------------------------------------------------------
    # Windows along-track
    #--------------------------------------------------------------------------
    x = np.arange(vec.size) #vector index
    #xa = x[:x.size-winsize:sampling] #windows starting coordinate
    xa = x[:np.int(x.size-winsize):np.int(sampling)] #windows starting coordinate
    xb = xa + winsize-1 #window end coordinate
    if xb[-1] > x[-1]: xb[-1] = x[-1] #cut last window in limb
    xo = [val+(xb[i]-val)/2. for i, val in enumerate(xa)]

    #--------------------------------------------------------------------------
    # Table to hold the results
    #--------------------------------------------------------------------------
    columns = ['xa', 'xb', 'xo', 'pt', 'pc', 'pn', 'crl', 'chisqr', 'mu',
               'eps', 'sh', 'flag']
    index = np.arange(xa.size)
    table = DataFrame({'xa':xa, 'xb':xb, 'xo':xo},
                      index=index, columns=columns)

    #--------------------------------------------------------------------------
    # Fit for all the windows
    #--------------------------------------------------------------------------
    for i, val in enumerate(xo):
        if verbose is True:
            print('ITER '+ str(i+1) + '/' + str(xa.size) +
            ' (observations ' + str(xa[i]) + ':' + str(xb[i]) + ')')

        sample = vec[int(xa[i]):int(xb[i])]
        p = fit.lmfit(sample, fit_model=fit_model, bins=bins)

        table.loc[i, 'pt'] = p.power()['pt']
        table.loc[i, 'pc'] = p.power()['pc']
        table.loc[i, 'pn'] = p.power()['pn']
        table.loc[i, 'crl'] = p.crl()
        table.loc[i, 'chisqr'] = p.chisqr
        table.loc[i, 'mu'] = p.values['mu']
        table.loc[i, 'eps'] = p.invert(frq=frq, method=inv)['eps']
        table.loc[i, 'sh'] = p.invert(frq=frq, method=inv)['sh']
        table.loc[i, 'flag'] = p.flag()

        if verbose is True:
            p.report(frq=frq)

    #--------------------------------------------------------------------------
    # Output
    #--------------------------------------------------------------------------
    elapsed = time.time() - start
    print('DURATION: %4.1f min.' % (elapsed/60.))

    if save is not None:
        ext = '.'+fit_model+'.'+inv
        table.to_csv(save+ext+'.txt', sep='\t', index=False, float_format='%.3f')

    return table


def plot_inline(a, frq=60e6, title=''):
    """Plot infos from a DataFrame ceated by inline_estim

    Arguments
    ---------
    a : DataFrame
        inline_estim output
    """
    w = np.where(a.flag)
    x = a.xo.values.astype('float')[w]
    crl = a.crl.values.astype('float')[w]
    pt = a.pt.values.astype('float')[w]
    pc = a.pc.values.astype('float')[w]
    pn = a.pn.values.astype('float')[w]
    eps = a.eps.values.astype('float')[w]
    sh = a.sh.values.astype('float')[w]

    plt.figure(figsize=(15,10))

    #--------------------------------------------------------------------------
    # Correlation Coefficient
    #--------------------------------------------------------------------------
    ax_crl = plt.subplot2grid((5, 1), (0, 0))
    plt.plot(x, crl, 'o-', color='k')
    plt.grid(alpha=.5)
    plt.ylabel(r'Correl. Coeff.', size=17)
    plt.xticks(size='10')
    plt.yticks(size='15')
    plt.title(title, size='15')

    #--------------------------------------------------------------------------
    # Signal components
    #--------------------------------------------------------------------------
    ax_pwr = plt.subplot2grid((5,1), (1, 0), rowspan=2)
    ax_pwr.fill_between(x, pc, pn, where=pc>=pn, facecolor='k', alpha=.05, interpolate=True)
    ax_pwr.fill_between(x, pc, pn, where=pc<=pn, facecolor='k', alpha=.4, interpolate=True)
    plt.plot(x, pc, color='k', lw=3, alpha=.9, label=r'Reflectance $(P_c)$')
    plt.plot(x, pn, color='k', lw=3, alpha=.6, label=r'Scattering $(P_n)$')
    plt.ylim([-40,0])
    plt.grid(alpha=.5)
    plt.ylabel(r'Power $[dB]$', size=17)
    plt.yticks(size='15')
    plt.xticks(size='10')
    plt.legend(loc='lower right', fancybox=True).get_frame().set_alpha(0.5)

    #--------------------------------------------------------------------------
    # Permittivity
    #--------------------------------------------------------------------------
    ax_eps = plt.subplot2grid((5,1), (3, 0), rowspan=2)
    plt.semilogy(x, eps, color='k', lw=3, alpha=.9, label=r'Permittivity $(\epsilon)$')
    plt.ylim(1,100)
    plt.grid(True, which='both', alpha=.5)
    plt.ylabel('Permittivity', size=17)
    plt.xticks(size='10')
    plt.xlabel('Frame #', size=12)
    plt.yticks(size='15')
    ax_eps.set_yticks([1, 10, 100])
    ax_eps.set_yticklabels(['1', '10', '100'])

    #--------------------------------------------------------------------------
    # RMS height
    #--------------------------------------------------------------------------
    ax_sh = ax_eps.twinx()
    plt.semilogy(x, sh, '-', color='k', lw=3, alpha=.3, label=r'RMS height $(\sigma_h)$')
    plt.semilogy(x, eps, color='k', lw=3, alpha=.9, label=r'Permittivity $(\epsilon)$')
    plt.ylim(0.01,1)
    plt.ylabel(r'RMS height $[m]$', size=17)
    plt.yticks(size='15')
    ax_sh.set_yticks([.01, .1, 1])
    ax_sh.set_yticklabels(['0.01', '0.1', '1'])
    ax_sh.set
    plt.legend(loc='upper right', fancybox=True).get_frame().set_alpha(0.5)
