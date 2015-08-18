"""
Various tools for application of the rsr
Author: Cyril Grima <cyril.grima@gmail.com>
"""

import numpy as np
import pdf, fit, invert
import time
from pandas import DataFrame
import matplotlib.pyplot as plt



def inline_estim(vec, stat='hk', inv='spm', winsize=1000., sampling=100.,
                 frq=60e6, save='.inline_estim_last', verbose=True, **kws):
    """Histogram statistical estimation over windows sliding along a vector
    
    Arguments
    ---------
    vec : sequence
        A vector of linear amplitude values
    
    Keywords
    --------
    stat : string
        stat to use to estimate the histogram statistics (in fit.)
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

    x = np.arange(vec.size) #vector index
    xa = x[:x.size-sampling/2.:sampling] #windows starting coordinate
    xb = xa + winsize-1 #window end coordinate
    if xb[-1] > x[-1]: xb[-1] = x[-1] #cut last window in limb
    xo = [val+(xb[i]-val)/2. for i, val in enumerate(xa)]

    columns = ['xa', 'xb', 'xo', 'pt', 'pc', 'pn', 'crl', 'mu', 'eps', 'sh',
               'flag']
    index = np.arange(xa.size)
    table = DataFrame({'xa':xa, 'xb':xb, 'xo':xo},
                      index=index, columns=columns) # Table to hold the results

    for i, val in enumerate(xo): # Histogram estimation
        if verbose is True:
            print('ITER '+ str(i+1) + '/' + str(xa.size) +
            ' (observations ' + str(xa[i]) + ':' + str(xb[i]) + ')')
            
        sample = vec[int(xa[i]):int(xb[i])]
        param0 = getattr(fit, stat+'_param0')(sample)
        p = getattr(fit, stat)(sample, param0=param0, kws=kws)
        v = p.invert(frq=frq, method=inv)
        
        table.loc[i, 'pt'] = p.power()['pt']
        table.loc[i, 'pc'] = p.power()['pc']
        table.loc[i, 'pn'] = p.power()['pn']
        table.loc[i, 'crl'] = p.crl()
        table.loc[i, 'mu'] = p.values['mu']
        table.loc[i, 'eps'] = v['eps']
        table.loc[i, 'sh'] = v['sh']
        table.loc[i, 'flag'] = int(p.success*p.crl() > 0)
        
        #table.set_value('pt', i, p.power()['pt'])
        #table.set_value('pc', i, p.power()['pc'])
        #table.set_value('pn', i, p.power()['pn'])
        #table.set_value('crl', i, p.crl())
        #table.set_value('mu', i, p.values['mu'])
        #table.set_value('eps', i, v['eps'])
        #table.set_value('sh', i, v['sh'])
        #table.set_value('flag', i, int(p.success*p.crl() > 0))
        
        if verbose is True:
            p.report(frq=frq)

    elapsed = time.time() - start
    print('DURATION: %4.1f min.' % (elapsed/60.))

    if save is not None:
        ext = '.'+stat+'.'+inv
        table.to_csv(save+ext+'.txt', sep='\t', index=False, float_format='%.3f')

    return table


def plot_inline(a, frq=60e6):
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
    inv = invert.spm(60e6, pc, pn)
    eps = inv['eps']
    sh = inv['sh']
    
    plt.figure(figsize=(15,10))
    
    ax_crl = plt.subplot2grid((5, 1), (0, 0)) # Correlation Coefficient
    plt.plot(x, crl, 'o-', color='k')
    plt.grid(alpha=.5)
    plt.ylabel(r'Correl. Coeff.', size=17)
    plt.xticks(size='10')
    plt.yticks(size='15')
    #plt.tick_params(labelbottom=False)

    ax_pwr = plt.subplot2grid((5,1), (1, 0), rowspan=2) # Signal components
    #plt.plot(x, pt, lw=10, color='k', alpha=.2, label=r'Total $(P_t)$')
    ax_pwr.fill_between(x, pc, pn, where=pc>=pn, facecolor='k', alpha=.05, interpolate=True)
    ax_pwr.fill_between(x, pc, pn, where=pc<=pn, facecolor='k', alpha=.4, interpolate=True) 
    plt.plot(x, pc, color='k', lw=3, alpha=.9, label=r'Reflectance $(P_c)$')
    plt.plot(x, pn, color='k', lw=3, alpha=.6, label=r'Scattering $(P_n)$')
    plt.ylim([-40,0])
    plt.grid(alpha=.5)
    plt.ylabel(r'Power $[dB]$', size=17)
    plt.yticks(size='15')
    plt.xticks(size='10')
    #plt.tick_params(labelbottom=False)
    plt.legend(loc='lower right', fancybox=True).get_frame().set_alpha(0.5)
    
    ax_eps = plt.subplot2grid((5,1), (3, 0), rowspan=2) # Permittivity
    plt.semilogy(x, eps, color='k', lw=3, alpha=.9, label=r'Permittivity $(\epsilon)$')
    plt.ylim(1,100)
    plt.grid(True, which='both', alpha=.5)
    plt.ylabel('Permittivity', size=17)
    plt.xticks(size='10')
    plt.xlabel('Frame #', size=12)
    plt.yticks(size='15')
    ax_eps.set_yticks([1, 10, 100])
    ax_eps.set_yticklabels(['1', '10', '100'])

    ax_sh = ax_eps.twinx() #RMS height
    plt.semilogy(x, sh, '-', color='k', lw=3, alpha=.3, label=r'RMS height $(\sigma_h)$')
    plt.semilogy(x, eps, color='k', lw=3, alpha=.9, label=r'Permittivity $(\epsilon)$')
    plt.ylim(0.01,1)
    #plt.xlabel('Frame', size=17)
    plt.ylabel(r'RMS height $[m]$', size=17)
    #ax_sh.set_xticklabels([],size='15', rotation=90)
    plt.yticks(size='15')
    ax_sh.set_yticks([.01, .1, 1])
    ax_sh.set_yticklabels(['0.01', '0.1', '1'])
    ax_sh.set
    plt.legend(loc='upper right', fancybox=True).get_frame().set_alpha(0.5)


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
