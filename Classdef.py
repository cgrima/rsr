"""Various python classes for rsr package
Author: Cyril Grima <cyril.grima@gmail.com>
"""

import numpy as np
import pdf, fit, invert
import matplotlib.pyplot as plt
from scipy import interpolate, stats
from astroML.plotting import hist


class Statfit:
    """Class holding statistical fit results
    """
    def __init__(self, sample, func, values, params, chisqr, redchi,
                 elapsed, nfev, message, success, residual, x, n, edges, bins):
        self.sample = sample
        self.func = func
        self.values = values
        self.params = params
        self.chisqr = chisqr
        self.redchi = redchi
        self.elapsed = elapsed
        self.nfev = nfev
        self.message = message
        self.success = success
        self.residual = residual
        self.x = x
        self.n = n
        self.edges = edges
        self.bins = bins


    def power(self, db=True):
        """Total (pt), coherent (pc), and incoherent (pn) components in power
        """
        pt, pc, pn = np.average(self.sample)**2, self.values['a']**2, \
                     2*self.values['s']**2
        if db is True:
            pt, pc, pn = 10*np.log10(pt), 10*np.log10(pc), 10*np.log10(pn)
        return {'pt':pt, 'pc':pc, 'pn':pn, 'pc-pn':pc-pn}


    def crl(self, **kwargs):
        """Correlation coefficient between distribution and theoretical fit
        """
        try:
            out = np.corrcoef(self.n, self.n+self.residual)[0,1]
        except:
            out = np.nan

        if np.isfinite(out) is False:
            out = np.nan
        return out


    def invert(self, frq=60e6, method='spm'):
        """Invert signal components into physical properties
        """
        return getattr(invert, method)(frq, self.power()['pc'],
                       self.power()['pn'])


    def plot(self, ylabel='Normalized Probability', color='k', ls='-',
        bins=None, fbins=100, alpha=.1, method='compound',
        histtype='stepfilled', xlim=None):
        """Plot histogram and pdf
        """
        if xlim is None: xlim = [np.min(self.edges), np.max(self.edges)]
        if bins is None: bins = self.bins

        x = np.linspace(xlim[0], xlim[1], fbins)
        hist(self.sample, bins=bins, color=color, edgecolor=color,
             alpha=alpha, histtype=histtype, normed=True, range=xlim)
        plt.plot(x, self.func(self.values, x, method=method), color=color,
                 ls=ls, linewidth=2)
        plt.xlim(xlim)
        plt.ylabel(ylabel, size=17)
        plt.xlabel('Amplitude', size=17)
        plt.yticks(size='17')
        plt.xticks(size='17')


    def report(self, frq=60e6, inv='spm'):
        """Print a report for the fit
        """
        buff = []
        add = buff.append

        add('[%i] [%6.2f s.] [%3.0f eval.] [%3.3f]\n'
            % (self.flag(), self.elapsed, self.nfev, self.crl()))

        add('%s\n' % (self.message))

        for i, key in enumerate(self.values.keys()):
            add('%s = %.3e, ' % (key, self.params[key].value))

        add('\n')

        for i, key in enumerate(self.power().keys()):
            add('%s = %3.1f dB, ' % (key, self.power()[key]))

        add('\n')

        p = self.invert(frq=frq, method=inv)
        add('%s @ %.0f MHz gives, eps = %.3f, sh = %.2e m' % (inv.upper(),
            frq*1e-6, p['eps'], p['sh']))

        add('\n')

        out = "".join(buff)
        print(out)


    def flag(self):
        """0 is bad data, 1 is good data
        """
        out = self.success * np.isfinite(self.crl()) * (self.crl() > 0)
        return(int(out))
