"""Various python classes for rsr package
"""

from typing import Callable, Dict
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

import subradar as sr
from . import invert
import lmfit

@dataclass
class Statfit:
    """Class holding statistical fit results
    """

    sample: np.ndarray
    # PDF function used
    func: Callable
    # lmfit minimization results
    values: Dict[str, float]
    params: lmfit.Parameters
    chisqr: float
    redchi: float
    elapsed: float
    nfev: float
    message: str
    success: bool
    residual: np.ndarray
    # Histogram bin centers and values
    x: np.ndarray
    n: np.ndarray
    edges: np.ndarray
    bins: np.ndarray

    def power(self, db=True):
        """Total (pt), coherent (pc), and incoherent (pn) components in power
        """
        pt, pc, pn = np.average(self.sample)**2, self.values['a']**2, \
                     2*self.values['s']**2*self.values['mu']
        mu = self.values['mu']
        if db:
            pt, pc, pn = 10*np.log10(pt), 10*np.log10(pc), 10*np.log10(pn)
        if not self.success:
            pt, pc, pn, mu = 0, 0, 0, 0 # TODO: should these be floats?
        return {'pt':pt, 'pc':pc, 'pn':pn, 'pc-pn':pc-pn, 'mu':mu}


    def crl(self, **kwargs):
        """Correlation coefficient between distribution and theoretical fit
        """
        try:
            out = np.corrcoef(self.n, self.n+self.residual)[0,1]
        except: # TODO: make this more specific
            out = 0.

        if (not np.isfinite(out)) or (not self.success):
            out = 0.
        return out


    def invert(self, frq=60e6, th_max=1e-4, cl_logrange=[5], n=1e4,
        ep_range=[1,100], method='iem', approx='Small_S', ):
        """Invert signal components into physical properties
        """
        if method == 'spm':
            out =  getattr(invert, method)(frq, self.power()['pc'],
                   self.power()['pn'])
        if method == 'iem':
            out = sr.invert.power2srf_norminc(method, approx,
            self.power()['pc'], self.power()['pn'], ep_range=ep_range,
            th_max=th_max, wf=frq, cl_logrange=cl_logrange, n=n)
        return out


    def plot(self, ylabel='Normalized Probability', xlabel='Amplitude',
        color='k', ls='-', bins=None, fbins=100, alpha=.1,
        method='compound', histtype='stepfilled', xlim=None):
        """Plot histogram and pdf
        """
        if bins is None:
            bins = self.bins

        # TODO: fix unused histogram arguments fbins, alpha, histtype
        _, edges, _ = plt.hist(self.sample, bins=bins, density=True)
        x = [ val-(val-edges[i-1])/2. for i, val in enumerate(edges) ][1:]

        plt.plot(x, self.func(self.values, x, method=method), color=color,
                 ls=ls, linewidth=2)
        plt.xlim(xlim)
        plt.ylabel(ylabel, size=17)
        plt.xlabel(xlabel, size=17)
        plt.yticks(size='17')
        plt.xticks(size='17')


    def report(self, frq=60e6, inv='spm', file=None):
        """Print a report for the fit
        """
        buff = []
        add = buff.append

        add('[%i] [%6.2f s.] [%3.0f eval.] [%3.3f]\n'
            % (self.flag(), self.elapsed, self.nfev, self.crl()))

        add('%s\n' % (self.message))

        pwr = self.power()
        for key, val in pwr.items():
            add('%s = %3.1f dB, ' % (key, val))

        add('\n')

        p = self.invert(frq=frq, method=inv)
        add('%s @ %.0f MHz gives, eps = %.3f, sh = %.2e m' % (inv.upper(),
            frq*1e-6, p['eps'], p['sh']))

        add('\n')

        out = "".join(buff)
        print(out, file=file)


    def flag(self):
        """0 is bad data, 1 is good data
        """
        out = self.success * np.isfinite(self.crl()) * (self.crl() > 0)
        return int(out)

