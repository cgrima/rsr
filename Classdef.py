"""Various python classes for rsr package
Author: Cyril Grima <cyril.grima@gmail.com>
"""

import numpy as np
import pdf, fit
import matplotlib.pyplot as plt
from scipy import interpolate

class Statfit:
    """Class holding statistical fit results
    """
    def __init__(self, sample, func, kws, range, bins, values, params, chisqr,
                 redchi, elapsed, nfev, message, success):
        self.sample = sample
        self.func = func
        self.kws = kws
        self.range = range
        self.bins = bins
        self.values = values
        self.params = params
        self.chisqr = chisqr
        self.redchi = redchi
        self.elapsed = elapsed
        self.nfev = nfev
        self.message = message
        self.success = success


    def power(self, db=True):
        """coherent (pc) and incoherent (pn) components in power 
        """
        pt, pc, pn = np.average(self.sample)**2, self.values['a']**2, 2*self.values['s']**2
        if db is True:
            pt, pc, pn = 10*np.log10(pt), 10*np.log10(pc), 10*np.log10(pn)
        return {'pt':pt, 'pc':pc, 'pn':pn}


    def histogram(self, bins=None):
        """Coordinates for the histogram
        """
        if bins is None:
            bins = self.bins
        return np.histogram(self.sample, bins=bins, range=self.range,
                            density=True)


    def yfunc(self, x=None, method='compound'):
        """coordinates for the theoretical fit
        Can change the x coordinates (initial by default)
        """
        if x is None:
            y, edges = self.histogram()
            x = edges[1:] # - abs(edges[1]-edges[0])/2
        return self.func(self.values, x, method=method), x


    def crl(self, **kwargs):
        """Correlation coefficient between distribution and theoretical fit
        """
        ydata, x = self.histogram(**kwargs)
        y, tmp = self.yfunc()
        return np.corrcoef(y, ydata)[0,1]


    def plot(self, ylabel='Probability', color='k', alpha=.1,
             method='compound', bins=None):
        """Plot histogram and pdf
        """
        y, edges = self.histogram(bins=bins)
        width = np.array([abs(edges[i+1] - edges[i]) for i, val in enumerate(y)])
        xplot = np.linspace(0,1,100)
        yplot, xplot = self.yfunc(x=xplot, method=method)
        if ylabel is 'Occurences':
            factor = self.sample.size*width
            factor2 = interpolate.interp1d(edges[0:-1], factor,
                                          bounds_error=False)(xplot)
        if ylabel is 'Probability':
            factor = width
            factor2 = interpolate.interp1d(edges[0:-1], factor,
                                          bounds_error=False)(xplot)
        if ylabel is 'Normalized_probability':
            factor = factor2 = 1.

        plt.bar(edges[0:-1], y*factor, width=width,
                color=color, edgecolor=color, alpha=alpha)
        plt.plot(xplot, yplot*factor2, color=color, linewidth=2)
        plt.xlim((0,1))
        plt.ylabel(ylabel, size=17)
        plt.xlabel('Amplitude', size=17)
        plt.yticks(size='17')
        plt.xticks(size='17')


    def report(self):
        """Print a report for the fit
        """
        buff = []
        add = buff.append

        add('[%6.2f s.] [%3.0f eval.] [%s] %s\n' 
            % (self.elapsed, self.nfev, self.success, self.message))
        for i, key in enumerate(self.values.keys()):
            add('%s = %5.3f, ' % (key, self.params[key].value))
        add('crl = %3.3f\n' % (self.crl()))
        for i, key in enumerate(self.power().keys()):
            add('%s = %3.1f dB, ' % (key, self.power()[key]))
        add("\n")
        out = "".join(buff)
        print(out)
