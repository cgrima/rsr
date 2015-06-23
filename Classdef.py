"""Various python classes for rsr package
Author: Cyril Grima <cyril.grima@gmail.com>
"""

import numpy as np
import pdf
import matplotlib.pyplot as plt

class Statfit:
	"""Class holding statistical fit results
	"""
	def __init__(self, sample, func, kws, range, bins, values, params, chisqr,
	             redchi):
		self.sample = sample
		self.func = func
		self.kws = kws
		self.range = range
		self.bins = bins
		self.values = values
		self.params = params
		self.chisqr = chisqr
		self.redchi = redchi
		
	def power(self, db=True):
	    """coherent (pc) and incoherent (pn) components in power 
	    """
	    pc, pn = self.values['a']**2, 2*self.values['s']**2
	    if db is True:
	        pc, pn = 10*np.log10(pc), 10*np.log10(pn)
	    return {'pc':pc, 'pn':pn}
	    
	def histogram(self, **kwargs):
	    """Coordinates for the histogram
	    """
	    return np.histogram(self.sample, bins=self.bins, range=self.range,
	                        density=True, **kwargs)
	    
	def yfunc(self, x=None, **kwargs):
	    """coordinates for the theoretical fit
	    Can change the x coordinates (initial by default)
	    """
	    if x is None:
	        y, edges = self.histogram(**kwargs)
	        x = edges[1:] - abs(edges[1]-edges[0])/2
	    return self.func(self.values, x, **kwargs), x
	    
	def corrcoef(self, **kwargs):
	    """Correlation coefficient between distribution and theoretical fit
	    """
	    ydata, x = self.histogram(**kwargs)
	    y, tmp = self.yfunc()
	    return np.corrcoef(y, ydata)[0,1]
	    
	def plot(self, ylabel='Probability'):
	    y, edges = self.histogram()
	    width = edges[1]-edges[0]
	    xplot = np.linspace(0,1,100)
	    yplot, tmp = self.yfunc(x=xplot)
	    
	    if ylabel is 'Occurences':
	        factor = self.sample.size*width
	    if ylabel is 'Probability':
	        factor = width
	    if ylabel is 'Normalized probability':
	        factor = 1.
	    
	    plt.bar(edges[0:-1], y*factor, width=width, color='.9', edgecolor='.7')
	    plt.plot(xplot, yplot*factor, color='k', linewidth=2)
	    plt.xlim((0,1))
	    plt.ylabel(ylabel, size=17)
	    plt.xlabel('Amplitude', size=17)
	    plt.yticks(size='17')
	    plt.xticks(size='17')
	    
	def report(self):
	    buff = []
	    add = buff.append
	    p = self.params
	    for i, key in enumerate(self.values.keys()):
	        add("\t"+key +"\t= " + " %5.3f" % p[key].value +
	            " +/- " + " %5.3f" % p[key].stderr)
	    return '\n'.join(buff)
