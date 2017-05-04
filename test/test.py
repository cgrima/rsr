from rsr import pdf, fit, utils
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
%pylab

# Open data file (values are powers in dB)
file_srf = 'rsr/test/data.txt'
a = read_csv(file_srf, sep='\t')

# convert signal int0 linear amplitude
amp = 10**(a['PDB'].values/20)

# Apply RSR to a given subset of amplitude
sample = amp[80000:85000]
f = fit.lmfit(sample, fit_model='hk', bins='knuth')
f.report() # Display result
f.plot(method='analytic') # Plot results.

# Apply RSR along a vector of successive amplitude
b2 = utils.inline_estim(amp, winsize=1000, sampling=30000)
utils.plot_inline(b2) # Plot results
