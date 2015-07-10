from rsr import pdf, fit, utils
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt

# Open data file
file_srf = 'rsr/test/MIS_JKB2e_Y37a.srf_cyg.txt'
a = read_csv(file_srf, sep='\t')

# convert signal inot linear amplitude
amp = 10**(a['PDB']/20)

# Apply RSR to a given subset of amplitude
sample = amp[41000:41999]
f = fit.hk(sample, bins=50, param0=fit.hk_param0(sample))
f.plot() # Display result
f.report() # Plot resuts. Set method='analytic' if curve is bad

# Apply RSR along a vector of successive amplitude
b2 = utils.inline_estim(amp, winsize=1000, sampling=30000)
utils.plot_inline(b2) # Plot results

# dB x-axis scale
#s2 = insert(10**(np.linspace(-60,0,100)/20.), 0, 0)
#f = fit.hk(sample, bins=s2, param0=fit.hk_param0(sample))

