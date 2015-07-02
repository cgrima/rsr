from rsr import pdf, fit, utils
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt

file = 'rsr/test/MIS_JKB2e_X48a.srf_cyg.txt'
a = read_csv(file, sep='\t')

amp = 10**(a['PDB']/20)

#b = utils.inline_estim(amp, winsize=1000, sampling=500)

sample = amp[1500:2499]

f = fit.hk(sample, bins=50, param0=fit.hk_param0(sample))

f.report()
f.plot()

