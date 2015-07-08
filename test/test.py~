from rsr import pdf, fit, utils
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt

file_srf = 'rsr/test/MIS_JKB2e_Y37a.srf_cyg.txt'
file_rsr = 'rsr/test/MIS_JKB2e_Y37a.srf_cyg.rsr.txt'
a = read_csv(file_srf, sep='\t')
b = read_csv(file_rsr, sep='\t')

amp = 10**(a['PDB']/20)

#b = utils.inline_estim(amp, winsize=1000, sampling=500)

sample = amp[46000:46999]

f = fit.hk(sample, bins=50, param0=fit.hk_param0(sample))

f.report()
f.plot()

