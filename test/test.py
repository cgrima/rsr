from rsr import pdf, fit, utils
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt

file_srf = 'rsr/test/MIS_JKB2e_Y37a.srf_cyg.txt'
file_rsr = 'rsr/test/MIS_JKB2e_Y37a.srf_cyg.rsr.txt'
a = read_csv(file_srf, sep='\t')
b1 = read_csv(file_rsr, sep='\t')

amp = 10**(a['PDB']/20)
sample = amp[41000:41999]

#
f = fit.hk(sample, bins=50, param0=fit.hk_param0(sample))

#
b2 = utils.inline_estim(amp, winsize=1000, sampling=250)

#
plt.plot(b2.xo, b2.pt, lw=15, color='k', alpha=.2)
plt.ylim([-50,0])
plt.grid(alpha=.5)
plt.xlabel('Frame')
plt.ylabel('Power [dB]')

#plt.plot(b1.XB, b1.HK_PC, color='blue', lw=5, alpha=.2)
#plt.plot(b1.XB, b1.HK_PN, '-', color='red', lw=5, alpha=.2)

w2 = np.where(b2.crl > .8, True, False)
plt.plot(b2.xo[w2], b2.pc[w2], color='blue', lw=5, alpha=.3)
plt.plot(b2.xo[w2], b2.pn[w2], '-', color='red', lw=5, alpha=.3)


#dB x-axis scale
s2 = insert(10**(np.linspace(-60,0,100)/20.), 0, 0)
f = fit.hk(sample, bins=s2, param0=fit.hk_param0(sample))

