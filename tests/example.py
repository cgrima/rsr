#!/usr/bin/env python3

import sys
sys.path.insert(1, '..')
import rsr
#import wget
import pandas as pd
import matplotlib.pyplot as plt

# Load data (example is non-calibrated surface echo linear amplitudes from SHARAD orbit 0887601)
#data_filename = wget.download('https://raw.githubusercontent.com/cgrima/rsr/master/rsr/data.txt')
data_filename = 'data.txt'
data = pd.read_csv(data_filename)
amp = data['amp'].values

# Apply RSR to a given subset of amplitude.
sample = amp[50000:85000]
f = rsr.run.processor(sample, fit_model='hk')
_ = f.crl()
f.report()
f.plot()
plt.close()
