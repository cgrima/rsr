# Presentation

This is a Python package providing basic utilities for applying the Radar Statistical Reconnaissance.


# Example

The signal is assumed to be calibrated, so that the amplitude received from a perfect flat mirror would be unity.

Note: the data stored in the data.txtfile are powers in dB


```python

import rsr
import numpy as np
import matplotlib.pyplot as plt
%pylab

# Unit conversion (dB powers to linear amplitudes)
pdb = np.loadtxt('rsr/data.txt')
amp = 10**(pdb/20.)

# Apply RSR to a given subset of amplitude. RSR processing with Homodyned K-distribution fitting, over a histogram binned with the Freedman-Diaconis Rule rule
sample = amp[80000:85000]
f = rsr.fit.lmfit(sample, fit_model='hk', bins='freedman')
f.report() # Display result
f.plot(method='analytic') # Plot results

# Apply RSR along a vector of successive amplitude. The RSR is applied on windows made of 1000 values. Each window is separated by 500 samples (can be time consuming).
b2 = rsr.utils.inline_estim(amp, winsize=1000, sampling=250)
rsr.utils.plot_inline(b2) # Plot results

```


# Citation

Grima, C., Schroeder, D. M., Blankenship, D. D., and Young, D. A. (2014) [Planetary landing zone reconnaissance using ice penetrating radar data: concept validation in Antarctica][1]. Planetary and Space Science 103, 191-204.



  [1]: http://www.sciencedirect.com/science/article/pii/S0032063314002244

