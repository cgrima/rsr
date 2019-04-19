# Presentation

This is a Python package providing basic utilities for applying the Radar Statistical Reconnaissance.


# Requirements


numpy > 1.11.0



# Example




```python

import rsr
import numpy as np
import matplotlib.pyplot as plt
%pylab

# Load data (example is non-calibrated surface echo linear amplitudes from SHARAD orbit 0887601)
data = np.genfromtxt('rsr/data.txt', dtype=float, delimiter=',', names=True)
amp = data['amp']

# Apply RSR to a given subset of amplitude.
sample = amp[80000:85000]
f = rsr.run.processor(sample, fit_model='hk')
f.plot() # Plot results

# Apply RSR along a vector of successive amplitude.
# The RSR is applied on windows made of 1000 values. Each window is separated by
# 500 samples (can be time consuming).
a = rsr.run.along(amp, winsize=1000, sampling=250, nbcores=2)
rsr.utils.plot_along(a) # Plot results
```




# Citation

Grima, C., Schroeder, D. M., Blankenship, D. D., and Young, D. A. (2014) [Planetary landing zone reconnaissance using ice penetrating radar data: concept validation in Antarctica][1]. Planetary and Space Science 103, 191-204.



  [1]: http://www.sciencedirect.com/science/article/pii/S0032063314002244

