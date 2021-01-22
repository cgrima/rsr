# Presentation

This is a Python package providing basic utilities for applying the Radar Statistical Reconnaissance.

# Installation

`pip install rsr`

# Example

Below, 'f' is a Statfit python Class that holds the results of the rsr processing ([description here](https://github.com/cgrima/rsr/blob/master/rsr/Classdef.py)) 

```python

import rsr
import wget
import pandas as pd
import matplotlib.pyplot as plt
%pylab

# Load data (example is non-calibrated surface echo linear amplitudes from SHARAD orbit 0887601)
data_filename = wget.download('https://raw.githubusercontent.com/cgrima/rsr/master/rsr/data.txt')
data = pd.read_csv(data_filename)
amp = data['amp'].values

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
