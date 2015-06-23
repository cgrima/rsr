"""
Various tools for application of the rsr
Author: Cyril Grima <cyril.grima@gmail.com>
"""

import numpy as np
import pdf


def hk_pdf_sample(params, x):
    """Generate a hk distribution with noise. Use it fit testing.
    """
    # Extract parameters
    a = params['s']
    s = params['s']
    mu = params['mu']
    # Generate noisy hk distribution
    x = np.linspace(0,1,100)
    y = pdf.hk(params, x)
    noise = np.random.normal(0,max(y)*.05,x.size)
    sample = (y-noise) *((y-noise) >= 0)
    return sample
