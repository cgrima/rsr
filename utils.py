"""
Various tools for application of the rsr
"""

import numpy as np
import pdf


def hk_pdf_sample(a, s, mu):
    """Generate a hk distribution with noise. Use it fit testing.
    """
    x = np.linspace(0,1,100)
    y = pdf.hk(x, a, s, mu)
    noise = np.random.normal(0,y.max()*.05,x.size)
    sample = (y-noise) *((y-noise) >= 0)
    return x, sample
