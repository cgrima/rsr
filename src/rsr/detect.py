"""
Various tools for detecting specific signals in a radargram
"""

import numpy as np

def surface(rdg, method='maxvec', loc=None, ywinwidth=None):
    """Detect surface echo in a radargram

    Arguments
    ---------
    rdg : Array
        Radargram

    Keywords
    --------
    method : string
        Method to use for surface detection
    loc : vec
        A coordinate guess of where the surface should be
    ywinwidth : [int, int]
        Window limit around loc where the algo should search for the surface

    Output
    ------
    y : coordinate for the surface location
    val : value of te surface echo
    ok : Whether the detection is estimated to be good (1) or not (0)
    """

    n = rdg.shape

    loc = np.zeros(n[0]).astype('int') if loc is None else loc.astype('int')
    ywinwidth = np.array([0,n[1]-1]).astype('int') if ywinwidth is None else ywinwidth.astype('int')

    if ywinwidth[0] < 0:
        ywinwidth[0] = 0
    if ywinwidth[-1] > n[1]-1:
        ywinwidth[-1] = n[1]-1

    return globals()[method](rdg, loc, ywinwidth)


def maxvec(rdg, loc, ywinwidth):
    """
    Select the maximum value for each range line in a radargram
    """
    n = len(loc)
    y = np.zeros(n).astype('int')
    val = np.zeros(n).astype('int')
    ok = np.zeros(n).astype('int')
    for i, loci in enumerate(loc):
        pls = np.abs(rdg[i, :])
        itv = pls[loci+ywinwidth[0]:loci+ywinwidth[1]]
        y[i] = loci + ywinwidth[0] + np.argmax(itv)
        val[i] = pls[y[i]]
        # No quality metrics for now
        ok[i] = 1
    return {'y':y, 'val':val, 'ok':ok}


def maxprd(rdg, loc, ywinwidth):
    """
    Select the maximum of the val*val/dt for each range line in a radargram
    """
    n = len(loc)
    y = np.zeros(n).astype('int')
    val = np.zeros(n).astype('int')
    ok = np.zeros(n).astype('int')
    for i, loci in enumerate(loc):
        pls = np.abs(rdg[i, :])
        prd = np.abs(np.roll(np.gradient(pls), 2) * pls)
        itv = prd[loci+ywinwidth[0]:loci+ywinwidth[1]]
        y[i] = loci + ywinwidth[0] + np.argmax(itv)
        val[i] = pls[y[i]]
        # No Quality metrics for now
        ok[i] = 1
    return {'y':y, 'val':val, 'ok':ok}
