"""
Wrappers for running RSR processing
"""

from . import fit
from .Classdef import Async
import numpy as np
import pandas as pd
import time
from sklearn.neighbors import KDTree


def timing(func):
    """Outputs the time a function takes to execute.
    """
    def func_wrapper(*args, **kwargs):
        t1 = time.time()
        func(*args, **kwargs)
        t2 = time.time()
        print("- Processed in %.1f s.\n" % (t2-t1))
    return func_wrapper


def scale(amp):
    """Provide a factor to scale a set of amplitudes between 0 and 1
    for correct rsr processing through fit.lmfit
    """
    y, x = np.histogram(np.abs(amp), bins='fd')
    pik = x[y.argmax()]
    out = 1/(pik*10)
    return out


def processor(amp, gain=0., bins='stone', fit_model='hk', scaling=True, **kwargs):
    """Apply RSR over a sample of amplitudes

    Arguments
    ---------
    amp: float array
        Linear amplitudes

    Keywords
    --------
    gain: float
        Gain (in dB power) to add to the amplitudes
    bins: string
        Method to compute the bin width (inherited from numpy.histogram)
    fit_model: string
        Name of the function (in pdf module) to use for the fit
    scaling: boolean
        Whether to scale the amplitudes before processing.
        That ensures a correct fit in case the amplitudes are << 1
    output: string
        Format of the output
    Return
    ------
    A Statfit class
    """
    # Remove zero values
    amp = amp[amp > 0]

    # Gain and Scaling
    amp = amp * 10**(gain/20.)
    scale_amp = scale(amp) if scaling is True else 1
    amp = amp*scale_amp

    # Fit
    a = fit.lmfit( np.abs(amp), bins=bins, fit_model=fit_model)

    # Remove Scaling
    pc = 10*np.log10( a.values['a']**2  ) - 20*np.log10(scale_amp)
    pn = 10*np.log10( 2*a.values['s']**2  ) - 20*np.log10(scale_amp)
    a.sample = amp/scale_amp
    a.values['a'] = np.sqrt( 10**(pc/10.) )
    a.values['s'] = np.sqrt( 10**(pn/10.)/2. )

    # Output
    if 'ID' in kwargs:
        a.values['ID'] = kwargs['ID']
    else:
        a.values['ID'] = -1

    return a


def cb_processor(a):
    """
    Callback function for processor

    Argument:
    ---------
    a: class
        Results from "processor" (Statfit class)
    """
    p = a.power()
    #print(p)
    print("#%d\tCorrelation: %.3f\tPt: %.1f dB   Pc: %.1f dB   Pn: %.1f dB" %
            (a.values['ID'], a.crl(), p['pt'], p['pc'], p['pn'] ) )
    return a


def frames(x ,winsize=1000., sampling=250, **kwargs):
    """
    Defines along-track frames coordinates for rsr application

    Arguments
    ---------
    x: float array
        vector index

    Keywords
    --------
    winsize: int
        Number of elements in a window
    sampling: int
        Sampling step
    """
    # Window first and last id
    xa = x[:np.int(x.size-winsize):np.int(sampling)]
    xb = xa + winsize-1

    # Cut last window in limb
    if xb[-1] > x[-1]: xb[-1] = x[-1]
    xo = [val+(xb[i]-val)/2. for i, val in enumerate(xa)]

    # Output
    out = [ np.array([xa[i], xb[i]]).astype('int64') for i in np.arange(xa.size)  ]

    out = {'xa':np.array(xa, dtype=np.int64),
           'xb':np.array(xb, dtype=np.int64),
           'xo':np.array(xo, dtype=np.float64),
           }

    return out

#@timing
def along(amp, nbcores=1, verbose=True, **kwargs):
    """
    RSR applied on windows sliding along a vector of amplitudes

    Arguments
    ---------
    amp: Float array
        A vector of amplitudes

    Keywords
    --------
    nbcores: int
        number of cores
    verbose: boolean
        print results
    Any keywords accepted by 'processor' and 'frames'

    Return
    ------

    """
    t1 = time.time()

    #-----------
    # Parameters
    #-----------

    # Windows along-track
    x = np.arange( len(amp) ) #vector index
    w = frames(x, **kwargs)
    ID = np.arange(w['xa'].size)

    # Jobs Definition
    args, kwgs = [], []
    for i in ID:
        args.append( amp[w['xa'][i]: w['xb'][i]] )
        #kwgs.append( dict(**kwargs, i=w['xo'][i])  )

    #-----------
    # Processing
    #-----------

    # Do NOT use the multiprocessing package
    if nbcores== -1:
        results = pd.DataFrame()
        for i in ID:
            a = processor(args[i], **kwargs, ID=w['xo'][i])
            cb_processor(a)
            b = {**a.values, **a.power(), 'crl':a.crl(), 'chisqr':a.chisqr,}
            results = results.append(b, ignore_index=True)
        out = results

    # Do use the multiprocessing package
    if nbcores > 0:
        results = []
        if verbose is True:
            async_inline = Async(processor, cb_processor, nbcores=nbcores)
        elif verbose is False:
            async_inline = Async(processor, None, nbcores=nbcores)

        for i in ID:
            results.append( async_inline.call(args[i], **kwargs, ID=w['xo'][i]) )
        async_inline.wait()
        # Sorting Results
        out = pd.DataFrame()
        for i in results:
            a = i.get()
            b = {**a.values, **a.power(), 'crl':a.crl(), 'chisqr':a.chisqr,}
            out = out.append(b, ignore_index=True)
        out = out.sort_values('ID')

        out['xa'] = w['xa']
        out['xb'] = w['xb']
        out['xo'] = w['xo']
        out = out.drop('ID', 1)

        t2 = time.time()
        if verbose is True:
            print("- Processed in %.1f s.\n" % (t2-t1))

    return out


#@timing
def incircles(amp, amp_x, amp_y, circle_x, circle_y, circle_r, leaf_size=None,
              nbcores=1, verbose=True, **kwargs):
    """
    RSR applied over data within circles

    Arguments
    ---------
    amp: Float array
        A vector of amplitudes
    amp_x: Float array
        X coordinates for amp
    amp_y: Float array
        Y coordinates for amp
    circle_x: Float array
        X coordinates for circles
    circle_y: Float array
        Y_coordinates for circles
    circle_r: Float
        Radius of the circles
    leaf_size: Integer (Default: None)
        Set the leaf size for the KD-Tree. Inherits from sklearn.
        If None, use a brute force technique

    Keywords
    --------
    leaf_size: Integer (Default: None)
        Set the leaf size for the KD-Tree. Inherits from sklearn.
        If None, use a brute force technique
    nbcores: int
        number of cores
    verbose: boolean
        print results
    Any keywords accepted by 'processor' and 'frames'

    Return
    ------

    """
    t1 = time.time()

    #-----------
    # Parameters
    #-----------

    # Coordinates units
    #if deg is True:
    #    metrics = 'haversine'
    #    amp_x = np.deg2rad(amp_x)
    #    amp_y = np.deg2rad(amp_y)
    #    circle_x = np.deg2rad(circle_x)
    #    circle_y = np.deg2rad(circle_y)
    #    circle_r = np.deg2rad(circle_r)
    #else:
    #    metrics = 'euclidian'

    # KD-Tree
    if leaf_size is None:
        leaf_size = len(amp)
    amp_xy = np.array(list(zip(amp_x, amp_y)))
    tree = KDTree(amp_xy, leaf_size=leaf_size)

    # Radius Query
    circle_xy = np.array(list(zip(circle_x, circle_y)))
    ind = tree.query_radius(circle_xy, r=circle_r)
    
    # Jobs Definition
    ID, args, kwgs = [], [], []
    for i, data_index in enumerate(ind):
        if data_index.size != 0:
            data = np.take(amp, data_index)
            args.append(data)
            ID.append(i)
        #kwgs.append( dict(**kwargs, i=w['xo'][i])  )

    #-----------
    # Processing
    #-----------

    # Do NOT use the multiprocessing package
    if nbcores == -1:
        results = pd.DataFrame()
        for i, orig_i in enumerate(ID):
            a = processor(args[i], **kwargs, ID=orig_i)
            cb_processor(a)
            b = {**a.values, **a.power(), 'crl':a.crl(), 'chisqr':a.chisqr,}
            results = results.append(b, ignore_index=True)
        out = results

    # Do use the multiprocessing package
    if nbcores > 0:
        results = []
        if verbose is True:
            async_inline = Async(processor, cb_processor, nbcores=nbcores)
        elif verbose is False:
            async_inline = Async(processor, None, nbcores=nbcores)

        for i, orig_i in enumerate(ID):
            results.append( async_inline.call(args[i], **kwargs, ID=orig_i) )
        async_inline.wait()
        # Sorting Results
        out = pd.DataFrame()
        for i in results:
            a = i.get()
            b = {**a.values, **a.power(), 'crl':a.crl(), 'chisqr':a.chisqr,}
            out = out.append(b, ignore_index=True)
        out = out.sort_values('ID')

        #out['xa'] = w['xa']
        #out['xb'] = w['xb']
        #out['xo'] = w['xo']
        #out = out.drop('ID', 1)

        t2 = time.time()
        if verbose is True:
            print("- Processed in %.1f s.\n" % (t2-t1))

    return out
