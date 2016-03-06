# -*- coding: utf-8 -*-
"""
Miscellaneous useful functions, including:

resample_coupling: calculate significance with resampling
_z2p: convert z score to p value
getjetrgb: get the color values to plot in jet colors
linfit - calculate the linear fit of 2D data
norm01 - normalize a series of numbers to have a minimum of 0 and max of 1
"""

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def resample_coupling(x1, x2, couplingfn,
                      cfn_dict = {}, Nshuff=100, min_change=.1):
    """Do shuffle resampling"""
    Nsamp = len(x1)
    y_real = couplingfn(x1,x2,**cfn_dict)
    y_shuff = np.zeros(Nsamp)
    for n in range(Nshuff):
        offsetfract = min_change + (1-2*min_change)*np.random.rand()
        offsetsamp = np.int(Nsamp*offsetfract)
        x2_shuff = np.roll(x2,offsetsamp)
        y_shuff[n] = couplingfn(x1,x2_shuff,**cfn_dict)
        
    z = (y_real - np.mean(y_shuff)) / np.std(y_shuff)
    return _z2p(z)
    

def _z2p(z):
    """Convert z score to p value"""
    import scipy.integrate as integrate
    p, _ = integrate.quad(lambda x: 1/np.sqrt(2*np.pi)*np.exp(-x**2/2),-np.inf,z)
    return np.min((p,1-p))
    

def getjetrgb(N):
    """Get the RGB values of N colors across the jet spectrum"""
    from matplotlib import cm
    return cm.jet(np.linspace(0,255,N).astype(int))
    
    
def linfit(x,y):
    mb = np.polyfit(x,y,1)
    xs = np.array([np.min(x),np.max(x)])
    yfit = mb[1] + xs*mb[0]
    return xs, yfit
    
    
def norm01(x):
    return (x - np.min(x))/(np.max(x)-np.min(x))