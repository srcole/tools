# -*- coding: utf-8 -*-
"""
Miscellaneous functions for simulations

pha2r : Create a time series that is a nonlinear mapping of phase
simphase : simulate an oscillation and its phase by bandpass filtering white noise
spikes2lfp : Convolve a spike train with a synaptic potential to simulate a local field potential (LFP)
"""

from __future__ import division
import numpy as np
import scipy as sp
from scipy import signal

def pha2r(pha, method, mod_frac, firing_rate, sqprc=10.0, normstd = 1):
    '''
    Create a time-varying firing rate from a phase time series
    
    Parameters
    ----------
    pha : array
        time series of phases
    method : string ('sin', 'sq', or 'gauss')
        option for phase-to-firing rate transform
    mod_frac : float
        Fraction of the total firing rate to be phase-modulated
    firing_rate : float
        Average firing rate
    sqprc : float (between 0 and 100)
        Percentage of the cycle to bias the firing if using the 'sq' method
    normstd : float
        Standard deviation (in radians) of the gaussian that maps phase to
        firing rate
        
    Returns
    -------
    r : array
        time series of instantaneous firing rates
    '''
    # Generate time series of desired shape with mean 1 (1Hz)
    if method == 'sin':
        r_dep = np.sin(pha+np.pi/2) + 1
        
    elif method == 'sq':
        sqpha_thresh = np.percentile(pha,sqprc)
        t_bias = pha < sqpha_thresh
        r_dep = t_bias * 100.0 / sqprc
        
    elif method == 'gauss':
        import matplotlib.mlab as mlab
        r_dep = mlab.normpdf(pha,0,normstd)
        r_dep = r_dep / np.mean(r_dep)
        
    # Normalize for firing rate
    r_dep = r_dep * firing_rate * mod_frac
    r_indep = firing_rate*(1-mod_frac) * np.ones(len(pha))
    return r_dep + r_indep


def simphase(T, flo, dt=.001, returnwave=False):
    """ Simulate the phase of an oscillation
    The first and last second of the oscillation are simulated and taken out
    in order to avoid edge artifacts in the simulated phase
    
    Parameters
    ----------
    T : float
        length of time of simulated oscillation
    flo : 2-element array (lo,hi)
        frequency range of simulated oscillation
    dt : float
        time step of simulated oscillation
    returnwave : boolean
        option to return the simulated oscillation
    """
    from tools.spec import firfedge
    whitenoise = np.random.rand(int((T+2)/dt))
    theta = firfedge(whitenoise, flo, fs=1/dt, w=3)
    
    if returnwave:
        return np.angle(signal.hilbert(theta[int(1/dt):int((T+1)/dt)])), theta[int(1/dt):int((T+1)/dt)]
    else:
        return np.angle(signal.hilbert(theta[int(1/dt):int((T+1)/dt)]))
    

def spikes2lfp(spikes,
               gmax = 1, Tpsp = 100, tau_rise = 0.3, tau_decay = 2):
    """Simulate an LFP by convolving spikes with a synaptic potential"""
    # Create synaptic potential kernel
    t_dexp = np.arange(Tpsp)
    psp = gmax * (np.exp(-t_dexp/tau_decay) - np.exp(-t_dexp/tau_rise))
    return np.convolve(spikes, psp, mode='same')