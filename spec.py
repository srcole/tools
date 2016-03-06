# -*- coding: utf-8 -*-
"""
Miscellaneous functions for spectral analysis

1. fftmed: calculate the PSD by taking fourier transform followed by median filter
2. slope: calculate slope of power spectrum
3. centerfreq: calculate the center frequency of an oscillation
4. calcpow: calculate the power in a frequency band
5. firfedge: band-pass filter a signal with an FIR filter and don't remove edge artifacts
6. myhipass: high-pass filter a signal with an FIR filter
"""

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
    

def fftmed(x, Fs=1000, Hzmed=10, zeropad=False, usehanning = False, usemedfilt = True):
    '''
    Calculate the power spectrum of a signal by first taking its FFT and then
    applying a median filter
    
    Parameters
    ----------
    x : array
        temporal signal
    Fs : integer
        sampling rate
    Hzmed : float
        Frequency width of the median filter
    zeropad : Boolean
        False to calculate the FFT using the number of points as the length of
        the signal. True to increase the number of points to the next power of
        2 by zero padding the signal
        
    Returns
    -------
    f : array
        frequencies corresponding to the PSD output
    psd : array
        power spectrum
    '''
    if zeropad:
        import math
        N = 2**(int(math.log(len(x), 2))+1)
        x = np.hstack([x,np.zeros(N-len(x))])
    else:
        N = len(x)
        
    
    # Calculate PSD
    f = np.arange(0,Fs/2,Fs/N)
    if usehanning:
        win = np.hanning(N)
        rawfft = np.fft.fft(x*win)
    else:
        rawfft = np.fft.fft(x)
    psd = np.abs(rawfft[:len(f)]**2)
    
    # Median filter
    if usemedfilt:
        from scipy.signal import medfilt
        sampmed = np.argmin(np.abs(f-Hzmed/2.0))
        psd = medfilt(psd,sampmed*2+1)
    
    return f, psd

    
def slope(f, psd, fslopelim = (80,200), flatten_thresh = 0):
    '''
    Calculate the slope of the power spectrum
    
    Parameters
    ----------
    f : array
        frequencies corresponding to power spectrum
    psd : array
        power spectrum
    fslopelim : 2-element list
        frequency range to fit slope
    flatten_thresh : float
        See foof.utils
        
    Returns
    -------
    slope : float
        slope of psd
    slopelineP : array
        linear fit of the PSD in log-log space (includes information of offset)
    slopelineF : array
        frequency array corresponding to slopebyf
    
    '''
    fslopeidx = np.logical_and(f>=fslopelim[0],f<=fslopelim[1])
    slopelineF = f[fslopeidx]
    
    x = np.log10(slopelineF)
    y = np.log10(psd[fslopeidx])

    from sklearn import linear_model
    lm = linear_model.RANSACRegressor(random_state=42)
    lm.fit(x[:, np.newaxis], y)
    slopelineP = lm.predict(x[:, np.newaxis])
    psd_flat = y - slopelineP.flatten()
    mask = (psd_flat / psd_flat.max()) < flatten_thresh
    psd_flat[mask] = 0
    slopes = lm.estimator_.coef_
    slopes = slopes[0][0]

    return slopes, slopelineP, slopelineF
    

def centerfreq(x, frange= [13,30], Fs = 1000, Hzmed = 10, plot_psd = False,
               importpsd = False, f = None, psd = None):
    
    # Calculate PSD
    if importpsd == False:
        f, psd = fftmed(x, Hzmed = Hzmed)
    
    # Calculate center frequencies
    frangeidx = np.logical_and(f>frange[0], f<frange[1])
    psdbeta = psd[frangeidx]
    cfs_idx = psdbeta.argmax() + np.where(frangeidx)[0][0]
    cf = f[cfs_idx]
    
    # Plot PSDs for subjects ECoG during task
    if plot_psd:
        plt.figure()
        plt.plot(f, np.log10(psd), 'k')
        plt.plot([cf, cf], [min(np.log10(psd)), max(np.log10(psd))], 'k--')
        plt.xlim(0,40)
        plt.xlabel('f [Hz]')
        plt.ylabel('log power')
    
    return cf


def calcpow(f, psd, flim):
    '''
    Calculate the power in a frequency range
    
    Parameters
    ----------
    f : array
        frequencies corresponding to power spectrum
    psd : array
        power spectrum
    flim : (lo, hi)
        limits of frequency range
        
    Returns
    -------
    pow : float
        power in the range
    '''
    fidx = np.logical_and(f>=flim[0],f<=flim[1])
    return np.sum(psd[fidx])
    
    
def firfedge(x, f_range, fs=1000, w=3):
    """
    Filter signal with an FIR filter but don't remove edge artifacts

    x : array-like, 1d
        Time series to filter
    f_range : (low, high), Hz
        Cutoff frequencies of bandpass filter
    fs : float, Hz
        Sampling rate
    w : float
        Length of the filter in terms of the number of cycles 
        of the oscillation whose frequency is the low cutoff of the 
        bandpass filter

    Returns
    -------
    x_filt : array-like, 1d
        Filtered time series
    """
    from scipy.signal import firwin, filtfilt
    nyq = np.float(fs / 2)
    Ntaps = np.floor(w * fs / f_range[0])
    taps = firwin(Ntaps, np.array(f_range) / nyq, pass_zero=False)
    return filtfilt(taps, [1], x)

    
def myhipass(x,cf,Fs, w = 3):
    from scipt.signal import firwin, filtfilt
    numtaps = w * Fs / np.float(cf)
    taps = firwin(numtaps, cf / np.float(Fs) * 2, pass_zero=False)
    return filtfilt(taps,[1],x)