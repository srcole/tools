# -*- coding: utf-8 -*-
"""
Miscellaneous functions for spectral analysis

1. fftmed: calculate the PSD by taking fourier transform followed by median filter
2. slope: calculate slope of power spectrum
3. centerfreq: calculate the center frequency of an oscillation
4. calcpow: calculate the power in a frequency band
5. firfedge: band-pass filter a signal with an FIR filter and don't remove edge artifacts
6. myhipass: high-pass filter a signal with an FIR filter
7. notch: notch filtera signal with an FIR filter
8. rmvedge: remove edges from a signal prone to artifacts
9. nmppc: n:m phase-phase coupling
"""

from __future__ import division
import numpy as np
import scipy as sp
from scipy import signal
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
    nyq = np.float(fs / 2)
    Ntaps = np.floor(w * fs / f_range[0])
    taps = signal.firwin(Ntaps, np.array(f_range) / nyq, pass_zero=False)
    return signal.filtfilt(taps, [1], x)

    
def myhipass(x,cf,Fs, w = 3):
    numtaps = w * Fs / np.float(cf)
    taps = signal.firwin(numtaps, cf / np.float(Fs) * 2, pass_zero=False)
    return signal.filtfilt(taps,[1],x)


def notch(x, cf, bw, Fs=1000, order=3):
    '''
    Notch Filter the time series x with a butterworth with center frequency cf
    and bandwidth bw
    '''
    nyq_rate = Fs / 2.0
    f_range = [cf - bw / 2.0, cf + bw / 2.0]
    Wn = (f_range[0] / nyq_rate, f_range[1] / nyq_rate)
    b, a = signal.butter(order, Wn, 'bandstop')
    return signal.filtfilt(b, a, x)

def rmvedge(x, cf, Fs, w = 3):
    """
    Calculate the number of points to remove for edge artifacts
    x : array
        time series to remove edge artifacts from
    N : int
        length of filter
    """
    N = np.int(np.floor(w * Fs / cf))
    return x[N:-N]
    
def nmppc(x, flo, fhi, nm, Fs):
    """
    Calculate n:m phase-phase coupling between two oscillations
    Method from Palva et al., 2005 J Neuro
    * Morlet filter for the two frequencies
    * Use Hilbert to calculate phase and amplitude
    
    Parameters
    ----------
    x : np array
        time series of interest
    flo : 2-element list
        low and high cutoff frequencies for the low frequency band of interest
    fhi : 2-element list
        low and high cutoff frequencies for the high frequency band of interest
    nm : 2-element list of ints (n,m)
        n:m is the ratio of low frequency to high frequency (e.g. if flo ~= 8 and fhi ~= 24, then n:m = 1:3)
    Fs : float
        Sampling rate
        
    Returns
    -------
    plf : float
        n:m phase-phase coupling value (phase-locking factor)
    """
    
    from pacpy.pac import pa_series, _trim_edges
    phalo, _ = pa_series(x, x, flo, flo, fs = Fs)
    phahi, _ = pa_series(x, x, fhi, fhi, fs = Fs)
    phalo, phahi = _trim_edges(phalo, phahi)
    
    phadiffnm = phalo*nm[1] - phahi*nm[0]
    
    plf = np.sum(np.exp(1j*phadiffnm))
    return plf
    
    
def nmppcmany(x, floall, bw, M, Fs):
    """Calculate n:m coupling for many frequencies and values of 'm' for
    a single signal"""
    n_flo = len(floall)
    plfs = np.zeros((n_flo,M-1))
    for f in range(n_flo):
        for midx in range(M-1):
            m = midx + 2
            flo = (floall[f]-bw,floall[f]+bw)
            fhi = (floall[f]*m-m*bw,floall[f]*m+m*bw)
            plfs[f,midx] = nmppc(x, flo, fhi, (1,m),Fs)
            
    return plfs
    

def nmppcplot(plfs, floall, M, bw):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    # Realign plfs
    plfs2 = np.zeros((len(floall)+1,M))
    plfs2[:len(floall),:M-1] = plfs

    clim1 = (0,1)
    plt.figure(figsize=(5,5))
    cax = plt.pcolor(range(2,M+2), np.append(floall,100), plfs2, cmap=cm.jet)
    cbar = plt.colorbar(cax, ticks=clim1)
    cbar.ax.set_yticklabels(clim1,size=20)
    cbar.ax.set_ylabel('Phase locking factor', size=20)
    plt.clim(clim1)
    plt.axis([2, M+1, floall[0],floall[-1]+10])
    plt.xlabel('M', size=20)
    plt.ylabel('Frequency (Hz)', size=20)
    ax = plt.gca()
    ax.set_yticks(np.array(floall)+bw)
    ax.set_yticklabels(["%d" % n for n in floall],size=20)
    plt.xticks(np.arange(2.5,M+1),["%d" % n for n in np.arange(2,M+1)],size=20)
    plt.tight_layout()