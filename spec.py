# -*- coding: utf-8 -*-
"""
Miscellaneous functions for spectral analysis

0. bandpass_default: default bandpass filter
1. fftmed: calculate the PSD by taking fourier transform followed by median filter
2. slope: calculate slope of power spectrum
3. centerfreq: calculate the center frequency of an oscillation
4. calcpow: calculate the power in a frequency band
5. firfedge: band-pass filter a signal with an FIR filter and don't remove edge artifacts
6. myhipass: high-pass filter a signal with an FIR filter
7. notch: notch filtera signal with an FIR filter
8. rmvedge: remove edges from a signal prone to artifacts
9. nmppc: n:m phase-phase coupling
10. morletT: continuous morlet transform (uses morletf)
11. plot_filter: plot the frequency respponse of a filter
"""

from __future__ import division
import numpy as np
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt


def bandpass_default(x, f_range, Fs, rmv_edge = True, w = 3, plot_frequency_response = False):
    """
    Default bandpass filter
    
    Parameters
    ----------
    x : array-like 1d
        voltage time series
    f_range : (low, high), Hz
        frequency range for narrowband signal of interest
    Fs : float
        The sampling rate
    w : float
        Length of filter order, in cycles. Filter order = ceil(Fs * w / f_range[0])
        
    Returns
    -------
    x_filt : array-like 1d
        filtered time series
    taps : array-like 1d
        filter kernel
    """
    
    # Design filter
    from scipy import signal
    Ntaps = np.ceil(Fs*w/f_range[0])
    # Force Ntaps to be odd
    if Ntaps % 2 == 0:
        Ntaps = Ntaps + 1
    taps = sp.signal.firwin(Ntaps, np.array(f_range) / (Fs/2.), pass_zero=False)
    
    # Apply filter
    x_filt = np.convolve(taps,x,'same')
    
    # Plot frequency response
    if plot_frequency_response:
        w, h = signal.freqz(taps)
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,2)
        plt.title('Kernel')
        plt.plot(taps)
        
        plt.subplot(1,2,1)
        plt.plot(w*Fs/(2.*np.pi), 20 * np.log10(abs(h)), 'b')
        plt.title('Frequency response')
        plt.ylabel('Attenuation (dB)', color='b')
        plt.xlabel('Frequency (Hz)')

    # Remove edge artifacts
    N_rmv = int(Ntaps/2.)
    if rmv_edge:
        return x_filt[N_rmv:-N_rmv], Ntaps
    else:
        return x_filt, taps


def fftmed(x, Fs=1000, Hzmed=0, zeropad=False, usehanning = False, usemedfilt = True):
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
    psd = np.abs(rawfft[:len(f)])**2
    
    # Median filter
    if usemedfilt:
        sampmed = np.argmin(np.abs(f-Hzmed/2.0))
        psd = signal.medfilt(psd,sampmed*2+1)
    
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
    slopes = slopes[0]

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
    return np.sum(psd[fidx])/np.float(len(f)*2)
    
    
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
    
    plf = np.abs(np.mean(np.exp(1j*phadiffnm)))
    return plf
    
    
def nmppcmany(x, floall, bw, M, Fs):
    """Calculate n:m coupling for many frequencies and values of 'm' for
    a single signal"""
    n_flo = len(floall)
    plfs = np.zeros((n_flo,M-1))
    for f in range(n_flo):
        for midx in range(M-1):
            m = midx + 2
            fhi = (floall[f]-bw,floall[f]+bw)
            flo = (floall[f]/m-bw/m,floall[f]/m+bw/m)
            plfs[f,midx] = nmppc(x, flo, fhi, (1,m),Fs)
            
    return plfs
    

def nmppcplot(plfs, floall, M, bw, clim1=(0,1)):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    # Realign plfs
    plfs2 = np.zeros((len(floall)+1,M))
    plfs2[:len(floall),:M-1] = plfs

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
    
    
def morletT(x, f0s, w = 7, fs = 1000, s = 1):
    """
    Calculate the time-frequency representation of the signal 'x' over the
    frequencies in 'f0s' using morlet wavelets
    Parameters
    ----------
    x : array
        time series
    f0s : array
        frequency axis
    w : float
        Length of the filter in terms of the number of cycles of the oscillation
        whose frequency is the center of the bandpass filter
    Fs : float
        Sampling rate
    s : float
        Scaling factor
    Returns
    -------
    mwt : 2-D array
        time-frequency representation of signal x
    """
    if w <= 0:
        raise ValueError('Number of cycles in a filter must be a positive number.')
        
    T = len(x)
    F = len(f0s)
    mwt = np.zeros([F,T],dtype=complex)
    for f in range(F):
        mwt[f] = morletf(x, f0s[f], fs = fs, w = w, s = s)

    return mwt


def morletf(x, f0, fs = 1000, w = 7, s = 1, M = None, norm = 'sss'):
    """
    Convolve a signal with a complex wavelet
    The real part is the filtered signal
    Taking np.abs() of output gives the analytic amplitude
    Taking np.angle() of output gives the analytic phase
    x : array
        Time series to filter
    f0 : float
        Center frequency of bandpass filter
    Fs : float
        Sampling rate
    w : float
        Length of the filter in terms of the number of cycles of the oscillation
        with frequency f0
    s : float
        Scaling factor for the morlet wavelet
    M : integer
        Length of the filter. Overrides the f0 and w inputs
    norm : string
        Normalization method
        'sss' - divide by the sqrt of the sum of squares of points
        'amp' - divide by the sum of amplitudes divided by 2
    Returns
    -------
    x_trans : array
        Complex time series
    """
    if w <= 0:
        raise ValueError('Number of cycles in a filter must be a positive number.')
        
    if M == None:
        M = 2 * s * w * fs / f0

    morlet_f = signal.morlet(M, w = w, s = s)
    morlet_f = morlet_f
    
    if norm == 'sss':
        morlet_f = morlet_f / np.sqrt(np.sum(np.abs(morlet_f)**2))
    elif norm == 'abs':
        morlet_f = morlet_f / np.sum(np.abs(morlet_f))
    else:
        raise ValueError('Not a valid wavelet normalization method.')

    mwt_real = np.convolve(x, np.real(morlet_f), mode = 'same')
    mwt_imag = np.convolve(x, np.imag(morlet_f), mode = 'same')

    return mwt_real + 1j*mwt_imag
    

def plot_filter(taps, Fs):
    w, h = signal.freqz(taps)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,6))
    plt.plot(w*Fs/(2.*np.pi), 20 * np.log10(abs(h)), 'b')
    plt.ylabel('Attenuation (dByy)', color='b')
    plt.xlabel('Frequency (Hz)')