# -*- coding: utf-8 -*-
"""
Miscellaneous functions for plotting

1. bar : create a bar chart with error bars
2. viztime : plot a pretty time series
3. scatt_2cond : scatter plot that compares the x and y values for each point
4. unpair_2cond : plot to compare the distribution of two sets of values
5. scatt_corr : plot a correlation
6. viz_ecog : plot multiple channels of channel x time data in an interactive plot
7. color2d : plot a matrix with values encoded in a colormap
8. spectrogram : plot a spectrogram using pcolormesh
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Slider

def bar(y, yerr, xlab, ylab,
            y2 = None, yerr2 = None, legend = None,
            ylim=None,yticksvis=True,figsize=(3,6),
            fontsize=15):
    if ylim is None:
        ylim = (np.min(y-yerr),np.max(y+yerr))
        
    x = np.arange(len(y))
    if y2 is None:
        x_width = .5
        Nextra=0
    else:
        if len(np.shape(y2)) == 1:
            Nextra = 1
        else:
            Nextra = np.int(np.shape(y2)[0])
        x_width = .8 / np.float(Nextra+1)
    
    plt.figure(figsize=figsize)
    plt.bar(x,y,x_width, color='k', yerr=yerr,ecolor='k')
    if y2 is not None:
        if Nextra == 1:
            plt.bar(x+x_width,y2,x_width, color='r', yerr=yerr2,ecolor='k')
        else:
            colorlist = ('r','b','g','y','c')
            for e in range(Nextra):
                plt.bar(x+x_width*(e+1),y2[e],x_width,color=colorlist[e], yerr=yerr2[e],ecolor='k')
        plt.legend(legend, loc='best',fontsize=fontsize)
        plt.xticks(x+x_width*Nextra/2.,xlab,size=fontsize)
    else:
        plt.xticks(x+x_width/2.,xlab,size=fontsize)
    
    plt.xlim((x[0]-x_width,x[-1]+x_width*(Nextra+2)))
    plt.ylim(ylim)
    
    plt.yticks(ylim,visible=yticksvis,size=fontsize)
    plt.ylabel(ylab,size=fontsize)
    
    plt.tight_layout()


def viztime(x, y,
            xlim = None, ylim = None,
            xticks = None, yticks = None,
            xlabel = '', ylabel = '',
            figsize = (12,4),
            returnax = False):
    if xlim is None:
        xlim = (np.min(x),np.max(x))
    if ylim is None:
        ylim = (np.min(y),np.max(y))
    if xticks is None:
        xticks = xlim
    if yticks is None:
        yticks = ylim

    plt.figure(figsize=figsize)
    plt.plot(x,y,'k-')

    plt.xlim(xlim)
    plt.xticks(xticks,size=15)
    plt.xlabel(xlabel,size=20)

    plt.ylim(ylim)
    plt.yticks(yticks,fontsize=15)
    plt.ylabel(ylabel,size=20)

    plt.tight_layout()

    if returnax:
        return plt.gca()


def scatt_2cond(x, y, ms = 12,
            lims = None, ticks = None,
            xlabel = '', ylabel = '',
            figsize = (5,5),
            returnax = False):
    if lims is None:
        lims = (np.min(np.hstack((x,y))),np.max(np.hstack((x,y))))
    if ticks is None:
        ticks = lims

    plt.figure(figsize=figsize)
    plt.plot(x,y,'k.', ms = ms)
    plt.plot(lims, lims,'k-')

    plt.xlim(lims)
    plt.xticks(ticks,size=15)
    plt.xlabel(xlabel,size=20)

    plt.ylim(lims)
    plt.yticks(ticks,fontsize=15)
    plt.ylabel(ylabel,size=20)

    plt.tight_layout()

    if returnax:
        return plt.gca()


def unpair_2cond(y1, y2, xlabs, ms = 12,
            ylim = None, yticks = None,
            ylabel = '',
            figsize = (3,5),
            returnax = False):
    if ylim is None:
        ylim = (np.min(np.hstack((y1,y2))),np.max(np.hstack((y1,y2))))
    if yticks is None:
        yticks = ylim

    plt.figure(figsize=figsize)
    plt.plot(np.zeros(len(y1)),y1,'k.', ms=ms)
    plt.plot(np.ones(len(y2)),y2,'k.', ms=ms)

    plt.xlim((-1,2))
    plt.xticks([0,1], xlabs,size=20)

    plt.ylim(ylim)
    plt.yticks(yticks,fontsize=15)
    plt.ylabel(ylabel,size=20)

    plt.tight_layout()

    if returnax:
        return plt.gca()


def scatt_corr(x, y, ms = 12,
            xlim = None, ylim = None,
            xticks = None, yticks = None,
            xlabel = '', ylabel = '',
            showrp = False, ploc = (0,0), rloc = (0,1), corrtype = 'Pearson',
            showline = False,
            figsize = (5,5),
            returnax = False):
    if xlim is None:
        xlim = (np.min(x),np.max(x))
    if ylim is None:
        ylim = (np.min(y),np.max(y))
    if xticks is None:
        xticks = xlim
    if yticks is None:
        yticks = ylim

    plt.figure(figsize=figsize)
    plt.plot(x,y,'k.', ms = ms)

    if showline:
        from tools.misc import linfit
        linplt = linfit(x,y)
        plt.plot(linplt[0],linplt[1], 'k--')

    if showrp:
        if corrtype == 'Pearson':
            r, p = sp.stats.pearsonr(x,y)
        elif corrtype == 'Spearman':
            r, p = sp.stats.spearmanr(x,y)
        ax = plt.gca()
        ax.text(rloc[0], rloc[1], '$r^2 = $' + np.str(np.round(r**2,2)), fontsize=15)
        ax.text(ploc[0], ploc[1], '$p = $' + np.str(np.round(p,3)), fontsize=15)


    plt.xlim(xlim)
    plt.xticks(xticks,size=20)
    plt.xlabel(xlabel,size=20)

    plt.ylim(ylim)
    plt.yticks(yticks,fontsize=20)
    plt.ylabel(ylabel,size=20)

    plt.tight_layout()

    if returnax:
        return plt.gca()
        
def viz_ecog(x, t, tmax = 30,
             Nch_plot = 6, init_t_len = 1, init_ch_start = 0, init_t_start = 0, figsize=(20,10)):
    """
    Visualize ECoG data 
    
    Parameters
    ----------
    x : 2-d array
        channels by time
    t : 1-d array
        time indices corresponding to columns of x
    Nch_plot : int
        Number of channels to plot in the figure
    init_t_len : float
        Initial value for length of time to plot
    init_ch_start : int
        Initial value for the first channel plotted (channels plotted sequentially)
    init_t_start : float
        Initial value for start of plotting
    """             
             
    # Init figure
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(left=0.04, right=0.96, top=0.98, bottom=0.1)
    
    # make first figure
    tplt = np.where(np.logical_and(t>=init_t_start,t<init_t_start+init_t_len))[0]
    for ch in range(Nch_plot):
        plt.subplot(Nch_plot,1,ch+1)
        plt.plot(t[tplt],x[ch+init_ch_start][tplt])
        plt.ylabel(str(ch+init_ch_start))
        plt.xlim((t[tplt[0]],t[tplt[-1]]))
        if ch == Nch_plot-1:
            plt.xlabel('Time (s)')
    
    #% Update functions
    def update(val):
        cur_t_len = sTlen.val
        cur_ch_start = int(sCh.val)
        cur_t_start = sTstart.val
        
        tplt = np.where(np.logical_and(t>=cur_t_start,t<cur_t_start+cur_t_len))[0]
        for ch in range(Nch_plot):
            plt.subplot(Nch_plot,1,ch+1)
            plt.cla()
            plt.plot(t[tplt],x[ch+cur_ch_start][tplt])
            plt.ylabel(str(ch+cur_ch_start))
            plt.xlim((t[tplt[0]],t[tplt[-1]]))
            if ch == Nch_plot-1:
                plt.xlabel('Time (s)')
            
        fig.canvas.draw_idle()
    
    # Make sliders and buttons
    axcolor = 'lightgoldenrodyellow'
    axTlen  = plt.axes([0.13, 0.01, 0.1, 0.03], axisbg=axcolor)
    axCh = plt.axes([0.28, 0.01, 0.3, 0.03], axisbg=axcolor)
    axTstart  = plt.axes([0.65, 0.01, 0.3, 0.03], axisbg=axcolor)
    
    sTlen = Slider(axTlen, '$t_{len}$', 0.1, 10, valinit=init_t_len)
    sCh = Slider(axCh, 'Chans', 0, np.shape(x)[0]-Nch_plot, valinit=init_ch_start)
    sTstart = Slider(axTstart, '$t_{start}$', 0, tmax, valinit=init_t_start)
    
    sTlen.on_changed(update)
    sCh.on_changed(update)
    sTstart.on_changed(update)
    
    plt.show()
    
    
def color2d(X, cmap=None, clim=None, cticks=None, figsize=(8,8), color_label='', plot_title='',
            plot_xlabel='', plot_ylabel='',
            plot_xticks_locs=[], plot_xticks_labels=[],
            plot_yticks_locs=[], plot_yticks_labels=[],
            interpolation='none', fontsize_major=20, fontsize_minor=10):
    """Plot the matrix X using a 2-dimensional color matrix

    Note you can put this in a subplot. it does not have to be its own figure"""
    if cmap is None:
        cmap = cm.viridis
    if clim is None:
        clim = [None, None]
    if cticks is None:
        if clim is not None:
            cticks=clim
        
        
    # Plot colored matrix and colormap bar
    cax = plt.imshow(X, cmap=cmap, interpolation=interpolation,vmin=clim[0], vmax=clim[1])
    cbar = plt.colorbar(cax, ticks=cticks)
    cbar.ax.set_yticklabels(cticks,size=fontsize_minor)
    
    # Format plot
    cbar.ax.set_ylabel(color_label, size=fontsize_major)
    plt.title(plot_title, size=fontsize_major)
    plt.ylabel(plot_ylabel, size=fontsize_major)
    plt.xlabel(plot_xlabel, size=fontsize_major)
    plt.yticks(plot_yticks_locs, plot_yticks_labels, size=fontsize_minor)
    plt.xticks(plot_xticks_locs, plot_xticks_labels, size=fontsize_minor,rotation='vertical')
    plt.tight_layout()
    
    
def spectrogram(t, f, spec,
                figsize=(10,4), vmin=0, vmax=5, ylim=(0,100)):
    x_mesh, y_mesh = np.meshgrid(t, f)
    plt.figure(figsize=figsize)
    plt.pcolormesh(x_mesh, y_mesh, spec, cmap=cm.jet, vmin=vmin, vmax=vmax)
    plt.ylim(ylim)
    plt.colorbar()
    
    
def histogram(xs, bins, labels = None, figsize = (4,4),
              xlabel='', return_ax = False):
    plt.figure(figsize=figsize)
    if type(xs[0]) in [list, np.ndarray]:
        if labels is None:
            raise ValueError('Must provide labels')
        for i in range(len(xs)):
            plt.hist(xs[i],bins, label=labels[i], alpha=.5)
            plt.legend(loc='best',fontsize=10)
    else:
        plt.hist(xs[i],bins,color='k',alpha=0.5)
    plt.ylabel('Count',size=15)
    plt.xlabel(xlabel,size=15)
    plt.yticks(size=12)
    plt.xticks(size=12)
    plt.tight_layout()
    if return_ax:
        return plt.gca()