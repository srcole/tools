# -*- coding: utf-8 -*-
"""
Miscellaneous functions for plotting

bar : create a bar chart with error bars
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

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
    plt.bar(x,y,x_width, color='b', yerr=yerr,ecolor='k')
    if y2 is not None:
        if Nextra == 1:
            plt.bar(x+x_width,y2,x_width, color='r', yerr=yerr2,ecolor='k')
        else:
            colorlist = ('r','g','y','c','k')
            for e in range(Nextra):
                plt.bar(x+x_width*(e+1),y2[e],x_width,color=colorlist[e], yerr=yerr2[e],ecolor='k')
        plt.legend(legend, loc='best',fontsize=fontsize)
        plt.xticks(x+x_width*Nextra/2.,xlab,size=fontsize)
    else:
        plt.xticks(x+x_width/2.,xlab,size=fontsize)
    
    plt.xlim((x[0]-x_width,x[-1]+x_width*(Nextra+2)))
    plt.ylim(ylim)
    
    plt.yticks(ylim,visible=yticksvis)
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
        raise ValueError('Implement showline')

    if showrp:
        if corrtype == 'Pearson':
            r, p = sp.stats.pearsonr(x,y)
        elif corrtype == 'Spearman':
            r, p = sp.stats.spearmanr(x,y)
        ax = plt.gca()
        ax.text(rloc[0], rloc[1], '$r^2 = $' + np.str(np.round(r**2,2)), fontsize=15)
        ax.text(ploc[0], ploc[1], '$p = $' + np.str(np.round(p,3)), fontsize=15)


    plt.xlim(xlim)
    plt.xticks(xticks,size=15)
    plt.xlabel(xlabel,size=20)

    plt.ylim(ylim)
    plt.yticks(yticks,fontsize=15)
    plt.ylabel(ylabel,size=20)

    plt.tight_layout()

    if returnax:
        return plt.gca()
    
    
def vizecog(ecog, t, chanperplt = 5):
    """
    Visualize ecog data that is ecog[channel][sample]
    """
    from matplotlib.widgets import Slider
    
    # Initial plot
    fig, ax = plt.subplots(figsize=(20,10))
    plt.subplots_adjust(left=0.04, right=0.96, top=0.98, bottom=0.08)
    
    # Make sliders and buttons
    axcolor = 'lightgoldenrodyellow'
    axDur = plt.axes([0.28, 0.01, 0.3, 0.03], axisbg=axcolor)
    axTstart  = plt.axes([0.65, 0.01, 0.3, 0.03], axisbg=axcolor)
    
    sDur = Slider(axDur, '$t_{len}$', 0.1, 30, 30)
    sTstart = Slider(axTstart, '$t_{start}$', 0, 30, valinit=0)
    
    # Time details
    Fs = 1/np.float(t[1])
    t_start = 0
    t_len = 30
    sampmax = len(ecog[0])    
    trange = np.logical_and(t>=t_start,t<np.min([(t_start+t_len),sampmax/Fs]))
    
    # Channel details
    C = len(ecog)
    Nplt = np.ceil(C/np.float(chanperplt))
 
    for c in range(C):
        if c == 0:
            ax1 = plt.subplot(np.int(Nplt),1,np.int(np.ceil((c+1)/np.float(chanperplt))))
            
        elif c % chanperplt == 0:
            ax1.legend(loc='best')
            ax1 = plt.subplot(np.int(Nplt),1,np.int(np.ceil((c+1)/np.float(chanperplt))))
        ax1.plot(t[trange],ecog[c][trange],label=str(c))
    ax1.legend(loc='best')
    
    # Update functions
    def update(val):
        # Time input and details
        t_start = sTstart.val
        t_len = sDur.val
        sampmax = len(ecog[0])    
        trange = np.logical_and(t>=t_start,t<np.min([(t_start+t_len),sampmax/Fs]))
        
        # Channel details
        C = len(ecog)
        
        # Initial plot
        for c in range(C):
            if c == 0:
                ax1 = plt.subplot(np.int(Nplt),1,np.int(np.ceil((c+1)/np.float(chanperplt))))
                ax1.cla()
            if c % chanperplt == 0:
                plt.legend(loc='best')
                ax1 = plt.subplot(np.int(Nplt),1,np.int(np.ceil((c+1)/np.float(chanperplt))))
                ax1.cla()
            ax1.plot(t[trange],ecog[c][trange],label=str(c))
        ax1.legend(loc='best')
        
        fig.canvas.draw_idle()
    
    sDur.on_changed(update)
    sTstart.on_changed(update)
    
    plt.show()