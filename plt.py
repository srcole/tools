# -*- coding: utf-8 -*-
"""
Miscellaneous functions for plotting

bar : create a bar chart with error bars
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

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