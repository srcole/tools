# -*- coding: utf-8 -*-
"""
Miscellaneous useful functions, including:

resample_coupling: calculate significance with resampling
_z2p: convert z score to p value
getjetrgb: get the color values to plot in jet colors
linfit - calculate the linear fit of 2D data
regressout - regress 1 variable out of another
norm01 - normalize a series of numbers to have a minimum of 0 and max of 1
pearsonp - calculate the p-value for a pearson correlation form r and n

added some new functions that need to be formatted especially:
pdf2text - extract text from a pdf
emailme - email me when a script is done
add2path - add a directory to the file path
addenvvar - add an environmental variable
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
    
def regressout(x,y):
    mb = np.polyfit(x,y,1)
    return y - mb[1] - x*mb[0]
    
    
def norm01(x):
    return (x - np.min(x))/(np.max(x)-np.min(x))
    

def pearsonp(r, n):
    from scipy.stats import betai
    if abs(r) == 1:
        return 0
    else:
        df = n-2
        t_squared = r*r * (df / ((1.0 - r) * (1.0 + r)))
        return betai(0.5*df, 0.5, df / (df + t_squared))
    
    
def pdf2text(filename):
    """
    Function to convert the text on each page of a pdf to
    an element in a numpy array
    """
    from PyPDF2 import PdfFileReader
    filename = 'C:/gh/data/sfnabstract/SFN2014_Abstracts_PDF_Sat_PM.pdf'
    pdf = PdfFileReader(open(filename, "rb"))
    nPages = pdf.getNumPages()
    text1 = np.zeros(nPages, dtype=object)
    for p in range(nPages):
        pg = pdf.getPage(p)
        try:
            text1[p] = pg.extractText()
        except KeyError:
            text1[p] = False
            
    return text1

from datetime import datetime
def emailme(starttime=datetime.now(), msgtxt = 'Default message',
            usr='srcolepy', psw=np.str(np.load('c:/gh/data/misc/emailmepsw.npy')), 
            fromaddr='srcolepy@gmail.com', toaddr='scott.cole0@gmail.com'):
    """
    Adapted from: http://drewconway.com/zia/2013/3/26/u9utnymvh5ieja2plmwlwywekp37wf
    Sends an email message through GMail once the script is completed.  
    Developed to be used with AWS so that instances can be terminated 
    once a long job is done. Only works for those with GMail accounts.
    
    starttime : a datetime() object for when to start run time clock

    usr : the GMail username, as a string

    psw : the GMail password, as a string 
    
    fromaddr : the email address the message will be from, as a string
    
    toaddr : a email address, or a list of addresses, to send the 
             message to
    """
 
    from datetime import datetime
    import smtplib
    # Calculate run time
    runtime=datetime.now() - starttime
    
    # Initialize SMTP server
    server=smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login(usr,psw)
    
    # Send email
    senddate=datetime.strftime(datetime.now(), '%Y-%m-%d')
    subject="Your job has completed"
    m="Date: %s\r\nFrom: %s\r\nTo: %s\r\nSubject: %s\r\nX-Mailer: My-Mail\r\n\r\n" % (senddate, fromaddr, toaddr, subject)
    msg='''
    
    Job runtime: '''+str(runtime) + '\n\n' + msgtxt
    
    server.sendmail(fromaddr, toaddr, m+msg)
    server.quit()

def add2path(foldertoadd):
#    for d in sys.path:
#        print d
#    foldertoadd = "C:\\gh\\bv"
    import sys
    sys.path.append(foldertoadd)
    
def addenvvar(varname, varval):
    #os.environ['PDDATA'] = "C:\gh\_dataPYTHON\PD\data"
    import os
    os.environ[varname] = varval
    #print os.environ['PATH'] #or see sys.path
    #print os.environ['PDDATA']