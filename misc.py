# -*- coding: utf-8 -*-
"""
Miscellaneous useful functions, including:

1. resample_coupling: calculate significance with resampling
2. _z2p: convert z score to p value
3. getjetrgb: get the color values to plot in jet colors
4. linfit - calculate the linear fit of 2D data
5. regressout - regress 1 variable out of another
6. norm01 - normalize a series of numbers to have a minimum of 0 and max of 1
7. pearsonp - calculate the p-value for a pearson correlation form r and n
8. subplot_dims - calculate dimensions for a subplot of a certain number of plots

9. pdf2text - extract text from a pdf
10. emailme - email me when a script is done
11. add2path - add a directory to the file path
12. addenvvar - add an environmental variable
13. run_pdb_on_break - run python debugger when the input function breaks
14. save_features_general - save features as done for seizure data
"""

from __future__ import division
import numpy as np
import scipy as sp

import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

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
    """Regress x out of y to get a new y value"""
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
    

def subplot_dims(N):
    N_rows = np.floor(np.sqrt(N))
    N_cols = np.ceil(N/N_rows)
    return N_rows, N_cols
    
    
def run_pdb_on_break(fun, args):
    """Run a function and open up pdb when it breaks
    * If only 1 arg, must be entered in a list, i.e. [args]
    """
    import pdb, traceback, sys
    try:
        fun(*args)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
        
        
def save_features_general(feature_fn, feature_kwargs,
                          load_data_fn, load_data_kwargs,
                          load_bulk_data = True,
                          load_filenames_fn=None, load_filenames_kwargs=None,
                          output_folder = 'C:/gh/data2/temp', output_filename_append='',
                          print_progress = False):
    """
    Function to save a feature of many signals to a dataframe
    
    Inputs
    ------
    
    Outputs
    -------
    
    Notes
    -----
    * If load_data_fn returns None, then use the previous filename
    """
    
    # Load filenames or data
    if load_bulk_data:
        data_all = load_data_fn(**load_data_kwargs)
        N = len(data_all)
    else:
        filenames = load_filenames_fn(**load_filenames_kwargs)
        N = len(filenames)
        
    # Calculate features
    features_dict = np.zeros(N,dtype=dict)
    for n in range(N):
        if load_bulk_data:
            features_dict[n] = feature_fn(data_all[n],**feature_kwargs)
        else:
            # Load data
            data_temp = load_data_fn(filenames[n], **load_data_kwargs)
            # Deal with case of missing data
            n_use = n - 1
            while data_temp is None:
                data_temp = load_data_fn(filenames[n_use], **load_data_kwargs)
                n_use -= 1
            # Calculate features
            features_dict[n] = feature_fn(data_temp, **feature_kwargs)
        if print_progress:
            print 'Calculating features for file '+str(n+1)+'/'+str(N)
        
    # Combine dicts by key
    features_dict_all = {}
    for k in features_dict[0].iterkeys():
        features_dict_all[k] = tuple(d[k] for d in features_dict)
    
    # Convert dicts to df
    df = pd.DataFrame.from_dict(features_dict_all)
    
    # Derive output filename from feature_fn
    output_filename = output_folder + feature_fn.func_name + output_filename_append + '.csv'
    
    # Save df to csv
    df.to_csv(output_filename)
    
    
def load_features_general(features_filepath):
    """Load a set of features from csvs into dataframes"""
    # Get filenames for all features
    filenames = glob.glob(features_filepath+'*.csv')
    # Load all csvs into one data frame
    dfs = [0]*len(filenames)
    for n, fi in enumerate(filenames):
        dfs[n] = pd.read_csv(fi,index_col=0)
    df = pd.concat(dfs,axis=1)
    return df
    

def logreg_crossval_general(df, features_pred, y_dict,  y_colname='class',
                   test_fraction = .3, N_iter = 10, random_state_offset=0,
                   equalize_classes = False):
    """
    Perform logistic regression to predict the 'class' in the df from the features in features_pred
    """
    # Force same number of training examples in each class
    if equalize_classes:
        df = _equalize_training_classes(df, np.array(df['class']==y_dict.keys()[1])*1)

    # Make the classes and training feature set
    df, _, y = df_separate_example_submit(df, y_colname, y_dict)

    # Select specific features to be used for prediction
    df_feat = df[features_pred]
    
    # Normalize features
    for k in df_feat.keys():
        df_feat[k] = sp.stats.zscore(df_feat[k])

    # Calculate results for multiple random initilializations
    predicted = np.zeros(N_iter,dtype=object)
    probs = np.zeros(N_iter,dtype=object)
    df_weights = np.zeros(N_iter,dtype=object)
    scores = {}
    y_test = np.zeros(N_iter,dtype=object)
    scores_ks = ['acc','auc','confuse','report']
    for k in scores_ks:
        scores[k] = np.zeros(N_iter,dtype=object)

    for i in range(N_iter):
        # Split data into training and testing
        X_train, X_test, y_train, y_test[i] = train_test_split(df_feat, y, test_size=test_fraction, random_state=i+random_state_offset)
        # Train a logistic regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)
        # Classify test data
        predicted[i] = model.predict(X_test)
        probs[i] = model.predict_proba(X_test)
        # Characterize classifier performance
        scores['acc'][i] = metrics.accuracy_score(y_test[i], predicted[i])
        scores['auc'][i] = metrics.roc_auc_score(y_test[i], probs[i][:, 1])
        scores['confuse'][i] = metrics.confusion_matrix(y_test[i], predicted[i])
        scores['report'][i] = metrics.classification_report(y_test[i], predicted[i])
        # Save classifier weights
        df_weights[i] = pd.DataFrame(zip(df_feat.columns, np.transpose(model.coef_)))

    return model, scores, df_weights


def _equalize_training_classes(df, y):
    """Make the same number of training examples for eyes open and eyes closed"""
    # Find examples to keep
    y_vals = np.unique(y)
    N_keep = np.min((sum(y==y_vals[0]),sum(y==y_vals[1])))
    idx_keep = np.hstack((np.where(y==y_vals[1])[0][:N_keep],np.where(y==y_vals[0])[0][:N_keep]))
    # Filter df for these examples
    df2 = df.reset_index()
    df2 = df2.drop('index',1)
    df_equal = df2.loc[idx_keep]
    df_equal = df_equal.reset_index()
    df_equal = df_equal.drop('index',1)
    return df_equal


def df_separate_example_submit(df, y_colname, y_dict):
    df_examples = df[df[y_colname]!='None']
    df_submit = df[df[y_colname]=='None']
    y = df_examples[y_colname].values
    y_dict = dict((v, k) for k, v in y_dict.iteritems())
    y = (y==y_dict[1])*1
    df_examples = df_examples.drop(y_colname,1)
    df_submit = df_submit.drop(y_colname,1)
    return df_examples, df_submit, y
    

def df_stats_general(df, y_dict, y_colname='class'):
    """Calculate stats from features in a data frame, related to the class"""
    # Differentiate feature dataframe and class value
    df, _, y = df_separate_example_submit(df, y_colname, y_dict)
    
    
    feats = df.keys().values
    stats = {}
    stats['t_stats'] = {}
    stats['t_stats_p'] = {}
    stats['mean_0'] = {}
    stats['mean_1'] = {}
    for f in feats:
        feats = df[f]
        feat_class0 = feats[y==0]
        feat_class1 = feats[y==1]
        stats['t_stats'][f], stats['t_stats_p'][f] = sp.stats.ttest_ind(feat_class0,feat_class1)
        stats['mean_0'][f] = np.mean(feat_class0)
        stats['mean_1'][f] = np.mean(feat_class1)
        
    return stats
    
    
def logreg_model_general(Xtrain, y, Xtest):
    """Perform logistic regression on training data and return
    estimated probabilities for test data"""
    model = LogisticRegression()
    model = model.fit(Xtrain,y)
    probs = model.predict_proba(Xtrain)
    train_auc = metrics.roc_auc_score(y, probs[:, 1])
    test_probs = model.predict_proba(Xtest)[:,1]
    return model, train_auc, test_probs