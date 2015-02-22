#! /usr/bin/env python

# this libary/script contains a bunch of tools for empirically evaluating model parameter
# convergence in the presence of feature colinearity.  
#
# The libary focuses on one illustrative example (predicting income as a function of 
# residence, which could be complicated by a colinear feature (favorite football team).
#
# Includes a data generator for the example, an OLS fit calculation, and a number of 
# wrappers designed to allow the user to examine convergence as a function of 
# number of trianing examples and feature colinearity.
#
# Also includes some visualizations.
#
# note dependence on numpp, scipy, pandas

import sys
import os
import argparse
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import inspect
# example root formula: 100000 + 25000*seattle_resident + (-10000)*detroit_resident + normal(0, 10000)
#
# data generating process: 
#    P(seattle_resident, seahawks_fan) = 0.39
#    P(seattle_resident, lions_fan) = 0.01
#    P(detroit_resident, lions_fan) = 0.58
#    P(detroit_resident, seahawks_fan) = 0.02

ssprob = 0.39
slprob = 0.01
dlprob = 0.58
dsprob = 0.02

error_scale = 10000
seattle_effect = 25000
detroit_effect = -10000

n_examples = 100

fractional_starting_kwargs = {"ssprob": 0.5,
                              "slprob": 0.0,
                              "dlprob": 0.5,
                              "dsprob": 0.0,
                              "bias": 1,
                              "seattle_effect": 0.1,
                              "detroit_effect": 0,
                              "error_scale": 0.1,
                              "n_examples": None}
physical_starting_kwargs = {"ssprob": 0.5,
                            "slprob": 0.0,
                            "dlprob": 0.5,
                            "dsprob": 0.0,
                            "bias": 100000,
                            "seattle_effect": 10000,
                            "detroit_effect": 0,
                            "error_scale": 10000,
                            "n_examples": None}

def generate_income_df(ssprob=0.35, slprob=0.05, dlprob=0.53, dsprob=0.07, 
                       bias=100000, seattle_effect=8000, detroit_effect=-10000, 
                       error_scale=10000, n_examples=40):
    '''generates a data frame of the form:
    
            bias  residence  team  seattle  detroit         income
        0      1          0     0        0        1   96997.170127
        1      1          0     0        0        1   66512.824446
        2      1          0     0        0        1   99385.679061
        3      1          0     0        0        1   98587.756121
        4      1          0     0        0        1   93577.574512
        5      1          1     1        1        0   99145.438985
        6      1          0     0        0        1   85918.102276
        7      1          0     0        0        1   98622.818392
        8      1          0     0        0        1   82421.718979
        9      1          0     0        0        1   90238.190944

    Where  bias = 1 in all cases
           residence = 0 if residence is detroit, 1 if seattle
           team = 0 --> seahawks, team = 1 --> lions
           income is drawn based on the following equation:
           income = bias + city_effect + Normal(0, error_scale)
    '''
    feature_list = []
    for ii in xrange(n_examples):
        rval = np.random.random()
        if rval < ssprob:
            feature_list += [{"residence": "seattle", "team": "seahawks"}]
        elif rval < ssprob + slprob:
            feature_list += [{"residence": "seattle", "team": "lions"}]
        elif rval < ssprob + slprob + dlprob:
            feature_list += [{"residence": "detroit", "team": "lions"}]
        else:
            feature_list += [{"residence": "detroit", "team": "seahawks"}]

    x_mat = []
    y_vec = []
    df = []
    for f_dict in feature_list:
        income = bias
        if f_dict["residence"]=="seattle": # seattle
            income += seattle_effect
            if f_dict['team']=="seahawks": # seattle + seahawks fan
                f_val = [1,1,1,1,0]
            else: # seattle + lions fan
                f_val = [1,1,0,1,0]
        else: # detroit
            income += detroit_effect
            if f_dict['team']=="lions": # detroit + lions fan
                f_val = [1,0,0,0,1]
            else: # detroit + seahawks fan
                f_val = [1,0,1,0,1]
        income += np.random.normal(0, error_scale) 
        f_dict["income"] = income

        x_mat += [f_val]
        y_vec += [income]
        df += [f_val + [income]]
    df = pd.DataFrame(df)
    df = df.rename(columns={0:"bias", 1:"residence", 2:"team", 3:"seattle", 4:"detroit", 5:"income"})
    x_mat = np.matrix(x_mat)
    y_vec = np.matrix(y_vec)
    return df

def print_args():
    for argname in inspect.getargspec(generate_income_df).args:
        print '"' + argname + '"'

def fit(df, xcols, ycol):
    '''fits a dataframe column (specified by ycol), with a linear model on the 
       specified xcols, using OLS regression.'''
    xx = np.matrix(df[xcols])
    yy = np.matrix(df[[ycol]])
    wmat = (xx.T.dot(xx).I).dot(xx.T).dot(yy)
    return dict(zip(xcols, [float(ww[0][0]) for ww in wmat]))

def calculate_fit_stats(df, xcols, ycol, w_dict=None):
    '''calculates some statistics on the linear model fit of ycol using xcol.
       If user specifies a model via w_dict, this model is used, but if not,
       the model is first fit, then the stats are calculated.
    '''
    if not w_dict:
        w_dict = fit(df, xcols, ycol)
    prediction_series = pd.Series(np.zeros(len(df[ycol])))
    for col in xcols:
        prediction_series += w_dict[col]*df[col]
    resid = df[ycol]-prediction_series
    ss_resid = float((resid**2).sum())
    ss_total = float(((df[ycol] - df[ycol].mean())**2).sum())
    R_sqd = 1 - ss_resid/ss_total
    output_dict = {'ss_resid': ss_resid, 
                   'ss_total': ss_total, 
                   'R_sqd': R_sqd}
    return output_dict

def generate_w_df(n_models=10, xcols=['residence','team'], ycol='income', **data_kwargs):
    '''Generates income data frames, and for each one fits ycol to xcols as 
    specified.  Returns a data frame containing model coefficients for each model. 
    Example output:
                residence          team
          0  69953.911082  43735.183098
          1  55167.325574  57493.361794
          2  18683.545623  89598.094608
          3  44823.133612  66726.378727
          4  55014.262639  61292.746966
          5  36603.242185  71809.430342
          6  23727.272028  86803.608559
          7  73631.874571  45364.038503
          8  48580.017897  64387.205151
          9  68511.190721  50043.093398

    Purpose of this function is to empirically convergence of model paramters
    as a function of parameters in data_kwargs (e.g. number of training examples,
    colinearity of the features), which are passed through to the 
    generate_income_df().
    '''
    output_dict = defaultdict(list)
    for ii in xrange(n_models):
        try:
            df = generate_income_df(**data_kwargs)
            fit_dict = fit(df, xcols=xcols, ycol=ycol)
            for key in fit_dict:
                output_dict[key] += [fit_dict[key]]
        except np.linalg.LinAlgError:
            continue
    return pd.DataFrame(output_dict)

def do_fit_routine(max_power=10, xcols=['bias', 'residence'], starting_kwargs=fractional_starting_kwargs):
    '''wraps a sequence of calls to generate_w_df() where there number of 
    training examples are varied (logarithmically) starting from 100, so
    the user can test the convergence as a function of number of training examples.
    '''
    n_examples_list = [int(100*1.5**ee) for ee in range(0,max_power)]
    out_std_dict = defaultdict(list)
    for n_examples in n_examples_list:
        starting_kwargs['n_examples'] = n_examples
        w_df = generate_w_df(n_models=1000, xcols=xcols, ycol='income', **starting_kwargs)
        print n_examples
        for col in xcols:
            print col
            print w_df[col].std()
            print w_df[col].mean()
        out_std_dict['n_examples'] += [n_examples]
        for col in xcols:
            out_std_dict[col+ '_std'] += [w_df[col].std()]
    return out_std_dict

def run_sdev_sim(n_simulations=4, max_power=10, starting_kwargs=fractional_starting_kwargs):
    '''
    runs a simulation program to develop observations on parameter estimation convergence
    for linear models derived by fitting data from generate_income_df().  Essentially
    a wrapper for do_fit_routine() that runs several time and calculates averages.  
    '''
    bias_dict = {}
    residence_dict = {}
    n_examples_dict = {}
    for ii in xrange(n_simulations):
        print 'simulation ' + str(ii)
        std_dict = do_fit_routine(max_power=max_power, starting_kwargs=starting_kwargs)
        bias_dict[ii] = std_dict['bias_std']
        residence_dict[ii] = std_dict['residence_std']
        n_examples_dict[ii] = std_dict['n_examples']
    bias_df = pd.DataFrame(bias_dict)
    residence_df = pd.DataFrame(residence_dict)
    bias_df = bias_df.set_index(np.array(n_examples_dict[0]))
    residence_df = residence_df.set_index(np.array(n_examples_dict[0]))
    bias_df['mean'] = bias_df.T.mean()
    residence_df['mean'] = residence_df.T.mean()
    
    bias_prefactor = np.sqrt(2)*starting_kwargs['error_scale']
    residence_prefactor = np.sqrt(2)*np.sqrt(2)*starting_kwargs['error_scale']

    bias_df['theory'] = bias_prefactor/np.sqrt(np.array(residence_df.index))
    residence_df['theory'] = residence_prefactor/np.sqrt(np.array(residence_df.index))
    return bias_df, residence_df

def generate_cplot(n_models, **data_kwargs):
    '''
    generates a visualization of model parameter estimates for linear models made
    by fitting the data from generate_income_df().
    TO DO: make fit columns specifiable in the function call.
    '''
    wmat_list = []
    for ii in xrange(n_models):
        df = generate_income_df(**data_kwargs)
        xx = np.matrix(df[['bias', 'residence', 'team']])
        yy = np.array(df[['income']])
        #print xx, yy
        try:
            wmat = (xx.T.dot(xx).I).dot(xx.T).dot(yy)
        except Exception:
            continue
        wmat_list += [wmat]
    bias = [w.item(0) for w in wmat_list]
    residence = [w.item(1) for w in wmat_list]
    team = [w.item(2) for w in wmat_list]
    print np.array(bias).mean() ,np.array(bias).std()
    print np.array(residence).mean(), np.array(income).std()
    print np.array(team).mean(), np.array(team).std()
    print np.array(team).mean(), np.array(team).std()

    plt.plot(bias)
    plt.plot(residence)
    plt.plot(team)
    plt.show()

    return np.array(bias), np.array(residence), np.array(team)

def generate_chists(bins=100, n_models=1000, **gen_kwargs):
    '''
    histogram generator for data derived in generate_cplot() above.
    '''
    bias, residence, team = generate_cplot(n_models=n_models, **gen_kwargs)
    plt.hist(bias, bins=bins)
    plt.hist(residence, bins=bins)
    plt.hist(team, bins=bins)
    plt.show()

###############################################
# A one-off calculation to prove to myself I'm not crazy.  
# commiting code in case I want to come back to this later, but would need some
# cleanup, renaming, etc.
xxx = np.matrix(zip(list(np.ones(100)), list(np.ones(50)) + list(np.zeros(50))))
corr=xxx.T.dot(xxx)
xxx_alt = np.matrix(zip(list(np.ones(100)), list(np.ones(51)) + list(np.zeros(49))))
corr_alt = xxx_alt.T.dot(xxx_alt)

xxo = np.matrix(zip(list(np.zeros(50)) + list(np.ones(50)), list(np.ones(50)) + list(np.zeros(50))))
coro=xxo.T.dot(xxo)
xxo_alt = np.matrix(zip(list(np.zeros(51)) + list(np.ones(49)), list(np.ones(51)) + list(np.zeros(49))))
coro_alt = xxo_alt.T.dot(xxo_alt)

yyy = np.matrix(list(np.ones(50)) + list(np.ones(50) + 0.1))
yyy_alt = np.matrix(list(np.ones(51)) + list(np.ones(49) + 0.1))

xxo_yyy=xxo.T.dot(yyy.T)
xxo_yyy_alt=xxo_alt.T.dot(yyy_alt.T)
xxx_yyy=xxx.T.dot(yyy.T)
xxx_yyy_alt=xxx_alt.T.dot(yyy_alt.T)
###############################################

# When called from teh command line this script calls run_sdev_() and handles 
# naming/storing output from that function
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, required=True, 
                          help="directory to store output files")
    parser.add_argument('--nSimulations', type=int, required=True, 
                          help="number of simulations to run")
    parser.add_argument('--maxPower', type=int, required=True, 
                          help="maximum power to use within each sim (sets max # of training examples per model)")
    parser.add_argument('--skew', type=float, default=0.5, 
                          help="sets ssprob")
    #parser.add_argument('-S', '--sepspace', action="store_true", 
    #                      help="sets separator to whitespace")
    starting_kwargs = {"ssprob": 0.5,
                        "slprob": 0.0,
                        "dlprob": 0.5,
                        "dsprob": 0.0,
                        "bias": 1,
                        "seattle_effect": 0.1,
                        "detroit_effect": 0,
                        "error_scale": 0.1,
                        "n_examples": None}
    
    args = parser.parse_args()
    if ((args.skew > 1) or (args.skew < 0)):
        raise ValueError('skew must be between 0 and 1')
    starting_kwargs['ssprob'] = args.skew
    starting_kwargs['dlprob'] = 1 - args.skew
    dir_name = args.directory
    n_simulations = args.nSimulations
    max_power = args.maxPower
    subdir_name = '__'.join([str(key) + '-' + str(starting_kwargs[key]) for key in sorted(starting_kwargs.keys())])
    subdir_name = os.path.join(dir_name, subdir_name)
       
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if not os.path.exists(subdir_name):
        os.makedirs(subdir_name)
    bdf_fname = os.path.join(subdir_name, 'bdf_' + str(n_simulations) + '_' + str(max_power) + '.out')
    rdf_fname = os.path.join(subdir_name, 'rdf_' + str(n_simulations) + '_' + str(max_power) + '.out')
    bdf, rdf = run_sdev_sim(n_simulations=n_simulations, max_power=max_power, starting_kwargs=starting_kwargs)
    bdf.to_csv(bdf_fname, sep = '\t')
    rdf.to_csv(rdf_fname, sep = '\t')

