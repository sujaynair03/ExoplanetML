import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams; rcParams["figure.dpi"] = 150
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
# from ELCA_C import transit

import os
import pickle
from sklearn import preprocessing
from tensorflow.keras.layers import Input, Dense, Conv1D, AveragePooling1D, Concatenate, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.patches as mpatches
from transitleastsquares import transitleastsquares
import pylightcurve_torch
from pylightcurve_torch import TransitModule

import pdb

def genData(size, noise, tolerance, TLS = False, ptc = False, fold=True):
#     assert(False)
    nimages = 27.4*24*60 / 30
    time = np.linspace(0,27.4,int(nimages)) # [day]
    fluxs =[]
    phases = []
    per_tolerance = tolerance
    fakeperiods = []
    negphases=[]
    negfluxes = []
    sorted_idxs = []
    sfluxs = []
    TLSresults = []
    count = 0
    trueprms = []
    for i in range(0,size):
        if ptc:
            prior = { 
            'rp': np.random.random()*.05 + .05,         # Rp/Rs           [0.05-0.1]
            'a': np.random.random()*4 + 12,         # a/Rs            [12-16]
            'P': np.random.uniform(4.0, 7.0),             # Period [day]    [4-7]
            'i':90,          # Inclination [deg] [90]
            'e':0,            # Eccentricity
            'w':0,          # Arg of periastron
            't0':np.random.uniform(0.0, 1.0), #0.75,         # time of mid transit [0,1]
            'method':'quad',
            'ldc': [0.3, 0.01]
            }
            # prior['t0'] = 0.5*prior['P'] #to be in middle of plot
            rpname = 'rp'
            pername = 'P'
            t0name = 't0'
            arsname= 'a'
        else:
            prior = { 
                'rprs':np.random.random()*.05 + .05,         # Rp/Rs           [0.05-0.1]
                'ars':np.random.random()*4 + 12,         # a/Rs            [12-16]
                'per':np.random.uniform(4.0, 7.0),             # Period [day]    [0.5-14]
                'inc':90,          # Inclination [deg] [90]
                'u1': 0.3, 'u2': 0.01, # limb darkening (linear, quadratic)
                'ecc':0,            # Eccentricity
                'omega':0,          # Arg of periastron
                'tmid':np.random.uniform(0.0, 1.0), #0.75         # time of mid transit [0,1]
            } 
            # prior['tmid'] = 0.5*prior['per']
            rpname = 'rprs'
            pername = 'per'
            t0name = 'tmid'
            arsname = 'ars'
            
        trueprms.append([prior[rpname], prior[arsname], prior[pername], prior[t0name]]) # true params for later prediction

        phase = (time-prior[t0name] + 0.5*prior[pername])/prior[pername] % 1 # same as before
        phases.append(phase)
        n = (noise*np.random.random()+noise)* np.random.normal(0, prior[rpname]**2, len(time))
        
        if ptc: #if ptc use pylightcurvetorch transit function, else use standard one from before
            tm = TransitModule(time, **prior) 
            # flux = (tm()).squeeze().cpu().numpy() 
            flux = (tm() + n).squeeze().cpu().numpy()
        else:
            # flux = transit(time, prior)
            flux = transit(time, prior) + n
        
        
        if(TLS): #from previous study
            print(size)
            count+=1
            print(count)
            model = transitleastsquares(time, flux)
            result = model.power()
            TLSresults.append(result)
            
        fluxs.append(flux)
        
        fake_period = np.clip(np.random.uniform(prior[pername]-per_tolerance, prior[pername]+per_tolerance), 4.0, 7.0)
        fakeperiods.append(fake_period)
        phase = (time-prior[t0name] + 0.5*fakeperiods[i])/fakeperiods[i] % 1
        negphases.append(phase)
        s_i = np.argsort(phase)
        negfluxes.append(fluxs[i][s_i])
        
        
    for i in range(0,size):
        sorted_idxs.append(np.argsort(phases[i]))
        sfluxs.append(fluxs[i][sorted_idxs[i]])
#     for i in range(0,len(sfluxs)):
#         sfluxs[i] = preprocessing.scale(sfluxs[i])

    # pdb.set_trace()
    negfluxsarr  = np.array(negfluxes)
    trueprms = np.array(trueprms)
    trueprms[:,0] = (trueprms[:,0] - 0.05)/(0.1 - 0.05)
    trueprms[:,1] = (trueprms[:,1] - 12)/(16 - 12)
    trueprms[:,2] = (trueprms[:,2] - 4)/(7 - 4)
    trueprms[:,3] = (trueprms[:,3] - 0)/(1 - 0)
    # pdb.set_trace()

#     for i in range(0,len(negfluxsarr)):
#         negfluxsarr[i] = preprocessing.scale(negfluxsarr[i])
    if fold:
        return sfluxs, negfluxsarr, TLSresults, trueprms #return trueparams because model may use this in training
    else:
        return np.array(fluxs), None, TLSresults, trueprms



def pn_rates(y_pred, y, thres = 0.5):
#     y_pred = (model.predict(X) > thres) * 1.0

    pos_idx = y==1
    neg_idx = y==0

    tp = np.sum(y_pred[pos_idx]> thres)/y_pred.shape[0]
    fn = np.sum(y_pred[pos_idx] <= thres)/y_pred.shape[0]

    tn = np.sum(y_pred[neg_idx] <= thres)/y_pred.shape[0]
    fp = np.sum(y_pred[neg_idx]> thres)/y_pred.shape[0]

    return fn,fp,tn,tp
  
  
def softclamp(x, mn, mx):
    return mn + (mx-mn)*F.sigmoid(x)
