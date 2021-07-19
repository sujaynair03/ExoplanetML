import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams; rcParams["figure.dpi"] = 150
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from ELCA_C import transit

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

class CNN(nn.Module): #identical to original model - just in pt
    def __init__(self, rnn = False):
        super().__init__()
        self.rnn = rnn
        self.sigm = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.c1 = nn.Conv1d(1, 16, 5)
        self.pool = nn.AvgPool1d(5, 2)
        self.c2 = nn.Conv1d(16, 8, 5)
        self.fc1 = nn.Linear(8*323, 64)
        self.drp = nn.Dropout(0.1)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32,8)
        self.fc4 = nn.Linear(8,1)
        if self.rnn:
            self.lstm = nn.LSTM(8,8)
        
        
    def forward(self, data):
        h = self.c1(data)
        h = self.pool(h)
        h = self.c2(h)
        h = self.pool(h)
        if self.rnn:
            h = h.permute(2,0,1)
            h, _ = self.lstm(h)
            h = h.permute(1,2,0)
        h = h.reshape(data.shape[0], -1) #making into one vector
        
        h = self.drp(self.relu(self.fc1(h))) #play around with dropout
        h = self.drp(self.relu(self.fc2(h)))
        h = self.drp(self.relu(self.fc3(h)))
        h = self.sigm(self.fc4(h))
        return h, None
    
    
class CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigm = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.c1 = nn.Conv1d(1, 16, 5)
        self.pool = nn.AvgPool1d(5, 2)
        self.c2 = nn.Conv1d(16, 8, 5)
        self.fc1 = nn.Linear(8*323, 64)
        self.drp = nn.Dropout(0.1)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32,8)
        self.fc4 = nn.Linear(8,1)
        nimages = 27.4*24*60 / 30
        self.time = np.linspace(0,27.4,int(nimages))
        self.param_prediction = nn.Linear(8, 3)
        
        self.c1_res = nn.Conv1d(1, 16, 5) #structured same as before
        self.pool_res = nn.AvgPool1d(5, 2)
        self.c2_res = nn.Conv1d(16, 8, 5)
        self.fc1_res = nn.Linear(8*323, 64)
        self.fc2_res = nn.Linear(64, 32)
        self.fc3_res = nn.Linear(32,8)
        self.fc4_res = nn.Linear(8,1)
        
        self.prior = { 
            'i':90,          # Inclination [deg] [90]
            'e':0,            # Eccentricity
            'w':0,          # Arg of periastron
            't0':0.75,         # time of mid transit [0,1] - could include this (pred) (need to change in data gen? ask)
            'method':'quad',
            'ldc': [0.3, 0.01]}
        

        self.tm = TransitModule(self.time, **self.prior).float()
        
    def forward(self, data, trueparams): 
        h = self.c1(data)
        h = self.pool(h)
        h = self.c2(h)
        h = self.pool(h)
        h = h.view(data.shape[0], -1)
        h = self.drp(self.relu(self.fc1(h)))
        h = self.drp(self.relu(self.fc2(h)))
        h = self.drp(self.relu(self.fc3(h)))
        
        prms0 = (self.param_prediction(h)) ## Predicted params
        predRP=prms0[:, 0] #separating, added clips here instead
        predARS=prms0[:, 1]
        predP=prms0[:, 2]
        truepredRP=trueparams[:, 0] #separating, added clips here instead
        truepredARS=trueparams[:, 1]
        truepredP=trueparams[:, 2]
        
        returnedprms = torch.stack([predRP, predARS, predP], -1) #saving for later
        truereturnedprms = torch.stack([truepredRP, truepredARS, truepredP], -1) #saving for later

#         timeminusprior = torch.FloatTensor(self.time-(0.5*predP)).unsqueeze(0).repeat(16, 1) original
        timeminusprior = torch.FloatTensor(self.time).unsqueeze(0).repeat(16, 1)-(0.5*predP.unsqueeze(1).repeat(1, 1315)) 
        truetimeminusprior = torch.FloatTensor(self.time).unsqueeze(0).repeat(16, 1)-(0.5*truepredP.unsqueeze(1).repeat(1, 1315))

        phase_new = ((timeminusprior.T/predP)+predP*0.5).T %1
        p_time_new = (phase_new.T*predP).T
        p_time_new2, _ = torch.sort(p_time_new, dim =1)
        flux = self.tm(time = p_time_new2.detach(),
                       rp=predRP,
                       a=predARS,
                       P=predP,
                       t0 = predP*0.5) 
        
        truephase_new = ((truetimeminusprior.T/truepredP)+truepredP*0.5).T %1
        truep_time_new = (truephase_new.T*truepredP).T
        truep_time_new2, _ = torch.sort(truep_time_new, dim =1)
        trueflux = self.tm(time = truep_time_new2.detach(),
                       rp=truepredRP,
                       a=truepredARS,
                       P=truepredP,
                       t0 = truepredP*0.5)

        delta = data - flux.unsqueeze(1).float() ##residuals
        truedelta = data - trueflux.unsqueeze(1).float()

        h = self.c1_res(delta)
        h = self.pool_res(h)
        h = self.c2_res(h)
        h = self.pool_res(h)
        h = h.view(data.shape[0], -1)
        h = self.drp(self.relu(self.fc1_res(h)))
        h = self.drp(self.relu(self.fc2_res(h)))
        h = self.drp(self.relu(self.fc3_res(h)))
        h = self.sigm(self.fc4_res(h)) ## final prediction
        return h, returnedprms, delta, flux, data, p_time_new2, trueflux
    
class CNN3(nn.Module):
    def __init__(self, rnn = False):
        super().__init__()
        self.rnn = rnn
        self.sigm = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.c1 = nn.Conv1d(1, 16, 5)
        self.pool = nn.AvgPool1d(5, 2) #could try max
        self.c2 = nn.Conv1d(16, 8, 5)
        self.fc1 = nn.Linear(8*323, 64)
        self.drp = nn.Dropout(0.1)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32,8)
        self.fc4 = nn.Linear(8,1)
        self.param_prediction = nn.Linear(8, 3)
        if self.rnn:
            self.lstm = nn.LSTM(8,8)
        
    def forward(self, data):
        h = self.c1(data)
        h = self.pool(h)
        h = self.c2(h)
        h = self.pool(h)
        if self.rnn:
            h = h.permute(2,0,1)
            h, _ = self.lstm(h)
            h = h.permute(1,2,0)
        h = h.reshape(data.shape[0], -1) #making into one vector
        
        h = self.drp(self.relu(self.fc1(h)))
        h = self.drp(self.relu(self.fc2(h)))
        h = self.drp(self.relu(self.fc3(h)))
        pred = self.sigm(self.fc4(h))
        prms0 = (self.param_prediction(h))
        
        return pred, prms0
    
