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

from Models import CNN2
from Utils import genData
from Utils import pn_rates
from Utils import softclamp
# data generation -train/test

# train
noise = 1/4
tolerance = 0.25

positives, negatives, l, trueparams = genData(1200, noise, tolerance, ptc=True)

X = np.vstack([positives, negatives])
trueparams = np.vstack([trueparams, trueparams])
X = X.reshape((X.shape[0], X.shape[1], 1))
y = np.hstack([np.ones(len(positives)), np.zeros(len(negatives))])


# test
noiseT = 1/4
toleranceT = .25

positivesT, negativesT, l, testparams = genData(1000, noiseT, toleranceT, ptc=True)

Xt = np.vstack([positivesT, negativesT])
testparams = np.vstack([testparams, testparams])
Xt = Xt.reshape((Xt.shape[0], Xt.shape[1], 1))
yt = np.hstack([np.ones(len(positivesT)), np.zeros(len(negativesT))])




# training loop
original = False #original true means no param prediction
rnn = False #rnn true means yes sequence stuff
nb_epoch = 100
batch_size = 16
# cnn = make_cnn(1315)
if original:
    cnn=CNN(rnn)
else:
    cnn = CNN2()


optimizer = torch.optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9) #adjust lr
bce = nn.BCELoss() 

count = 0

for epoch in range(nb_epoch):
    rand_index = np.random.choice(X.shape[0], X.shape[0], replace=False)
    X = X[rand_index]
    y = y[rand_index]
    trueparams = trueparams[rand_index]
    losses = []
    accs = []
    errs = []
    name = "epoch"
    for i in range(0, X.shape[0], batch_size): #might want to shuffle batches after each eepoch
        batch_X = torch.FloatTensor(X[i:(i+batch_size)]).permute(0, 2, 1)
        batch_y = torch.FloatTensor(y[i:(i+batch_size)])
        batch_tp = torch.FloatTensor(trueparams[i:(i+batch_size)])
#         print(batch_tp.shape)
        pred, prms, res, fluxx, inputd, p_time, trueflux = cnn(batch_X, batch_tp)
        
        pred = pred.squeeze()
        
        if original:
            loss = bce(pred, batch_y)
        else:
            err = (prms-batch_tp).abs().mean(0)
            errs.append(err.cpu().detach().numpy())
#             loss = (res**2).mean((1, 2))
#             loss = loss * batch_y
#             loss = loss.mean()
            loss = ((prms-batch_tp)**2).mean() 
    
        #((prms-batch_tp)**2).mean()
#             print("params: " , prms[:3])
#             + (res**2).mean()#lower weight on the error gives better acc _ bce(pred, batch_y) + 
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach().numpy())
        accs.append((((pred > 0.5) == batch_y)*1.0).mean().cpu().detach().numpy()) #maybe adjust this
    
    print("--------------------------------------------------------------------------")
    print(f"Epoch: {epoch}, train Loss: {np.mean(losses)}, train Accuracy: {np.mean(accs)}")#np.mean(losses), np.mean(accs))
#     fn,fp,tn,tp = pn_rates(pred, batch_y)
#     print(' FP:',fp)
#     print(' FN:',fn)
    if not original:
        errs = np.stack(errs)
        print("train Errors: ", errs.mean(0))
#     print("bce loss: " + str(bce(pred, batch_y)))
    print("params mean^2: " + str(((prms-batch_tp)**2).mean()))
    print("residual mean^2: " + str((res**2).mean()))
#     print("param vals: ", prms)
    if count % 5 == 0:
        plt.figure()
        plt.plot(fluxx[0].cpu().detach().numpy(), 'r.'); plt.plot(inputd[0,0].cpu().detach().numpy(), 'k.'); plt.plot(trueflux[0].cpu().detach().numpy(), 'g.'); plt.savefig(fname = ('Plots/' + str(name + str(count)))); 
    count+=1
    ################# test metrics
        
#     pred_test = []
#     pred_test_prmz = []
#     for i in range(0, Xt.shape[0], batch_size):
#         batch_Xt = torch.FloatTensor(Xt[i:(i+batch_size)]).permute(0, 2, 1)
#         p, prmz, res_t, _, _, _ = cnn(batch_Xt)
#         p.squeeze()
#         pred_test.append(p.cpu().detach().numpy())
#         if not original:
#             pred_test_prmz.append(prmz.cpu().detach().numpy())
#     pred_test = np.concatenate(pred_test).flatten()
#     if not original:
#         pred_test_prmz = np.concatenate(pred_test_prmz)
#     fn,fp,tn,tp = pn_rates(pred_test,yt)
#     print('Test accuracy:', (((pred_test > 0.5) == yt)*1.0).mean())
#     print('Test FP:',fp)
#     print('Test FN:',fn)
#     if not original:
#         print('Test error: ', np.abs(pred_test_prmz-testparams).mean(0))
