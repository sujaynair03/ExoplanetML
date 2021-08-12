import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams; rcParams["figure.dpi"] = 150
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import sys
sys.path.insert(1, '/Users/sujaynair/Documents/astroML')
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

from Models import CNN2, CNN4
from Utils import genData
from Utils import pn_rates
from Utils import softclamp
import sys
# data generation -train/test

### FLAGS
mode = int(sys.argv[2]) #False #original true means no param prediction # 0 - just classification, 1 - classification + regression, 2 - just regression
if mode == 0:
    fold = 1
elif mode == 1:
    fold = 1
elif mode == 2:
    fold = 0

# train
noise = 1/4
tolerance = 0.25
positives, negatives, l, trueparams = genData(1000, noise, tolerance, ptc=True, fold = fold)
if fold:
    X = np.vstack([positives, negatives])
    trueparams = np.vstack([trueparams, trueparams])
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = np.hstack([np.ones(len(positives)), np.zeros(len(negatives))])
else:
    X = positives
    X-=1
    X = X.reshape((X.shape[0], X.shape[1], 1))


# test
noiseT = 1/4
toleranceT = .25
positivesT, negativesT, l, testparams = genData(200, noiseT, toleranceT, ptc=True, fold = fold)
if fold:
    Xt = np.vstack([positivesT, negativesT])
    testparams = np.vstack([testparams, testparams])
    Xt = Xt.reshape((Xt.shape[0], Xt.shape[1], 1))
    yt = np.hstack([np.ones(len(positivesT)), np.zeros(len(negativesT))])
else:
    Xt = positivesT
    Xt-=1
    Xt = Xt.reshape((Xt.shape[0], Xt.shape[1], 1))




# training loop
rnn = False #rnn true means yes sequence stuff
nb_epoch = 1000
batch_size = int(sys.argv[1])
# cnn = make_cnn(1315)
if mode == 0:
    cnn = CNN(rnn)
elif mode == 1:
    cnn = CNN2()
elif mode == 2:
    cnn = CNN4()


optimizer = torch.optim.Adam(cnn.parameters(), lr=0.0001) #adjust lr
bce = nn.BCELoss() 

count = 0

l = []
r = []
p = []
test_p = []
test_r = []

for epoch in range(nb_epoch):
    rand_index = np.random.choice(X.shape[0], X.shape[0], replace=False)
    X = X[rand_index]
    if mode != 2:
        y = y[rand_index]
    trueparams = trueparams[rand_index]
    losses = []
    accs = []
    errs = []
    name = "epoch"
    # name1 = "losscurve"
    for i in range(0, X.shape[0], batch_size): #might want to shuffle batches after each eepoch
        batch_X = torch.FloatTensor(X[i:(i+batch_size)]).permute(0, 2, 1)
        if mode != 2:
            batch_y = torch.FloatTensor(y[i:(i+batch_size)])
        batch_tp = torch.FloatTensor(trueparams[i:(i+batch_size)])
#         print(batch_tp.shape)
        if mode == 2:
            _, prms, res, fluxx, inputd, _, trueflux = cnn(batch_X, batch_tp)
        else:
            pred, prms, res, fluxx, inputd, p_time, trueflux = cnn(batch_X, batch_tp)
            pred = pred.squeeze()
        
        if mode == 0:
            loss = bce(pred, batch_y)
        elif mode == 1:
            err = (prms-batch_tp).abs().mean(0)
            errs.append(err.cpu().detach().numpy())
#             loss = (res**2).mean((1, 2))
#             loss = loss * batch_y
#             loss = loss.mean()
            loss = ((prms-batch_tp)**2).mean()
        elif mode == 2:
            err = (prms-batch_tp).abs().mean(0)
            errs.append(err.cpu().detach().numpy())
            lossprm = (prms-batch_tp)
            residprm = (res**2).mean()
            loss = (lossprm**2).mean()
        
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach().numpy())
        if mode != 2:
            accs.append((((pred > 0.5) == batch_y)*1.0).mean().cpu().detach().numpy()) #maybe adjust this
    
    l.append(np.mean(losses))
    p.append(((prms-batch_tp)**2).mean())
    r.append((res**2).mean())
    print("--------------------------------------------------------------------------")
    print(f"Epoch: {epoch}, train Loss: {np.mean(losses)}, train Accuracy: {np.mean(accs)}")#np.mean(losses), np.mean(accs))
#     fn,fp,tn,tp = pn_rates(pred, batch_y)
#     print(' FP:',fp)
#     print(' FN:',fn)
    if mode != 0:
        errs = np.stack(errs)
        print("train Errors: ", errs.mean(0))
#     print("bce loss: " + str(bce(pred, batch_y)))
    print("params mean^2: " + str(((prms-batch_tp)**2).mean()))
    print("residual mean^2: " + str((res**2).mean()))
    print("param vals: ", prms[:2], batch_tp[:2])

    ################# test metrics
    with torch.no_grad():
        pred_test_prmz = []
        pred_test_res = []
        for i in range(0, Xt.shape[0], batch_size):
            batch_Xt = torch.FloatTensor(Xt[i:(i+batch_size)]).permute(0, 2, 1)
            batch_tpt = torch.FloatTensor(testparams[i:(i+batch_size)])
            _, prmz, res_t, flux_t, input_d_t, _, true_test = cnn(batch_Xt, batch_tpt)
            if mode == 2:
                pred_test_prmz.append(((prmz - batch_tpt)**2).mean().cpu().detach().numpy())
                pred_test_res.append((res_t**2).mean().cpu().detach().numpy())
        # if mode == 2:
        #     pred_test_prmz = np.concatenate(pred_test_prmz)
        #     pred_test_res = np.concatenate(pred_test_res)
        test_p.append(np.mean(pred_test_prmz))
        test_r.append(np.mean(pred_test_res))
    # plt.figure()
    # pdb.set_trace()
    # plt.plot(losses); plt.savefig(fname = ('Loss_Curves/' + str(name1 + str(count))))
    if count % 5 == 0:
        plt.figure()
        plt.plot(fluxx[0].cpu().detach().numpy(), 'r.')
        plt.plot(inputd[0,0].cpu().detach().numpy(), 'k.')
        plt.plot(trueflux[0].cpu().detach().numpy(), 'g.')
        plt.savefig(fname = ('Plots/' + str(name + str(count))))
        plt.close()

        plt.figure()
        randt = np.random.randint(0, flux_t.shape[0])
        plt.plot(flux_t[randt].cpu().detach().numpy(), 'r.')
        plt.plot(input_d_t[randt,0].cpu().detach().numpy(), 'k.')
        plt.plot(true_test[randt].cpu().detach().numpy(), 'g.')
        plt.savefig(fname = ('Plots/test_' + str(name + str(count))))
        plt.close()

        plt.figure()
        plt.plot(l[5:]); plt.savefig(fname = ('Plots/' + 'losscurve'))
        plt.close()
        plt.figure()
        plt.plot(p[5:]); plt.savefig(fname = ('Plots/' + 'paramcurve'))
        plt.close()
        plt.figure()
        plt.plot(r[5:]); plt.savefig(fname = ('Plots/' + 'residcurve'))
        plt.close()
        plt.figure()
        plt.plot(test_p[5:]); plt.savefig(fname = ('Plots/' + 'test_paramcurve'))
        plt.close()
        plt.figure()
        plt.plot(test_r[5:]); plt.savefig(fname = ('Plots/' + 'test_residcurve'))
        plt.close()
    count+=1
# pdb.set_trace()


    # pred_test = np.concatenate(pred_test).flatten()
#     if not original:
#         pred_test_prmz = np.concatenate(pred_test_prmz)
#     fn,fp,tn,tp = pn_rates(pred_test,yt)
#     print('Test accuracy:', (((pred_test > 0.5) == yt)*1.0).mean())
#     print('Test FP:',fp)
#     print('Test FN:',fn)
#     if not original:
#         print('Test error: ', np.abs(pred_test_prmz-testparams).mean(0))
