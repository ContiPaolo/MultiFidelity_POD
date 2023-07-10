#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Paolo Conti
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Reshape, Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import KFold
import numpy as np
from tensorflow.keras.optimizers import Adam,Nadam,Adamax, RMSprop
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.constraints import max_norm
from pyDOE import lhs
from tensorflow.keras.models import model_from_json
from celluloid import Camera
from matplotlib.animation import PillowWriter
from IPython.display import HTML
import scipy.io

from sklearn.utils import extmath
import time

def create_model(params, dim_input, dim_output):
    K.clear_session()

    X =  Input(shape=(None, dim_input))
    
    a = LSTM(params['nodes'], return_sequences = True)(X) 
    for i in range(params['lay']-1):
        a = Dropout(params['dropout'])(a)
        a = LSTM(params['nodes'], return_sequences = True)(a)
    for i in range(params['lay_dense']):
        a  = Dense(params['nodes_dense'])(a)
    
    output = Dense(dim_output,activation='linear')(a)

    model = Model(inputs = X, outputs = output)
    opti = getOpti(params['opti'],params['lr'])
    model.compile(loss = 'mse', optimizer = opti, metrics = ['mse'])
    return model

def train_model(params, model, x, y, verbose = 1):
    callback = tf.keras.callbacks.EarlyStopping(monitor='mse', patience=params['patience'], restore_best_weights=True)
    hist = model.fit(x, y, epochs=params['epochs'],batch_size=params['batch'], verbose = verbose, callbacks=[callback])
    return hist

def load_model(name):
    json_file = open(name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    model.load_weights(name + '.h5')
    print('Loaded ' + name)
    return model

def save_model(model,name):
    model_json = model.to_json()
    with open(name + ".json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights(name + ".h5")
    print("Saved " + name)

def compute_randomized_SVD(S, N_POD, N_h, n_channels, name='', verbose=False):
    if verbose:
        print('Computing randomized POD...')
    U = np.zeros((n_channels * N_h, N_POD))
    start_time = time.time()
    for i in range(n_channels):
        U[i * N_h: (i + 1) * N_h], Sigma, Vh = extmath.randomized_svd(S[i * N_h: (i + 1) * N_h, :],
                                                                      n_components=N_POD, transpose=False,
                                                                      flip_sign=False, random_state=123)
        if verbose:
            print('Done... Took: {0} seconds'.format(time.time() - start_time))

    if verbose:
        I = 1. - np.cumsum(np.square(Sigma)) / np.sum(np.square(Sigma))
        print(I[-1])

    if name:
        sio.savemat(name, {'V': U[:, :N_POD]})

    return U, Sigma


def getOpti(name,lr):
    if name == 'Adam':
        return Adam(learning_rate=lr,amsgrad=True)
    elif name == 'Nadam':
        return Nadam(learning_rate=lr)
    elif name == 'Adamax':
        return Adamax(learning_rate=lr)
    elif name == 'RMSprop':
        return RMSprop(learning_rate=lr)
    elif name == 'standardadam':
        return 'adam'

def sliding_windows(data_input, data_output, seq_length, freq=1):
    x = []
    y = []

    for i in range(data_input.shape[0]):
        for j in range(0, data_input.shape[1] - seq_length, freq):
            _x = data_input[i, j:(j + seq_length), :]
            _y = data_output[i, j:(j + seq_length), :]
            x.append(_x)
            y.append(_y)

    return np.array(x), np.array(y)

def reshape_lstm(params, times):
    N = len(params)
    Nt = len(times)
    
    params_lstm = np.tile(params.reshape(-1,1), Nt).reshape(N, Nt,1)
    times_lstm = np.transpose(np.tile(times.reshape(-1,1), N)).reshape(N,Nt,1)
    return np.concatenate((params_lstm, times_lstm), axis = 2)
    
def kCrossVal(p,N,Nepo,x,y,params,name):
    #K.clear_session()
    model = getModel(params,name)
    Nfold = int(N/p)
    score = np.zeros([Nfold])
    cv = KFold(n_splits=Nfold,shuffle=True)
    splits = cv.split(x)
    i=0
    for train_index, test_index in splits:
      x_train = x[train_index,:]
      y_train = y[train_index]
      x_val = x[test_index,:]
      y_val = y[test_index]
      #model = getModel(params,name)
      model.fit(x_train,y_train,epochs=Nepo,batch_size=N-p,verbose=0)
      score[i] = np.mean(np.square(y_val - model.predict(x_val)[:,0]))     
      i = i+1
    return np.mean(score)

def kCrossValSingle(N,Nepo,x,y,params,name):
    score = np.zeros([N])
    cv = KFold(N)
    splits = cv.split(x)
    i=0
    for train_index, test_index in splits:
      x_train = x[train_index]
      y_train = y[train_index]
      x_val = x[test_index]
      y_val = y[test_index]
      model = getModel(params,name)
      model.fit(x_train,y_train,epochs=Nepo,batch_size=N-1,verbose=0)
      score[i] = np.square(y_val - model.predict(x_val))     
      i = i+1
    return np.mean(score)

def kCrossValGP(p,Nhf,Nlf,Nepo,xhf,yhf,xlf,ylf,params,name):
    Nfolds = int(Nhf/p)
    score = np.zeros([Nfolds])    
    cv = KFold(n_splits = Nfolds, shuffle = True)
    splits = cv.split(xhf)
    i=0
    N = Nhf + Nlf
    model = getModel(params,name)
    for train_index, test_index in splits:
      xhf_train = xhf[train_index,:]
      yhf_train = yhf[train_index]
      xhf_val = xhf[test_index,:]
      yhf_val = yhf[test_index]
      x_train = np.concatenate((xhf_train,xlf))
      yhf_train = np.concatenate((yhf_train,np.full(Nlf,-10)))
      ylf_train = np.concatenate((np.full(Nhf-p,-10),ylf))    
      model.fit(x_train,[yhf_train,ylf_train],epochs=int(params['epochs'])*Nepo,batch_size=N-p,verbose=0)
      score[i] = np.mean(np.square(yhf_val - model.predict(xhf_val)[0][:,0]))
      i = i+1
    return np.mean(score)

def transfBestparam(bestparam,dic):
    for key in bestparam:
        if key in ['kernel_init','opt']:
            bestparam[key] = dic[key][bestparam[key]]
    return

def load_reaction_diffusion(params, fidelity, path, splitted = False):
    u_list = []
    for param in params:
        name = path + 'u_' + fidelity + '_' + "{:.3f}".format(param) 
        
        if splitted:
            u_test_list = []
            for i in [1,2]:
                u_test = scipy.io.loadmat(name + '_' + str(i) + '.mat')['u']
                u_test_list.append(u_test)
            u = np.concatenate(u_test_list, axis = 2)
        else:
            u = scipy.io.loadmat(name + '.mat')['u']
            
        u_list.append(u)
    
    data_u = np.stack(u_list, axis=3)
    x = scipy.io.loadmat(path + 'x_' + fidelity + '.mat')['x']
    t = scipy.io.loadmat(path + 't_' + fidelity + '.mat')['t']
    
    return data_u, x.flatten(), t.flatten()