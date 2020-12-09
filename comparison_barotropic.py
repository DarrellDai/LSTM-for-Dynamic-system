# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 20:27:55 2020

@author: Darrell Dai
"""
import torch.nn as nn
import torch.nn.parallel
import torch
import LSTMs as models 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.io import loadmat
import scipy.stats
#model_classic_LSTM=classic_LSTM.main()
#model_mLSTM=mLSTM.main()
#model_LSTM_attention=LSTM_attention.main()

def load_checkpoint(filepath, model, device='cuda'):
    if device=='cuda':
        model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = a.size
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    # m, se = np.mean(a), np.std(a)
    # h = se
    return m, h

def plot_shade(data, plt, color, label, linewidth, linestyle = '-', marker = None):
    num_points = np.shape(data)[0]
    mean = np.zeros(num_points)
    interval = np.zeros(num_points)

    for i in range(num_points):
        m, h = mean_confidence_interval(data[i])
        mean[i] = m
        interval[i] = h
    if label == 'full model':
        plt.plot(np.arange(data.shape[0])*0.05, mean, '-', color=color, label = label, linewidth = linewidth, 
                 linestyle = linestyle, marker = marker, markevery=50, markersize = 5)
    else:
        plt.plot(np.arange(data.shape[0]) * 0.05, mean, '-', color=color, label=label, linewidth=linewidth,
                 linestyle=linestyle, marker=marker, markevery=50, markersize=5, alpha = 0.9)

def plot_shade_small(data, plt, color, label, linewidth, linestyle='-', marker=None):
    num_points = np.shape(data)[0]
    mean = np.zeros(num_points)
    interval = np.zeros(num_points)

    for i in range(num_points):
        m, h = mean_confidence_interval(data[i])
        mean[i] = m
        interval[i] = h
    interval= range(data.shape[0]//30, data.shape[0]//5)    
    if label == 'full model':
        plt.plot(np.arange(data.shape[0])[interval] * 0.05, mean[interval], '-', color=color, label=label, linewidth=linewidth,
                 linestyle=linestyle, marker=marker, markevery=50, markersize=5)
    else:
        plt.plot(np.arange(data.shape[0])[interval] * 0.05, mean[interval], '-', color=color, label=label, linewidth=linewidth,
                 linestyle=linestyle, marker=marker, markevery=50, markersize=5, alpha=0.9)     
        
def get_ACF(data, Tcorr):
    trajectory = data
    seq_length = trajectory.shape[0]
    trajectory = trajectory- np.mean(trajectory, 0)
    ACF_U = np.zeros((Tcorr, 1))
    ACF_T = np.zeros((Tcorr, 1))
    ACF_v = np.zeros((Tcorr, 1))
    for s in range(seq_length-Tcorr):
        ACF_U[s: s+1] = np.mean(trajectory[:Tcorr, 0] * trajectory[s:Tcorr + s, 0], axis = 0, keepdims=True)
        ACF_T[s: s+1] = np.mean(trajectory[:Tcorr, 1] * trajectory[s:Tcorr + s, 1], axis = 0, keepdims=True)
        ACF_v[s: s+1] = np.mean(trajectory[:Tcorr, 2] * trajectory[s:Tcorr + s, 2], axis = 0, keepdims=True)
    ACF = np.concatenate((ACF_U, ACF_T, ACF_v),axis=1)    
    return ACF

def get_density(data):
    density=[]
    for i in range(data.shape[1]):
        density.append(scipy.stats.gaussian_kde(data[:,i]))  
    return density

def plot_trajectory(predict_classic_LSTM, predict_mLSTM, predict_LSTM_attention, true): 
    fig = plt.figure(figsize=(200,10))
    for i in range(predict_classic_LSTM.shape[1]):
        var_max=np.concatenate((predict_classic_LSTM[:,i],predict_mLSTM[:,i],predict_LSTM_attention[:,i],true[:,i]), axis=0).max()
        var_min=np.concatenate((predict_classic_LSTM[:,i],predict_mLSTM[:,i],predict_LSTM_attention[:,i],true[:,i]), axis=0).min()
        ax1 = fig.add_subplot(predict_classic_LSTM.shape[1], 1, i+1)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        
        axins = inset_axes(ax1, width=1.5, height=1.5, loc='upper right')
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.title("x" if i==0 else "y")
        # full model
        plot_shade(predict_classic_LSTM[:,i], ax1, 'c', 'classic_LSTM', 1.5, '-.')
        plot_shade(predict_mLSTM[:,i], ax1, 'r', 'mLSTM', 1.5, '--')
        plot_shade(predict_LSTM_attention[:,i], ax1, 'b', 'LSTM_attention', 1.5, ':')
        plot_shade(true[:,i], ax1, 'limegreen', 'full model', 2.5, '-')
        
        plot_shade_small(predict_classic_LSTM[:,i], axins, 'c', 'classic_LSTM', 1.5, '-.')
        plot_shade_small(predict_mLSTM[:,i], axins, 'r', 'mLSTM', 1.5, '--')
        plot_shade_small(predict_LSTM_attention[:,i], axins, 'b', 'LSTM_attention', 1.5, ':')
        plot_shade_small(true[:,i], axins, 'limegreen', 'full model', 2.5, '-')
        
        ax1.grid(linestyle='dotted')
        ax1.set_ylabel('Trajectory', fontsize = 18)
        ax1.set_xlabel('time', fontsize = 18)
        ax1.legend(prop={'size': 11}, loc='upper center', bbox_to_anchor=(0.36,0.95))
        # plt.legend()
        ax1.set_xlim((0,true.shape[0] * 0.05))
        ax1.set_ylim((var_min,var_max))
        axins.grid(linestyle='dotted')
        plt.title("U" if i==0 else "T" if i==1 else 'v')
    plt.savefig('plot/{}.png'.format('Trajectory'), bbox_inches='tight')
    plt.show() 
       
def plot_ACF(predict_classic_LSTM, predict_mLSTM, predict_LSTM_attention, true, Tcorr): 
    predict_classic_LSTM_ACF = get_ACF(predict_classic_LSTM, Tcorr)
    predict_mLSTM_ACF = get_ACF(predict_mLSTM, Tcorr)
    predict_LSTM_attention_ACF = get_ACF(predict_LSTM_attention, Tcorr)
    true_ACF = get_ACF(data_true, Tcorr)
    fig = plt.figure(figsize=(200,10))
    for i in range(predict_classic_LSTM.shape[1]):
        var_max=np.concatenate((predict_classic_LSTM_ACF[:,i],predict_mLSTM_ACF[:,i],predict_LSTM_attention_ACF[:,i],true_ACF[:,i]), axis=0).max()
        var_min=np.concatenate((predict_classic_LSTM_ACF[:,i],predict_mLSTM_ACF[:,i],predict_LSTM_attention_ACF[:,i],true_ACF[:,i]), axis=0).min()
        ax1 = fig.add_subplot(predict_classic_LSTM.shape[1], 1, i+1)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        
        axins = inset_axes(ax1, width=1.5, height=1.5, loc='upper right')
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.title("x" if i==0 else "y")
        # full model
        plot_shade(predict_classic_LSTM_ACF[:,i], ax1, 'c', 'classic_LSTM', 1.5, '-.')
        plot_shade(predict_mLSTM_ACF[:,i], ax1, 'r', 'mLSTM', 1.5, '--')
        plot_shade(predict_LSTM_attention_ACF[:,i], ax1, 'b', 'LSTM_attention', 1.5, ':')
        plot_shade(true_ACF[:,i], ax1, 'limegreen', 'full model', 2.5, '-')
        
        plot_shade_small(predict_classic_LSTM_ACF[:,i], axins, 'c', 'classic_LSTM', 1.5, '-.')
        plot_shade_small(predict_mLSTM_ACF[:,i], axins, 'r', 'mLSTM', 1.5, '--')
        plot_shade_small(predict_LSTM_attention_ACF[:,i], axins, 'b', 'LSTM_attention', 1.5, ':')
        plot_shade_small(true_ACF[:,i], axins, 'limegreen', 'full model', 2.5, '-')
        
        ax1.grid(linestyle='dotted')
        ax1.set_ylabel('ACF', fontsize = 18)
        ax1.set_xlabel('time', fontsize = 18)
        ax1.legend(prop={'size': 11}, loc='upper center', bbox_to_anchor=(0.36,0.95))
        # plt.legend()
        ax1.set_xlim((0, true_ACF.shape[0] * 0.05))
        ax1.set_ylim((var_min,var_max))
        axins.grid(linestyle='dotted')
        plt.title("x" if i==0 else "y")
    plt.savefig('plot/{}.png'.format('ACF'), bbox_inches='tight')
    plt.show()  
    
def plot_density(predict_classic_LSTM, predict_mLSTM, predict_LSTM_attention, true): 
    predict_classic_LSTM_density= get_density(predict_classic_LSTM)
    predict_mLSTM_density = get_density(predict_mLSTM)
    predict_LSTM_attention_density = get_density(predict_LSTM_attention)
    true_density= get_density(data_true)
    fig = plt.figure(figsize=(600,10))
    for i in range(predict_classic_LSTM.shape[1]):
        var_min=np.concatenate((predict_classic_LSTM[:,i],predict_mLSTM[:,i],predict_LSTM_attention[:,i],true[:,i]), axis=0).min()
        var_max=np.concatenate((predict_classic_LSTM[:,i],predict_mLSTM[:,i],predict_LSTM_attention[:,i],true[:,i]), axis=0).max()
        x = np.linspace(var_min, var_max, 100)
        
        ax1 = fig.add_subplot(predict_classic_LSTM.shape[1], 1, i+1)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        
        axins = inset_axes(ax1, width=1.5, height=1.5, loc='upper right')
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.title("x" if i==0 else "y")
        
        ax1.plot(x, predict_classic_LSTM_density[i](x), '-.', color='c', label='classic_LSTM', linestyle = 'dotted', linewidth = 1.5, alpha = 0.9)
        ax1.plot(x, predict_mLSTM_density[i](x), '--', color='r', label='mLSTM', linestyle = 'dotted', linewidth = 1.5, alpha = 0.9)
        ax1.plot(x, predict_LSTM_attention_density[i](x), '-.', color='b', label='LSTM_attention', linestyle = 'dotted', linewidth = 1.5, alpha = 0.9)
        ax1.plot(x, true_density[i](x), '-', color='limegreen', label='full model', linestyle = 'dotted', linewidth = 2.5)
        p_max=np.concatenate((predict_classic_LSTM_density[i](x),predict_mLSTM_density[i](x),predict_LSTM_attention_density[i](x),true_density[i](x)), axis=0).max()
        
        interval=range(25,75)
        axins.plot(x[interval], predict_classic_LSTM_density[i](x)[interval], '-.', color='c', label='classic_LSTM', linestyle = 'dotted', linewidth = 1.5, alpha = 0.9)
        axins.plot(x[interval], predict_mLSTM_density[i](x)[interval], '--', color='r', label='mLSTM', linestyle = 'dotted', linewidth = 1.5, alpha = 0.9)
        axins.plot(x[interval], predict_LSTM_attention_density[i](x)[interval], '-.', color='b', label='LSTM_attention', linestyle = 'dotted', linewidth = 1.5, alpha = 0.9)
        axins.plot(x[interval], true_density[i](x)[interval], '-', color='limegreen', label='full model', linestyle = 'dotted', linewidth = 2.5)
        
        ax1.grid(linestyle='dotted')
        ax1.set_ylabel("Density", fontsize = 18)
        ax1.set_xlabel('x', fontsize = 18)
        ax1.legend(prop={'size': 11}, loc='upper center', bbox_to_anchor=(0.36,0.95))
        # plt.legend()
        ax1.set_xlim((var_min,var_max))
        ax1.set_ylim((0,p_max))
        axins.grid(linestyle='dotted')
        plt.title("x" if i==0 else "y")
    plt.savefig('plot/Density.png', bbox_inches='tight')
    
ifcomplex=False    
filepath=['checkpoint/classic_LSTM/checkpoint.pth.tar','checkpoint/mLSTM/checkpoint_1.pth.tar','checkpoint/LSTM_attention/checkpoint_0_20.pth.tar']    
checkpoint = torch.load(filepath[2])    
attn_len = 5    
device='cpu'   
input_size=checkpoint['input_size']
output_size=input_size
hidden_size=checkpoint['hidden_size']
classic_LSTM = load_checkpoint(filepath[0], models.classic_LSTM(input_size = input_size, hidden_size = hidden_size, output_size=output_size))
mLSTM = load_checkpoint(filepath[1], models.mLSTM(input_size = input_size, hidden_size= hidden_size, output_size=output_size))
LSTM_attention = load_checkpoint(filepath[2], models.attn_LSTM(input_size = input_size, hidden_size_encoder=hidden_size, hidden_size_decoder=hidden_size, output_size = output_size, attn_len = attn_len), device='cpu')
data = loadmat('data_p1.mat')
U = data['Uout'].transpose()
T = data['Tkout'].transpose()
v = data['vkout'].transpose()
if ifcomplex:
    T = T.real
    v = v.real
else:
    T = T.imag
    v = v.imag
data = np.concatenate((U, T, v), axis=1)

seq_length=1000
half_seq_length = data.shape[0]//2 - 1
#idx_randint = torch.randint(low = 0, high = half_seq_length-seq_length, size = (1,))
idx_randint = 5000
data_1 = torch.from_numpy(np.expand_dims(data[idx_randint:idx_randint + seq_length], axis = 1)).float()
data_2 = torch.from_numpy(np.expand_dims(data[idx_randint -attn_len + 1:idx_randint + seq_length], axis = 1)).float()
data_true = torch.from_numpy(np.expand_dims(data[idx_randint + 1: idx_randint + seq_length + 1], axis = 1)).float()

predict_classic_LSTM = classic_LSTM(data_1)
predict_mLSTM = mLSTM(data_1)
predict_LSTM_attention = LSTM_attention(data_2)
fd=(0,1,11)
predict_classic_LSTM = predict_classic_LSTM.cpu().detach().numpy().squeeze(1)[:,fd]
predict_mLSTM = predict_mLSTM.cpu().detach().numpy().squeeze(1)[:,fd]
predict_LSTM_attention = predict_LSTM_attention.cpu().detach().numpy().squeeze(1)[:,fd]
data_true = np.array(data_true, dtype='f').squeeze(1)[:,fd]


plot_trajectory(predict_classic_LSTM, predict_mLSTM, predict_LSTM_attention, data_true)
plot_ACF(predict_classic_LSTM, predict_mLSTM, predict_LSTM_attention, data_true, Tcorr=500)
plot_density(predict_classic_LSTM, predict_mLSTM, predict_LSTM_attention, data_true)

