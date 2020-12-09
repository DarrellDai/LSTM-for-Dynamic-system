# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 16:48:28 2020

@author: Darrell Dai
"""

import torch
from torch.utils import data
import os
import numpy as np
from scipy.io import loadmat

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, dir, seq_length, ifcomplex=True):
        'Initialization'
        self.root = dir
        self.seq_length = seq_length
        self.data = loadmat('data_p1.mat')
        self.U = self.data['Uout'].transpose()
        self.T = self.data['Tkout'].transpose()
        self.v = self.data['vkout'].transpose()
        if ifcomplex:
            self.T = self.T.real
            self.v = self.v.real
        else:
            self.T = self.T.imag
            self.v = self.v.imag        
        self.list_IDs = os.listdir(dir)
        self.size=self.U.shape[1]+self.T.shape[1]+self.v.shape[1]
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # select sample
        ID = self.list_IDs[index]
        # load data and get label
        half_seq_length = self.U.shape[0]//2 - 1
#        idx_randint = torch.randint(low = 0, high = half_seq_length-self.seq_length, size = (1,))
        idx_randint = 5000
        U_t = self.U[idx_randint:idx_randint + self.seq_length]
        T_t = self.T[idx_randint:idx_randint + self.seq_length]
        v_t = self.v[idx_randint:idx_randint + self.seq_length]
        U_tt = self.U[idx_randint + 1:idx_randint + 1 + self.seq_length]
        T_tt = self.T[idx_randint + 1:idx_randint + 1 + self.seq_length]
        v_tt = self.v[idx_randint + 1:idx_randint + 1 + self.seq_length]
        return U_t, T_t, v_t, U_tt, T_tt, v_tt# index, idx_randint