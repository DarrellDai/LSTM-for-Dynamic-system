# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 14:24:50 2020

@author: Darrell Dai
"""

from __future__ import print_function

import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils import data
from dataloader_barotropic import Dataset
import LSTMs as models 

from utils import Logger, AverageMeter, accuracy, mkdir_p

import math
import random
import numpy as np

import argparse
import os


# hyper parameters for training
parser = argparse.ArgumentParser(description='hyper parameters')
parser.add_argument('--epoch', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--seq_length', default=500, type=int, metavar='N',
                    help='length of training sequence')
parser.add_argument('--iters', default=20, type=int, metavar='N',
                    help='number of iters each epoch to run')
parser.add_argument('--train-batch', default=50, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[15, 25],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--complex', '--cplx', default=0, type=int,
                    help='real or complex number')
# checkpoints setting
parser.add_argument('-c', '--checkpoint', default='checkpoint/classic_LSTM', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# # Architecture
# parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

if not os.path.isdir(args.checkpoint):
    mkdir_p(args.checkpoint)

# save config
with open(args.checkpoint + "/Config.txt", 'w') as f:
    for (k, v) in args._get_kwargs():
        f.write(k + ' : ' + str(v) + '\n')

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)

    
def train(trainloader, model, criterion, optimizer, use_cuda):

    model.train()
    batch_time = AverageMeter() # track training status
    losses = AverageMeter() # track training status
    end = time.time()

    for idx, (U_t, T_t, v_t, U_tt, T_tt, v_tt) in enumerate(trainloader):
        # U_t of size batch size * seq_length
        inputs = torch.cat((U_t, T_t, v_t), 2).transpose(0,1)   # seq_length * batch size * 2
        targets = torch.cat((U_tt, T_tt, v_tt), 2).transpose(0,1)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs.float()), torch.autograd.Variable(targets.float())

        # compute output
        outputs= model(inputs)
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        LR = optimizer.param_groups[0]['lr']
        suffix = 'Train_Loss::{loss:.4f} lr::{lr:.8f}'.format(loss = losses.avg, lr = LR)
        print(suffix)
    return losses.avg

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in state['schedule']:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

def save_checkpoint(state, is_best = False, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    # if is_best:
    #     torch.save(state, filepath)

if args.complex:
    it=2
    Epoch=["Epoch1","Epoch2"]
else: 
    it=1
    Epoch = ["Epoch"]
for i in range(it):    
    # data loader
    training_set = Dataset('data_barotropic/', seq_length=args.seq_length, ifcomplex=i)
    sampler = torch.utils.data.RandomSampler(training_set, replacement=True, num_samples=args.train_batch * args.iters)
    trainloader = data.DataLoader(training_set, sampler=sampler, batch_size=args.train_batch, num_workers=0)
    
    # model
    model = models.classic_LSTM(input_size = training_set.size, hidden_size = training_set.size*2, output_size=training_set.size)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    with open(args.checkpoint + "/Config.txt", 'w') as f:
        f.write('Total params: %.2f' % (sum(p.numel() for p in model.parameters())) + '\n')
    print('    Total params: %.2f' % (sum(p.numel() for p in model.parameters())))
    
    # loss function and optimizer
    criterion = nn.MSELoss()
    #    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # logger
    logger = Logger(os.path.join(args.checkpoint,'log.txt'), title = 'log')
    logger.set_names(['Learning Rate.', 'Train Loss.'])
    
    
    for epoch in range(args.epoch):
        adjust_learning_rate(optimizer, epoch)
        print(Epoch[i],': [%d | %d] LR: %f' % (epoch + 1, state['epoch'], state['lr']))
        train_loss = train(trainloader, model, criterion, optimizer, use_cuda = True)
    
        # append logger file
        logger.append([state['lr'], train_loss])
        save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, False, checkpoint=state['checkpoint'],filename='checkpoint_%d.pth.tar'%(i+1))
    logger.close()