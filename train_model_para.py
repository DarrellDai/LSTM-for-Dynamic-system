''' this script works on GPUs with parallel
'''
from __future__ import print_function

import time

import torch
import torch.nn as nn
from geomloss import SamplesLoss
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import random
import numpy as np
import scipy.io

import argparse
import os
import gflags
import sys
import gc

from utils import Logger, AverageMeter
from models import LSTMupdate, BaroTopo

# hyper parameters for training
parser = argparse.ArgumentParser(description='model configuration')
# data loading parameters
parser.add_argument('--train_length', default=10109, type=int, metavar='T',
                    help='sequence length for training samples')
parser.add_argument('--input_length', default=100, type=int, metavar='L',
                    help='model input state size')
parser.add_argument('--iters', default=100, type=int, metavar='I',
                    help='number of iterations for each epoch')
parser.add_argument('--train-batch', default=100, type=int, metavar='B',
                    help='each training batch size')
parser.add_argument('--nskip', default=10, type=int, metavar='nk',
                    help='time step skip in the loaded raw data, dt=1e-2')
parser.add_argument('--npred', default=10, type=int, metavar='Np',
                    help='number of iterations to measure in the loss func.')
# (train-batch * iters = train_length - input_length - npred+1)
parser.add_argument('--seq_length_pred', default=10000, type=int, metavar='T',
                    help='length of prediction sequence')
# model parameters
parser.add_argument('--epoch', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--nhid', default=50, type=int, metavar='nh',
                    help='hidden layer size in the network cell')
parser.add_argument('--nlag', default=1, type=int, metavar='nl',
                    help='lag steps included in the prediction of next state')
parser.add_argument('--nloss', default=50, type=int, metavar='nv',
                    help='number of steps to measure in the loss function')
# optimization parameters
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[50, 80],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.5, 
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--loss-type', '--lt', default='w2', type=str, metavar='LT',
                    help='choices of loss functions (kld, w2, mse, mixed, obsd)')

# checkpoints/data setting
parser.add_argument('-c', '--checkpoint', default='checkpoint/BARO2_h10sU10', type=str, metavar='C_PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--data-file', default='data/baro2_h10sU10_dk1dU1', type=str, metavar='DATA_PATH',
                    help='path to data set (default: baro2)')
parser.add_argument('--resume', default=False, type=bool, metavar='R_PATH',
                    help='path to latest checkpoint (default: none)')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')

args = parser.parse_args()
cfg = {k: v for k, v in args._get_kwargs()}

if not os.path.isdir(args.checkpoint):
    os.makedirs(args.checkpoint)
        
# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)

# save config
fname = 'ls4stg0hs{}lag{}nl{}_seq{}nsk{}np{}_epoch{}_'.format(args.nhid, args.nlag,  args.nloss,
            args.input_length, args.nskip, args.npred, args.epoch) + args.loss_type
with open(args.checkpoint + "/config_"+fname+".txt", 'w') as f:
    for (k, v) in args._get_kwargs():
        f.write(k + ' : ' + str(v) + '\n')
    f.write('\n')

def floss(out, tag, criteron):
    if len(out.size())==2:
        out=out.unsqueeze(0)
        tag=tag.unsqueeze(0)
    distances = [criteron(out[:,i,:], tag[:,i,:]) for i in range(tag.shape[1])]    
    return torch.mean(torch.stack(distances),0)
    
def main(pretrained = False, valid = False):
    # models for unresolved processes
    model1 = LSTMupdate(input_size = 5, hidden_size = args.nhid, output_size = 4, 
                              nlags = args.nlag, nlayers = 4, nstages = 0).double()
    model2 = LSTMupdate(input_size = 5, hidden_size = args.nhid, output_size = 4, 
                              nlags = args.nlag, nlayers = 4, nstages = 0).double()
    # load model on GPU
    dev1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dev2 = torch.device("cuda:1" if torch.cuda.device_count()>1 else "cuda:0")
    device = (dev1, dev2)
    gc.collect()
    print('This code is run by {} and {}: {} GPU(s)'.format(dev1, dev2, torch.cuda.device_count()))
    # if  torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model).to(device)
    cudnn.benchmark = True
    cudnn.enabled = True
    
    if pretrained:
        # load the pretrained model
        model_path1 = torch.load(os.path.join(cfg['checkpoint'], 'model1_'+fname), map_location=dev1)
        model1.load_state_dict(model_path1['model_state_dict'])
        model_path2 = torch.load(os.path.join(cfg['checkpoint'], 'model2_'+fname), map_location=dev2)
        model2.load_state_dict(model_path2['model_state_dict'])
    ires = 0
    if args.resume == True:
        model_path1 = torch.load(os.path.join(cfg['checkpoint'], 'model1_'+fname), map_location=dev1)
        model1.load_state_dict(model_path1['model_state_dict'])
        model_path2 = torch.load(os.path.join(cfg['checkpoint'], 'model2_'+fname), map_location=dev2)
        model2.load_state_dict(model_path2['model_state_dict'])
        log = np.loadtxt(os.path.join(cfg['checkpoint'], 'log_'+fname+'.txt'), skiprows=1)
        ires = int(log[-2, 0]) + 1
        args.lr = log[-2,1]
    model1.to(dev1)
    model2.to(dev2)
    model = (model1, model2)
    
    # loss function and optimizer
    if args.loss_type == 'mse' or args.loss_type == 'obsd' or args.loss_type == 'mse_s' or args.loss_type == 'mse_s1':
        criterion = nn.MSELoss(reduction='mean')
    elif args.loss_type == 'kld':
        criterion = nn.KLDivLoss(reduction='batchmean')    
    elif args.loss_type == 'w2':
        criterion = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
    elif args.loss_type == 'mixed' or args.loss_type == 'obsd_m' or args.loss_type == 'mixed_s' or args.loss_type == 'mixed_s1':
        crion1 = nn.KLDivLoss(reduction='batchmean')
        crion2 = nn.MSELoss(reduction='mean')
        criterion = (crion1, crion2)

        
    optim1 = optim.SGD(model1.parameters(), lr = args.lr, momentum = args.momentum, 
                          weight_decay = args.weight_decay)
    optim2 = optim.SGD(model2.parameters(), lr = args.lr, momentum = args.momentum, 
                          weight_decay = args.weight_decay)
    #optim1 = torch.optim.Adam(model1.parameters(), lr=args.lr, betas=(0.9,.99), weight_decay=args.weight_decay, amsgrad = True)
    #optim2 = torch.optim.Adam(model2.parameters(), lr=args.lr, betas=(0.9,.99), weight_decay=args.weight_decay, amsgrad = True)
    optimizer = (optim1, optim2)

    # logger
    logger = Logger(os.path.join(args.checkpoint, 'log_'+fname+'.txt'), title = 'log', resume=args.resume)
    if ires == 0:
        logger.set_names(['Epoch', '        Learning Rate.', 'Train Loss.', 'Accu. w1','Accu. w2','Accu. T1','Accu. T2','Accu. U'])
    
    # load dataset
    data_load = scipy.io.loadmat(args.data_file)
    # model statistics
    dt       = data_load.get('Dt')[0,0]
    autocorr_v = np.transpose(data_load.get('autocorr_v'), (1,0))
    autocorr_t = np.transpose(data_load.get('autocorr_T'), (1,0))
    # model data
    tt = np.transpose(data_load.get('tout'), (1,0))
    samp_u = np.transpose(data_load.get('Uout'), (1,0))
    samp_w = np.transpose(data_load.get('wout'), (1,0))
    samp_t = np.transpose(data_load.get('Trout'), (1,0))
    noise  = np.transpose(data_load.get('noise'), (1,0))
    # load data in the observed step
    nskip = args.input_length
    tt     =     tt[:nskip * args.train_length*args.nskip:args.nskip]
    samp_u = samp_u[:nskip * args.train_length*args.nskip:args.nskip, 0]
    samp_w = samp_w[:nskip * args.train_length*args.nskip:args.nskip, :]
    samp_t = samp_t[:nskip * args.train_length*args.nskip:args.nskip, :]
    noise  =  np.sum(noise[1:1 + nskip * args.train_length*args.nskip].reshape(-1,args.nskip), axis=1)
    nk = int(np.around((tt[1,0] - tt[0,0]) / dt))
    autocorr_v = abs(autocorr_v[:args.npred*nk:nk, [0,1,3]])
    autocorr_t = abs(autocorr_t[:args.npred*nk:nk, [0,2]])
    Nsamp = (args.train_length-args.input_length - args.npred+1)
    train_set  = torch.zeros(args.input_length + args.npred-1, Nsamp, 9, dtype=torch.double)
    target_set = torch.zeros(args.input_length + args.npred-1, Nsamp, 9, dtype=torch.double)
    noise_set  = torch.zeros(args.input_length + args.npred-1, Nsamp,    dtype=torch.double)
    for i in range(Nsamp):
        train_set[:, i, 0]   = torch.from_numpy(samp_u[i*nskip    :i*nskip + args.input_length + args.npred-1])
        train_set[:, i, 1:5] = torch.from_numpy(samp_w[i*nskip    :i*nskip + args.input_length + args.npred-1])
        train_set[:, i, 5:]  = torch.from_numpy(samp_t[i*nskip    :i*nskip + args.input_length + args.npred-1])
        target_set[:,i, 0]   = torch.from_numpy(samp_u[i*nskip + 1:i*nskip + args.input_length + args.npred])
        target_set[:,i, 1:5] = torch.from_numpy(samp_w[i*nskip + 1:i*nskip + args.input_length + args.npred])
        target_set[:,i, 5:]  = torch.from_numpy(samp_t[i*nskip + 1:i*nskip + args.input_length + args.npred])
        noise_set[:,i]       = torch.from_numpy( noise[i*nskip    :i*nskip + args.input_length + args.npred-1])
    train_loader = (train_set, target_set, noise_set)
    autocorr = (autocorr_v, autocorr_t)
    del(data_load,samp_u,samp_w,samp_t,noise)
    

    # training performance measure
    epoch_loss = np.zeros((args.epoch,2))
    epoch_accu = np.zeros((args.epoch,2))
    epoch_accv1 = np.zeros((args.epoch,2))
    epoch_accv2 = np.zeros((args.epoch,2))
    epoch_accT1 = np.zeros((args.epoch,2))
    epoch_accT2 = np.zeros((args.epoch,2))
    for epoch in range(ires, args.epoch):
        adjust_learning_rate(optimizer, epoch)
        print('\nEpoch: [{} | {}] LR: {:.8f} loss: {}'.format(epoch + 1, cfg['epoch'], cfg['lr'], cfg['loss_type']))
        train_loss,vloss, train_acc1,vacc1, train_acc2,vacc2, train_accT1,vaccT1, train_accT2,vaccT2, \
        train_accu,vaccu = train(train_loader, autocorr, model, criterion, optimizer, device)

        # save accuracy
        epoch_loss[epoch,0]  = train_loss
        epoch_accu[epoch,0]  = train_accu
        epoch_accv1[epoch,0] = train_acc1
        epoch_accv2[epoch,0] = train_acc2
        epoch_accT1[epoch,0] = train_accT1
        epoch_accT2[epoch,0] = train_accT2
        epoch_loss[epoch,1]  = vloss
        epoch_accu[epoch,1]  = vaccu
        epoch_accv1[epoch,1] = vacc1
        epoch_accv2[epoch,1] = vacc2
        epoch_accT1[epoch,1] = vaccT1
        epoch_accT2[epoch,1] = vaccT2
        
        # append logger file
        logger.append([epoch, cfg['lr'], train_loss, train_acc1, train_acc2, train_accT1, train_accT2, train_accu])
        filepath1 = os.path.join(cfg['checkpoint'], 'model1_' + fname)
        torch.save({'model_state_dict': model1.state_dict(), 
                    'optimizer_state_dict': optim1.state_dict(),}, filepath1)
        filepath2 = os.path.join(cfg['checkpoint'], 'model2_' + fname)
        torch.save({'model_state_dict': model2.state_dict(), 
                    'optimizer_state_dict': optim2.state_dict(),}, filepath2)

        datapath = os.path.join(cfg['checkpoint'], 'train_' + fname)
        np.savez(datapath, tt = tt, epoch_loss = epoch_loss, epoch_accu = epoch_accu, epoch_accv1 = epoch_accv1,
             epoch_accT1 = epoch_accT1, epoch_accv2 = epoch_accv2, epoch_accT2 = epoch_accT2)
        
    datapath = os.path.join(cfg['checkpoint'], 'train_' + fname)
    np.savez(datapath, tt = tt, epoch_loss = epoch_loss, epoch_accu = epoch_accu, epoch_accv1 = epoch_accv1,
             epoch_accT1 = epoch_accT1, epoch_accv2 = epoch_accv2, epoch_accT2 = epoch_accT2) 
    
    # evaluating model in prediction data set
    if valid:
        # load evaluation dataset
        data_load = scipy.io.loadmat(args.data_file)
        tt = np.transpose(data_load.get('tout'), (1,0))
        samp_u = np.transpose(data_load.get('Uout'), (1,0))
        samp_w = np.transpose(data_load.get('wout'), (1,0))
        samp_t = np.transpose(data_load.get('Trout'), (1,0))
        noise  = np.transpose(data_load.get('noise'), (1,0))
        nskip = 50
        tt     = tt[:nskip * args.train_length*args.nskip:args.nskip]
        samp_u = samp_u[:nskip * args.train_length*args.nskip:args.nskip]
        samp_w = samp_w[:nskip * args.train_length*args.nskip:args.nskip]
        samp_t = samp_t[:nskip * args.train_length*args.nskip:args.nskip]
        noise  = np.sum(noise[1:1 + nskip * args.train_length*args.nskip].reshape(-1,args.nskip), axis=1)
        # samp_u = samp_u[nskip * args.train_length*args.nskip:nskip * (args.seq_length_pred+args.train_length)*args.nskip:args.nskip, :]
        # samp_w = samp_w[nskip * args.train_length*args.nskip:nskip * (args.seq_length_pred+args.train_length)*args.nskip:args.nskip, :]
        
        traj_u = torch.from_numpy(samp_u)
        traj_w = torch.from_numpy(samp_w)
        traj_t = torch.from_numpy(samp_t)
        traj_n = torch.from_numpy(noise)[:,None]
        traj_set = torch.cat([traj_u,traj_w,traj_t,traj_n], 1)
        npred = 500
        Nsamp = (args.train_length-args.input_length - (npred-1))
        init_set = torch.zeros(args.input_length, Nsamp, 9, dtype=torch.double)
        for i in range(Nsamp):
            init_set[:, i, 0]   = torch.from_numpy(samp_u[i*nskip :i*nskip + args.input_length,0])
            init_set[:, i, 1:5] = torch.from_numpy(samp_w[i*nskip :i*nskip + args.input_length])
            init_set[:, i, 5:]  = torch.from_numpy(samp_t[i*nskip :i*nskip + args.input_length])
        eval_loader = (init_set, traj_set)
        
        logger.file.write('\n')
        logger.set_names(['Model eval.', 'MSE', '        error full', 'error ome','error T'])
        valid_pred, valid_err, err_stat = prediction(eval_loader, npred,nskip, model, logger, device)
        
        datapath = os.path.join(cfg['checkpoint'], 'pred_' + fname)
        np.savez(datapath, tt = tt, pred = valid_pred[:,:,:,0], gt = valid_pred[:,:,:,1],
                      valid_err = valid_err, err_stat = err_stat)

    logger.close()
    
def prediction(eval_loader, npred,nskip, model, logger, device):
    model1, model2 = model
    dev1, dev2 = device
    with torch.no_grad():
        model1.eval()
        model2.eval()
    baroU = BaroTopo(dt = args.nskip*0.01, H0 = 10, device=dev1)
 
    init_set, traj_set = eval_loader
    Nsamp = init_set.size(1)
    rs = 10
    valid_pred = np.zeros((npred, rs, 8, 2))
    valid_err  = np.zeros((npred, 5))
    err_stat   = np.zeros((npred, 9, 2))

    target = torch.empty(args.input_length, Nsamp, 9, dtype=torch.double)
    noise  = torch.empty(args.input_length, Nsamp,    dtype=torch.double, device=dev1)
    istate1 = init_set[:,:,[0,1,2,5,6]].to(dev1)
    istate2 = init_set[:,:,[0,3,4,7,8]].to(dev2)
    inputs  = init_set[:,:,:5].to(dev1)
    hidden1, hidden2 = (), ()
    with torch.no_grad():
        for istep in range(npred):
            # target set data
            for it in range(Nsamp):
                target[:,it, :] = traj_set[it*nskip + (istep+1): it*nskip + args.input_length + (istep+1), :9]
                noise[:,it]  = traj_set[it*nskip + (istep): it*nskip + args.input_length + (istep), 9]

            # run model in one forward iteration
            output1, hidden1 = model1(istate1, hidden1, device=dev1)
            output2, hidden2 = model2(istate2, hidden2, device=dev2)
            inputs[:,:,1:3] = (inputs[:,:,1:3]+ output1[:,:,:2]) / 2
            inputs[:,:,3:]  = (inputs[:,:,3:] + output2[:,:,:2].to(dev1)) / 2
            outputU = inputs[:,:,0] + baroU.baro_euler(inputs, noise)

            pred1 = output1.data.cpu().numpy()[-1]
            pred2 = output2.data.cpu().numpy()[-1]
            targ  = target.data.cpu().numpy()[-1]
            predU = outputU.data.cpu().numpy()[-1]
            valid_pred[istep, :,:2,0]  = pred1[:rs*50:50,:2]
            valid_pred[istep, :,2:4,0] = pred2[:rs*50:50,:2]
            valid_pred[istep, :,4:6,0] = pred1[:rs*50:50,2:]
            valid_pred[istep, :,6:,0]  = pred2[:rs*50:50,2:]
            valid_pred[istep, :,:,1]   = targ[:rs*50:50,1:]
            
            valid_err[istep,0] = np.sqrt( np.square(predU   - targ[:,0]).mean())
            valid_err[istep,1] = np.sqrt( np.square(pred1[:,:2]  - targ[:,1:3]).mean(0).sum())
            valid_err[istep,2] = np.sqrt( np.square(pred2[:,:2]  - targ[:,3:5]).mean(0).sum())
            valid_err[istep,3] = np.sqrt( np.square(pred1[:,2:]  - targ[:,5:7]).mean(0).sum())
            valid_err[istep,4] = np.sqrt( np.square(pred2[:,2:]  - targ[:,7:]).mean(0).sum())
            
            err_stat[istep, [1,2,5,6],0] = np.square(pred1.mean(0) - targ[:,[1,2,5,6]].mean(0)) / targ[:,[1,2,5,6]].var(0)
            err_stat[istep, [1,2,5,6],1] = np.abs(pred1.var(0) - targ[:,[1,2,5,6]].var(0)) / targ[:,[1,2,5,6]].var(0) 
            err_stat[istep, [3,4,7,8],0] = np.square(pred2.mean(0) - targ[:,[3,4,7,8]].mean(0)) / targ[:,[3,4,7,8]].var(0)
            err_stat[istep, [3,4,7,8],1] = np.abs(pred2.var(0) - targ[:,[3,4,7,8]].var(0)) / targ[:,[3,4,7,8]].var(0) 
            err_stat[istep, 0,0] = np.square(predU.mean(0) - targ[:,0].mean(0)) / targ[:,0].var(0)
            err_stat[istep, 0,1] = np.abs(predU.var(0) - targ[:,0].var(0)) / targ[:,0].var(0)

            istate1[:-1,:,:] = istate1[1:,:,:].clone()
            istate1[-1,:,1:] = output1[-1]
            istate1[-1,:,0] = target[-1,:,0]
            istate2[:-1,:,:] = istate2[1:,:,:].clone()
            istate2[-1,:,1:] = output2[-1]
            istate2[-1,:,0] = target[-1,:,0]
            inputs[:,:,:3] = istate1[:,:,:3].clone()
            inputs[:,:,3:] = istate2[:,:,1:3].clone()

            inputs[-1,:,0]  = outputU[-1]
            istate1[-1,:,0] = outputU[-1]
            istate2[-1,:,0] = outputU[-1]
            error1 = np.square(pred1-targ[:,[1,2,5,6]]).mean()
            error2 = np.square(pred2-targ[:,[3,4,7,8]]).mean()
            print('step {}: error1 = {:.6f} error2 = {:.6f}'.format(istep, error1, error2))
            logger.append([istep, error1+error2, valid_err[istep,0], valid_err[istep,1:3].sum(), valid_err[istep,3:].sum()])
        
    return valid_pred, valid_err, err_stat
    
def train(train_loader, autocorr, model, criterion, optimizer, device):
    dev1, dev2 = device
    model1, model2 = model
    optim1, optim2 = optimizer
    model1.train()
    model2.train()
    baroU = BaroTopo(dt = args.nskip*0.01, H0 = 10, device=dev1)
    
    batch_time = AverageMeter()
    losses     = AverageMeter()
    accsv1     = AverageMeter()
    accsv2     = AverageMeter()
    accsT1     = AverageMeter()
    accsT2     = AverageMeter()
    accsu      = AverageMeter()
    end = time.time()
    
    input_full, target_full, noise_full = train_loader
    autocorr_v, autocorr_t = autocorr
    dsize = args.train_batch*args.iters
    s_idx = random.sample(range(0,input_full.size(1)), dsize)
    input_iter  = input_full[:, s_idx,:].pin_memory()
    target_iter = target_full[:,s_idx,:].pin_memory()
    noise_iter  = noise_full[:, s_idx].pin_memory()
    
    for ib in range(0, args.iters):
        inputs1  = input_iter[:, ib*args.train_batch:(ib+1)*args.train_batch,[0,1,2,5,6]].to(dev1, non_blocking=True)
        inputs2  = input_iter[:, ib*args.train_batch:(ib+1)*args.train_batch,[0,3,4,7,8]].to(dev2, non_blocking=True)
        targets1 = target_iter[:,ib*args.train_batch:(ib+1)*args.train_batch,[1,2,5,6]].to(dev1, non_blocking=True)
        targets2 = target_iter[:,ib*args.train_batch:(ib+1)*args.train_batch,[3,4,7,8]].to(dev2, non_blocking=True)
        noise    =  noise_iter[:,ib*args.train_batch:(ib+1)*args.train_batch].to(dev1, non_blocking=True)
        inputs   = input_iter[:, ib*args.train_batch:(ib+1)*args.train_batch, :5].to(dev1, non_blocking=True)
        targets  = target_iter[:,ib*args.train_batch:(ib+1)*args.train_batch,0].to(dev1, non_blocking=True)
        
        optim1.zero_grad()
        optim2.zero_grad()  # zero the gradient buffers
        # iteration the model in npred steps
        hidden1, hidden2 = (), ()
        istate1 = inputs1[:args.input_length]
        istate2 = inputs2[:args.input_length]
        inputs  =  inputs[:args.input_length]
        
        pred1 = torch.empty(args.input_length+args.npred, args.train_batch, 4, dtype=torch.double, device=dev1)
        pred2 = torch.empty(args.input_length+args.npred, args.train_batch, 4, dtype=torch.double, device=dev2)
        predU = torch.empty(args.input_length+args.npred, args.train_batch,    dtype=torch.double, device=dev1)
        pred1[:args.input_length] = inputs1[:args.input_length,:,1:]
        pred2[:args.input_length] = inputs2[:args.input_length,:,1:]
        predU[:args.input_length] =  inputs[:args.input_length,:,0]
        loss1, loss2, lossU = 0, 0, 0
        for ip in range(args.npred):
            output1, hidden1 = model1(istate1, hidden1, device=dev1)
            output2, hidden2 = model2(istate2, hidden2, device=dev2)
            inputs[:,:,1:3] = (inputs[:,:,1:3]+ output1[:,:,:2]) / 2
            inputs[:,:,3:]  = (inputs[:,:,3:] + output2[:,:,:2].to(dev1)) / 2
            output = inputs[:,:,0] + baroU.baro_euler(inputs, noise[ip:ip+args.input_length])
            
            pred1[args.input_length+ip] = output1[-1]
            pred2[args.input_length+ip] = output2[-1]
            predU[args.input_length+ip] = output[-1]
            
            if ip < args.npred-1:
                istate1 = torch.empty_like(istate1)
                istate2 = torch.empty_like(istate2)
                inputs  = torch.empty_like(inputs)
                # update with final model output
                #istate1[:,:,0]  = predU[ip+1:args.input_length+ip+1]
                #istate2[:,:,0]  = predU[ip+1:args.input_length+ip+1]
                #istate1[:,:,1:] = pred1[ip+1:args.input_length+ip+1]
                #istate2[:,:,1:] = pred2[ip+1:args.input_length+ip+1]
                #inputs[:,:,0]   = predU[ip+1:args.input_length+ip+1]
                #inputs[:,:,1:3] = pred1[ip+1:args.input_length+ip+1]
                #inputs[:,:,3:]  = pred2[ip+1:args.input_length+ip+1]
                # update with full model output
                istate1[:,:,1:] = output1.clone()
                istate2[:,:,1:] = output2.clone()
                inputs[:,:,1:3] = output1[:,:,:2].clone()
                inputs[:,:,3:]  = output2[:,:,:2].clone() 
                istate1[:,:,0] = inputs1[ip+1:args.input_length+ip+1,:,0]
                istate2[:,:,0] = inputs1[ip+1:args.input_length+ip+1,:,0]
                inputs[:,:,0]  = inputs1[ip+1:args.input_length+ip+1,:,0]
            output1 = torch.transpose(output1, 0,1)
            output2 = torch.transpose(output2, 0,1)
            output  = torch.transpose(output, 0,1)
            target1 = torch.transpose(targets1[ip:args.input_length+ip], 0,1)
            target2 = torch.transpose(targets2[ip:args.input_length+ip], 0,1)
            target  = torch.transpose(targets[ip:args.input_length+ip], 0,1)
            if args.loss_type == 'mse' or args.loss_type == 'mse_s' or args.loss_type == 'mse_s1':
                out = output1[:, -args.nloss:,:2]
                tag = target1[:, -args.nloss:,:2]
                loss1 += autocorr_v[ip,1] * criterion(out, tag)
                out = output1[:, -args.nloss:,2:]
                tag = target1[:, -args.nloss:,2:]
                loss1 += autocorr_t[ip,0] * criterion(out, tag)
                
                out = output2[:, -args.nloss:,:2]
                tag = target2[:, -args.nloss:,:2]
                loss2 += autocorr_v[ip,2] * criterion(out, tag)
                out = output2[:, -args.nloss:,2:]
                tag = target2[:, -args.nloss:,2:]
                loss2 += autocorr_t[ip,1] * criterion(out, tag)
                
                out = output[:, -args.nloss:]
                tag = target[:, -args.nloss:]
                lossU += autocorr_v[ip,0] * criterion(out, tag)
            elif args.loss_type == 'kld':
                out = output1[:, -args.nloss:,:2]
                tag = target1[:, -args.nloss:,:2]
                out_p = F.log_softmax(1. * out, dim=1)
                tag_p = F.softmax(1. * tag, dim=1)
                out_n = F.log_softmax(-1. * out, dim=1)
                tag_n = F.softmax(-1. * tag, dim=1)
                loss1 += autocorr_v[ip,1] * (criterion(out_p, tag_p) + 1. * criterion(out_n, tag_n))
                out = output1[:, -args.nloss:,2:]
                tag = target1[:, -args.nloss:,2:]
                out_p = F.log_softmax(1. * out, dim=1)
                tag_p = F.softmax(1. * tag, dim=1)
                out_n = F.log_softmax(-1. * out, dim=1)
                tag_n = F.softmax(-1. * tag, dim=1)
                loss1 += autocorr_t[ip,0] * (criterion(out_p, tag_p) + 1. * criterion(out_n, tag_n))
                
                out = output2[:, -args.nloss:,:2]
                tag = target2[:, -args.nloss:,:2]
                out_p = F.log_softmax(1. * out, dim=1)
                tag_p = F.softmax(1. * tag, dim=1)
                out_n = F.log_softmax(-1. * out, dim=1)
                tag_n = F.softmax(-1. * tag, dim=1)
                loss2 += autocorr_v[ip,2] * (criterion(out_p, tag_p) + 1. * criterion(out_n, tag_n))
                out = output2[:, -args.nloss:,2:]
                tag = target2[:, -args.nloss:,2:]
                out_p = F.log_softmax(1. * out, dim=1)
                tag_p = F.softmax(1. * tag, dim=1)
                out_n = F.log_softmax(-1. * out, dim=1)
                tag_n = F.softmax(-1. * tag, dim=1)
                loss2 += autocorr_t[ip,1] * (criterion(out_p, tag_p) + 1. * criterion(out_n, tag_n))
                
                out = output[:, -args.nloss:]
                tag = target[:, -args.nloss:]
                out_p = F.log_softmax(1. * out, dim=1)
                tag_p = F.softmax(1. * tag, dim=1)
                out_n = F.log_softmax(-1. * out, dim=1)
                tag_n = F.softmax(-1. * tag, dim=1)
                lossU += autocorr_v[ip,0] * (criterion(out_p, tag_p) + 1. * criterion(out_n, tag_n))
            elif args.loss_type == "w2":
                out = output1[:, -args.nloss:,:2]
                tag = target1[:, -args.nloss:,:2]
                out_p = F.log_softmax(1. * out, dim=1)
                tag_p = F.softmax(1. * tag, dim=1)
                out_n = F.log_softmax(-1. * out, dim=1)
                tag_n = F.softmax(-1. * tag, dim=1)
                loss1 += autocorr_v[ip,1] * (floss(out_p, tag_p, criterion) + 1. * floss(out_n, tag_n, criterion))
                out = output1[:, -args.nloss:,2:]
                tag = target1[:, -args.nloss:,2:]
                out_p = F.log_softmax(1. * out, dim=1)
                tag_p = F.softmax(1. * tag, dim=1)
                out_n = F.log_softmax(-1. * out, dim=1)
                tag_n = F.softmax(-1. * tag, dim=1)
                loss1 += autocorr_t[ip,0] * (floss(out_p, tag_p, criterion) + 1. * floss(out_n, tag_n, criterion))
                
                out = output2[:, -args.nloss:,:2]
                tag = target2[:, -args.nloss:,:2]
                out_p = F.log_softmax(1. * out, dim=1)
                tag_p = F.softmax(1. * tag, dim=1)
                out_n = F.log_softmax(-1. * out, dim=1)
                tag_n = F.softmax(-1. * tag, dim=1)
                loss2 += autocorr_v[ip,2] * (floss(out_p, tag_p, criterion) + 1. * floss(out_n, tag_n, criterion))
                out = output2[:, -args.nloss:,2:]
                tag = target2[:, -args.nloss:,2:]
                out_p = F.log_softmax(1. * out, dim=1)
                tag_p = F.softmax(1. * tag, dim=1)
                out_n = F.log_softmax(-1. * out, dim=1)
                tag_n = F.softmax(-1. * tag, dim=1)
                loss2 += autocorr_t[ip,1] * (floss(out_p, tag_p, criterion) + 1. * floss(out_n, tag_n, criterion))
                
                out = output[:, -args.nloss:]
                tag = target[:, -args.nloss:]
                out_p = F.log_softmax(1. * out, dim=1)
                tag_p = F.softmax(1. * tag, dim=1)
                out_n = F.log_softmax(-1. * out, dim=1)
                tag_n = F.softmax(-1. * tag, dim=1)
                lossU += autocorr_v[ip,0] * (floss(out_p, tag_p, criterion) + 1. * floss(out_n, tag_n, criterion))
            elif args.loss_type == 'mixed' or args.loss_type == 'mixed_s' or args.loss_type == 'mixed_s1':
                crion1, crion2 = criterion
                
                out = output1[:, -args.nloss:, :2]
                tag = target1[:, -args.nloss:, :2]
                out_p = F.log_softmax(1. * out, dim=1)
                tag_p = F.softmax(1. * tag, dim=1)
                out_n = F.log_softmax(-1. * out, dim=1)
                tag_n = F.softmax(-1. * tag, dim=1)
                loss1 += autocorr_v[ip,1] * (crion1(out_p,tag_p)+1.*crion1(out_n,tag_n) + .1*crion2(out,tag))
                out = output1[:, -args.nloss:, 2:]
                tag = target1[:, -args.nloss:, 2:]
                out_p = F.log_softmax(1. * out, dim=1)
                tag_p = F.softmax(1. * tag, dim=1)
                out_n = F.log_softmax(-1. * out, dim=1)
                tag_n = F.softmax(-1. * tag, dim=1)
                loss1 += autocorr_t[ip,0] * (crion1(out_p,tag_p)+1.*crion1(out_n,tag_n) + .1*crion2(out,tag))
                
                out = output2[:, -args.nloss:, :2]
                tag = target2[:, -args.nloss:, :2]
                out_p = F.log_softmax(1. * out, dim=1)
                tag_p = F.softmax(1. * tag, dim=1)
                out_n = F.log_softmax(-1. * out, dim=1)
                tag_n = F.softmax(-1. * tag, dim=1)
                loss2 += autocorr_v[ip,2] * (crion1(out_p,tag_p)+1.*crion1(out_n,tag_n) + .1*crion2(out,tag))
                out = output2[:, -args.nloss:, 2:]
                tag = target2[:, -args.nloss:, 2:]
                out_p = F.log_softmax(1. * out, dim=1)
                tag_p = F.softmax(1. * tag, dim=1)
                out_n = F.log_softmax(-1. * out, dim=1)
                tag_n = F.softmax(-1. * tag, dim=1)
                loss2 += autocorr_t[ip,1] * (crion1(out_p,tag_p)+1.*crion1(out_n,tag_n) + .1*crion2(out,tag))
                
                out = output[:, -args.nloss:]
                tag = target[:, -args.nloss:]
                out_p = F.log_softmax(1. * out, dim=1)
                tag_p = F.softmax(1. * tag, dim=1)
                out_n = F.log_softmax(-1. * out, dim=1)
                tag_n = F.softmax(-1. * tag, dim=1)
                lossU += autocorr_v[ip,0] * (crion1(out_p,tag_p)+1.*crion1(out_n,tag_n) + .1*crion2(out,tag))
            elif args.loss_type == 'obsd':
                out = output[:, -args.nloss:]
                tag = target[:, -args.nloss:]
                lossU += autocorr_v[ip,0] * criterion(out, tag)
            elif args.loss_type == 'obsd_m':
                crion1, crion2 = criterion
                
                out = output1[:, -args.nloss:, 2:]
                tag = target1[:, -args.nloss:, 2:]
                out_p = F.log_softmax(1. * out, dim=1)
                tag_p = F.softmax(1. * tag, dim=1)
                out_n = F.log_softmax(-1. * out, dim=1)
                tag_n = F.softmax(-1. * tag, dim=1)
                loss1 += autocorr_t[ip,0] * (crion1(out_p,tag_p)+1.*crion1(out_n,tag_n) + .1*crion2(out,tag))
                
                out = output2[:, -args.nloss:, 2:]
                tag = target2[:, -args.nloss:, 2:]
                out_p = F.log_softmax(1. * out, dim=1)
                tag_p = F.softmax(1. * tag, dim=1)
                out_n = F.log_softmax(-1. * out, dim=1)
                tag_n = F.softmax(-1. * tag, dim=1)
                loss2 += autocorr_t[ip,1] * (crion1(out_p,tag_p)+1.*crion1(out_n,tag_n) + .1*crion2(out,tag))
                
                out = output[:, -args.nloss:]
                tag = target[:, -args.nloss:]
                out_p = F.log_softmax(1. * out, dim=1)
                tag_p = F.softmax(1. * tag, dim=1)
                out_n = F.log_softmax(-1. * out, dim=1)
                tag_n = F.softmax(-1. * tag, dim=1)
                lossU += autocorr_v[ip,0] * (crion1(out_p,tag_p)+1.*crion1(out_n,tag_n) + .1*crion2(out,tag))
            elif args.loss_type == 'obsd_v':
                crion1, crion2 = criterion
                
                out = output[:, -args.nloss:]
                tag = target[:, -args.nloss:]
                out_p = F.log_softmax(1. * out, dim=1)
                tag_p = F.softmax(1. * tag, dim=1)
                out_n = F.log_softmax(-1. * out, dim=1)
                tag_n = F.softmax(-1. * tag, dim=1)
                lossU += autocorr_v[ip,0] * (crion1(out_p,tag_p)+1.*crion1(out_n,tag_n) + .1*crion2(out,tag))
        
        if args.loss_type == 'obsd_v':
            lossU.backward()
            loss = lossU
        elif args.loss_type == 'mixed_s' or args.loss_type == 'mse_s' or args.loss_type == 'kld':
            loss1.backward(retain_graph=True)
            loss2.backward()
            loss = loss1 + loss2
        elif args.loss_type == 'mixed_s1' or args.loss_type == 'mse_s1':
            lossU.backward(retain_graph=True)
            loss1.backward()
            loss = lossU + loss1
        else:
            loss1.backward(retain_graph=True)
            loss2.backward(retain_graph=True)
            lossU.backward()
            loss = lossU + loss1 + loss2
        optim1.step()
        optim2.step()
        
        # get trained output
        losses.update(loss.item() )
        pred_out1 = pred1[args.input_length:]
        pred_out2 = pred2[args.input_length:]
        pred_outU = predU[args.input_length:]
        gt_out1    = targets1[args.input_length-1:]
        gt_out2    = targets2[args.input_length-1:]
        gt_outU    = targets[args.input_length-1:]
        accsv1.update( ((pred_out1[:,:,:2]-gt_out1[:,:,:2]).square().sum() / gt_out1[:,:,:2].square().sum()).item() )
        accsv2.update( ((pred_out2[:,:,:2]-gt_out2[:,:,:2]).square().sum() / gt_out2[:,:,:2].square().sum()).item() )
        accsu.update( ((pred_outU-gt_outU).square().sum() / gt_outU.square().sum()).item() )
        accsT1.update( ((pred_out1[:,:,2:]-gt_out1[:,:,2:]).square().sum() / gt_out1[:,:,2:].square().sum()).item() )
        accsT2.update( ((pred_out2[:,:,2:]-gt_out2[:,:,2:]).square().sum() / gt_out2[:,:,2:].square().sum()).item() )
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        torch.cuda.empty_cache()
        
        #LR = optimizer.param_groups[0]['lr']
        suffix = 'Iter{iter}: Loss = {loss:.5e} v1 = {acc1:.5f} v2 = {acc2:.5f} T1 = {accT1:.5f} T2 = {accT2:.5f} acc_U = {accU:.5e} run_time = {bt:.2f}'.format(
                  iter = ib, loss = losses.val, acc1=accsv1.val,acc2=accsv2.val, accT1=accsT1.val,accT2=accsT2.val, accU = accsu.val, bt = batch_time.sum)
        if ib % 20 == 0:
            print(suffix)
            
    # get trained output
    # pred = torch.cat([output1, output2.to(dev1)], 2).data.cpu().numpy()[:10,:,:]
    # gt   = torch.cat([target1, target2.to(dev1)], 2).data.cpu().numpy()[:10,:,:]

    return losses.avg,losses.var, accsv1.avg,accsv1.var, accsv2.avg,accsv2.var, accsT1.avg,accsT1.var, accsT2.avg,accsT2.var, accsu.avg,accsu.var  #, pred, gt
    
def adjust_learning_rate(optimizer, epoch):
    global cfg
    if epoch in cfg['schedule']:
        cfg['lr'] *= args.gamma
        for ioptim in optimizer:
            for param_group in ioptim.param_groups:
                param_group['lr'] = cfg['lr']


if __name__ == '__main__':
    for name in list(gflags.FLAGS):
        delattr(gflags.FLAGS, name)
    gflags.FLAGS(sys.argv)
    gflags.DEFINE_boolean('pretrained', False, 'Use pretrained model')
    gflags.DEFINE_boolean('eval', True, 'Run tests with the network')
    
    main(pretrained = gflags.FLAGS.pretrained, valid = gflags.FLAGS.eval)
