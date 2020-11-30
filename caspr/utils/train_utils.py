import os, sys, time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch

from .torch_utils import torch_to_numpy

def plot_train_stats(train_losses, cnf_losses, tnocs_losses,
                    train_steps, val_losses, val_steps, out_dir):
    # plot training and validation curves
    fig = plt.figure(figsize=(12, 8))

    plt.plot(np.array(train_steps), np.array(train_losses), '-', label='train loss')
    plt.plot(np.array(train_steps), np.array(cnf_losses), '--', label='NLL loss')
    plt.plot(np.array(train_steps), np.array(tnocs_losses), '--', label='TNOCS loss')

    plt.plot(np.array(val_steps), np.array(val_losses), '-', label='val loss')

    plt.xlabel('optim steps')

    plt.legend()
    plt.title('Training Curves')
    plt.savefig(os.path.join(out_dir, 'train_curve.png'))
    plt.close(fig)

def log(log_out, write_str):
    with open(log_out, 'a') as f:
        f.write(str(write_str) + '\n')
    print(write_str)

def print_stats(log_out, epoch, cur_batch, num_batches, total_loss, 
                cnf_err, tnocs_pos_err, tnocs_time_err,
                type_id='TRAIN', nfe=None):
    log(log_out, '[Epoch %d: Batch %d/%d] %s Mean loss: %f' % (epoch, cur_batch, num_batches, type_id, total_loss))
    log(log_out, '                    %s Mean CNF NLL: %f' % (type_id, cnf_err))
    log(log_out, '                    %s Mean TNOCS Pos (m): %f, Mean TNOCS time: %f' % (type_id, tnocs_pos_err, tnocs_time_err))

    if nfe is not None:
        log(log_out, '                    %s Mean NFE (latent-ode, decoder): (%f, %f)' % (type_id, nfe[0], nfe[1]))

class TrainLossTracker():
    '''
    Tracks training and validation set losses during training.

    Keeps arrays of losses throughout training and plots when desired.
    '''
    def __init__(self):
        # total weighted loss actually used to optimize the net
        self.train_losses = []
        # optimization step at which the loss was recorded
        self.train_steps = []

        self.cnf_losses = []
        self.tnocs_losses = []

        # total weighted loss acutally used to optimize
        self.val_losses = []
        # optimization step at which loss was recorded
        self.val_steps = []

    def record_train_step(self, train_loss, cnf_loss, tnocs_loss, step_idx):
        self.train_losses.append(train_loss)
        self.cnf_losses.append(cnf_loss)
        self.tnocs_losses.append(tnocs_loss)

        self.train_steps.append(step_idx)

    def record_val_step(self, val_loss, step_idx):
        self.val_losses.append(val_loss)
        self.val_steps.append(step_idx)

    def plot_cur_loss_curves(self, out_dir):
        plot_train_stats(self.train_losses, self.cnf_losses, self.tnocs_losses,
                        self.train_steps, self.val_losses, 
                         self.val_steps, out_dir)

def run_one_epoch(model, 
                 data_loader,
                 device, 
                 optimizer,
                 cnf_loss_weight,
                 tnocs_loss_weight,
                 epoch,
                 loss_tracker,
                 log_out,
                 mode='train',
                 print_stats_every=10):
    '''
    Runs through the given dataset once to train or test the model depending on the mode given.
    '''
    if mode not in ['train', 'val', 'test']:
        print('Most must be train or test!')
        exit()
    
    is_parallel = isinstance(model, torch.nn.DataParallel)

    if mode == 'train':
        model = model.train()
        batch_losses = []
    else:
        model = model.eval()

    for i, data in enumerate(data_loader):
        pcl_in, nocs_out = data[0] # world point cloud, corresponding nocs point cloud
        pcl_in = pcl_in.to(device) # B x T x N x 4 (x,y,z,t)
        nocs_out = nocs_out.to(device) # B x T x N x 4 (x,y,z,t)

        # print(pcl_in.size())
        B, T, N, _ = nocs_out.size()

        if is_parallel and B % 2 == 1:
            # skip this batch otherwise screws up splitting across GPUs
            continue

        if mode == 'train':
            # zero the gradients
            optimizer.zero_grad()

        # forward
        losses = model(pcl_in, nocs_out)

        #
        # Compute aggregate loss
        #
        per_point_nll = per_step_nll = None
        per_point_tnocs = None
        if len(losses) == 1:
            # just pretraining tnocs
            per_point_tnocs = losses[0]
        elif len(losses) == 2:
            per_point_nll, per_point_tnocs = losses
        else:
            print('unexpected number of losses returned')
            exit()

        # get number of function evals
        if per_point_nll is not None:
            if is_parallel:
                cur_nfe = model.module.get_nfe()
            else:
                cur_nfe = model.get_nfe()
        else:
            cur_nfe = 0.0

        # loss calculation
        loss = torch.zeros(1).to(device)
        if per_point_nll is not None:
            per_step_nll = per_point_nll.sum(2)
            cnf_loss = cnf_loss_weight * per_step_nll.mean()
            # print(cnf_loss)
            loss += cnf_loss
        else:
            cnf_loss = torch.zeros(1)
            per_point_nll = torch.zeros(B, T, N)
            
        if per_point_tnocs is not None:
            # we're using tnocs regression loss
            tnocs_loss = tnocs_loss_weight * per_point_tnocs[:,:,:,:4].mean()

            loss += tnocs_loss
            # print(tnocs_loss)
        else:
            tnocs_loss = torch.zeros(1)
            per_point_tnocs = torch.zeros(B, T, N, 4)

        if mode == 'train':
            # backward + optimize
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.to('cpu').item())

            # log stats and print update
            if i % print_stats_every == 0:
                cur_mean_loss = np.mean(np.array(batch_losses))

                cur_mean_cnf_loss = cnf_loss.to('cpu').item()
                cur_mean_tnocs_loss = tnocs_loss.to('cpu').item()

                loss_tracker.record_train_step(cur_mean_loss, cur_mean_cnf_loss, cur_mean_tnocs_loss,
                                                epoch * len(data_loader) + i)

                cur_cnf_error = torch_to_numpy([per_point_nll])[0]
                cur_cnf_error = np.mean(cur_cnf_error)

                cur_tnocs_error = torch_to_numpy([per_point_tnocs])[0]
                cur_tnocs_spatial_error = cur_tnocs_error[:,:,:,:3].reshape((-1, 3))
                cur_tnocs_spatial_error = np.linalg.norm(cur_tnocs_spatial_error, axis=1)
                cur_tnocs_spatial_error = np.mean(cur_tnocs_spatial_error)

                if cur_tnocs_error.shape[3] > 3:
                    cur_tnocs_time_error = cur_tnocs_error[:,:,:,3].reshape((-1))
                    cur_tnocs_time_error = np.mean(cur_tnocs_time_error)
                else:
                    cur_tnocs_time_error = 0.0


                print_stats(log_out, epoch, i, len(data_loader), 
                            cur_mean_loss, cur_cnf_error, cur_tnocs_spatial_error, 
                            cur_tnocs_time_error,
                            'TRAIN')

                batch_losses = []
        else:
            # log stats and print update
            loss_scalar = loss.to('cpu').item() 
            
            cnf_err, tnocs_err = torch_to_numpy([per_point_nll, per_point_tnocs])

            B, T, N, _ = tnocs_err.shape
            tnocs_pos_err = np.linalg.norm(tnocs_err[:,:,:,:3].reshape((-1, 3)), axis=1)
            if tnocs_err.shape[3] > 3:
                tnocs_time_err = tnocs_err[:,:,:,3].reshape((-1))
            else:
                tnocs_time_err = np.zeros((B*T*N))

            loss_tracker.record_stats(loss_scalar, cnf_err, tnocs_pos_err, tnocs_time_err, cur_nfe)

            if i % print_stats_every == 0:
                print('%s batch %d/%d...' % (mode, i, len(data_loader)))

                mean_losses = loss_tracker.get_mean_stats()
                total_loss_out, mean_cnf_err, mean_tnocs_pos_err, mean_tnocs_time_err, mean_nfe = mean_losses
                
                # print stats
                print_stats(log_out, epoch, i, len(data_loader), total_loss_out, mean_cnf_err, mean_tnocs_pos_err, 
                            mean_tnocs_time_err, mode, mean_nfe)

    torch.cuda.empty_cache()