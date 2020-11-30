'''

This script can be used to train the CaSPR model. Use:

python train.py --help

'''
import os, sys
import argparse
import math

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from models.caspr import CaSPR

from utils.torch_utils import get_device, count_params, load_encoder_weights_from_full, load_weights
from utils.train_utils import print_stats, log, TrainLossTracker, run_one_epoch
from utils.test_utils import TestStatTracker
from utils.config_utils import get_general_options, get_train_options

from data.caspr_dataset import DynamicPCLDataset

def parse_args(args):
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser = get_general_options(parser)
    parser = get_train_options(parser)

    flags = parser.parse_known_args()
    flags = flags[0]

    return flags

def train(flags):
    # General options
    num_workers = flags.num_workers

    model_out_path = flags.out

    data_cfg = flags.data_cfg
    batch_size = flags.batch_size
    seq_len = flags.seq_len
    num_pts = flags.num_pts

    augment_quad = flags.augment_quad
    augment_pairs = flags.augment_pairs

    pretrain_tnocs = flags.pretrain_tnocs

    model_in_path = flags.weights
    radii_list = flags.radii
    local_feat_size = flags.local_feat_size
    latent_feat_size = flags.latent_feat_size
    ode_hidden_size = flags.ode_hidden_size
    motion_feat_size = flags.motion_feat_size
    cnf_blocks = flags.cnf_blocks
    regress_tnocs = flags.regress_tnocs

    cnf_loss_weight = flags.cnf_loss
    tnocs_loss_weight = flags.tnocs_loss

    # Train-only options
    parallel_train = flags.use_parallel

    num_epochs = flags.epochs
    val_every = flags.val_every
    save_every = flags.save_every
    print_stats_every = flags.print_every

    lr = flags.lr
    betas = (flags.beta1, flags.beta2)
    eps = flags.eps
    weight_decay = flags.decay

    # prepare output
    if not os.path.exists(model_out_path):
        os.mkdir(model_out_path)
    log_out = os.path.join(model_out_path, 'train_log.txt')
    log(log_out, flags)

    # load train and validation sets
    train_dataset = DynamicPCLDataset(data_cfg, split='train', train_frac=0.8, val_frac=0.1,
                                num_pts=num_pts, seq_len=seq_len,
                                shift_time_to_zero=(not pretrain_tnocs),
                                random_point_sample=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers, pin_memory=True, drop_last=True,
                                worker_init_fn=lambda _: np.random.seed()) # get around numpy RNG seed bug

    val_dataset = DynamicPCLDataset(data_cfg, split='val', train_frac=0.8, val_frac=0.1,
                                num_pts=num_pts, seq_len=seq_len,
                                shift_time_to_zero=(not pretrain_tnocs),
                                random_point_sample=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers, pin_memory=True, drop_last=True,
                                    worker_init_fn=lambda _: np.random.seed())

    if parallel_train:
        log(log_out, 'Attempting to use all available GPUs for parallel training...')
    # gets GPU 0 if available, else CPU
    device = get_device()

    # create caspr model
    model = CaSPR(radii_list=radii_list,
                    local_feat_size=local_feat_size,
                    latent_feat_size=latent_feat_size,
                    ode_hidden_size=ode_hidden_size,
                    pretrain_tnocs=pretrain_tnocs,
                    augment_quad=augment_quad,
                    augment_pairs=augment_pairs,
                    cnf_blocks=cnf_blocks,
                    motion_feat_size=motion_feat_size,
                    regress_tnocs=regress_tnocs
                    )

    if pretrain_tnocs and model_in_path != '':
        # load in only pretrained tnocs weights
        print('Loading weights for pre-trained canonicalizer from %s...' % (model_in_path))
        loaded_state_dict = torch.load(model_in_path, map_location=device)
        load_encoder_weights_from_full(model, loaded_state_dict)
    elif model_in_path != '':
        print('Loading model weights from %s...' % (model_in_path))
        loaded_state_dict = torch.load(model_in_path, map_location=device)
        load_weights(model, loaded_state_dict)

    if parallel_train:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.Adam(model.parameters(),
                            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    params =  count_params(model)
    log(log_out, 'Num model params: ' + str(params))

    loss_tracker = TrainLossTracker()

    for epoch in range(num_epochs):
        # train
        run_one_epoch(model, train_loader, device, optimizer, 
                        cnf_loss_weight, tnocs_loss_weight, 
                        epoch, loss_tracker, log_out,
                        mode='train', print_stats_every=print_stats_every)

        # validate
        if epoch % val_every == 0:
            with torch.no_grad(): # must do this to avoid running out of memory
                val_stat_tracker = TestStatTracker()

                run_one_epoch(model, val_loader, device, None,
                                cnf_loss_weight, tnocs_loss_weight, 
                                epoch, val_stat_tracker, log_out,
                                mode='val', print_stats_every=print_stats_every)

                # get final aggregate stats
                mean_losses = val_stat_tracker.get_mean_stats()
                total_loss_out, mean_cnf_err, mean_tnocs_pos_err, mean_tnocs_time_err, mean_nfe = mean_losses

                # early stopping - save if it's the best so far
                if not math.isnan(total_loss_out):
                    if len(loss_tracker.val_losses) == 0:
                        min_loss_so_far = True
                    else:
                        min_loss_so_far = total_loss_out < min(loss_tracker.val_losses)

                    # record loss curve and print stats
                    loss_tracker.record_val_step(total_loss_out, epoch * len(train_loader))
                    print_stats(log_out, epoch, 0, 0, total_loss_out, mean_cnf_err, 
                                mean_tnocs_pos_err, mean_tnocs_time_err,
                                'VAL', mean_nfe)

                    if min_loss_so_far:
                        log(log_out, 'BEST Val loss so far! Saving checkpoint...')
                        save_name = 'BEST_time_model.pth'
                        save_file = os.path.join(model_out_path, save_name)
                        torch.save(model.state_dict(), save_file)

            # viz loss curve
            loss_tracker.plot_cur_loss_curves(model_out_path)

        if epoch % save_every == 0:
            # save model parameters
            save_name = 'time_model_%d.pth' % (epoch)
            save_file = os.path.join(model_out_path, save_name)
            torch.save(model.state_dict(), save_file)

def main(flags):
    train(flags)

if __name__=='__main__':
    flags = parse_args(sys.argv[1:])
    # print(flags)
    main(flags)