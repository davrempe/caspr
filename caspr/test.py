'''

This script can be used to evaluate the CaSPR model. Use:

python test.py --help

'''

import os, sys
import argparse
import math

import numpy as np

import torch
from torch.utils.data import DataLoader

from models.caspr import CaSPR

from utils.torch_utils import get_device, load_encoder_weights_from_full, load_weights
from utils.train_utils import print_stats, log, TrainLossTracker, run_one_epoch
from utils.test_utils import TestStatTracker
import utils.evaluations as eval_utils
from utils.evaluations import test_shape_recon, test_tnocs_regression, test_observed_camera_pose_ransac
from utils.config_utils import get_general_options, get_test_options

from data.caspr_dataset import DynamicPCLDataset

def parse_args(args):
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser = get_general_options(parser)
    parser = get_test_options(parser)

    flags = parser.parse_known_args()
    flags = flags[0]

    return flags

def test(flags):
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

    # Test-specific options
    test_log_out_name = flags.log

    shuffle_test = flags.shuffle_test

    eval_full_test = flags.eval_full_test
    eval_shape_recon_observed = flags.eval_shape_recon_observed
    eval_shape_recon_unobserved = flags.eval_shape_recon_unobserved
    eval_tnocs_regression = flags.eval_tnocs_regression
    eval_pose_observed_ransac = flags.eval_pose_observed_ransac
    show_pose_viz = flags.show_pose_viz

    if not os.path.exists(model_out_path):
        os.mkdir(model_out_path)

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

    model.to(device)
    
    # then run on test set
    test_dataset = DynamicPCLDataset(data_cfg, split='test', train_frac=0.8, val_frac=0.1,
                                    num_pts=num_pts, seq_len=seq_len,
                                    shift_time_to_zero=(not pretrain_tnocs),
                                    random_point_sample=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers,
                                    worker_init_fn=lambda _: np.random.seed()) # get around numpy RNG seed bug

    log_out = os.path.join(model_out_path, test_log_out_name)
    log(log_out, flags)

    # run through the full test set and calculate same metrics reported during training for comparison
    if eval_full_test:
        with torch.no_grad():
            test_stat_tracker = TestStatTracker()

            run_one_epoch(model, test_loader, device, None,
                          cnf_loss_weight, tnocs_loss_weight, 
                          0, test_stat_tracker, log_out,
                          mode='test', print_stats_every=1)

            # get final aggregate stats
            mean_losses = test_stat_tracker.get_mean_stats()
            total_loss_out, mean_cnf_err, mean_tnocs_pos_err, mean_tnocs_time_err, mean_nfe = mean_losses
            
            # print stats
            print_stats(log_out, 0, 0, 0, total_loss_out, mean_cnf_err, mean_tnocs_pos_err, 
                        mean_tnocs_time_err, 'TEST', mean_nfe)

    # other evaluations
    with torch.no_grad():
        if eval_shape_recon_observed:
            observed_steps = eval_utils.ALL_OBSERVED_STEPS
            unobserved_steps = eval_utils.ALL_UNOBSERVED_STEPS
            test_shape_recon(model, 
                             test_loader,
                             device,
                             log_out,
                             observed_steps,
                             unobserved_steps)
        if eval_shape_recon_unobserved:
            observed_steps = eval_utils.SPLIT_OBSERVED_STEPS
            unobserved_steps = eval_utils.SPLIT_UNOBSERVED_STEPS
            test_shape_recon(model,
                             test_loader,
                             device,
                             log_out,
                             observed_steps,
                             unobserved_steps)
        if eval_tnocs_regression:
            test_tnocs_regression(model,
                                  test_loader,
                                  device,
                                  log_out)
        if eval_pose_observed_ransac:
            test_observed_camera_pose_ransac(model, 
                                             test_loader,
                                             device,
                                             log_out,
                                             show=show_pose_viz)

def main(flags):
    test(flags)

if __name__=='__main__':
    flags = parse_args(sys.argv[1:])
    # print(flags)
    main(flags)