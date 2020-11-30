'''

This script can be used to visualize CaSPR model results. Use:

python viz.py --help

'''
import os, sys
import argparse

import numpy as np

import torch
from torch.utils.data import DataLoader

from models.caspr import CaSPR

from utils.torch_utils import get_device, load_encoder_weights_from_full, load_weights
from utils.viz_utils import test_viz, VizConfig
from utils.config_utils import get_general_options, get_viz_options

from data.caspr_dataset import DynamicPCLDataset

def parse_args(args):
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser = get_general_options(parser)
    parser = get_viz_options(parser)

    flags = parser.parse_known_args()
    flags = flags[0]

    return flags

def viz(flags):
    # General options
    num_workers = flags.num_workers

    data_cfg = flags.data_cfg
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

    # Viz-specific options
    shuffle_test = flags.shuffle_test

    viz_tnocs = flags.viz_tnocs
    viz_observed = flags.viz_observed
    viz_interpolated = flags.viz_interpolated

    device = get_device()

    print('Setting batch size to 1 for visualization...')
    batch_size = 1

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
    
    # visualize results on test set
    test_dataset = DynamicPCLDataset(data_cfg, split='test', train_frac=0.8, val_frac=0.1,
                                    num_pts=num_pts, seq_len=seq_len,
                                    shift_time_to_zero=(not pretrain_tnocs),
                                    random_point_sample=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers,
                                    worker_init_fn=lambda _: np.random.seed()) # get around numpy RNG seed bug

    # visualize predictions
    viz_cfg = VizConfig(flags)
    with torch.no_grad():
        test_viz(viz_cfg, model, test_dataset, test_loader, device)

def main(flags):
    train(flags)

if __name__=='__main__':
    flags = parse_args(sys.argv[1:])
    print(flags)
    viz(flags)