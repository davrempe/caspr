import sys, time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist

import numpy as np

sys.path.append('.')
sys.path.append('..')

from .tpointnet2 import TPointNet2
from .latent_ode_model import LatentODE
from .flow import get_point_cnf, count_nfe, PointCNFArgs

from .utils import standard_normal_logprob, sample_gaussian

from utils.transform_utils import sphere_surface_points


class CaSPR(nn.Module):
    def __init__(self, radii_list=[0.02, 0.05, 0.1, 0.2, 0.4, 0.8], # radii to use for pointnet++
                       local_feat_size=512, # size of per-point features from PointNet++
                       latent_feat_size=1600, # size of feature from TPointNet++ intermediate features (dynamic + static feature size)
                       ode_hidden_size=512, # size of dynamics net hidden state for latent ode
                       motion_feat_size=64, # size of the latent feature to go through ODE (dynamic feature)
                       pretrain_tnocs=False, # If true, forward pass and loss will only compute TNOCS regression
                       augment_quad=True, # If true, augments the raw point cloud input with quadratic terms
                       augment_pairs=True, # if true, augments raw point cloud input with pairwise mult terms
                       cnf_blocks=1, # number of normalizing flow blocks to use
                       regress_tnocs=True, # if false, does not regress tnocs or use tnocs loss
                       ):
        '''
        Main CaSPR architecture.
        '''
        super(CaSPR, self).__init__()

        self.pretrain_tnocs = pretrain_tnocs
        self.augment_quad = augment_quad
        self.augment_pairs = augment_pairs
        self.motion_feat_size = motion_feat_size
        self.regress_tnocs = regress_tnocs

        self.tnocs_point_size = 4 # by default x,y,z,t
        # Encoder and T-NOCS regression
        self.encoder = TPointNet2(radii_list, 
                                    local_feat_size=local_feat_size, 
                                    out_feat_size=latent_feat_size,
                                    augment_quad=self.augment_quad,
                                    augment_pairs=self.augment_pairs,
                                    tnocs_point_size=self.tnocs_point_size,
                                    regress_tnocs=self.regress_tnocs)

        if self.pretrain_tnocs:
            # do not need below here if we're pretraining
            return

        # Dynamic feature advection
        ode_input_size = self.motion_feat_size
        self.latent_ode = LatentODE(input_size=ode_input_size,
                                    hidden_size=ode_hidden_size,
                                    num_layers=2, # num hidden layers
                                    nonlinearity=nn.Tanh)

        # CNF Decoder
        self.cnf_args = PointCNFArgs()
        self.cnf_args.zdim = latent_feat_size
        self.cnf_args.num_blocks = cnf_blocks
        self.point_cnf = get_point_cnf(self.cnf_args)

        # print(self.encoder)
        # print(self.latent_ode)
        # print(self.point_cnf)

    def forward(self, x, sample_points, aggregate_points=None):
        '''
        x               : B x T x N x 4 space-time point cloud with timestamps
        sample_points   : B x T x N x 4 sampled points on object surface in T-NOCS
                                NOTE: these must correspond exactly to the input points if
                                        using TNOCs regression supervision!
        '''
        # get initial state of latent trajectory
        # z0 is B x H
        z0, tnocs_pred = self.encode(x)

        B, H = z0.size()
        _, T, N, _ = sample_points.size()

        # get loss if regressing TNOCS before the latent ODE
        tnocs_loss = None
        if self.regress_tnocs:
            tnocs_loss = self.encoder.loss(
                                    tnocs_pred[:,:,:,:self.tnocs_point_size],
                                    sample_points[:,:,:,:self.tnocs_point_size]) # only want loss on x,y,z,t

        if self.pretrain_tnocs:
            # return nocs loss only
            return tuple([tnocs_loss])

        ode_feat_dim = self.cnf_args.zdim

        # collect times we need to solve for
        # NOTE: this assumes all batches have the same init time t0
        #       i.e all z0 are for the same timestamp
        all_times = sample_points[:,:,0,3] # B x T
        # solve the ODE forward in time
        sample_feats = self.aggregate_and_solve_latent(z0, all_times) # B x T x H
        sample_feats = sample_feats.view(B*T, ode_feat_dim)
        z = sample_feats

        # now sample with CNF
        sample_points = sample_points.view((B*T, N, 4))[:,:,:3].clone() # don't need timestamps
        init_logprob = torch.zeros(B*T, N, 1).to(sample_points)
        # run flow
        cnf_result = self.point_cnf(sample_points, z, init_logprob)
        # get loss
        recon_loss = self.get_nll_loss(cnf_result, B, T)

        return_list = [recon_loss, tnocs_loss]

        return tuple(return_list)

    def get_nll_loss(self, cnf_result_list, B, T):
        '''
        Compute negative log-likelihood loss for normalizing flow.
        '''
        batch_size = B*T        
        y, delta_log_py = cnf_result_list
        cloud_dim = y.size()[1]

        # likelihood under base distribution
        log_py = standard_normal_logprob(y)
        log_py = log_py.sum(2)
        # change in volume term
        delta_log_py = delta_log_py.view(batch_size, cloud_dim)
        # log likelihood
        log_px = log_py - delta_log_py

        # Loss
        recon_loss = -log_px # negative log likelihood

        batch_dim = B
        recon_loss = recon_loss.view((batch_dim, T, -1))

        return recon_loss

    def encode(self, x):
        '''
        x               : B x T x N x 4 space-time point cloud with timestamps
        '''
        # get initial trajectory state from encoder
        # z0 is B x H
        z0, tnocs_pred = self.encoder(x)
        return z0, tnocs_pred

    def aggregate_and_solve_latent(self, z0, time_tensor):
        '''
        Given a time tensor B x T containing all the timestamps we want to
        solve for (they may not be unique). Finds all unique times, solves
        the ODE forward in time, and maps back to the original time step to
        return latent z's of size B x T x H
        '''
        B, T = time_tensor.size()
        # get unique times
        solve_t, time_map = torch.unique(time_tensor, sorted=True, return_inverse=True)

        # factorize input feature into static and dynamic feature 
        z_init = z0[:,:self.latent_ode.input_size]      # dynamic
        z_global = z0[:,self.latent_ode.input_size:]    # static

        # solve ODE for necessary times
        pred_z = self.gen_latent(z_init, solve_t)

        batch_inds = torch.arange(B).view((-1, 1)).repeat((1, T))
        # map result latent states back to input sampled points
        sample_feats = pred_z[batch_inds, time_map, :] # B x T x H

        B_global, H_global = z_global.size()
        z_global = z_global.unsqueeze(1).expand(B_global, sample_feats.size()[1], H_global)
        sample_feats = torch.cat([sample_feats, z_global], dim=2)

        return sample_feats

    def gen_latent(self, z0, timestamps):
        '''
        Generates the latent code at the given timestamps starting at the initial condition
        z0 by solving an ODE forward in time.

        z0 : B x H
        timestamps : T

        returns B x T x H
        '''
        pred_z = self.latent_ode(z0, timestamps)
        return pred_z

    def get_nfe(self):
        '''
        Returns the number of function evaluations needed for the most recent forward pass.
        '''
        return np.array([count_nfe(self.latent_ode), count_nfe(self.point_cnf)])

    def decode(self, z, 
                num_points=1024,
                constant_in_time=False,
                truncate_std=None,
                sample_contours=None,
                ):
        '''
        Given latent vectors at various time steps, samples points on the object surface

        z          : latent codes for each step B x T x H
        num_points : how many points to sample
        constant_in_time : if True, samples a single Nx3 point cloud for each batch and uses this same
                           sampling for every timestep
        truncate_std : the number of standard deviations to truncate the sampling from.
        sample_contours : if given a list of floats e.g. [0.25, 0.5, 1.0, 1.5, 2.25, 3.0] samples will come
                          from these gaussian contours rather than randomly

        Returns
        y      : points sampled from the standard normal
        logp_y : the log probability of the sampled points
        x      : points after decoding
        '''
        # transform points from the prior to a point cloud, conditioned on a shape code
        B, T, H = z.size()
        samp_batch = B if constant_in_time else B*T
        input_dim = self.cnf_args.input_dim
        samp_size = (samp_batch, num_points, input_dim)

        if sample_contours is not None:
            radii = sample_contours
            contours = []
            nsamp_pts = 0
            for radius in radii:
                if radius == radii[-1]:
                    cur_npts = samp_batch*(num_points - nsamp_pts)
                else:
                    cur_npts = samp_batch*(num_points//len(radii))
                rand_surf_points = sphere_surface_points(cur_npts, radius=radius)
                if radius == radii[-1]:
                    rand_surf_points = rand_surf_points.reshape((samp_batch, num_points - nsamp_pts, 3))
                else:
                    rand_surf_points = rand_surf_points.reshape((samp_batch, num_points//len(radii), 3))
                contours.append(rand_surf_points)
                nsamp_pts += num_points // len(radii)
            y = np.concatenate(contours, axis=1)
            y = torch.from_numpy(y).to(z)
            y = y.view(samp_size)
        else:
            y = sample_gaussian(samp_size, truncate_std, device=z.device)

        if constant_in_time:
            y = y.view((B, 1, num_points, input_dim)).expand((B, T, num_points, self.cnf_args.input_dim))
            y = y.reshape((B*T, num_points, input_dim))

        logp_y = standard_normal_logprob(y).view(B*T, num_points, -1).sum(2)

        z = z.view((B*T, H))

        x = self.point_cnf(y, z, reverse=True) #.view(*y.size())
        x = x.view((B, T, num_points, input_dim))
        y = y.view((B, T, num_points, input_dim))
        logp_y = logp_y.view((B, T, num_points))
        
        return y, logp_y, x

    def reconstruct(self, x, 
                    num_points=1024,
                    constant_in_time=False, 
                    timestamps=None,
                    max_timestamp=5.0,
                    truncate_std=None,
                    sample_contours=None,
                    ):
        '''
        Reconstructs a given point cloud sequence with CaSPR.

        x               : B x T x N x 4 space-time point cloud with timestamps to condition reconstruction
        num_points      : the number of points to sample at each step
        constant_in_time : if True, samples a single Nx3 point cloud for each batch and uses this same
                           sampling for every timestep
        timestamps      : (T) the times to sample at, if None, uses those from the input sequence
        max_timestamp   : the max_timestamps to normalize input sequences times by if timestamps is None
        truncate_std    : the number of standard deviations to truncate the sampling from.
        sample_contours : if given a list of floats e.g. [0.25, 0.5, 1.0, 1.5, 2.25, 3.0] samples will come
                            from these gaussian contours rather than randomly

        Returns
        y      : points sampled from the standard normal (base distribution)
        logp_y : the log probability of the sampled points
        x      : points after decoding (the reconstruction)
        tnocs_pred : the TNOCS regression if applicable
        '''
        B, T, N, _ = x.size()
        z0, tnocs_pred = self.encode(x)

        if timestamps is None:
            all_times = x[:,:,0,3] / max_timestamp
        else:
            all_times = timestamps.view((1, -1)).repeat((B, 1))
        # print(all_times)

        z = self.aggregate_and_solve_latent(z0, all_times)
        y, logp_y, x = self.decode(z, num_points, constant_in_time, truncate_std, sample_contours)

        return y, logp_y, x, tnocs_pred