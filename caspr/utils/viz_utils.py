import os, sys

import numpy as np
import torch

from .torch_utils import torch_to_numpy
from .pcl_viewer import viz_pcl_seq

from .emd import earth_mover_distance as emd
from .evaluations import eval_reconstr_frames

# std contours for sampling from gaussian
SAMPLE_CONTOURS_RADII = [0.25, 0.5, 1.0, 1.5, 2.25, 3.0]
# amount to offset prediction and base distribution visualizations
PRED_OFFSET = [1.0, 0.0, 0.0]
BASE_OFFSET = [2.5, 0.5, 0.5]

class VizConfig():
    def __init__(self, flags):
        '''
        - flags : argparse flags
        '''
        # TNOCS
        self.viz_tnocs = flags.viz_tnocs

        # Observed
        self.viz_observed = flags.viz_observed

        # Interpolated
        self.viz_interpolated = flags.viz_interpolated

        self.num_sampled_pts = flags.num_sampled_pts
        self.num_sampled_steps = flags.num_sampled_steps
        self.constant_in_time = flags.constant_in_time
        self.show_base_sampling = flags.show_base_sampling
        self.sample_contours = flags.sample_contours
        self.show_input_seq = flags.show_input_seq
        self.tnocs_error_map = flags.tnocs_error_map
        self.show_nocs_cubes = flags.show_nocs_cubes
        self.base_color_map = flags.base_color_map
        self.prob_color_map = flags.prob_color_map

def test_viz(cfg,
            model,
            test_dataset,
            test_loader,
            device):
    '''
    Visualize CaSPR results
    '''

    from tk3dv.extern.chamfer import ChamferDistance # only import if we need it
    chamfer_dist = ChamferDistance()

    model.eval()
    for i, data in enumerate(test_loader):
        print('Batch: %d / %d' % (i, len(test_loader)))
        pcl_in, nocs_out = data[0] # world point cloud, corresponding nocs point cloud
        pcl_in = pcl_in.to(device) # B x T x N x 4 (x,y,z,t)
        nocs_out = nocs_out.to(device) # B x T x N x 4 (x,y,z,t)

        # print(nocs_out.size())

        B, T, N, _ = pcl_in.size()

        if B != 1:
            print('batch size must be 1 to visualize!')
            exit()

        cur_model_id = data[1][0]
        cur_seq_id = data[2][0]
        print('Model %s' % (cur_model_id))
        print('Seq %s' % (cur_seq_id))

        #
        # first show quantitative eval for context
        #
        pred_tnocs = None
        if cfg.viz_tnocs and not (cfg.viz_observed or cfg.viz_interpolated):
            _, pred_tnocs = model.encode(pcl_in) 
        else: # we need sampling predictions
            samp_pcl, logprob_samp_pcl, pred_pcl, pred_tnocs  = model.reconstruct(pcl_in, 
                                                                                num_points=cfg.num_sampled_pts,
                                                                                constant_in_time=cfg.constant_in_time, 
                                                                                max_timestamp=test_dataset.max_timestamp,
                                                                                sample_contours=SAMPLE_CONTOURS_RADII if cfg.sample_contours else None)
        if cfg.viz_tnocs:
            nocs_err = torch.norm(pred_tnocs[:,:,:,:3] - nocs_out[:,:,:,:3], dim=3).mean().to('cpu').item()
            print('Cur L2 nocs spatial error: %f' % (nocs_err))
                
        if cfg.viz_observed or cfg.viz_interpolated:
            quant_num_pts = min([cfg.num_sampled_pts, N])
            observed_tnocs_gt = nocs_out[:,:,:quant_num_pts,:3].view((B*T, quant_num_pts, 3)) # don't need time stamp for reconstruction
            observed_reconstr = pred_pcl[:,:,:quant_num_pts,:].view((B*T, quant_num_pts, 3))
            mean_chamfer, cur_emd = eval_reconstr_frames(observed_reconstr, observed_tnocs_gt, chamfer_dist)
            print('Cur Mean Chamfer: %f' % (np.mean(mean_chamfer)*1000))
            print('Cur Mean EMD: %f' % (np.mean(cur_emd)*1000))

        #
        # Visualize
        #

        # needed by all visualizations
        pcl_in_np, gt_nocs_np = torch_to_numpy([pcl_in, nocs_out])
        viz_gt_nocs = np_to_list(gt_nocs_np)
        viz_pcl_in = np_to_list(pcl_in_np)
        gt_nocs_rgb = copy_pcl_list(viz_gt_nocs)

        base_seq_to_viz = [viz_gt_nocs]
        base_rgb_to_viz = [gt_nocs_rgb]
        if cfg.show_input_seq:
            base_seq_to_viz.append(viz_pcl_in)
            base_rgb_to_viz.append(gt_nocs_rgb)
        
        # TNOCS regression visualization
        if cfg.viz_tnocs:
            print('Visualizing TNOCS Regression Prediction...')
            pred_nocs_np = torch_to_numpy([pred_tnocs])[0]

            viz_pred_nocs = np_to_list(pred_nocs_np)
            if cfg.tnocs_error_map:
                pred_nocs_rgb = [get_error_colors(viz_pred_nocs[idx], viz_gt_nocs[idx]) for idx in range(gt_nocs_np.shape[1])]
            else:
                pred_nocs_rgb = copy_pcl_list(viz_pred_nocs)
            # translate to be in predicted viz cube
            viz_pred_nocs = shift_pcl_list(viz_pred_nocs, PRED_OFFSET)

            seq_to_viz = base_seq_to_viz + [viz_pred_nocs]
            rgb_to_viz = base_rgb_to_viz + [pred_nocs_rgb]

            viz_pcl_seq(seq_to_viz, rgb_seq=rgb_to_viz, fps=T, autoplay=True, draw_cubes=cfg.show_nocs_cubes)

        # Observed sampling visualization
        if cfg.viz_observed:
            print('Visualizing CaSPR Observed Reconstruction Sampling...')
            viz_caspr_reconstruction(cfg, samp_pcl, logprob_samp_pcl, pred_pcl,
                             base_seq_to_viz, base_rgb_to_viz, T)

        # Interpolated sampling visualization
        if cfg.viz_interpolated:
            print('Visualizing CaSPR Interpolated Reconstruction Sampling...')
            timstamps = torch.linspace(0.0, 1.0, cfg.num_sampled_steps).to(pcl_in)
            # rerun reconstruction on higher-res timestamps
            samp_pcl, logprob_samp_pcl, pred_pcl, _ = model.reconstruct(pcl_in, 
                                                                        timestamps=timstamps,
                                                                        num_points=cfg.num_sampled_pts,
                                                                        constant_in_time=cfg.constant_in_time,
                                                                        sample_contours=SAMPLE_CONTOURS_RADII if cfg.sample_contours else None)

            # naively subsample observations to visualize with interpolated result
            subsampled_gt_nocs = []
            subsampled_pcl_in = []
            subsampled_times = []
            subsamples_per_step = int(float(cfg.num_sampled_steps) / T)
            for time_idx in range(T):
                for repeat_idx in range(subsamples_per_step):
                    subsampled_gt_nocs.append(gt_nocs_np[0, time_idx, :, :3])
                    subsampled_pcl_in.append(pcl_in_np[0, time_idx, :, :3])
                    subsampled_times.append(gt_nocs_np[0, time_idx, 0, 3])
            # fill any extras
            while len(subsampled_gt_nocs) < cfg.num_sampled_steps:
                subsampled_gt_nocs.append(gt_nocs_np[0, T-1, :, :3])
                subsampled_pcl_in.append(pcl_in_np[0, T-1, :, :3])
                subsampled_times.append(gt_nocs_np[0, T-1, 0, 3])

            viz_gt_nocs = subsampled_gt_nocs
            viz_pcl_in = subsampled_pcl_in
            gt_nocs_rgb = copy_pcl_list(viz_gt_nocs)

            cur_base_seq_to_viz = [viz_gt_nocs]
            cur_base_rgb_to_viz = [gt_nocs_rgb]
            if cfg.show_input_seq:
                cur_base_seq_to_viz.append(viz_pcl_in)
                cur_base_rgb_to_viz.append(gt_nocs_rgb)

            viz_caspr_reconstruction(cfg, samp_pcl, logprob_samp_pcl, pred_pcl,
                             cur_base_seq_to_viz, cur_base_rgb_to_viz, cfg.num_sampled_steps)

def viz_caspr_reconstruction(cfg, samp_pcl, logprob_samp_pcl, pred_pcl,
                             base_seq_to_viz, base_rgb_to_viz, fps):
    '''
    Visualizes full CaSPR reconstruction pipeline results
    '''
    samp_pcl_np, logprob_samp_pcl_np, pred_pcl_np = torch_to_numpy([samp_pcl, logprob_samp_pcl, pred_pcl])

    # sampling from caspr
    viz_pred = np_to_list(pred_pcl_np)
    pred_nocs_rgb = copy_pcl_list(viz_pred)
    # sampling from base gaussian given to caspr
    viz_samp = np_to_list(samp_pcl_np)
    viz_samp_rgb = pred_nocs_rgb

    if cfg.sample_contours:
        # get different colors for each contour
        pred_nocs_rgb = viz_samp_rgb = get_sphere_samp_colors(-logprob_samp_pcl_np[0])
    elif cfg.base_color_map:
        # color based on location of base point rather than final location
        cur_gaussian = samp_pcl_np[0]
        cur_gaussian = cur_gaussian / 4.5
        cur_gaussian = cur_gaussian + 0.5
        pred_nocs_rgb = viz_samp_rgb = [cur_gaussian[idx] for idx in range(cur_gaussian.shape[0])]
    elif cfg.prob_color_map:
        # color based on log probability of base distribution sample
        pred_nocs_rgb = viz_samp_rgb = get_logprob_colors(-logprob_samp_pcl_np[0])

    viz_pred = shift_pcl_list(viz_pred, PRED_OFFSET)
    # have to (approximately) scale & translate manually b/c base sampling did not go through normalzation layer yet
    viz_samp = [(viz_samp[idx] / 15.0) + np.array([BASE_OFFSET]) for idx in range(len(viz_samp))]

    seq_to_viz = base_seq_to_viz + [viz_pred]
    rgb_to_viz = base_rgb_to_viz + [pred_nocs_rgb]
    if cfg.show_base_sampling:
        seq_to_viz.append(viz_samp)
        rgb_to_viz.append(viz_samp_rgb)

    viz_pcl_seq(seq_to_viz, rgb_seq=rgb_to_viz, fps=fps, autoplay=True, draw_cubes=cfg.show_nocs_cubes)

#
# Viz helper functions
#

def np_to_list(np_array):
    ''' Turns B x T x N x D np array into a list of Nx3 arrays for visualization. Uses batch 1. '''
    return [np_array[0,i,:,:3] for i in range(np_array.shape[1])]

def copy_pcl_list(pcl_list):
    return [pcl_list[idx] for idx in range(len(pcl_list))]

def shift_pcl_list(pcl_list, offset):
    '''
    Shifts a list of Nx3 np array point clouds by the given offset translation [x,y,z].
    '''
    return [pcl_list[idx] + np.array([offset]) for idx in range(len(pcl_list))]

def get_error_colors(predicted, gt):
    worst_error = 0.07
    # color the predicted_nocs based on error
    pt_errors = np.linalg.norm(predicted - gt, axis=1)
    pt_colors = np.ones_like(predicted)
    pt_colors[:,0] = np.minimum(1.0, pt_errors / worst_error)  # based on error
    pt_colors[:,1] = 27.0 / 255.0
    pt_colors[:,2] = 116.0 / 255.0

    return pt_colors

def get_logprob_colors(logprob_y, low_prob=2.0, high_prob=9.0):
    '''
    logprob_y : T x N
    '''
    # print(logprob_y)
    trans_logprob = logprob_y - low_prob
    high_prob -= low_prob
    low_prob = 0.0
    T, N = logprob_y.shape
    pt_colors = np.ones((T, N, 3))
    pt_colors[:,:,0] = np.minimum(1.0, trans_logprob / high_prob)
    pt_colors[:,:,1] = 27.0 / 255.0
    pt_colors[:,:,2] = 116.0 / 255.0

    logprob_colors = [pt_colors[idx] for idx in range(pt_colors.shape[0])]

    return logprob_colors

def get_sphere_samp_colors(logprob_y):
    '''
    logprob_y : T x N
    '''

    prob_colors = np.array([[153.0, 0.0, 76.0],
                            [102.0, 0.0, 0.0],
                            [204.0, 102.0, 0.0],
                            [0.0, 102.0, 0.0],
                            [0.0, 102.0, 204.0],
                            [102.0, 0.0, 204.0]])
    prob_colors /= 255.0

    T, N = logprob_y.shape
    sorted_probs, inv_map = np.unique(logprob_y.round(decimals=4), return_inverse=True)

    pt_colors = np.ones((T*N, 3))
    pt_colors[:,:] = prob_colors[inv_map,:]
    pt_colors = pt_colors.reshape((T, N, 3))

    logprob_colors = [pt_colors[idx] for idx in range(pt_colors.shape[0])]
    return logprob_colors