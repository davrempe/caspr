'''

This script contains functions for running the primary evaluations from the paper

'''

import os, sys, csv, time

import numpy as np

import torch
import torch.nn.functional as F

from tk3dv.extern import quaternions

sys.path.append('.')
from .torch_utils import torch_to_numpy
from .pcl_viewer import viz_pcl_seq
from .train_utils import log
from data.caspr_dataset import load_seq_path

from .emd import earth_mover_distance as emd
from tk3dv.extern.chamfer import ChamferDistance # only import if we need it

# protocol for evaluations in the paper
PROTOCOL_NUM_STEPS = 10
PROTOCOL_NUM_PTS = 2048

ALL_OBSERVED_STEPS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
ALL_UNOBSERVED_STEPS = []

# 3 steps input
SPLIT_OBSERVED_STEPS = [0, 5, 9]
SPLIT_UNOBSERVED_STEPS = [1, 2, 3, 4, 6, 7, 8]

def eval_reconstr_frames(pred, gt, chamfer_dist):
    '''
    Evaluates chamfer, one-way chamfer, and emd and returns as np arrays.
    '''
    dist1, dist2 = chamfer_dist(pred, gt)
    mean_dist_pred2gt = torch.mean(dist1, dim=1)
    mean_dist_gt2pred = torch.mean(dist2, dim=1)
    mean_dist =  mean_dist_pred2gt + mean_dist_gt2pred

    cur_emd = emd(pred, gt, transpose=False)
    cur_emd = cur_emd / pred.size(1)

    results = [mean_dist, cur_emd]
    return tuple([res.cpu().data.numpy() for res in results])

def test_shape_recon(model, test_loader, device, log_out, observed_steps, unobserved_steps):
    '''
    Evaluates shape reconstruction from CNF.
    '''
    test_dataset = test_loader.dataset

    chamfer_dist = ChamferDistance()

    log(log_out, 'Observed steps [%s]' % (','.join([str(idx) for idx in observed_steps])))
    log(log_out, 'Unobserved steps [%s]' % (','.join([str(idx) for idx in unobserved_steps])))

    use_unobserved_steps = len(unobserved_steps) > 0

    model.eval()

    nfe_stats = []
    model_ids = []
    seq_ids = []
    observed_stats = {
        'chamfer' : [],
        'emd' : [],
        'infer_time' : []
    }
    unobserved_stats = {
        'chamfer' : [],
        'emd' : []
    }
    num_batches_total = 0
    for i, data in enumerate(test_loader):
        print('Batch: %d / %d' % (i, len(test_loader)))
        pcl_in, nocs_out = data[0] # world point cloud, corresponding nocs point cloud
        pcl_in = pcl_in.to(device) # B x T x N x 4 (x,y,z,t)
        nocs_out = nocs_out.to(device) # B x T x N x 4 (x,y,z,t)

        cur_model_ids = data[1]
        cur_seq_ids = data[2]
        model_ids.extend(cur_model_ids)
        seq_ids.extend(cur_seq_ids)

        # print(cur_model_ids)
        # print(cur_seq_ids)

        B, T, N, _ = pcl_in.size()
        num_batches_total += B
        T_observed = len(observed_steps)
        T_unobserved = len(unobserved_steps)

        if T != PROTOCOL_NUM_STEPS:
            print('Test protocol requires %d steps, but %d given!' % (PROTOCOL_NUM_STEPS, T))
            exit()
        if N != PROTOCOL_NUM_PTS:
            print('Test protocol requires %d points, but %d given!' % (PROTOCOL_NUM_PTS, N))
            exit()
        
        # only use the observed steps as input
        observed_pcl_in = pcl_in[:,observed_steps,:,:]

        elapsed = 0.0
        start_t = time.time()
        # reconstruct at all time steps, both observed and unobserved
        _, _, pred_pcl, _  = model.reconstruct(observed_pcl_in, 
                                            num_points=N,
                                            timestamps=nocs_out[0, :, 0, 3],
                                            constant_in_time=False)
        elapsed = time.time() - start_t

        cur_nfe = model.get_nfe()
        nfe_stats.append(cur_nfe)

        # evaluate Chamfer and EMD
        # first observed
        observed_tnocs_gt = nocs_out[:,observed_steps,:,:3].view((B*T_observed, N, 3)) # don't need time stamp for reconstruction
        observed_reconstr = pred_pcl[:,observed_steps,:,:].view((B*T_observed, N, 3))

        mean_chamfer, cur_emd = eval_reconstr_frames(observed_reconstr, observed_tnocs_gt, chamfer_dist)
        observed_stats['chamfer'].extend(mean_chamfer.tolist())
        observed_stats['emd'].extend(cur_emd.tolist())
        observed_stats['infer_time'].append(elapsed)

        print('==== OBSERVED ====')
        print('Shape Recon Mean Chamfer: %f' % (np.mean(observed_stats['chamfer'])*1000))
        print('Shape Recon Median Chamfer: %f' % (np.median(observed_stats['chamfer'])*1000))
        print('Shape Recon Mean EMD: %f' % (np.mean(observed_stats['emd'])*1000))
        print('Shape Recon Median EMD: %f' % (np.median(observed_stats['emd'])*1000))
        print('NFE Mean: (%f, %f)' % tuple(np.mean(nfe_stats, axis=0).tolist()))
        print('Infer time mean: %f' % (np.mean(observed_stats['infer_time'])))


        if use_unobserved_steps:
            unobserved_tnocs_gt = nocs_out[:,unobserved_steps,:,:3].view((B*T_unobserved, N, 3)) # don't need time stamp for reconstruction
            unobserved_reconstr = pred_pcl[:,unobserved_steps,:,:].view((B*T_unobserved, N, 3))

            mean_chamfer, cur_emd = eval_reconstr_frames(unobserved_reconstr, unobserved_tnocs_gt, chamfer_dist)
            unobserved_stats['chamfer'].extend(mean_chamfer.tolist())
            unobserved_stats['emd'].extend(cur_emd.tolist())

            print('==== UNOBSERVED ====')
            print('Shape Recon Mean Chamfer: %f' % (np.mean(unobserved_stats['chamfer'])*1000))
            print('Shape Recon Median Chamfer: %f' % (np.median(unobserved_stats['chamfer'])*1000))
            print('Shape Recon Mean EMD: %f' % (np.mean(unobserved_stats['emd'])*1000))
            print('Shape Recon Median EMD: %f' % (np.median(unobserved_stats['emd'])*1000))
            print('NFE Mean: (%f, %f)' % tuple(np.mean(nfe_stats, axis=0).tolist()))

        # print(len(observed_stats['chamfer']))
        # print(len(unobserved_stats['chamfer']))

    # aggregate
    stats_list = [observed_stats, unobserved_stats] if use_unobserved_steps else [observed_stats]
    stats_names = ['OBSERVED', 'UNOBSERVED'] if use_unobserved_steps else ['OBSERVED']
    for stat_dict, stats_name in zip(stats_list, stats_names):
        mean_chamfer_err = np.mean(stat_dict['chamfer'])*1000.0
        median_chamfer_err = np.median(stat_dict['chamfer'])*1000.0
        std_chamfer_err = np.std(stat_dict['chamfer'])*1000.0
        mean_emd_err = np.mean(stat_dict['emd'])*1000.0
        median_emd_err = np.median(stat_dict['emd'])*1000.0
        std_emd_err = np.std(stat_dict['emd'])*1000.0

        log(log_out, '================  %s SAMPLING RECONSTR EVAL =====================' % (stats_name))
        log(log_out, 'mean CHAMFER error (x1000): %f +- %f, median: %f' % (mean_chamfer_err, std_chamfer_err, median_chamfer_err))
        log(log_out, 'mean EMD error (x1000): %f +- %f, median: %f' % (mean_emd_err, std_emd_err, median_emd_err))
    log(log_out, 'NFE Mean: (%f, %f)' % tuple(np.mean(nfe_stats, axis=0).tolist()))
    log(log_out, 'mean Inference time: %f' % (np.mean(observed_stats['infer_time'])))

    # save the evaluation data
    per_seq_data_out = log_out[:-len('txt')] + 'npz'
    np.savez(per_seq_data_out, observed_chamfer=observed_stats['chamfer'],
                               observed_emd=observed_stats['emd'],
                               unobserved_chamfer=unobserved_stats['chamfer'],
                               unobserved_emd=unobserved_stats['emd'])

    # log per-sequence performance
    per_seq_log = log_out[:-len('txt')] + 'csv'
    print('Per seq performance being saved to %s...' % (per_seq_log))
    stats_steps = [T_observed, T_unobserved]
    with open(per_seq_log, 'w', newline='') as csvfile:
        # write header
        csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        header = ['type', 'model_id', 'seq_id', 'chamfer', 'emd']
        csvwriter.writerow(header)
        for stat_dict, stats_name, stats_T in zip(stats_list, stats_names, stats_steps):
            per_seq_chamfer = np.array(stat_dict['chamfer']).reshape((num_batches_total, stats_T))
            per_seq_chamfer = np.mean(per_seq_chamfer, axis=1)
            per_seq_emd = np.array(stat_dict['emd']).reshape((num_batches_total, stats_T))
            per_seq_emd = np.mean(per_seq_emd, axis=1)

            for line_idx in range(len(model_ids)):
                cur_line = [stats_name, model_ids[line_idx], seq_ids[line_idx], per_seq_chamfer[line_idx], per_seq_emd[line_idx]]
                csvwriter.writerow(cur_line)

    return

def test_tnocs_regression(model, test_loader, device, log_out):
    '''
    EVAL only TNOCS regression.
    '''
    test_dataset = test_loader.dataset

    model.eval()

    model_ids = []
    seq_ids = []
    stat_dict = {
        'space' : [], # l2 loss in space
        'time' : [] # abs difference in time
    }
    num_batches_total = 0
    for i, data in enumerate(test_loader):
        print('Batch: %d / %d' % (i, len(test_loader)))
        pcl_in, nocs_out = data[0] # world point cloud, corresponding nocs point cloud
        pcl_in = pcl_in.to(device) # B x T x N x 4 (x,y,z,t)
        nocs_out = nocs_out.to(device) # B x T x N x 4 (x,y,z,t)

        cur_model_ids = data[1]
        cur_seq_ids = data[2]
        model_ids.extend(cur_model_ids)
        seq_ids.extend(cur_seq_ids)

        # print(cur_model_ids)
        # print(cur_seq_ids)

        B, T, N, _ = pcl_in.size()
        num_batches_total += B

        if T != PROTOCOL_NUM_STEPS:
            print('Test protocol requires %d steps, but %d given!' % (PROTOCOL_NUM_STEPS, T))
            exit()
        if N != PROTOCOL_NUM_PTS:
            print('Test protocol requires %d points, but %d given!' % (PROTOCOL_NUM_PTS, N))
            exit()

        # only tnocs predictions
        _, pred_tnocs = model.encode(pcl_in)

        # calculate distance error (with correspondences)
        diff = pred_tnocs[:,:,:,:3] - nocs_out[:,:,:,:3]
        dist = torch.mean(torch.norm(diff, dim=3), dim=2) # B x T
        stat_dict['space'].extend(dist.to('cpu').data.numpy().reshape((-1)).tolist())

        # calulcate time error
        if pred_tnocs.size(3) > 3:
            time_diff = torch.abs(pred_tnocs[:,:,:,3] - nocs_out[:,:,:,3])
            time_diff = torch.mean(time_diff, dim=2) # B x T
            stat_dict['time'].extend(time_diff.to('cpu').data.numpy().reshape((-1)).tolist())

        print('==== CURRENT ERROR ====')
        print('mean SPATIAL error (l2 distance) %f' % (np.mean(stat_dict['space'])))
        print('mean TIME error (absolute diff): : %f' % (np.mean(stat_dict['time'])))

    
    mean_space_err = np.mean(stat_dict['space'])
    median_space_err = np.median(stat_dict['space'])
    std_space_err = np.std(stat_dict['space'])

    mean_time_err = np.mean(stat_dict['time'])
    median_time_err = np.median(stat_dict['time'])
    std_time_err = np.std(stat_dict['time'])

    log(log_out, '================  TNOCS REGRESSION EVAL =====================')
    log(log_out, 'mean SPATIAL error (l2 distance): %f +- %f, median: %f' % (mean_space_err, std_space_err, median_space_err))
    log(log_out, 'mean TIME error (absolute diff): %f +- %f, median: %f' % (mean_time_err, std_time_err, median_time_err))

     # save the evaluation data
    per_seq_data_out = log_out[:-len('txt')] + 'npz'
    np.savez(per_seq_data_out, space=stat_dict['space'],
                               time=stat_dict['time'])

    # log per-sequence performance
    per_seq_log = log_out[:-len('txt')] + 'csv'
    print('Per seq performance being saved to %s...' % (per_seq_log))
    with open(per_seq_log, 'w', newline='') as csvfile:
        # write header
        csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        header = ['model_id', 'seq_id', 'space', 'time']
        csvwriter.writerow(header)

        per_seq_space = np.array(stat_dict['space']).reshape((num_batches_total, T))
        per_seq_space = np.mean(per_seq_space, axis=1)
        per_seq_time = np.array(stat_dict['time']).reshape((num_batches_total, T))
        per_seq_time = np.mean(per_seq_time, axis=1)

        for line_idx in range(len(model_ids)):
            cur_line = [model_ids[line_idx], seq_ids[line_idx], per_seq_space[line_idx], per_seq_time[line_idx]]
            csvwriter.writerow(cur_line)

def test_observed_camera_pose_ransac(model, test_loader, device, log_out, show=False):
    import open3d as o3d

    '''
    Evaluate camera pose estimation at observed time steps. With just the TNOCS regression.
    '''
    test_dataset = test_loader.dataset
    test_dataset.set_return_pose_data(True)        

    model.eval()

    model_ids = []
    seq_ids = []
    stat_dict = {
        'trans_RANSAC' : [], # l2 loss on translation
        'rot_RANSAC' : [],  # abs degree difference of rotation
        'point_RANSAC' : [], # pointwise distance after transforming predicted NOCS with optimal pose
        'point_mean_RANSAC' : [], # pointwise distance after transforming predicted NOCS with optimal pose
    }
    num_batches_total = 0

    for i, data in enumerate(test_loader):
        print('Batch: %d / %d' % (i, len(test_loader)))
        pcl_in, nocs_out = data[0] # world point cloud, corresponding nocs point cloud
        pcl_in = pcl_in.to(device) # B x T x N x 4 (x,y,z,t)
        nocs_out = nocs_out.to(device) # B x T x N x 4 (x,y,z,t)

        pose_data = data[1]

        cur_model_ids = data[2]
        cur_seq_ids = data[3]
        model_ids.extend(cur_model_ids)
        seq_ids.extend(cur_seq_ids)

        # print(cur_model_ids)
        # print(cur_seq_ids)

        B, num_steps, N, _ = pcl_in.size()
        num_batches_total += B

        if num_steps != PROTOCOL_NUM_STEPS:
            print('Test protocol requires %d steps, but %d given!' % (PROTOCOL_NUM_STEPS, T))
            exit()
        if N != PROTOCOL_NUM_PTS:
            print('Test protocol requires %d points, but %d given!' % (PROTOCOL_NUM_PTS, N))
            exit()

        # only tnocs predictions
        _, pred_tnocs = model.encode(pcl_in)

        for batch_idx in range(pred_tnocs.shape[0]):
            norm_pred_nocs = pred_tnocs[batch_idx,:,:,:3] - 0.5
            norm_gt_nocs = nocs_out[batch_idx,:,:,:3] - 0.5
            cur_input_points = pcl_in[batch_idx,:,:,:3]

            cam_transform_seq_RANSAC = []
            gt_cam_transform_seq = []

            pred_depth_seq_RANSAC = []
            gt_depth_seq = []

            for step_idx in range(norm_pred_nocs.shape[0]):
                # solve for rigid transform using RANSAC
                pcd1 = o3d.geometry.PointCloud()
                pcd2 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(norm_pred_nocs[step_idx].to('cpu').data.numpy())
                pcd2.points = o3d.utility.Vector3dVector(cur_input_points[step_idx].to('cpu').data.numpy())

                # Correspondences are already sorted
                corr_idx = np.tile(np.expand_dims(np.arange(len(pcd1.points)),1),(1,2))
                corrs = o3d.utility.Vector2iVector(corr_idx)

                distance_threshold = 0.015
                result_ransac = o3d.registration.registration_ransac_based_on_correspondence(
                                source=pcd1, target=pcd2, corres=corrs, 
                                max_correspondence_distance=distance_threshold,
                                estimation_method=o3d.registration.TransformationEstimationPointToPoint(False),
                                ransac_n=4,
                                criteria=o3d.registration.RANSACConvergenceCriteria(50000, 5000))

                trans_param = result_ransac.transformation

                R_pred_RANSAC = trans_param[0:3,0:3]
                T_pred_RANSAC = trans_param[0:3,3]#.reshape(-1,1)
                

                #
                # Compute errors
                #
                input_data_np = cur_input_points[step_idx].cpu().numpy()
                # get GT object pose
                R_gt = pose_data[batch_idx, step_idx, :3, :3].data.numpy()
                T_gt = pose_data[batch_idx, step_idx, :3, 3].data.numpy()

                # Use ground truth TNOCS as not to compound error from NOCS estimation into point-wise error
                norm_gt_nocs_np = norm_gt_nocs[step_idx].cpu().numpy()
                gt_depth_points = np.dot(R_gt, norm_gt_nocs_np.T).T + T_gt
                pred_depth_points_RANSAC = np.dot(R_pred_RANSAC, norm_gt_nocs_np.T).T + T_pred_RANSAC

                gt_depth_seq.append(gt_depth_points)
                pred_depth_seq_RANSAC.append(pred_depth_points_RANSAC)

                # calculate RANSAC distance error (with correspondences)
                diff_RANSAC = pred_depth_points_RANSAC - input_data_np
                dist_RANSAC = np.median(np.linalg.norm(diff_RANSAC, axis=1))
                stat_dict['point_RANSAC'].append(dist_RANSAC.tolist())
                dist_RANSAC = np.mean(np.linalg.norm(diff_RANSAC, axis=1))
                stat_dict['point_mean_RANSAC'].append(dist_RANSAC.tolist())

                # predicted camera pose RANSAC
                cam_R_RANSAC = R_pred_RANSAC.T
                cam_T_RANSAC = np.dot(cam_R_RANSAC, -T_pred_RANSAC)
                cam_transform_RANSAC = np.eye(4)
                cam_transform_RANSAC[:3,:3] = cam_R_RANSAC
                cam_transform_RANSAC[:3, 3] = cam_T_RANSAC
                cam_transform_seq_RANSAC.append(cam_transform_RANSAC)                

                # GT camera pose
                gt_cam_transform = np.eye(4)
                gt_cam_transform[:3,:3] = R_gt.T
                gt_cam_transform[:3, 3] = np.dot(R_gt.T, -T_gt)
                gt_cam_transform_seq.append(gt_cam_transform)

                # calculate error RANSAC
                trans_err_RANSAC = np.linalg.norm(T_pred_RANSAC - T_gt)
                rot_mat_diff_RANSAC = (np.trace(np.dot(R_pred_RANSAC.T, R_gt)) - 1.0) / 2.0
                rot_mat_diff_RANSAC = np.clip(rot_mat_diff_RANSAC, -1.0, 1.0)
                rot_err_RANSAC = np.degrees(np.arccos(rot_mat_diff_RANSAC))

                stat_dict['trans_RANSAC'].append(trans_err_RANSAC)
                stat_dict['rot_RANSAC'].append(rot_err_RANSAC)

            pred_cam_transform_seq_RANSAC = np.stack(cam_transform_seq_RANSAC, axis=0)
            gt_cam_transform_seq = np.stack(gt_cam_transform_seq, axis=0)
            
            pred_depth_seq_RANSAC = np.stack(pred_depth_seq_RANSAC, axis=0)
            gt_depth_seq = np.stack(gt_depth_seq, axis=0)

            if show:
                # Visualization shows:
                #   - predicted NOCS in RGB
                #   - GT NOCS transformed by the predicted pose (in blue)
                #   - GT input and NOCS point clouds (in green)
                #   - GT camera pose (in green)
                #   - pred camera pose (in red)
                pred_nocs_viz = [norm_pred_nocs[idx].to('cpu').data.numpy() for idx in range(norm_pred_nocs.size()[0])]
                pred_nocs_rgb = [pred_tnocs[batch_idx, idx, :, :3].to('cpu').data.numpy().astype(np.float) for idx in range(len(pred_nocs_viz))]
                pred_depth_viz = [pred_depth_seq_RANSAC[idx] for idx in range(pred_depth_seq_RANSAC.shape[0])]
                blue_rgb = np.zeros_like(pred_depth_seq_RANSAC[0])
                blue_rgb[:,2] = 1.0
                pred_depth_rgb = [blue_rgb for idx in range(pred_depth_seq_RANSAC.shape[0])]
                gt_depth_viz = [gt_depth_seq[idx] for idx in range(gt_depth_seq.shape[0])]
                green_rgb = np.zeros_like(gt_depth_seq[0])
                green_rgb[:,1] = 1.0
                gt_depth_rgb = [green_rgb for idx in range(gt_depth_seq.shape[0])]
                gt_nocs_viz = [norm_gt_nocs[idx].to('cpu').data.numpy() for idx in range(norm_gt_nocs.size()[0])]
                gt_nocs_rgb = [green_rgb for idx in range(len(gt_nocs_viz))]

                viz_pcl_seq([pred_nocs_viz, pred_depth_viz, gt_depth_viz, gt_nocs_viz], 
                            rgb_seq=[pred_nocs_rgb, pred_depth_rgb, gt_depth_rgb, gt_nocs_rgb], 
                            fps=10, autoplay=True, cameras=[gt_cam_transform_seq, pred_cam_transform_seq_RANSAC], 
                            draw_cubes=False)

        print('==== CURRENT ERROR ====')
        print('mean Pos error RANSAC (l2 distance) %f' % (np.mean(stat_dict['trans_RANSAC'])))
        print('mean Rot error RANSAC (degrees): %f' % (np.mean(stat_dict['rot_RANSAC'])))
        print('mean-median Point error RANSAC (L2 distance): %f' % (np.mean(stat_dict['point_RANSAC'])))
        print('mean-mean Point error RANSAC (L2 distance): %f' % (np.mean(stat_dict['point_mean_RANSAC'])))

    mean_pos_err_RANSAC = np.mean(stat_dict['trans_RANSAC'])
    median_pos_err_RANSAC = np.median(stat_dict['trans_RANSAC'])
    std_pos_err_RANSAC = np.std(stat_dict['trans_RANSAC'])

    mean_rot_err_RANSAC = np.mean(stat_dict['rot_RANSAC'])
    median_rot_err_RANSAC = np.median(stat_dict['rot_RANSAC'])
    std_rot_err_RANSAC = np.std(stat_dict['rot_RANSAC'])

    mean_point_err_RANSAC = np.mean(stat_dict['point_RANSAC'])
    median_point_err_RANSAC = np.median(stat_dict['point_RANSAC'])
    std_point_err_RANSAC = np.std(stat_dict['point_RANSAC'])

    mean_mean_point_err_RANSAC = np.mean(stat_dict['point_mean_RANSAC'])
    median_mean_point_err_RANSAC = np.median(stat_dict['point_mean_RANSAC'])
    std_mean_point_err_RANSAC = np.std(stat_dict['point_mean_RANSAC'])

    log(log_out, 'mean POS error RANSAC (l2 distance): %f +- %f, median: %f' % (mean_pos_err_RANSAC, std_pos_err_RANSAC, median_pos_err_RANSAC))
    log(log_out, 'mean ROT error RANSAC (degrees): %f +- %f, median: %f' % (mean_rot_err_RANSAC, std_rot_err_RANSAC, median_rot_err_RANSAC))
    log(log_out, 'mean POINT(median) error RANSAC (l2 distance): %f +- %f, median: %f' % (mean_point_err_RANSAC, std_point_err_RANSAC, median_point_err_RANSAC))
    log(log_out, 'mean POINT(mean) error RANSAC (l2 distance): %f +- %f, median: %f' % (mean_mean_point_err_RANSAC, std_mean_point_err_RANSAC, median_mean_point_err_RANSAC))

    # save the evaluation data
    per_seq_data_out_RANSAC = log_out[:-len('.txt')] + '_RANSAC.npz'
    np.savez(per_seq_data_out_RANSAC, trans=stat_dict['trans_RANSAC'],
                               rot=stat_dict['rot_RANSAC'],
                               point=stat_dict['point_RANSAC'],
                               point_mean=stat_dict['point_mean_RANSAC'])

    # log per-sequence performance of RANSAC
    per_seq_log_RANSAC = log_out[:-len('.txt')] + '_RANSAC.csv'
    print('Per seq performance of RANSAC being saved to %s...' % (per_seq_log_RANSAC))

    with open(per_seq_log_RANSAC, 'w', newline='') as csvfile:
        # write header
        csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        header = ['model_id', 'seq_id', 'pos', 'rot', 'point']
        csvwriter.writerow(header)

        per_seq_pos_RANSAC = np.array(stat_dict['trans_RANSAC']).reshape((num_batches_total, num_steps))
        per_seq_pos_RANSAC = np.mean(per_seq_pos_RANSAC, axis=1)
        per_seq_rot_RANSAC = np.array(stat_dict['rot_RANSAC']).reshape((num_batches_total, num_steps))
        per_seq_rot_RANSAC = np.mean(per_seq_rot_RANSAC, axis=1)
        per_seq_point_RANSAC = np.array(stat_dict['point_RANSAC']).reshape((num_batches_total, num_steps))
        per_seq_point_RANSAC = np.mean(per_seq_point_RANSAC, axis=1)

        for line_idx in range(len(model_ids)):
            cur_line = [model_ids[line_idx], seq_ids[line_idx], per_seq_pos_RANSAC[line_idx], per_seq_rot_RANSAC[line_idx], per_seq_point_RANSAC[line_idx]]
            csvwriter.writerow(cur_line)