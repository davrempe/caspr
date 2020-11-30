import os, sys
import glob
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# models that didn't actually render. They're just spheres.
BAD_MODELS = ['93ce8e230939dfc230714334794526d4',
              '207e69af994efa9330714334794526d4',
              '2307b51ca7e4a03d30714334794526d4']

# raw world point cloud sequences will be given timestamps from 0 to this
DEFAULT_MAX_TIMESTAMP = 5.0 
# numbser of steps in each sequence
DEFAULT_EXPECTED_SEQ_LEN = 10
# number of points at each tep
DEFAULT_EXPECTED_NUM_PTS = 4096

class SplitLineParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        return arg_line.split()

def parse_dataset_cfg(cfg_file_path):
    parser = SplitLineParser(fromfile_prefix_chars='@', allow_abbrev=False)
    parser.add_argument('--data', type=str, nargs='+', required=True, help='Paths to dataset roots e.g. ../data/cars')
    parser.add_argument('--splits', type=str, nargs='+', default=None, help='Directory to optionally load data splits from e.g. ../data/splits/car_splits. Otherwise will split based on fractions given.')    
    parser.add_argument('--max-timestamp', type=float, default=DEFAULT_MAX_TIMESTAMP, help='Max timestamp to label the data with')
    parser.add_argument('--expected-num-pts', type=int, default=DEFAULT_EXPECTED_NUM_PTS, help='The expected number of points in each frame of the dataset to load in.')
    parser.add_argument('--expected-seq-len', type=int, default=DEFAULT_EXPECTED_SEQ_LEN, help='The expected number of frames in each sequence.')
    args = parser.parse_args(['@' + cfg_file_path])
    return args

def load_time_data(data_roots, split, train_frac, val_frac, splits_dirs=None, data_seq_len=DEFAULT_EXPECTED_SEQ_LEN):
    '''
    Collects all the necessary data paths contained in the
    given data_roots which will be used to load in the
    data lazily when requested.
    '''
    print('Loading time data from %d sources...' % (len(data_roots)))
    all_nocs_seq_paths = []
    for data_src_idx, data_root in enumerate(data_roots):
        print(data_root)

        if not os.path.exists(data_root):
            print('Could not find %s!' % (data_root))
            exit()

        cur_split_dir = None
        split_list = None
        if splits_dirs is not None:
            cur_split_dir = splits_dirs[data_src_idx]
            cur_split_file = os.path.join(cur_split_dir, split + '_split.txt')
            print(cur_split_file)
            if not os.path.exists(cur_split_file):
                print('There is no split file for the requested split!')
                exit()
            with open(cur_split_file, 'r') as f:
                split_str = f.read()
            split_list = split_str.split('\n')

        # go through each model in this dataset
        if split_list is None:
            model_dirs = [os.path.join(data_root, f) for f in sorted(os.listdir(data_root)) if f[0] != '.']
            model_dirs = [f for f in model_dirs if os.path.isdir(f)]
        else:
            model_dirs = [os.path.join(data_root, split_model) for split_model in split_list if split_model != '']

        print('Found %d models from this data source.' % (len(model_dirs)))
        if split_list is not None:
            print('NOTE: this is just for the single requested split.')

        all_model_ids = []
        nocs_seq_paths = [] # all model lists
        for model_path in model_dirs:
            model_id = model_path.split('/')[-1]
            if cur_split_dir is not None and not os.path.exists(model_path):
                print('WARNING: Could not find model %s requested in the split file! Skipping...' % (model_id))
                continue
            if model_id in BAD_MODELS:
                print('Skipping model %s, bad model...' % (model_id))
                continue
            cur_model_paths = [] # all the sequences for this model
            # go through each sequence
            seq_dirs = [os.path.join(model_path, f) for f in sorted(os.listdir(model_path)) if f[0] != '.']
            seq_dirs = [f for f in seq_dirs if os.path.isdir(f)]

            # collect paths for full dataset
            for seq_path in seq_dirs:
                nocs_pc_files = sorted(glob.glob(os.path.join(seq_path, '*frame*.npz')))

                num_frames = len(nocs_pc_files)
                if num_frames != data_seq_len:
                    print('Found %d frames at %s...skipping!' % (num_frames, seq_path))
                    continue

                cur_model_paths.append(nocs_pc_files)
            nocs_seq_paths.append(cur_model_paths)
            all_model_ids.append(model_id)
        all_model_ids = np.array(all_model_ids)
        
        print('Sequences are of length %d...' % (len(nocs_seq_paths[0][0])))
        print('Loading %s split...' % (split))

        # only load in necessary split

        num_models = len(nocs_seq_paths)
        print('This data source has %d sequences per model...' % (len(nocs_seq_paths[0])))

        if splits_dirs is None:
            # split by model
            if train_frac + val_frac > 1.0:
                print('Training and validation fraction must be less than 1.0!')
                exit()

            train_inds = np.arange(int(train_frac*num_models))
            # print(train_inds)
            val_inds = np.arange(train_inds[-1]+1, train_inds[-1]+1 + int(val_frac*num_models))
            # print(val_inds)
            test_inds = np.arange(val_inds[-1]+1, num_models)
            # print(test_inds)

            split_inds = train_inds
            if split == 'val':
                split_inds = val_inds
            elif split == 'test':
                split_inds = test_inds

        else:
            split_inds = np.arange(len(nocs_seq_paths))

        print('Split size (num models): %d' % (split_inds.shape[0]))

        # only take the models from this split
        nocs_seq_paths = [nocs_seq_paths[i] for i in split_inds.tolist()]
        # combine all together
        for model_seq in nocs_seq_paths:
            all_nocs_seq_paths.extend(model_seq)

    print('Split size (num seqs): %d' % (len(all_nocs_seq_paths)))


    return all_nocs_seq_paths


def load_seq_path(seq_path_list, max_timestamp=DEFAULT_MAX_TIMESTAMP, expected_num_pts=4096):
    '''
    Given a list of data files making up a sequence, loads in all data for the sequence.
    '''
    pc_seq_files = seq_path_list

    seq_len = len(pc_seq_files)
    if seq_len == 1:
        step_size = 0.0
    else:
        step_size = 1.0 / (seq_len-1)

    # print(seq_path_list)

    # load data for this sequence
    nocs_seq = np.zeros((seq_len, expected_num_pts, 4)) # x,y,z,t
    depth_seq = np.zeros((seq_len, expected_num_pts, 4)) # x,y,z,t
    pose_seq = np.zeros((seq_len, 4, 4)) # 4x4 transformation matrices
    for step_idx, pc_file in enumerate(pc_seq_files):
        pc_data = np.load(pc_file)
        nocs_pc = pc_data['nocs_data']
        depth_pc = pc_data['depth_data']
        pose = pc_data['obj_T']

        # account for missing data (from i.e. warping cars data)
        if depth_pc.size == 0:
            # if we don't have any depth data, use NOCS point cloud as input
            depth_pc = nocs_pc
        if pose.size == 0:
            pose = np.zeros((4,4))

        # print(nocs_pc.shape)
        # print(depth_pc.shape)
        # print(nocs_pc)
        # print(depth_pc)
        if np.count_nonzero(nocs_pc) == 0:
            # has a blank frame, don't use this sequence
            # print('BLANK FRAME')
            break

        if nocs_pc.shape[0] < expected_num_pts:
            # print('Needs padding: %d' % (nocs_pc.shape[0]))
            # need to pad end to get the expected point cloud size
            pad_size = expected_num_pts - nocs_pc.shape[0]
            while pad_size > 0:
                nocs_pc = np.concatenate([nocs_pc, nocs_pc[:pad_size].reshape((-1,3))], axis=0)
                depth_pc = np.concatenate([depth_pc, depth_pc[:pad_size].reshape((-1,3))], axis=0)
                pad_size = expected_num_pts - nocs_pc.shape[0]
        
        pose_seq[step_idx] = pose

        # add time stamp min 0, max 1 based on number of steps
        time_stamp = np.ones((nocs_pc.shape[0], 1))*step_size*step_idx # NOCS time (0, 1)
        nocs_pc = np.concatenate([nocs_pc, time_stamp], axis=1)
        nocs_seq[step_idx] = nocs_pc
        # world time (0, max_time)
        time_stamp = max_timestamp*np.ones((depth_pc.shape[0], 1))*step_size*step_idx
        depth_pc = np.concatenate([depth_pc, time_stamp], axis=1)
        depth_seq[step_idx] = depth_pc

    return nocs_seq, depth_seq, pose_seq


class DynamicPCLDataset(Dataset):
    '''
    Dataset of point cloud sequences
    '''

    def __init__(self,  data_cfg, 
                        split='train',
                        train_frac=0.8, 
                        val_frac=0.1, 
                        num_pts=1024,
                        seq_len=5,
                        shift_time_to_zero=False,
                        random_point_sample=True,
                        random_point_sample_per_step=False):
        '''
        - data_cfg                      : path to dataset configuration file
        - split                         : "train", "test", or "val"
        - train_frac                    : percentage of dataset to use for training (if a splits file is not given)
        - val_frac                      : percentage of dataset to use for validation (if a splits file is not given)
        - num_pts                       : number of points to sample for input/output of TNOCS at each time step
        - seq_len                       : length of sequences to sample. Cannot be more than the sequence length of the data.
        - shift_time_to_zero            : if True, will shift all the timestamps to zero in the returned sequences. For example
                                              if the initial samplest steps are [0.2, 0.6, 0.7] the returned stamps would be
                                             shifted to [0.0, 0.4, 0.5] so that it starts at 0.
        - random_point_sample           : if True, randomly chooses the set of points returned at each timestep. if Fals, the first num_pts are returned.
        - random_point_sample_per_step  : if True, chooses a different sampling at each time step (only matters for data where points correspond over time, e.g warping cars)
        '''
        data_args = parse_dataset_cfg(data_cfg)
        print(data_args)
        self.data_paths = data_args.data
        self.split_paths = data_args.splits
        self.data_seq_len = data_args.expected_seq_len
        self.expected_num_pts = data_args.expected_num_pts
        self.max_timestamp = data_args.max_timestamp

        self.split = split
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.num_pts = num_pts
        self.seq_len = seq_len
        self.shift_time_to_zero = shift_time_to_zero
        self.random_point_sample = random_point_sample
        self.random_point_sample_per_step = random_point_sample_per_step

        # optional data to return
        self.return_pose_data = False
        self.return_first_steps = False # If true, returns the first seq_len steps in the sequence instead of random sampling

        if self.split not in ['train', 'test', 'val']:
            print('Split %s is not a valid option. Choose train, test, or val.' % (split))
            exit()

        print('Expected sequence length is set to %d!!' % (self.data_seq_len))

        # gets the file path to all sequences to load for this split
        self.seq_data_paths = load_time_data(self.data_paths, 
                                            self.split, 
                                            self.train_frac,
                                            self.val_frac,
                                            self.split_paths,
                                            data_seq_len=self.data_seq_len)
        self.data_len = len(self.seq_data_paths)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        '''
        Returns list of tuples of (input, output) for TNOCS regression and reconstruction training.

        The input is the world point cloud sequence B x T x N x 4
        and the output is the corresponding TNOCS point cloud sequence B x T x N x 4.
        '''
        # load in the data
        model_id = self.seq_data_paths[idx][0].split('/')[-3]
        seq_id = self.seq_data_paths[idx][0].split('/')[-2]
        all_data = load_seq_path(self.seq_data_paths[idx], 
                                max_timestamp=self.max_timestamp,
                                expected_num_pts=self.expected_num_pts)
        full_nocs_seq, full_depth_seq, full_pose_seq = all_data

        # randomly subsample time steps to get the desired sequence length
        if self.return_first_steps:
            sampled_steps = np.arange(self.seq_len)
        else:
            sampled_steps = np.random.choice(full_nocs_seq.shape[0], self.seq_len, replace=False)
        
        sampled_steps = sorted(sampled_steps)
                
        if self.random_point_sample or self.random_point_sample_per_step:
            # randomly subsample the same points
            if self.random_point_sample:
                sampled_pts = np.random.choice(full_nocs_seq.shape[1], self.num_pts, replace=False)
            elif self.random_point_sample_per_step:
                sampled_pts = [np.random.choice(full_nocs_seq.shape[1], self.num_pts, replace=False) for i in range(full_nocs_seq.shape[0])]
                sampled_pts = np.stack(sampled_pts, axis=0)  
        else:
            sampled_pts = np.arange(self.num_pts)

        # for TNOCS
        if not self.random_point_sample_per_step:
            input_data = full_depth_seq[sampled_steps,:,:].copy()
            input_data = input_data[:,sampled_pts,:]
            output_data = full_nocs_seq[sampled_steps,:,:].copy()
            output_data = output_data[:,sampled_pts,:]
        else:
            time_inds = np.repeat(np.arange(sampled_pts.shape[0]), sampled_pts.shape[1])
            pt_inds = sampled_pts.reshape((-1))

            input_data = full_depth_seq[sampled_steps,:,:].copy()
            input_data = input_data[time_inds, pt_inds, :].reshape((sampled_pts.shape[0], sampled_pts.shape[1], -1))
            output_data = full_nocs_seq[sampled_steps,:,:].copy()
            output_data = output_data[time_inds,pt_inds,:].reshape((sampled_pts.shape[0], sampled_pts.shape[1], -1))

        if self.shift_time_to_zero:
            # subtract out the min time from the timestamps so it starts at 0
            input_data[:,:,-1] -= np.min(input_data[:,:,-1])
            output_data[:,:,-1] -= np.min(output_data[:,:,-1])

        # to torch
        input_data = torch.from_numpy(input_data.astype(np.float32))
        output_data = torch.from_numpy(output_data.astype(np.float32))
        cur_item = (input_data, output_data)

        output_list = [cur_item]

        if self.return_pose_data:
            pose_data = full_pose_seq[sampled_steps, :]
            output_list.append(pose_data)

        output_list.extend([model_id, seq_id])

        return tuple(output_list)

    def set_return_pose_data(self, return_pose):
        self.return_pose_data = return_pose

    def set_return_first_steps(self, return_first_steps):
        self.return_first_steps = return_first_steps