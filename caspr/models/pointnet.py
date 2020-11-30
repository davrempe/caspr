'''
This PointNet implementation was adapted from:
https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py

'''


from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class PointNetfeat(nn.Module):
    '''
    Simple PointNet that extracts point-wise feature by concatenating local and global features.
    Uses group norm instead of batch norm.
    '''
    def __init__(self, input_dim=3, out_size=1024, layer_sizes=[64, 128]):
        super(PointNetfeat, self).__init__()
        self.output_size = out_size
        self.input_dim = input_dim
        self.conv1 = torch.nn.Conv1d(self.input_dim, layer_sizes[0], 1)
        self.conv2 = torch.nn.Conv1d(layer_sizes[0], layer_sizes[1], 1)
        self.conv3 = torch.nn.Conv1d(layer_sizes[1], self.output_size, 1)
        self.bn1 = nn.GroupNorm(16, layer_sizes[0])
        self.bn2 = nn.GroupNorm(16, layer_sizes[1])
        self.bn3 = nn.GroupNorm(16, self.output_size)

    def forward(self, x):
        n_pts = x.size()[2]

        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        max_op = torch.max(x, 2, keepdim=True)
        x = max_op[0]
        max_inds = max_op[1]
        x = x.view(-1, self.output_size, 1).repeat(1, 1, n_pts)
    
        return torch.cat([x, pointfeat], 1)