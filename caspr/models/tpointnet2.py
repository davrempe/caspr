import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet import PointNetfeat
from .pointnet2 import PointNet2feat as PointNet2

class TPointNet2(nn.Module):
    '''
    TPointNet++
    Extracts an initial z0 feature based on the given sequence and regresses TNOCS points.
    '''
    def __init__(self, radii_list=[0.02, 0.05, 0.1, 0.2, 0.4, 0.8], 
                        local_feat_size=512,  # size of the PointNet++ features
                        out_feat_size=1600,   # size of the output latent feature size from this model.
                        augment_quad=True,  # whether to augment quadratic terms to input of PointNet++ (x^2, y^x, z^2)
                        augment_pairs=True, # whether to augment pairwise multiplied terms (xy, xz, yz)
                        tnocs_point_size=4, 
                        regress_tnocs=True): # if true, regresses TNOCS points in addition to extracting latent feature
        super(TPointNet2, self).__init__()

        self.augment_quad = augment_quad 
        self.augment_pairs = augment_pairs
        self.tnocs_point_size = tnocs_point_size

        # PointNet++ feat output size
        self.local_feat_size =  local_feat_size
        self.local_bottleneck_size = self.local_feat_size

        # PointNet feat size
        self.global_feat_size = 1024
        self.space_time_pt_feat = 64

        # out feature size
        self.latent_feat_size = out_feat_size
        
        # PointNet++
        in_features = 0 # by default we only use x,y,z as input
        if self.augment_quad:
            print('Augmenting quadratic terms to input of PointNet++!')
            in_features += 3 # add quadratic terms
        if self.augment_pairs:
            print('Augmenting pairwise terms to input of PointNet++!')
            in_features += 3 # add pairwise terms
        self.local_extract = PointNet2(in_features=in_features,
                                        num_classes=self.local_feat_size, # size of the output
                                        batchnorm=False, # will use groupnorm instead
                                        use_xyz_feature=True, # also uses the coordinate as a feature
                                        use_random_ball_query=False,
                                        radii_list=radii_list,
                                        max_feat_prop_size=self.local_bottleneck_size
                                        )

        # PointNet
        self.global_extract = PointNetfeat(input_dim=4, out_size=self.global_feat_size)

        # layers to get space-time feature
        per_point_out_size = self.global_feat_size + self.space_time_pt_feat + self.local_feat_size
        self.conv1 = torch.nn.Conv1d(per_point_out_size, per_point_out_size, 1)
        self.conv2 = torch.nn.Conv1d(per_point_out_size, self.latent_feat_size, 1)
        self.bn1 = nn.GroupNorm(16, per_point_out_size)
        self.bn2 = nn.GroupNorm(16, self.latent_feat_size)

        # regress TNOCS afterward
        self.regress_tnocs = regress_tnocs
        if self.regress_tnocs:
            self.conv3 = torch.nn.Conv1d(self.latent_feat_size, self.tnocs_point_size, 1) # output latent features besides just (x,y,z,t)
            self.loss_func = torch.nn.L1Loss(reduce=False)

    def forward(self, x):
        B, T, N, _ = x.size()
        
        # Global spatio-temporal feature
        # output is the per-point features concatenated with global feature
        global_input = x.view(B, T*N, 4).transpose(2, 1).contiguous()
        global_feat = self.global_extract(global_input)

        # Local spatial feature for each timestep
        spatial_in = x.view(B*T, N, 4)[:,:,:3] # only want spatial inputs
        local_in = spatial_in
        if self.augment_quad:
            # concat quadratic terms
            quad_terms = spatial_in*spatial_in
            local_in = torch.cat([spatial_in, quad_terms], axis=2)
        if self.augment_pairs:
            # concat pairwise mult terms
            xz = spatial_in[:,:,0:1] * spatial_in[:,:,2:3]
            xy = spatial_in[:,:,0:1] * spatial_in[:,:,1:2]
            yz = spatial_in[:,:,2:3] * spatial_in[:,:,1:2]
            local_in = torch.cat([local_in, xz, xy, yz], axis=2)

        local_feat = self.local_extract(local_in).view(B, T, N, -1)
        local_feat = local_feat.view(B, T*N, self.local_feat_size).transpose(2, 1).contiguous()

        # concat global and local features
        feat = torch.cat([local_feat, global_feat], dim=1)

        # process to get latent features output
        feat = F.relu(self.bn1(self.conv1(feat)))
        feat = self.bn2(self.conv2(feat))

        # further process to get TNOCS regression output
        tnocs_regression = None
        if self.regress_tnocs:
            tnocs_out = self.conv3(F.relu(feat)) 
            tnocs_regression = torch.sigmoid(tnocs_out[:,:self.tnocs_point_size,:])
            tnocs_regression = tnocs_regression.transpose(2,1).contiguous() # B x T*N x 4
            tnocs_regression = tnocs_regression.view(B, T, N, self.tnocs_point_size)

        # max-pool over point-wise latent features to gets single output feature
        feat_max_op = torch.max(feat, 2, keepdim=False)
        feat = feat_max_op[0]
        feat_max_inds = feat_max_op[1]

        return feat, tnocs_regression

    def loss(self, outputs, gt):
        '''
        Computes the loss for TNOCS regression given the outputs of the network compared to GT
        TNOCS values. Returns unreduces loss values (per-point)
        '''
        loss = self.loss_func(outputs, gt)
        return loss