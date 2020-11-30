from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from kaolin.models.PointNet2 import separate_xyz_and_features, PointNet2GroupingLayer, furthest_point_sampling, fps_gather_by_index, three_nn, three_interpolate 

import kaolin.cuda as ext
import kaolin.cuda.furthest_point_sampling

NUM_GROUPS = 16 # for group norm

class PointNet2feat(nn.Module):
    """Modified PointNet++ segmentation network to give per-point features.

    .. note::

        If you use this code, please cite the original paper in addition to Kaolin.
        
        .. code-block::

            @article{qi2017pointnet2,
                title = {PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
                author = {Qi, Charles R. and Yi, Li and Su, Hao and Guibas, Leonidas J.},
                year = {2017},
                journal={arXiv preprint arXiv:1706.02413},
            }

    Args:
        in_features (int): Number of features (not including xyz coordinates) in
            the input point cloud (default: 0).
        num_classes (int): Number of classes (for the classification
            task) (default: 2).
        batchnorm (bool): Whether or not to use batch normalization.
            (default: True)
        use_xyz_feature (bool): Whether or not to use the coordinates of the
            points as feature.
        use_random_ball_query (bool): Whether or not to use random sampling when
            there are too many points per ball.
        radii_list (List[float]): List of ball radii to use for MSG. There are 2 scales at each SA layer
                                  and the radii overlap, so i.e. by default the first layer has 0.05 and 0.1 scales,
                                  the second layer has 0.1 and 0.2, and so on.

    """

    def __init__(self,
                 in_features=0,
                 num_classes=2,
                 batchnorm=True,
                 use_xyz_feature=True,
                 use_random_ball_query=False,
                 radii_list=[0.02, 0.05, 0.1, 0.2, 0.4, 0.8],
                 max_feat_prop_size=512):

        super(PointNet2feat, self).__init__()

        if len(radii_list) != 6:
            print('Radii list must be length 6, not %d!' % (len(radii_list)))
            exit()

        self.set_abstractions = nn.ModuleList()

        self.set_abstractions.append(
            PointNet2SetAbstraction(
                num_points_out=1024,
                pointnet_in_features=in_features,
                pointnet_layer_dims_list=[
                    [16, 16, 32],
                    [32, 32, 64],
                ],
                radii_list=[radii_list[-6], radii_list[-5]],
                num_samples_list=[16, 32],
                batchnorm=batchnorm,
                use_xyz_feature=use_xyz_feature,
                use_random_ball_query=use_random_ball_query
            )
        )

        self.set_abstractions.append(
            PointNet2SetAbstraction(
                num_points_out=512,
                pointnet_in_features=self.set_abstractions[-1].get_num_features_out(
                ),
                pointnet_layer_dims_list=[
                    [32, 32, 64],
                    [32, 32, 64],
                ],
                radii_list=[radii_list[-5], radii_list[-4]],
                num_samples_list=[16, 32],
                batchnorm=batchnorm,
                use_xyz_feature=use_xyz_feature,
                use_random_ball_query=use_random_ball_query
            )
        )

        self.set_abstractions.append(
            PointNet2SetAbstraction(
                num_points_out=256,
                pointnet_in_features=self.set_abstractions[-1].get_num_features_out(
                ),
                pointnet_layer_dims_list=[
                    [64, 64, 128],
                    [64, 96, 128],
                ],
                radii_list=[radii_list[-4], radii_list[-3]],
                num_samples_list=[16, 32],
                batchnorm=batchnorm,
                use_xyz_feature=use_xyz_feature,
                use_random_ball_query=use_random_ball_query
            )
        )

        self.set_abstractions.append(
            PointNet2SetAbstraction(
                num_points_out=64,
                pointnet_in_features=self.set_abstractions[-1].get_num_features_out(
                ),
                pointnet_layer_dims_list=[
                    [128, 196, 256] if batchnorm else [128, 256, 256],
                    [128, 196, 256] if batchnorm else [128, 256, 256],
                ],
                radii_list=[radii_list[-3], radii_list[-2]],
                num_samples_list=[16, 32],
                batchnorm=batchnorm,
                use_xyz_feature=use_xyz_feature,
                use_random_ball_query=use_random_ball_query
            )
        )

        self.set_abstractions.append(
            PointNet2SetAbstraction(
                num_points_out=16,
                pointnet_in_features=self.set_abstractions[-1].get_num_features_out(
                ),
                pointnet_layer_dims_list=[
                    [256, 256, 512],
                    [256, 384, 512] if batchnorm else [256, 256, 512],
                ],
                radii_list=[radii_list[-2], radii_list[-1]],
                num_samples_list=[16, 32],
                batchnorm=batchnorm,
                use_xyz_feature=use_xyz_feature,
                use_random_ball_query=use_random_ball_query
            )
        )

        self.feature_propagators = nn.ModuleList()

        layer_dims = [max([max_feat_prop_size, num_classes])]*2
        self.feature_propagators.append(
            PointNet2FeaturePropagator(
                num_features=self.set_abstractions[-2].get_num_features_out(),
                num_features_prev=self.set_abstractions[-1].get_num_features_out(),
                layer_dims=layer_dims,
                batchnorm=batchnorm,
            )
        )

        layer_dims = [max([max_feat_prop_size, num_classes])]*2
        self.feature_propagators.append(
            PointNet2FeaturePropagator(
                num_features=self.set_abstractions[-3].get_num_features_out(),
                num_features_prev=self.feature_propagators[-1].get_num_features_out(
                ),
                layer_dims=layer_dims,
                batchnorm=batchnorm,
            )
        )

        layer_dims = [max([max_feat_prop_size // 2, num_classes])]*2
        self.feature_propagators.append(
            PointNet2FeaturePropagator(
                num_features=self.set_abstractions[-4].get_num_features_out(),
                num_features_prev=self.feature_propagators[-1].get_num_features_out(
                ),
                layer_dims=layer_dims,
                batchnorm=batchnorm,
            )
        )

        layer_dims = [max([max_feat_prop_size // 2, num_classes])]*2
        self.feature_propagators.append(
            PointNet2FeaturePropagator(
                num_features=self.set_abstractions[-5].get_num_features_out(),
                num_features_prev=self.feature_propagators[-1].get_num_features_out(
                ),
                layer_dims=layer_dims,
                batchnorm=batchnorm,
            )
        )

        layer_dims = [max([max_feat_prop_size // 4, num_classes])]*2
        self.feature_propagators.append(
            PointNet2FeaturePropagator(
                num_features=in_features,
                num_features_prev=self.feature_propagators[-1].get_num_features_out(
                ),
                layer_dims=layer_dims,
                batchnorm=batchnorm,
            )
        )

        final_dim = layer_dims[0]
        final_layer_modules = [
            module for module in [
                nn.Conv1d(
                    self.feature_propagators[-1].get_num_features_out(), final_dim, 1),
                nn.BatchNorm1d(final_dim) if batchnorm else nn.GroupNorm(NUM_GROUPS, final_dim),
                nn.ReLU(),
                # nn.Dropout(0.5),
                nn.Conv1d(final_dim, num_classes, 1)
            ] if module is not None
        ]
        self.final_layers = nn.Sequential(*final_layer_modules)

    def forward(self, points):
        """
        Args:
            points (torch.Tensor): shape = (batch_size, num_points, 3 + in_features)
                The points to perform segmentation on.

        Returns:
            (torch.Tensor): shape = (batch_size, num_points, num_classes)
                The score of each point being in each class.
                Note: no softmax or logsoftmax will be applied.
        """
        xyz, features = separate_xyz_and_features(points)

        xyz_list, features_list = [xyz], [features]

        for module in self.set_abstractions:
            xyz, features = module(xyz, features)
            xyz_list.append(xyz)
            features_list.append(features)

        target_index = -2
        for module in self.feature_propagators:
            features_list[target_index] = module(
                xyz_list[target_index],
                xyz_list[target_index + 1],
                features_list[target_index],
                features_list[target_index + 1])

            target_index -= 1

        return (self.final_layers(features_list[0])
                .transpose(1, 2)
                .contiguous())


################ ADAPTED KAOLIN LAYERS TO ADD GROUPNORM OPTION AND MAKE PARALLEL-FRIENDLY ####################################

class PointNet2SetAbstraction(nn.Module):
    """A single set-abstraction layer for the PointNet++ architecture.
    Supports multi-scale grouping (MSG).

    .. note::

        If you use this code, please cite the original paper in addition to Kaolin.
        
        .. code-block::

            @article{qi2017pointnet2,
                title = {PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
                author = {Qi, Charles R. and Yi, Li and Su, Hao and Guibas, Leonidas J.},
                year = {2017},
                journal={arXiv preprint arXiv:1706.02413},
            }

    Args:
        num_points_out (int|None): The number of output points.
            If None, group all points together.

        pointnet_in_features (int): The number of features to input into pointnet.
            Note: if use_xyz_feature is true, this value will be increased by 3.

        pointnet_layer_dims_list (List[List[int]]): The pointnet MLP dimensions list for each scale.
            Note: the first (input) dimension SHOULD NOT be included in each list,
            while the last (output) dimension SHOULD be included in each list.

        radii_list (List[float]|None): The grouping radius for each scale.
            If num_points_out is None, this value is ignored.

        num_samples_list (List[int]|None): The number of samples in each ball query for each scale.
            If num_points_out is None, this value is ignored.

        batchnorm (bool): Whether or not to use batch normalization.

        use_xyz_feature (bool): Whether or not to use the coordinates of the
            points as feature.

        use_random_ball_query (bool): Whether or not to use random sampling when
            there are too many points per ball.
    """

    def __init__(self,
                num_points_out,
                pointnet_in_features,
                pointnet_layer_dims_list,
                radii_list=None,
                num_samples_list=None,
                batchnorm=True,
                use_xyz_feature=True,
                use_random_ball_query=False):

        super(PointNet2SetAbstraction, self).__init__()

        if num_points_out is None:
            radii_list = [None]
            num_samples_list = [None]
        else:
            assert isinstance(radii_list, list) and isinstance(
                num_samples_list, list), 'radii_list and num_samples_list must be lists'

        assert (len(radii_list) == len(num_samples_list) == len(pointnet_layer_dims_list)), (
            'Dimension of radii_list ({}), num_samples_list ({}), pointnet_layer_dims_list ({}) must match'
            .format(len(radii_list), len(num_samples_list), len(pointnet_layer_dims_list)))

        self.num_points_out = num_points_out
        self.pointnet_layer_dims_list = pointnet_layer_dims_list
        # self.sub_modules = nn.ModuleList()
        self.grouper_modules = nn.ModuleList()
        self.pointnet_modules = nn.ModuleList()
        self.layers = []
        self.pointnet_in_channels = pointnet_in_features + \
            (3 if use_xyz_feature else 0)

        num_scales = len(radii_list)
        for i in range(num_scales):
            radius = radii_list[i]
            num_samples = num_samples_list[i]
            pointnet_layer_dims = pointnet_layer_dims_list[i]

            assert isinstance(pointnet_layer_dims, list), 'Each pointnet_layer_dims must be a list, got {} instead'.format(
                pointnet_layer_dims)
            assert len(
                pointnet_layer_dims) > 0, 'Each pointnet_layer_dims must have at least one element'

            grouper = PointNet2GroupingLayer(
                radius, num_samples, use_xyz_feature=use_xyz_feature,
                use_random_ball_query=use_random_ball_query)

            pointnet = PointNetFeatureExtractor(
                in_channels=self.pointnet_in_channels,
                feat_size=pointnet_layer_dims[-1],
                layer_dims=pointnet_layer_dims[:-1],
                global_feat=True,
                batchnorm=batchnorm,
                transposed_input=True
            )

            # Register sub-modules
            # self.sub_modules.append(grouper)
            # self.sub_modules.append(pointnet)
            self.grouper_modules.append(grouper)
            self.pointnet_modules.append(pointnet)

            self.layers.append(num_samples)

    def forward(self, xyz, features=None):
        """
        Args:
            xyz (torch.Tensor): shape = (batch_size, num_points_in, 3)
                The 3D coordinates of each point.

            features (torch.Tensor|None): shape = (batch_size, num_features, num_points_in)
                The features of each point.

        Returns:
            new_xyz (torch.Tensor|None): shape = (batch_size, num_points_out, 3)
                The new coordinates of the grouped points.
                If self.num_points_out is None, new_xyz will be None.

            new_features (torch.Tensor): shape = (batch_size, out_num_features, num_points_out)
                The features of each output point.
                If self.num_points_out is None, new_features will have shape:
                (batch_size, num_features_out)
        """
        batch_size = xyz.shape[0]

        new_xyz = None
        if self.num_points_out is not None:
            new_xyz_idx = furthest_point_sampling(xyz, self.num_points_out)
            new_xyz = fps_gather_by_index(
                xyz.transpose(1, 2).contiguous(), new_xyz_idx)
            new_xyz = new_xyz.transpose(1, 2).contiguous()

        new_features_list = []
        for i, num_samples in enumerate(self.layers):
            new_features = self.grouper_modules[i](xyz, new_xyz, features)
            # shape = (batch_size, num_points_out, self.pointnet_in_channels, num_samples)
            # if num_points_out is None:
            # shape = (batch_size, self.pointnet_in_channels, num_samples)

            if self.num_points_out is not None:
                new_features = new_features.view(-1,
                                                self.pointnet_in_channels, num_samples)

            # new_features = pointnet(new_features)
            new_features = self.pointnet_modules[i](new_features)

            # shape = (batch_size * num_points_out, feat_size)
            # if num_points_out is None:
            # shape = (batch_size, feat_size)

            if self.num_points_out is not None:
                new_features = new_features.view(
                    batch_size, self.num_points_out, -1).transpose(1, 2)
                # shape = (batch_size, feat_size, num_points_out)

            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1)
        # shape = (batch_size, num_features_out, num_points_out)
        # if num_points_out is None:
        # shape = (batch_size, num_features_out)

        return new_xyz, new_features

    def get_num_features_out(self):
        return sum([lst[-1] for lst in self.pointnet_layer_dims_list])

class PointNet2FeaturePropagator(nn.Module):
    """A single feature-propagation layer for the PointNet++ architecture.

    Used for segmentation.

    .. note::

        If you use this code, please cite the original paper in addition to Kaolin.
        
        .. code-block::

            @article{qi2017pointnet2,
                title = {PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
                author = {Qi, Charles R. and Yi, Li and Su, Hao and Guibas, Leonidas J.},
                year = {2017},
                journal={arXiv preprint arXiv:1706.02413},
            }

    Args:
        num_features (int): The number of features in the current layer.
            Note: this is the number of output features of the corresponding
            set abstraction layer.

        num_features_prev (int): The number of features from the previous
            feature propagation layer (corresponding to the next layer during
            feature extraction).
            Note: this is the number of output features of the previous feature
            propagation layer (or the number of output features of the final set
            abstraction layer, if this is the very first feature propagation
            layer)

        layer_dims (List[int]): Sizes of the MLP layer.
            Note: the first (input) dimension SHOULD NOT be included in the list,
            while the last (output) dimension SHOULD be included in the list.

        batchnorm (bool): Whether or not to use batch normalization.
    """

    def __init__(self, num_features, num_features_prev, layer_dims, batchnorm=True):
        super(PointNet2FeaturePropagator, self).__init__()

        self.layer_dims = layer_dims

        unit_pointnets = []
        in_features = num_features + num_features_prev
        for out_features in layer_dims:
            unit_pointnets.append(
                nn.Conv1d(in_features, out_features, 1))

            if batchnorm:
                unit_pointnets.append(nn.BatchNorm1d(out_features))
            else:
                unit_pointnets.append(nn.GroupNorm(NUM_GROUPS, out_features))

            unit_pointnets.append(nn.ReLU())
            in_features = out_features

        self.unit_pointnet = nn.Sequential(*unit_pointnets)

    def forward(self, xyz, xyz_prev, features=None, features_prev=None):
        """
        Args:
            xyz (torch.Tensor): shape = (batch_size, num_points, 3)
                The 3D coordinates of each point at current layer,
                computed during feature extraction (i.e. set abstraction).

            xyz_prev (torch.Tensor|None): shape = (batch_size, num_points_prev, 3)
                The 3D coordinates of each point from the previous feature
                propagation layer (corresponding to the next layer during
                feature extraction).
                This value can be None (i.e. for the very first propagator layer).

            features (torch.Tensor|None): shape = (batch_size, num_features, num_points)
                The features of each point at current layer,
                computed during feature extraction (i.e. set abstraction).

            features_prev (torch.Tensor|None): shape = (batch_size, num_features_prev, num_points_prev)
                The features of each point from the previous feature
                propagation layer (corresponding to the next layer during
                feature extraction).

        Returns:
            (torch.Tensor): shape = (batch_size, num_features_out, num_points)
        """
        num_points = xyz.shape[1]
        if xyz_prev is None:  # Very first feature propagation layer
            new_features = features_prev.expand(
                *(features.shape + [num_points]))

        else:
            dist, idx = three_nn(xyz, xyz_prev)
            # shape = (batch_size, num_points, 3), (batch_size, num_points, 3)
            inverse_dist = 1.0 / (dist + 1e-8)
            total_inverse_dist = torch.sum(inverse_dist, dim=2, keepdim=True)
            weights = inverse_dist / total_inverse_dist
            new_features = three_interpolate(features_prev, idx, weights)
            # shape = (batch_size, num_features_prev, num_points)

        if features is not None:
            new_features = torch.cat([new_features, features], dim=1)

        return self.unit_pointnet(new_features)

    def get_num_features_out(self):
        return self.layer_dims[-1]


class PointNetFeatureExtractor(nn.Module):
    r"""PointNet feature extractor (extracts either global or local, i.e.,
    per-point features).

    Based on the original PointNet paper:.

    .. note::

        If you use this code, please cite the original paper in addition to Kaolin.
        
        .. code-block::

            @article{qi2016pointnet,
              title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
              author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
              journal={arXiv preprint arXiv:1612.00593},
              year={2016}
            }

    Args:
        in_channels (int): Number of channels in the input pointcloud
            (default: 3, for X, Y, Z coordinates respectively).
        feat_size (int): Size of the global feature vector
            (default: 1024)
        layer_dims (Iterable[int]): Sizes of fully connected layers
            to be used in the feature extractor (excluding the input and
            the output layer sizes). Note: the number of
            layers in the feature extractor is implicitly parsed from
            this variable.
        global_feat (bool): Extract global features (i.e., one feature
            for the entire pointcloud) if set to True. If set to False,
            extract per-point (local) features (default: True).
        activation (function): Nonlinearity to be used as activation
                    function after each batchnorm (default: F.relu)
        batchnorm (bool): Whether or not to use batchnorm layers
            (default: True)
        transposed_input (bool): Whether the input's second and third dimension
            is already transposed. If so, a transpose operation can be avoided,
            improving performance.
            See documentation for the forward method for more details.

    For example, to specify a PointNet feature extractor with 4 linear
    layers (sizes 6 -> 10, 10 -> 40, 40 -> 500, 500 -> 1024), with
    3 input channels in the pointcloud and a global feature vector of size
    1024, see the example below.

    Example:

        >>> pointnet = PointNetFeatureExtractor(in_channels=3, feat_size=1024,
                                           layer_dims=[10, 20, 40, 500])
        >>> x = torch.rand(2, 3, 30)
        >>> y = pointnet(x)
        print(y.shape)

    """

    def __init__(self,
                 in_channels: int = 3,
                 feat_size: int = 1024,
                 layer_dims: Iterable[int] = [64, 128],
                 global_feat: bool = True,
                 activation=F.relu,
                 batchnorm: bool = True,
                 transposed_input: bool = False):
        super(PointNetFeatureExtractor, self).__init__()

        if not isinstance(in_channels, int):
            raise TypeError('Argument in_channels expected to be of type int. '
                            'Got {0} instead.'.format(type(in_channels)))
        if not isinstance(feat_size, int):
            raise TypeError('Argument feat_size expected to be of type int. '
                            'Got {0} instead.'.format(type(feat_size)))
        if not hasattr(layer_dims, '__iter__'):
            raise TypeError('Argument layer_dims is not iterable.')
        for idx, layer_dim in enumerate(layer_dims):
            if not isinstance(layer_dim, int):
                raise TypeError('Elements of layer_dims must be of type int. '
                                'Found type {0} at index {1}.'.format(
                                    type(layer_dim), idx))
        if not isinstance(global_feat, bool):
            raise TypeError('Argument global_feat expected to be of type '
                            'bool. Got {0} instead.'.format(
                                type(global_feat)))

        # Store feat_size as a class attribute
        self.feat_size = feat_size

        # Store activation as a class attribute
        self.activation = activation

        # Store global_feat as a class attribute
        self.global_feat = global_feat

        # Add in_channels to the head of layer_dims (the first layer
        # has number of channels equal to `in_channels`). Also, add
        # feat_size to the tail of layer_dims.
        if not isinstance(layer_dims, list):
            layer_dims = list(layer_dims)
        layer_dims.insert(0, in_channels)
        layer_dims.append(feat_size)

        self.conv_layers = nn.ModuleList()
        # if batchnorm:
        self.bn_layers = nn.ModuleList()

        for idx in range(len(layer_dims) - 1):
            self.conv_layers.append(nn.Conv1d(layer_dims[idx],
                                              layer_dims[idx + 1], 1))
            if batchnorm:
                self.bn_layers.append(nn.BatchNorm1d(layer_dims[idx + 1]))
            else:
                self.bn_layers.append(nn.GroupNorm(NUM_GROUPS, layer_dims[idx + 1]))

        # Store whether or not to use batchnorm as a class attribute
        self.batchnorm = batchnorm

        self.transposed_input = transposed_input

    def forward(self, x: torch.Tensor):
        r"""Forward pass through the PointNet feature extractor.

        Args:
            x (torch.Tensor): Tensor representing a pointcloud
                (shape: :math:`B \times N \times D`, where :math:`B`
                is the batchsize, :math:`N` is the number of points
                in the pointcloud, and :math:`D` is the dimensionality
                of each point in the pointcloud).
                If self.transposed_input is True, then the shape is
                :math:`B \times D \times N`.

        """
        if not self.transposed_input:
            x = x.transpose(1, 2)

        # Number of points
        num_points = x.shape[2]

        # By default, initialize local features (per-point features)
        # to None.
        local_features = None

        # Apply a sequence of conv-batchnorm-nonlinearity operations

        # For the first layer, store the features, as these will be
        # used to compute local features (if specified).
        # if self.batchnorm:
        x = self.activation(self.bn_layers[0](self.conv_layers[0](x)))
        # else:
        #     x = self.activation(self.conv_layers[0](x))
        if self.global_feat is False:
            local_features = x

        # Pass through the remaining layers (until the penultimate layer).
        for idx in range(1, len(self.conv_layers) - 1):
            # if self.batchnorm:
            x = self.activation(self.bn_layers[idx](
                self.conv_layers[idx](x)))
            # else:
                # x = self.activation(self.conv_layers[idx](x))

        # For the last layer, do not apply nonlinearity.
        # if self.batchnorm:
        x = self.bn_layers[-1](self.conv_layers[-1](x))
        # else:
            # x = self.conv_layers[-1](x)

        # Max pooling.
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.feat_size)

        # If extracting global features, return at this point.
        if self.global_feat:
            return x

        # If extracting local features, compute local features by
        # concatenating global features, and per-point features
        x = x.view(-1, self.feat_size, 1).repeat(1, 1, num_points)
        return torch.cat((x, local_features), dim=1)