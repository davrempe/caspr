'''
argparse options for training, testing, and visualization scripts.
'''

def get_general_options(parser):
    '''
    Adds general options to the given argparse parser.
    These are options that are shares across train, test, and visualization time.
    '''
    # General
    parser.add_argument('--num-workers', type=int, default=2, help='for data loaders')

    # Output
    parser.add_argument('--out', type=str, default='./train_out', help='Directory to save model weights and logs to.')

    # Dataset
    parser.add_argument('--data-cfg', type=str, required=True, help='.cfg for the dataset to use')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for training.')
    parser.add_argument('--seq-len', type=int, default=5, help='Number of time steps to sample in each sequence that is used as input to the model.')
    parser.add_argument('--num-pts', type=int, default=1024, help='Number of point cloud points to sample for each step in the sequence that is used as input to the model.')

    # Input augmentation
    parser.add_argument('--no-augment-quad', dest='augment_quad', action='store_false', help="If given, won't augment raw input data with quadratic terms.")
    parser.set_defaults(augment_quad=True)
    parser.add_argument('--no-augment-pairs', dest='augment_pairs', action='store_false', help="If given, won't augment raw input data with pairwise multiplicative terms (xy, yz, xz).")
    parser.set_defaults(augment_pairs=True)

    # Model options
    parser.add_argument('--pretrain-tnocs', dest='pretrain_tnocs', action='store_true', help="If true, uses only TNOCS regression part of the model for training/testing.")
    parser.set_defaults(pretrain_tnocs=False)
    parser.add_argument('--weights', type=str, default='', help='Path to full model weights to start training from or load in for testing.')
    parser.add_argument('--radii', type=float, nargs='+', default=[0.02, 0.05, 0.1, 0.2, 0.4, 0.8], help='Radii list to use if using PointNet++ w/ PointNet+')
    parser.add_argument('--local-feat-size', type=int, default=512, help='Feature size of PointNet++ in PointNet+.')
    parser.add_argument('--cnf-blocks', type=int, default=1, help='Number of normalizing flow blocks to use.')
    parser.add_argument('--latent-feat-size', type=int, default=1600, help='Latent state size to extract from the encoder.')
    parser.add_argument('--ode-hidden-size', type=int, default=512, help='Hidden state size in dynamics network of LatentODE')
    parser.add_argument('--motion-feat-size', type=int, default=64, help='The size of the part of the feature that should be given to the latent ODE.')
    parser.add_argument('--no-regress-tnocs', dest='regress_tnocs', action='store_false', help="If given, will not regress or supervise TNOCS, instead using only the reconstruction loss.")
    parser.set_defaults(regress_tnocs=True)

    # Loss function options
    parser.add_argument('--cnf-loss', type=float, default=0.01, help='Weight for NLL loss')
    parser.add_argument('--tnocs-loss', type=float, default=100.0, help='Weight for TNOCS regression loss')

    return parser


def get_train_options(parser):
    '''
    Options specific to training time.
    '''

    # gpu
    parser.add_argument('--parallel', dest='use_parallel', action='store_true', help="If given, will use all available GPUs in parallel. Otherwise uses device 0 by default.")
    parser.set_defaults(use_parallel=False)

    # Training options
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training.')
    parser.add_argument('--val-every', type=int, default=3, help='Number of epochs between validations.')
    parser.add_argument('--save-every', type=int, default=10, help='Number of epochs between saving model checkpoint.')
    parser.add_argument('--print-every', type=int, default=10, help='Number of batches between printing stats.')

    # Optimizer options
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for ADAM')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for ADAM')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for ADAM')
    parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon rate for ADAM')
    parser.add_argument('--decay', type=float, default=0.0, help='Weight decay on params (except the dynamics network).')

    return parser

def get_test_options(parser):
    '''
    Options specific to testing time.
    '''

    # Output
    parser.add_argument('--log', type=str, default='test_log.txt', help='Name of the log to save - will be saved to the output directory.')

    # Dataset
    parser.add_argument('--shuffle-test', dest='shuffle_test', action='store_true', help="Shuffles ordering of test dataset.")
    parser.set_defaults(shuffle_test=False)

    # Different kinds of evaluations to run (any number of them can be given)
    parser.add_argument('--eval-test', dest='eval_full_test', action='store_true', help="Evaluate on the full test set with same metrics as during training.")
    parser.set_defaults(eval_full_test=False)
    parser.add_argument('--eval-shape-recon-observed', dest='eval_shape_recon_observed', action='store_true', help="evaluate shape reconstruction where all 10 steps of the sequence are given as input")
    parser.set_defaults(eval_shape_recon_observed=False)
    parser.add_argument('--eval-shape-recon-unobserved', dest='eval_shape_recon_unobserved', action='store_true', help="evaluate shape reconstruct when sparse set of steps is given as input and evaluates performance on both seen and unseen steps")
    parser.set_defaults(eval_shape_recon_unobserved=False)
    parser.add_argument('--eval-tnocs-regression', dest='eval_tnocs_regression', action='store_true', help="evaluate only the TNOCS regression")
    parser.set_defaults(eval_tnocs_regression=False)
    parser.add_argument('--eval-pose-observed-ransac', dest='eval_pose_observed_ransac', action='store_true', help="Perform and evaluate pose estimation at observed time steps using TNOCS estimates.")
    parser.set_defaults(eval_pose_observed_ransac=False)
    parser.add_argument('--show-pose-viz', dest='show_pose_viz', action='store_true', help="If doing pose estimation evaluation, show visualization of each result.")
    parser.set_defaults(show_pose_viz=False)

    return parser


def get_viz_options(parser):
    '''
    Options specific to visualization time.
    '''

    # Dataset
    parser.add_argument('--shuffle-test', dest='shuffle_test', action='store_true', help="Shuffles ordering of test dataset.")
    parser.set_defaults(shuffle_test=False)

    # Visualizations
    parser.add_argument('--viz-tnocs', dest='viz_tnocs', action='store_true', help="Visualizes TNOCS regression for each test sequence.")
    parser.set_defaults(viz_tnocs=False)
    parser.add_argument('--viz-observed', dest='viz_observed', action='store_true', help="Visualizes observed timestep reconstruction using CaSPR for each test sequence.")
    parser.set_defaults(viz_observed=False)
    parser.add_argument('--viz-interpolated', dest='viz_interpolated', action='store_true', help="Visualizes interpolated reconstruction using CaSPR for each test sequence.")
    parser.set_defaults(viz_interpolated=False)

    # options related to all
    parser.add_argument('--no-input-seq', dest='show_input_seq', action='store_false', help="If given, will not visualize the raw depth input sequence along with GT and predictiongs.")
    parser.set_defaults(show_input_seq=True)
    parser.add_argument('--no-nocs-cubes', dest='show_nocs_cubes', action='store_false', help="If given, will not visualize the NOCS cubes.")
    parser.set_defaults(show_nocs_cubes=True)

    # options related to tnocs
    parser.add_argument('--tnocs-err-map', dest='tnocs_error_map', action='store_true', help="If given, colors TNOCS predictions with an error map rather than TNOCS RGB.")
    parser.set_defaults(tnocs_error_map=False)

    # options related to observed/interp
    parser.add_argument('--num-sampled-pts', type=int, default=2048, help='Number of points to sample from CaSPR at each step of the sequence.')
    parser.add_argument('--num-sampled-steps', type=int, default=30, help='Number of timesteps to sample from CaSPR for interpolated reconstruction.')
    parser.add_argument('--no-constant', dest='constant_in_time', action='store_false', help="If given, samples random points from CaSPR at each step, rather than using the same base gaussian sampling for all timesteps.")
    parser.set_defaults(constant_in_time=True)
    parser.add_argument('--no-base-samples', dest='show_base_sampling', action='store_false', help="If given, will not visualize the sampling from the base distribution along with the CaSPR sampling.")
    parser.set_defaults(show_base_sampling=True)
    # affects coloring of visualization, can only choose one of these
    parser.add_argument('--sample-contours', dest='sample_contours', action='store_true', help="If true, CaSPR samples are taken from specific contours from the base gaussian rather than randomly sampled.")
    parser.set_defaults(sample_contours=False)
    parser.add_argument('--base-color-map', dest='base_color_map', action='store_true', help="If true, visualizes base and CaSPR samples using the location in the base gaussian rather than location in the NOCS cube.")
    parser.set_defaults(base_color_map=False)
    parser.add_argument('--prob-color-map', dest='prob_color_map', action='store_true', help="If true, visualizes base and CaSPR samples using the log probability in the base gaussian rather than location in the NOCS cube.")
    parser.set_defaults(prob_color_map=False)

    return parser