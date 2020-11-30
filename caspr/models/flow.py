#
# Adapted from https://github.com/stevenygd/PointFlow
#

from .odefunc import ODEfunc, ODEnet
from .normalization import MovingBatchNorm1d
from .cnf import CNF, SequentialFlow
from .latent_ode_model import LatentODE

def count_nfe(model):
    class AccNumEvals(object):

        def __init__(self):
            self.num_evals = 0

        def __call__(self, module):
            if isinstance(module, CNF) or isinstance(module, LatentODE):
                self.num_evals += module.num_evals()

    accumulator = AccNumEvals()
    model.apply(accumulator)
    return accumulator.num_evals


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_time(model):
    class Accumulator(object):

        def __init__(self):
            self.total_time = 0

        def __call__(self, module):
            if isinstance(module, CNF):
                self.total_time = self.total_time + module.sqrt_end_time * module.sqrt_end_time

    accumulator = Accumulator()
    model.apply(accumulator)
    return accumulator.total_time


def build_model(args, input_dim, hidden_dims, context_dim, num_blocks, conditional):
    def build_cnf():
        diffeq = ODEnet(
            hidden_dims=hidden_dims,
            input_shape=(input_dim,),
            context_dim=context_dim,
            layer_type=args.layer_type,
            nonlinearity=args.nonlinearity,
        )
        odefunc = ODEfunc(
            diffeq=diffeq,
        )
        cnf = CNF(
            odefunc=odefunc,
            T=args.time_length,
            train_T=args.train_T,
            conditional=conditional,
            solver=args.solver,
            use_adjoint=args.use_adjoint,
            atol=args.atol,
            rtol=args.rtol
        )
        return cnf

    chain = [build_cnf() for _ in range(num_blocks)]
    if args.batch_norm:
        bn_chain = [MovingBatchNorm1d(input_dim)]
        chain = bn_chain + chain
        chain += [MovingBatchNorm1d(input_dim)]
        
    model = SequentialFlow(chain, use_bn=args.batch_norm)

    return model


def get_point_cnf(args):
    dims = tuple(map(int, args.dims.split("-")))
    model = build_model(args, args.input_dim, dims, args.zdim, args.num_blocks, True).cuda()
    print("Number of trainable parameters of Point CNF: {}".format(count_parameters(model)))
    return model


class PointCNFArgs():
    def __init__(self):
        self.input_dim = 3 # Number of input dimensions (3 for 3D point clouds)
        self.dims = "512-512-512" # hidden dims of ode net (will be this many +1 (input layer))
        self.zdim = 512 # Dimension of the shape code
        self.num_blocks = 1 # Number of stacked CNFs
        self.layer_type = 'concatsquash' # type of layer to use in ode func
        self.nonlinearity = 'softplus' # nonlinearity in ode func
        self.time_length = 0.5 # if not training final time of flow, what to use
        self.train_T = True # whether to train time of ending flow
        self.solver = 'dopri5' # ODE solver to use
        self.use_adjoint = True # use adjoint method
        self.atol = 1e-5 # solver tolerences
        self.rtol = 1e-5
        self.batch_norm = True # whether to use moving batch norm