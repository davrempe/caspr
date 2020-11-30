import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torchdiffeq import odeint_adjoint as odeint

class LatentODE(nn.Module):
    ''' 
    Continuous ODE representation
    '''

    def __init__(self, input_size=1024, 
                       hidden_size=1024, 
                       num_layers=2, 
                       nonlinearity=nn.Tanh, 
                       augment_size=0):
        '''
        input_size : dimension of latent state
        hidden_size : size of hidden state to use within the dynamics net
        num_layers : number of hidden layers in the dynamics net
        nonlinearity : the nonlinearity to use in the network. Avoid non-smooth 
                        non-linearities such as ReLU and LeakyReLU. Prefer non-linearities
                        with a theoretically unique adjoint/gradient such as Softplus.
        augment_size : the number of zeros to augment to the last dimension of the init state.
                        If > 0, the returned output will be input_size + augment_size in this dim.
        '''
        super(LatentODE, self).__init__()

        self.input_size = input_size
        self.augment_size = augment_size
        self.output_size = input_size + self.augment_size
        self.ode_func = DynamicsNet(input_size=self.output_size, hidden_size=hidden_size, 
                                    num_layers=num_layers, nonlinearity=nonlinearity)
        self.solver = ODESolver(self.ode_func, method='dopri5', rtol=1e-3, atol=1e-4)

        init_network_weights(self.ode_func)

    def get_output_size(self):
        return self.output_size

    def forward(self, z0, t):
        '''
        z0 - initial state, B x H tensor
        t - times to evaluate ODE, size T 1-D tensor. The initial time corresponding to z0
            should be the first element of this sequence and each time must be larger than 
            the previous time

        Return:
        pred_z - latent state found by ODE solver at the requested times, B x T x H tensor
        '''
        self.ode_func._num_evals.fill_(0)

        # make all times relative to first (in case t0 != 0)
        rel_t = t - t[0]

        # augment initial state
        aug_z0 = z0
        if self.augment_size > 0:
            cur_device = z0.device
            augment_tens = torch.zeros(z0.size()[0], self.augment_size, dtype=z0.dtype).to(cur_device)
            aug_z0 = torch.cat([z0, augment_tens], dim=1) # B x (H+A)

        pred_z = self.solver(aug_z0, rel_t)
        pred_z = pred_z.permute(1,0,2) # first dimension of returned pred_z is time so must permute        

        return pred_z

    def num_evals(self):
        return self.ode_func._num_evals.item()   


class ODESolver(nn.Module):
	def __init__(self, ode_func, method='dopri5', rtol=1e-4, atol=1e-5):
		super(ODESolver, self).__init__()

		self.method = method
		self.ode_func = ode_func
		self.rtol = rtol
		self.atol = rtol

		if not isinstance(self.ode_func, nn.Module):
			raise ValueError('ode_func is required to be an instance of nn.Module to use the adjoint method')

	def forward(self, z0, t):
		'''
        z0 - initial state, N-D tensor
        t - times to evaluate ODE at size T 1-D tensor. The initial time corresponding to z0
            should be the first element of this sequence and each time must be larger than 
            the previous time

		Returns:
		pred_z - latent state found by the solver, tensor T x N-D
        '''
		pred_z = odeint(self.ode_func, z0, t, rtol=self.rtol, atol=self.atol, method=self.method)
		return pred_z 


class DynamicsNet(nn.Module):
    ''' 
    Network that ODE solver uses to query gradient. For a given
    latent state z, returns the gradient dz/dt.
    '''

    def __init__(self, input_size=1024, hidden_size=1024, num_layers=2, nonlinearity=nn.Tanh):
        '''
        input_size : dimension of latent state inputs. This is also the output size of the dynamics net.
        hidden_size : size of hidden state to use within the dynamics net
        num_layers : number of hidden layers in the dynamics net
        nonlinearity : the nonlinearity to use in the network. Avoid non-smooth 
                       non-linearities such as ReLU and LeakyReLU. Prefer non-linearities
                       with a theoretically unique adjoint/gradient such as Softplus.
        '''
        super(DynamicsNet, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.output_size = self.input_size

        self.dynamics_net = self.build_net()

        self.register_buffer("_num_evals", torch.tensor(0.)) # to work with dataparallel

    def build_net(self):
        layers = [nn.Linear(self.input_size, self.hidden_size)]
        for i in range(self.num_layers):
            layers.append(self.nonlinearity())
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        layers.append(self.nonlinearity())
        layers.append(nn.Linear(self.hidden_size, self.output_size))
        return nn.Sequential(*layers)


    def forward(self, t, z):
        '''
        t - current time to query gradient, scalar
        z - current state to query gradient, B x H tensor
        '''
        self._num_evals += 1

        dzdt = self.dynamics_net(z)
        return dzdt

    def num_evals(self):
        return self._num_evals

def init_network_weights(net, std = 0.1):
	for m in net.modules():
		if isinstance(m, nn.Linear):
			nn.init.normal_(m.weight, mean=0, std=std)
			nn.init.constant_(m.bias, val=0)