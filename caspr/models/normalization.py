#
# Adapted from https://github.com/stevenygd/PointFlow
#

import torch
import torch.nn as nn
from torch.nn import Parameter

__all__ = ['MovingBatchNorm1d']


class MovingBatchNormNd(nn.Module):
    def __init__(self, num_features, eps=1e-4, decay=0.1, affine=True):
        super(MovingBatchNormNd, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.decay = decay
        self.register_buffer('step', torch.zeros(1))
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        self.reset_parameters()

    @property
    def shape(self):
        raise NotImplementedError

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.zero_()
            self.bias.data.zero_()

    def update_running_mean(self, x):
        num_channels = x.size(-1)
        # compute batch statistics
        x_t = x.transpose(0, 1).reshape(num_channels, -1)
        batch_mean = torch.mean(x_t, dim=1)
        batch_var = torch.var(x_t, dim=1)
        self.running_mean -= self.decay * (self.running_mean - batch_mean.data)
        self.running_var -= self.decay * (self.running_var - batch_var.data)
        self.step += 1

    def forward(self, x, c=None, logpx=None, reverse=False):
        if reverse:
            return self._reverse(x, logpx)
        else:
            return self._forward(x, logpx)

    def _forward(self, x, logpx=None):
        used_mean = self.running_mean.clone().detach()
        used_var = self.running_var.clone().detach()
        
        if self.training:
            self.update_running_mean(x)

        # perform normalization
        used_mean = used_mean.view(*self.shape).expand_as(x)
        used_var = used_var.view(*self.shape).expand_as(x)

        y = (x - used_mean) * torch.exp(-0.5 * torch.log(used_var + self.eps))
        if self.affine:
            weight = self.weight.view(*self.shape).expand_as(x)
            bias = self.bias.view(*self.shape).expand_as(x)
            y = y * torch.exp(weight) + bias

        if logpx is None:
            return y
        else:
            log_out =  logpx - self._logdetgrad(x, used_var).sum(-1, keepdim=True)
            return y, log_out

    def _reverse(self, y, logpy=None):
        num_channels = y.size(-1)
        used_mean = self.running_mean.clone().detach()
        used_var = self.running_var.clone().detach()

        used_mean = used_mean.view(*self.shape).expand_as(y)
        used_var = used_var.view(*self.shape).expand_as(y)

        if self.affine:
            weight = self.weight.view(*self.shape).expand_as(y)
            bias = self.bias.view(*self.shape).expand_as(y)
            y = (y - bias) * torch.exp(-weight)
        
        x = y * torch.exp(0.5 * torch.log(used_var + self.eps)) + used_mean

        if logpy is None:
            return x
        else:
            log_out = logpy + self._logdetgrad(x, used_var).sum(-1, keepdim=True)
            return x, log_out

    def _logdetgrad(self, x, used_var):
        logdetgrad = -0.5 * torch.log(used_var + self.eps)
        if self.affine:
            weight = self.weight.view(*self.shape).expand(*x.size())
            logdetgrad += weight
        return logdetgrad

    def __repr__(self):
        return (
            '{name}({num_features}, eps={eps}, decay={decay}'
            ' affine={affine})'.format(name=self.__class__.__name__, **self.__dict__)
        )


class MovingBatchNorm1d(MovingBatchNormNd):
    @property
    def shape(self):
        return [1, -1]

    def forward(self, x, context=None, logpx=None, integration_times=None, reverse=False):
        ret = super(MovingBatchNorm1d, self).forward(
                x, context, logpx=logpx, reverse=reverse)
        if logpx is None:
            return ret

        return ret