#
# Adapted from https://github.com/stevenygd/PointFlow
#

from math import log, pi

import torch
import numpy as np

def standard_normal_logprob(z):
    log_z = -0.5 * log(2 * pi)
    return log_z - z.pow(2) / 2

# Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
def truncated_normal(tensor, mean=0, std=1, trunc_std=2):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

def sample_gaussian(size, truncate_std=None, device=None):
    y = torch.randn(*size).float()
    y = y if device is None else y.to(device)
    if truncate_std is not None:
        truncated_normal(y, mean=0, std=1, trunc_std=truncate_std)
    return y