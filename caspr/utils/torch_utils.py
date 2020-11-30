
import time

import torch

import numpy as np

def get_device():
    '''
    Returns GPU device if available, else CPU.
    '''
    gpu_device_str = 'cuda:0'
    device_str = gpu_device_str if torch.cuda.is_available() else 'cpu'
    if device_str == gpu_device_str:
        print('Using detected GPU!')
    else:
        print('No detected GPU...using CPU.')
    device = torch.device(device_str)
    return device

def torch_to_numpy(tensor_list):
    return [x.to('cpu').data.numpy() for x in tensor_list]

def torch_to_scalar(tensor_list):
    return [x.to('cpu').item() for x in tensor_list]

def load_weights(model, state_dict):
    '''
    Load weights for full model
    '''
    for k, v in state_dict.items():
        if k.split('.')[0] == 'module':
            # then it was trained with Data parallel
            print('Loading weights trained with DataParallel...')
            state_dict = {'.'.join(k.split('.')[1:]) : v for k, v in state_dict.items() if k.split('.')[0] == 'module'}
        break
    # 2. overwrite entries in the existing state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if len(missing_keys) > 0:
        print('WARNING: The following keys could not be found in the given state dict - ignoring...')
        print(missing_keys)
    if len(unexpected_keys) > 0:
        print('WARNING: The following keys were found in the given state dict but not in the current model - ignoring...')
        print(unexpected_keys)

def load_encoder_weights_from_full(model, state_dict):
    '''
    Given weights for the full TNOCS model, loads only those for pre-trained nocs encoder.
    '''
    for k, v in state_dict.items():
        if k.split('.')[0] == 'module':
            # then it was trained with Data parallel
            print('Loading weights trained with DataParallel...')
            state_dict = {'.'.join(k.split('.')[1:]) : v for k, v in state_dict.items() if k.split('.')[0] == 'module'}
        break
    # 1. filter out unnecessary keys
    state_dict = {'.'.join(k.split('.')[1:]) : v for k, v in state_dict.items() if k.split('.')[0] == 'encoder'}
    # 2. overwrite entries in the existing state dict
    model.encoder.load_state_dict(state_dict)
    return

def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params