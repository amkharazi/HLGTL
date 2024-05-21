# Author: A.M.Kharazi
# License: BSD 3 clause

import torch

def load_weight(model, weight_path):
    '''
    loads weights for a model

    ----------
    model : torch.nn.Module
        your pytorch model or layer

    weight_path : str
        path to your .pth saved weights file
    '''
    return model.load_state_dict(torch.load(weight_path))