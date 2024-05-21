# Author: A.M.Kharazi
# License: BSD 3 clause
# Counts the number of parameters in a nn.Module object
def count_parameters(model):
    '''
    Counts the number of parameters

    ----------
    model : torch.nn.Module
        your pytorch model or layer
    '''
    return sum(p.numel() for p in model.parameters())
