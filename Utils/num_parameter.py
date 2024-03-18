import torch
import tltorch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_linear_parameters(model):
    return sum(p.numel() for p in model.parameters() if isinstance(p, torch.nn.Linear))

def count_tcl_parameters(model):
    tcl_parameters = sum(p.numel() for p in model.parameters() if isinstance(p, tltorch.TCL))
    tcl_layers = sum(1 for _ in model.children() if isinstance(_, tltorch.TCL))
    return tcl_parameters, tcl_layers

def count_trl_parameters(model):
    trl_parameters = sum(p.numel() for p in model.parameters() if isinstance(p, tltorch.TRL))
    trl_layers = sum(1 for _ in model.children() if isinstance(_, tltorch.TRL))
    return trl_parameters, trl_layers

def count_tcl_trl_parameters(model):
    tcl_parameters, tcl_layers = count_tcl_parameters(model)
    trl_parameters, trl_layers = count_trl_parameters(model)
    return tcl_parameters, trl_parameters, tcl_parameters + trl_parameters, tcl_layers, trl_layers, 
