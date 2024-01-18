import torch
import pandas as pd
import numpy as np


def first_index(*args):
    if len(args) > 1:
        idx0 = []
        for arg in args:
            if isinstance(arg, pd.Series):
                idx0.append(arg.first_valid_index())
            else:
                idx0.append(0)
    else:
        if isinstance(args, pd.Series):
            idx0 = args[0].first_valid_index()
        else:
            idx0 = 0
    return idx0


def array_to_tensor(*args):
    tensors = torch.empty(0)
    for arg in args:
        if isinstance(arg, pd.Series):
            tensors = torch.cat((tensors,torch.tensor(arg.values).unsqueeze(0)))
        elif isinstance(arg,torch.Tensor):
            tensors = torch.cat((tensors,arg.unsqueeze(0)))
        else:
            tensors = torch.cat((tensors,torch.tensor(arg).unsqueeze(0)))
    return tensors.squeeze(0)