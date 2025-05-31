# circuit_tracer/utils/device.py
import torch

def get_default_device() -> torch.device:
    """Smart device detection - CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # MPS is not supported for sparse tensors
        #return torch.device("mps") 
        return torch.device("cpu")
    else:
        return torch.device("cpu")