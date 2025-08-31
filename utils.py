import numpy as np
import torch

if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

def select_device(device=""):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
