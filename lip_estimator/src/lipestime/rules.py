import operator

import torch
from torch import Tensor, nn


def get_linear_lip(weight: Tensor, *args, **kwargs)->float:
    r"""Spectral norm for lip-computation."""
    return torch.linalg.matrix_norm(weight, ord=2).item()

def get_conv2d_lip(weight: Tensor, input_shape: torch.Size)->float:
    r"""Leverage FFT for lip computation of convolution.

    Reference paper:
    `Sedghi Hanie et al. "The singular values of 
    convolutional layers." arXiv preprint arXiv:1805.10408 (2018).`
    """
    out_channels, in_channels, _, _ = weight.shape
    h,w = input_shape[-2], input_shape[-1]
    weight_fft = torch.fft.rfft2(weight, s=(h,w)).\
        permute(2, 3, 0, 1).\
        reshape(-1, out_channels, in_channels)
    return torch.linalg.svdvals(weight_fft).max().item()

def get_Linear_lip(module: nn.Module, *args, **kwargs)->float:
    r"""Return the lip of the module."""
    return get_linear_lip(module.weight.data)

def get_Conv2d_lip(module: nn.Module, input_shape: torch.Size, *args, **kwargs) -> float:
    r"""Return the lip of the Conv2d Module."""
    return get_conv2d_lip(module.weight.data, input_shape)

def get_BatchNorm2d_lip(module: nn.Module, *args, **kwargs) -> float:
    r"""Return the lip of the BatchNorm2d Module."""
    weight = module.weight.data / torch.sqrt(module.running_var.data +1e-5)
    return weight.abs().max().item()

# The Registry: Mapping torch modules to our functions
MODULE_RULES = {
    nn.Linear: get_Linear_lip,
    nn.Conv2d: get_Conv2d_lip,
    nn.BatchNorm2d: get_BatchNorm2d_lip,
    nn.ReLU: lambda *args: 1.0,
    nn.Sigmoid: lambda *args: 1.0,
    nn.Tanh: lambda *args: 1.0,
    nn.MaxPool2d: lambda *args: 1.0,
    nn.AdaptiveAvgPool2d: lambda *args: 1.0, # Theoretically is 1/k where k is the kernel size.
    nn.Flatten: lambda *args: 1.0,
    nn.Identity: lambda *args: 1.0
}

def get_module_lipfn(module: nn.Module):
    r"""Get the lip function for the given module."""
    for mod_type, lip_fn in MODULE_RULES.items():
        if isinstance(module, nn.AdaptiveAvgPool2d):
            print("Warning: Using default lip constant for AdaptiveAvgPool2d")
        if isinstance(module, mod_type):
            return lip_fn
    raise NotImplementedError(f"No lip function registered for module type {type(module)}")

# Registry for functional operations
FUNCTION_RULES = {
    torch.add: lambda *lips: sum(*lips),
    operator.add: lambda *lips: sum(*lips),
    torch.flatten: lambda l1: l1,
    torch.relu: lambda l1: l1,
    torch.cat: lambda *args: (sum(lip**2 for lip in args))**0.5
}

def get_function_lipfn(func):
    r"""Get the lip function for the given function."""
    for func_name, lip_fn in FUNCTION_RULES.items():
        if func == func_name:
            return lip_fn
    err_msg = f"No lip function registered for function {func}"
    raise NotImplementedError(err_msg)