import torch
import numpy as np
from torch import nn

# def L1_loss(x,y,nc=10):
    
#     y = torch.nn.functional.one_hot(y,nc)
#     loss = torch.mean(torch.abs(x-y))
    
#     return loss

def MAE_loss(x,y,nc=10):
    
    # Estimated probabilities
    p = nn.Softmax(dim=1)(x)

    y = torch.nn.functional.one_hot(y,nc)
    loss = torch.mean(torch.abs(p-y))
    
    return loss

# def MSE_loss(x,y,nc=10):
    
#     y = torch.nn.functional.one_hot(y,nc)
#     loss = torch.mean((x-y)**2)
    
#     return loss

def MSE_loss(x,y,nc=10):
    
    # Estimated probabilities
    p = nn.Softmax(dim=1)(x)

    y = torch.nn.functional.one_hot(y,nc)
    loss = torch.mean((p-y)**2)
    
    return loss

def CE_loss(x,y):
    eps = 1e-7

    # Estimated probabilities
    pred = nn.Softmax(dim=1)(x)

    # Advanced indexing: https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html
    py   = torch.clamp(pred[range(y.shape[0]), y], min=eps)
    loss = torch.mean(-torch.log(py))

    return loss

# def CE_x4_loss(x,y):
#     eps = 1e-7

#     # Estimated probabilities
#     pred = nn.Softmax(dim=1)(x)

#     # Advanced indexing: https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html
#     py   = torch.clamp(pred[range(y.shape[0]), y], min=eps)
#     loss = torch.mean(-4*torch.log(py))

#     return loss

def FR_loss(x, y):
    eps = 1e-7
    
    # Estimated probabilities
    pred = nn.Softmax(dim=1)(x)

    # Advanced indexing: https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html
    py   = torch.clamp(pred[range(y.shape[0]), y], min=eps)
    arg  = torch.clamp(torch.sqrt(py), eps, 1-eps)
    loss = torch.mean((torch.acos(arg))**2)

    return loss

def H_loss(x, y):
    eps = 1e-7
    
    # Estimated probabilities
    pred = nn.Softmax(dim=1)(x)

    # Advanced indexing: https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html
    py   = torch.clamp(pred[range(y.shape[0]), y], min=eps)
    loss = torch.mean(2*(1-torch.sqrt(py)))

    return loss