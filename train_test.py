import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

# Train routine
def train(dataloader, model, loss_fn, optimizer, device='cpu'):
    
    train_loss = []
    train_acc  = []
        
    model.train()
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
            
        # Compute prediction loss
        y_hat = model(X.float())
        loss  = loss_fn(y_hat, y)
        acc   = (y_hat.argmax(1) == y).type(torch.float).sum().item()/len(X)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            train_loss.append(loss.item())
            train_acc.append(acc)
            
    return train_loss, train_acc

# Test routine
def test(dataloader, model, loss_fn, device='cpu'):
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    model.eval()
    
    test_loss = []
    correct   = 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
                        
            y_hat = model(X.float())
            loss  = loss_fn(y_hat, y)
            
            test_loss.append(loss.item())
            correct += (y_hat.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss = np.array(test_loss).mean()
    test_acc = correct/size
    
    return test_loss, test_acc