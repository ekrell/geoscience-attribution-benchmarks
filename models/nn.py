'''
A simple Pytorch neural network.
'''

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class Data(Dataset):
  # Convert data to torch tensors
  def __init__(self, X, y):
    self.X = torch.from_numpy(X.astype(np.float32))
    self.y = torch.from_numpy(y.astype(np.float32))
    self.len = self.X.shape[0]

  def __getitem__(self, index):
    return self.X[index], self.y[index]

  def __len__(self):
    return self.len


class MLP(nn.Module):
  def __init__(self, input_size, layers_data: list):
    super().__init__()

    self.layers = nn.ModuleList()
    self.input_size = input_size

    self.kwargs = {'input_size' : input_size,
                   'layers_data' : layers_data,
    }

    for size, activation in layers_data:
      self.layers.append(nn.Linear(input_size, size))
      input_size = size
      self.layers.append(activation)
    self.layers.append(nn.Linear(input_size, 1))

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

