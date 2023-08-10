'''
A simple Pytorch neural network.
'''

import torch.nn as nn

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

