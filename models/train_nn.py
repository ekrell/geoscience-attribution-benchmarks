'''
This program is used to train a neural network given a set of samples and targets.

There are two input files, one containing the samples and another with targets. 
Each a Numpy '.npz' file. The variables are accessed by name, so it can be the same file. 

  - Samples file: contains float array with shape (n_samples, rows, cols, bands).
  - Targets file: contains float array with shape (n_samples).

'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from optparse import OptionParser
import matplotlib.pyplot as plt

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


def load_npz_var(path, var_name):
  # Load '.npz' file and extract variable by name
  data = np.load(path)
  return data[var_name]


def get_valid_cells(arr):
  # Subset only the non-NaN cells
  valid_idxs = np.argwhere(~np.isnan(arr[0])).flatten()
  sample_cells = arr[:, valid_idxs]
  n_valid_cells = len(valid_idxs)
  return sample_cells, valid_idxs, n_valid_cells


def split_train_valid(x, y, valid_frac):
  n_valid = int(-valid_frac*len(x))
  x_train = x[:n_valid]
  x_valid = x[n_valid:]
  y_train = y[:n_valid]
  y_valid = y[n_valid:]
  return x_train, x_valid, y_train, y_valid


def main():
  # Options  
  parser = OptionParser()
  parser.add_option("-s", "--samples_file",
                    help="Path to '.npz' file with samples.")
  parser.add_option(      "--samples_varname",
                    default="samples",
                    help="Name of variable with raster samples in samples file.")
  parser.add_option("-t", "--targets_file",
                    help="Path to '.npz' file with targets.")
  parser.add_option(      "--targets_varname",
                    default="y",
                    help="Name of variable with targets in targets file.")
  parser.add_option("-m", "--model_file",
                    help="Path to save trained model.")
  parser.add_option("-p", "--plot_loss_file",
                    help="Path to save plot of train and validation loss curves.") 
  parser.add_option("-c", "--loss_values_file",
                    help="Path to save train and validation loss history.")
  parser.add_option("-e", "--epochs",
                    default=50,
                    type="int",
                    help="Number of training epochs.")
  parser.add_option("-b", "--batch_size",
                    default=32,
                    type="int",
                    help="Batch size")
  parser.add_option("-l", "--learning_rate",
                    default=0.02,
                    type="float",
                    help="Training learning rate.")
  parser.add_option("-v", "--validation_fraction",
                    default="0.1",
                    type="float",
                    help="Fraction of samples to use as validation.")
  parser.add_option("-n", "--hidden_nodes",
                    default="512,256,128,64,32,16",
                    help="Comma-delimited list of hidden layer node sizes, from first to last hidden node (e.g. 512,128,32).")
  (options, args) = parser.parse_args()


  samples_npz_file = options.samples_file
  samples_npz_varname = options.samples_varname
  if samples_npz_file is None:
    print("Expected input '.npz' file with samples ('-s').\nExiting...")
    exit(-1)
  
  targets_npz_file = options.targets_file
  targets_npz_varname = options.targets_varname
  if targets_npz_file is None:
    print("Expected input '.npz' file with targets ('-t').\nExiting...")
    exit(-1)

  model_out_file = options.model_file
  if model_out_file is None:
    print("Expected output '.pt' file to save trained model ('-m').\nExiting...")
    exit(-1)
  
  loss_out_file = options.loss_values_file
  if loss_out_file is None:
    print("Expected output '.csv' file to save train and validation loss history ('-c').\nExiting...")
    exit(-1)

  plot_out_file = options.plot_loss_file

  hidden_sizes = np.array(options.hidden_nodes.split(",")).astype("int")
  validation_fraction = options.validation_fraction
  batch_size = options.batch_size
  epochs = options.epochs
  learning_rate = options.learning_rate

  # Setup device
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:0" if use_cuda else "cpu")

  # Load data
  samples = load_npz_var(samples_npz_file, samples_npz_varname)
  targets = load_npz_var(targets_npz_file, targets_npz_varname)
  n_samples, rows, cols, bands = samples.shape
  if targets.shape[0] != n_samples:
    print("The number of samples does not match the number of targets.\nExiting...")
    exit(-2)

  # Reshape maps to vector
  samples = np.reshape(samples, (n_samples, rows * cols * bands))
  
  # Subset only the valid (non-NaN) cells
  sample_cells, valid_idxs, n_valid_cells = get_valid_cells(samples) 

  print("Input data:  {} samples.".format(n_samples))
  print("      shape: {}x{}x{}.".format(rows, cols, bands))
  print("Device: {}".format(device))
  print("Hyperparameters:")
  print("  epochs: {}".format(epochs))
  print("  batch size: {}".format(batch_size))
  print("  learning rate: {}".format(learning_rate))
  print("  validation fraction: {}".format(validation_fraction))
  print("  hidden layer sizes: {}".format(hidden_sizes))  

  # Separate into train, validation
  x_train, x_valid, y_train, y_valid = split_train_valid(sample_cells, targets, validation_fraction)

  # Prepate data for torch
  data_train = Data(x_train, y_train)
  data_valid = Data(x_valid, y_valid)
  dataloader_valid = DataLoader(dataset=data_valid, batch_size=batch_size, shuffle=True)
  dataloader_train = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
  n_train_batches = len(dataloader_train)
  n_valid_batches = len(dataloader_valid)

  # Model hyperparameters 
  optimizer = optim.Adam
  input_size = n_valid_cells
  output_size = 1

  # Generate a sequence of layers (hidden, activation)
  layers = [(hs, nn.ReLU()) for hs in hidden_sizes]

  class MLP(nn.Module):
    def __init__(self, input_size, layers_data: list):
      super().__init__()

      self.layers = nn.ModuleList()
      self.input_size = input_size
      
      for size, activation in layers_data:
        self.layers.append(nn.Linear(input_size, size))
        input_size = size
        self.layers.append(activation)
      self.layers.append(nn.Linear(input_size, 1))

    def forward(self, x):
      for layer in self.layers:
        x = layer(x)
      return x

  model = MLP(input_size, layers)

  loss_func = torch.nn.MSELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

  print(model)
  # Training loop
  loss_values = np.zeros((epochs, 2))
  for epoch in range(epochs):

    # Update weights
    loss_accum = 0
    for x, y in dataloader_train:
      x, y = x.to(device), y.to(device)
      optimizer.zero_grad()
      pred = model(x)
      loss = loss_func(pred, y.unsqueeze(-1))
      loss_accum += loss.item()
      loss.backward()
      optimizer.step()
    loss_values[epoch, 0] = loss_accum / n_train_batches

    # Validation
    loss_accum = 0
    for x, y in dataloader_valid:
      x, y = x.to(device), y.to(device)
      pred = model(x)
      loss = loss_func(pred, y.unsqueeze(-1))
      loss_accum += loss.item()
    loss_values[epoch, 1] = loss_accum / n_valid_batches

    print("Epoch {}/{}.  training loss: {},   validation loss: {}".format(
      epoch + 1, epochs, loss_values[epoch,0], loss_values[epoch,1]))

  # Write model
  torch.save(model.state_dict(), model_out_file)

  # Write loss history
  import pandas as pd
  df_loss = pd.DataFrame(
    {'train_loss': loss_values[:,0],
     'valid_loss': loss_values[:,1]})
  df_loss.to_csv(loss_out_file, index=False)
  loss_out_file

  # Plot convergence curves
  plt.plot(loss_values[:,0], label="train loss")
  plt.plot(loss_values[:,1], label="valid loss")
  plt.title("Loss history")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend()
  if plot_out_file is not None:
    plt.savefig(plot_out_file)
  else:
    plt.show()


if __name__ == "__main__":
  main()