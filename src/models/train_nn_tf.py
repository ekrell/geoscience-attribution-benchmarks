'''
This program is used to train a neural network 
given a set of samples and targets.

- Samples file: 
  contains float array with shape (n_samples, rows, cols, bands).
- Targets file:
  contains float array with shape (n_samples).
'''

import math
import h5py
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import pandas as pd
from optparse import OptionParser
from sklearn.metrics import r2_score
import tensorflow as tf
import keras
import innvestigate
tf.compat.v1.disable_eager_execution()
from utils import get_valid_cells
import os

def split_train_valid(x, y, valid_frac):
  n_valid = int(-valid_frac*len(x))
  x_train = x[:n_valid]
  x_valid = x[n_valid:]
  y_train = y[:n_valid]
  y_valid = y[n_valid:]
  return x_train, x_valid, y_train, y_valid


def build_model(input_size, hidden_sizes):
  # Layers
  layers = [None for i in range(len(hidden_sizes) + 2)]
  # Input layer
  layers[0] = \
    keras.layers.Dense(hidden_sizes[0],
      input_shape=(int(input_size),), 
      activation='relu', 
      use_bias=True,
      kernel_initializer=keras.initializers.RandomNormal(stddev=1/input_size), 
      bias_initializer=keras.initializers.Zeros())
  # Hidden layers
  for hi in range(len(hidden_sizes) - 1):
    layers[hi + 1] = \
      keras.layers.Dense(hidden_sizes[hi + 1], 
        activation='relu', 
        use_bias=True,
        kernel_initializer='he_normal', 
        bias_initializer='he_normal')
  # Output layer
  layers[-2] = \
    keras.layers.Dense(1,
    use_bias=False,
    kernel_initializer='he_normal')
  layers[-1] = keras.layers.Activation('linear')

  # Build
  model = keras.models.Sequential(layers)
  return model


def main():

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
         help="Path to save plot of loss curves.")
  parser.add_option("-c", "--loss_values_file",
         help="Path to save train and validation loss history.")
  parser.add_option("-o", "--metrics_file",
         help="Path to save CSV of metrics.")
  parser.add_option("-e", "--epochs",
         default=50, type="int",
         help="Number of training epochs.")
  parser.add_option("-b", "--batch_size",
         default=32, type="int",
         help="Batch size")
  parser.add_option("-l", "--learning_rate",
         default=0.02, type="float",
         help="Training learning rate.")
  parser.add_option("-v", "--validation_fraction",
         default="0.1", type="float",
         help="Fraction of samples to use as validation.")
  parser.add_option("-n", "--hidden_nodes",
         default="512,256,128,64,32,16",
         help="Comma-delimited list of hidden layer node sizes, from first to last hidden node (e.g. 512,128,32).")
  parser.add_option("-q", "--quiet",
         action="store_true",
         help="Suppress printing each training epoch.")
  parser.add_option(      "--load_trained",
         action="store_true",
         help="Load trained instead.")
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

  metrics_out_file = options.metrics_file
  plot_out_file = options.plot_loss_file

  hidden_sizes = \
    np.array(options.hidden_nodes.split(",")).astype("int")
  validation_fraction = options.validation_fraction
  batch_size = options.batch_size
  epochs = options.epochs
  learning_rate = options.learning_rate

  quiet = options.quiet
  load_trained = options.load_trained

  # ---------- #
  # Setup Data #
  # ---------- #

  # Load data
  samples = np.load(samples_npz_file)[samples_npz_varname]
  targets = np.load(targets_npz_file)[targets_npz_varname]
  n_samples, rows, cols, bands = samples.shape
  if targets.shape[0] != n_samples:
    print("The number of samples and targets don't match.\nExit...")
    exit(-2)

  # Reshape maps to vector
  samples = np.reshape(samples, (n_samples, rows * cols * bands))

  shuffle = True
  verbose = 2 

  # Subset only the valid (non-NaN) cells
  sample_cells, valid_idxs, n_valid_cells = get_valid_cells(samples)

  # Prepare data for training
  X_train, X_validation, Y_train, Y_validation = \
    split_train_valid(sample_cells, targets, validation_fraction)

  # Number of NN inputs
  input_size = X_train.shape[-1]

  print("\nDataset:")
  print("  Training samples:   {}".format(X_train.shape))
  print("  Validation samples: {}\n".format(X_validation.shape))

  # ------------- #
  # Build Network #
  # ------------- #
   
  model = build_model(input_size, hidden_sizes)
  model.compile(
    optimizer=keras.optimizers.legacy.SGD(learning_rate=learning_rate),
    loss = 'mean_squared_error',  # MSE loss
    metrics=[keras.metrics.mean_absolute_error], # Track MAE
  )
  model.summary()

  # ------------- #
  # Train Network #
  # ------------- #

  if not load_trained:
    # Train
    history = model.fit(
      X_train, Y_train,
      validation_data=(X_validation, Y_validation),
      batch_size=batch_size,
      epochs=epochs,
      shuffle=shuffle,
      verbose=verbose)

    # Write model
    model.save(model_out_file)

    # Write loss history
    df_loss = pd.DataFrame({
      'train_loss': history.history["loss"],
      'valid_loss': history.history["val_loss"]})
    df_loss.to_csv(loss_out_file, index=False)

    # Plot convergence curves
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="valis loss")
    plt.title("Loss history")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    if plot_out_file is not None:
      plt.savefig(plot_out_file)
    else:
      plt.show() 

  else:
    # Load pretrained
    # Useful to quickly evaluate a dataset
    model = keras.models.load_model(model_out_file)

  # -------------- #
  # Evaluate Model #
  # -------------- #
  def get_metrics(model, x, y):
    preds = model.predict(x)
    loss, mae = model.evaluate(x, y, verbose=2)
    r2 = r2_score(y, preds)
    return preds, loss, mae, r2

  train_preds, train_loss, train_mae, train_r2 = \
    get_metrics(model, X_train, Y_train)
  valid_preds, valid_loss, valid_mae, valid_r2 = \
    get_metrics(model, X_validation, Y_validation)

  print("\nPerformance:")
  print("  Training:")
  print("    R2: {:4f},  MSE: {:6f},  MAE: {:6f}".format(
    train_r2, train_loss, train_mae))
  print("  Validation:")
  print("    R2: {:4f},  MSE: {:6f},  MAE: {:6f}".format(
    valid_r2, valid_loss, valid_mae))
  print("")

  # Write metrics
  if metrics_out_file is not None:
    types = ["training", "validation"]
    dfMetrics = pd.DataFrame({
      "dataset"  : ["training", "validation"], 
      "r-square" : [train_r2, valid_r2],
      "mse"      : [train_loss, valid_loss],
      "mae"      : [train_mae, valid_mae],
    })
    dfMetrics.to_csv(metrics_out_file, index=False)


if __name__ == "__main__":
  main()
