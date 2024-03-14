'''
This program is used to explain a model using XAI methods to
generate local explanations in the form of an attribution map.
Each grid cell of a gridded raster input is assigned a value
that represents the contribution of that cell toward the output.
'''


import numpy as np
import h5py
import matplotlib.pyplot as plt
from optparse import OptionParser
import tensorflow as tf
import keras
import innvestigate
tf.compat.v1.disable_eager_execution()
from utils import get_valid_cells

xai_methods = {
  "integrated_gradients"  : "integrated_gradients",
  "saliency"              : "gradient",
  "input_x_gradient"      : "input_t_gradient",
  "lrp",                  : "lrp.z",
}


def main():

  # Options
  parser = OptionParser()
  parser.add_option("-s", "--samples_file",
         help="Path to '.npz' with samples.")
  parser.add_option(      "--samples_varname",
         default="samples",
         help="Name of variable with raster samples in samples file.")
  parser.add_option("-m", "--model_file",
         help="Path to trained tensorflow model ('.h5').")
  parser.add_option("-i", "--indices",
         help="Comma-delimited list of sample indices to explain.")
  parser.add_option("-x", "--xai_method",
         default="integrated_gradients",
         help="Select which XAI method to use.")
  parser.add_option("-a", "--attributions_file",
         help="Path to save computed attributions.")
  (options, args) = parser.parse_args()

  print("\nSupported XAI methods:")
  for method in xai_methods:
    print("  - {}".format(method))
  print("")

  xai_method = options.xai_method
  if xai_method not in xai_methods:
    print("Could not find xai method '{}'.\nExiting...".format(xai_method))
    exit(-1)
  else:
    print("Selected method: {}".format(xai_method))
  print("")
  
  samples_file = options.samples_file
  samples_varname = options.samples_varname
  if samples_file is None:
    print("Expected input '.npz' file with samples ('-s').\nExiting...")
    exit(-1)

  model_file = options.model_file
  if model_file is None:
    print("Expected input file with trained tensorflow model ('-m').\nExiting...")
    exit(-1)

  attrib_file = options.attributions_file
  if attrib_file is None:
    print("Expected output file to save attributions ('-a').\nExiting...")
    exit(-1)

  indices = options.indices

  # Load model
  model = keras.models.load_model(model_file)

  # Load samples
  samples = np.load(samples_file)[samples_varname]
  if indices is not None:
    indices = np.array(indices.split(",")).astype("int")
  else:
    indices = range(samples.shape[0])
  # Subset 
  samples = samples[indices]
  # Get shapes
  n_samples, rows, cols, bands = samples.shape 
  # Reshape maps to vector
  samples = np.reshape(samples, (n_samples, rows * cols* bands))
  # Subset only the valid (non-NaN) cells
  sample_cells, valid_idxs, n_valid_cells = get_valid_cells(samples)

  print("Input data:  {} samples.".format(n_samples))
  print("      shape: {}x{}x{}.".format(rows, cols, bands))

  # Run selected XAI method
  analyzer = innvestigate.create_analyzer(xai_methods[xai_method], model)
  attribs = analyzer.analyze(sample_cells)

  attrmaps = np.zeros((n_samples, rows * cols * bands))
  attrmaps[:, valid_idxs] = attribs[0]
  attrmaps = np.reshape(attrmaps, 
    (n_samples, rows, cols, bands))
  

if __name__ == "__main__":
  main()
