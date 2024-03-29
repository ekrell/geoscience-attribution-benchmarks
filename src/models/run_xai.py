'''
This program is used to explain a model by applying multiple XAI methods
to generate local explanations in the form of an attribution map. That is,
each grid cell of a gridded raster input is assigned a value that represents
the contribution of that cell toward the model output.
'''

import numpy as np
import torch
from optparse import OptionParser
import matplotlib.pyplot as plt
from captum.attr import Occlusion, IntegratedGradients, LRP, DeepLift, \
                        FeaturePermutation, Lime, Saliency, KernelShap, \
                        InputXGradient
from nn import MLP
from utils import get_valid_cells
import matplotlib.pyplot as plt

def xai_integrated_gradients(X, model):
  ig = IntegratedGradients(model)
  X.requires_grad_()
  attrib = ig.attribute(X)
  attrib = attrib.detach().numpy()
  return attrib

def xai_saliency(X, model):
  saliency = Saliency(model)
  X.requires_grad_()
  attrib = saliency.attribute(X)
  return attrib

def xai_input_x_gradient(X, model):
  input_x_gradient = InputXGradient(model)
  X.requires_grad_()
  attrib = input_x_gradient.attribute(X)
  attrib = attrib.detach().numpy()
  return attrib

def xai_lrp(X, model):
  lrp = LRP(model)
  X.requires_grad_()
  attrib = lrp.attribute(X)
  attrib = attrib.detach().numpy()
  return attrib

xai_methods = {
  "integrated_gradients" : xai_integrated_gradients,
  "saliency" : xai_saliency,
  "input_x_gradient" : xai_input_x_gradient,
  "lrp" : xai_lrp,
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
                    help="Path to trained pytorch model.")
  parser.add_option("-i", "--indices",
                    help="Comma-delimited list of sample indices to include in XAI runs.")
  parser.add_option("-x", "--xai_method",
                    default="integrated_gradients",
                    help="Select which XAI method to use.")
  parser.add_option("-a", "--attributions_file",
                    help="Path to save computed attributions.")
  (options, args) = parser.parse_args()

  print("Supported XAI methods:")
  for method in xai_methods:
    print("  - {}".format(method))
  print("")

  xai_method = options.xai_method
  if xai_method not in xai_methods:
    print("Could not find xai method '{}'.\nExiting...".format(xai_method))
    exit(-1)
  else:
    print("Select method: {}".format(xai_method))
  print("")

  samples_file = options.samples_file
  samples_varname = options.samples_varname
  if samples_file is None:
    print("Expected input '.npz' file with samples ('-s').\nExiting...")
    exit(-1)

  model_file = options.model_file 
  if model_file is None:
    print("Expected input file with trained pytorch model ('-m').\nExiting...")
    exit(-1)

  attrib_file = options.attributions_file
  if attrib_file is None:
    print("Expected output file to save attributions ('-a').\nExiting...")
    exit(-1)

  plot_dir = options.plot_directory
  indices = options.indices

  # Load model data
  kwargs, state = torch.load(model_file)
  # Initialize model using saved arguments
  model = MLP(**kwargs)
  # Set model weights using saved state
  model.load_state_dict(state)

  # Load samples
  samples = np.load(samples_file)[samples_varname]
  # Subset samples
  if indices is not None:
    indices = np.array(indices.split(",")).astype("int")
    samples = samples[indices]
  else:
    indices = range(samples.shape[0])
  # Get shapes
  n_samples, rows, cols, bands = samples.shape
  # Reshape maps to vector
  samples = np.reshape(samples, (n_samples, rows * cols * bands))
  # Subset only the valid (non-NaN) cells
  sample_cells, valid_idxs, n_valid_cells = get_valid_cells(samples)
  # Convert to torch arrays
  X = torch.from_numpy(sample_cells.astype(np.float32))

  print("Input data:  {} samples.".format(n_samples))
  print("      shape: {}x{}x{}.".format(rows, cols, bands))

  # Run selected XAI method
  attrib = xai_methods[xai_method](X, model)

  # Reshape from vectors to maps
  attrib_maps = samples.copy()
  attrib_maps[:,valid_idxs] = attrib
  attrib_maps = np.reshape(attrib_maps, (n_samples, rows, cols, bands))

  # Write attributions
  np.savez(attrib_file, attributions=attrib, attribution_maps=attrib_maps)


if __name__ == "__main__":
  main()
