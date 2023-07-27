"""
This program is used to generate a benchmark function given a set of samples. 
The spatial relationships of the input rasters are used to define the weights
of a piece-wise linear function (PWL). The PWL is created so that the 
contribution (or, attribution) of each raster cell toward to sample's output
can be derived. By knowing the exact attribution of each cell, it can be used
as a benchmark for assessing the accuracy of XAI methods.

The extends code by Dr. Antonios Mamalakis:
 `Neural-Network-Attribution-Benchmark-for-Regression`
   (github.com/amamalak/Neural-Network-Attribution-Benchmark-for-Regression)
"""

import os
import numpy as np
from scipy.stats import norm
from optparse import OptionParser
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def gen_pwl(samples, n_breaks):
  """
  Function that generates a piece-wise linear function and attribution maps 
  based on a set of samples of raster data. The purpose is to create a benchmark
  function where there is a known attribution of each raster cell towards the 
  function output.
 
  Parameters
  ----------
  samples : 4D Numpy float array
    Set of samples of raster data with shape (n_samples, rows, cols, bands).
  n_breaks : Number of breakpoints in the piece-wise linear function

  Returns
  -------
  y : 1D Numpy float array
    Output of the function for each sample with shape (n_samples).
  attrib_maps : 4D Numpy float array
    Attribution maps with shape (n_samples, rows, cols, bands) 
    where each raster element's value is that element's contribution
    towards the sample's value in 'y'.
  weights : 2D Numpy float array
    The weights of the piece-wise linear function. Each raster cell has
    its own set of weights. There are weight for each bin in the PWL. 
    So the shape is (n_samples, n_breaks + 1).
  edges :  2D Numpy float array
    The locations of the function breakpoints. The shape is
    (n_samples, n_breaks + 2). The +2 is because we add "-INF" and "INF"
    to the first and last edges. 
  """

  # Get data sizes
  n_samples, rows, cols, bands = samples.shape

  # Reshape each (rows, cols, bands)-shaped sample to vector
  samples = np.reshape(samples, (n_samples, rows * cols * bands))

  # Subset only the valid (non-NaN) cells
  valid_idxs = np.argwhere(~np.isnan(samples[0])).flatten()
  sample_cells = samples[:, valid_idxs]
  n_valid_cells = len(valid_idxs)

  # Calculate correlation coefficient. This is used to generate random weights
  # so that there is spatial structure related to the original data between the 
  # elements even though the function itself treats each element independently.  
  cov = np.cov(sample_cells, rowvar=False)

  # Init PWL weights
  weights = np.ones((n_valid_cells, n_breaks + 1))
  # Init ground truth attribution maps
  attrib = np.ones((n_samples, n_valid_cells))

  # Make random PWL breakpoint weights (spatially correlated)
  for ki in range(n_breaks + 1):
    weights[:, ki] = np.random.multivariate_normal(
        np.zeros(n_valid_cells), # Breakpoint values for every cell
        cov,                     # Spatially related via covariance
        1)                       # One single breakpoint value (ki)

  # Random break points based on the spatial structure embedded in X
  # Thus, each pixel's breakpoints are spatially dependent
  breaks = np.random.multivariate_normal(
      np.zeros(n_valid_cells),   # Breakpoint values for every cell
      cov,                       # Spatially related via covariance
      n_breaks - 1)              # All but one breakpoints, other=0
  # Convert to probability to make between 0 - 1
  breaks = norm.cdf(breaks, 0, 1)

  # Calculate attribution and response for each sample at each valid cell
  for cell_idx in range(n_valid_cells):
    # Init attribution values
    attrib[:, cell_idx] = 0
    # Get cell values
    x = sample_cells[:, cell_idx]

    # Trivial case: if single breakpoint -> linear model
    if n_breaks == 1:
      # Not using breakpoints, so no need to define edges
      edges = None
      # Cell's attribution is weight * sample value
      attrib[:, cell_idx] = weights[cell_idx, 0] * x

    # Otherwise, multiple breakpoint case -> piece-wise linear
    else:
      # Set PWL breakpoint edges (each pixel's Ci has own edges)
      # These edges are via the emperical distribution of that pixel's values
      ecdf = ECDF(x)
      f1 = ecdf.y
      y1 = ecdf.x
      # For each breakpoint probability, get the x-value with that probability
      # Based on the emperical distribution of this pixel's value distribution
      l = interp1d(f1, y1)(breaks[:, cell_idx])
      edges = np.zeros(n_breaks + 2)
      # One edge must be -INF
      edges[0] = -1 * np.inf
      # One edge must be INF
      edges[1] = np.inf
      # One edge must be zero
      edges[2] = 0.0
      # Other edges are the random breaks
      edges[3:] = np.sort(l)
      # Sort from small to large to organize edges
      edges = np.sort(edges)
      edges = np.unique(edges)
      n_edges = len(edges)
      n_edges_leqzero = len(edges[edges <= 0])

      # Place each x value in a bin based on edges
      x_bin_idxs = np.digitize(x, edges) 
      # Make 0-based count for indices instead of 1-based
      x_bin_idxs = x_bin_idxs - 1

      # For each segment of the PWL function
      #for bin_idx in range(n_edges - 1):
      #  # Get the samples that fall into that segment (bin)
      #  sample_idxs = np.where(x_bin_idxs == bin_idx)

      #  attrib[sample_idxs, cell_idx] = weights[cell_idx, bin_idx] * x[sample_idxs] 

      for sample_idx in range(n_samples):
        value = x[sample_idx]
        bin_idx = x_bin_idxs[sample_idx]
 
        if x[sample_idx] > 0:
          while edges[bin_idx] >= 0:
            attrib[sample_idx, cell_idx] += weights[cell_idx, bin_idx] \
                                         * (value - edges[bin_idx])
            value = edges[bin_idx]
            bin_idx = bin_idx - 1

        if x[sample_idx] <= 0:
          while edges[bin_idx] < 0:
            attrib[sample_idx, cell_idx] += weights[cell_idx, bin_idx] \
                                         * (edges[bin_idx + 1] - value)
            value = edges[bin_idx + 1]
            bin_idx = bin_idx + 1

  #print(sample_cells.shape)
  #print(attrib.shape)
  #plt.scatter(sample_cells[:,0], attrib[:, 0])
  #plt.show()
  #exit(0)

  # Compute output y value for each sample
  y = np.zeros(n_samples)
  for i in range(n_samples):
    # Function output is sum of each cell's PWL output
    y[i] = np.nansum(attrib[i, :])

  # Place the attribution values on the map
  attrib_ = np.reshape(attrib, (n_samples, n_valid_cells))
  # Copy the original samples so that we have the NaN cells
  attrib_maps = samples.copy()
  # Replace sample values with attribution values
  attrib_maps[:, valid_idxs] = attrib_

  # Reshape from set of vectors to set of (rows, cols, bands) rasters
  attrib_maps = np.reshape(attrib_maps, (n_samples, rows, cols, bands))
  samples = np.reshape(samples,  (n_samples, rows, cols, bands))

  # Normalize based on number of valid cells
  y = y / len(valid_idxs)
  attrib[:] = attrib[:] / len(valid_idxs)

  return y, sample_cells, attrib, attrib_maps, weights, edges


# Options
parser = OptionParser()
parser.add_option("-f", "--samples_file",
                  help="Path to '.npz' file with benchmark samples.")
parser.add_option(      "--samples_varname",
                  default="samples",
                  help="Name of variable with raster samples in samples file.")
parser.add_option("-k", "--breakpoints",
                  default=5,
                  type="int",
                  help="Number of breakpoints in piece-wise linear function.")
parser.add_option("-o", "--output_file",
                  help="Path to '.npz' to write output data.")
parser.add_option("-p", "--plot_idxs",
                  help="Comma-delimited list of sample indices to plot.")
parser.add_option(     "--plot_cell_idxs",
                  help="Comma-delimited list of cell's PWL functions to plot.")
(options, args) = parser.parse_args()

samples_file = options.samples_file
if samples_file is None:
  print("Expected path to '.npz' file with raster samples.\nExiting...")
  exit(-1)
output_file = options.output_file
if output_file is None:
  print("Expected path to write a '.npz' file with generated function and attribution maps.\nExiting...")
  exit(-1)

samples_varname = options.samples_varname
n_breaks = options.breakpoints
plot_idxs = options.plot_idxs
n_plots = 0
if plot_idxs is not None:
  plot_idxs = np.array(plot_idxs.split(",")).astype("int")
  n_plots = len(plot_idxs)

plot_cell_idxs = options.plot_cell_idxs
n_cell_plots = 0
if plot_cell_idxs is not None:
  plot_cell_idxs = np.array(plot_cell_idxs.split(",")).astype("int")
  n_cell_plots = len(plot_cell_idxs)

# Load samples
samples_npz = np.load(samples_file)
samples = samples_npz[samples_varname]

# Generate piece-wise linear function and attribution maps
y, sample_cells, attrib, attrib_maps, weights, edges= gen_pwl(samples, n_breaks)

if n_plots > 0:
  fig, axs = plt.subplots(2, n_plots, figsize=(n_plots * 5, 6), squeeze=False)
  for i in range(n_plots):
    axs[0,i].imshow(samples[i])
    axs[1,i].imshow(attrib_maps[i], cmap="bwr")
  plt.tight_layout()
  plt.show()

if n_cell_plots > 0:
  fig, axs = plt.subplots(1, n_cell_plots, figsize=(n_cell_plots * 5, 6), squeeze=False)
  for i in range(n_cell_plots):
    axs[0,i].scatter(sample_cells[:, plot_cell_idxs[i]], attrib[:, plot_cell_idxs[i]])
  plt.tight_layout()
  plt.show()

# Write results
print("Writing function and attribution maps to: {}.".format(output_file))
np.savez(output_file, 
  y=y,                       # Function outputs for each sample
  attributions=attrib_maps,  # Attribution maps for each sample
  weights=weights,           # Weights of each PWL bin
  edges=edges)               # Edges that define the PWL bins 
