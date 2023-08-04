"""
This program is used to generate a benchmark function given a set of samples.
The spatial relationships of the input rasters are used to define the weights
of a piece-wise linear function (PWL). The PWL is created so that the
contribution (or, attribution) of each raster cell toward to sample's output
can be derived. By knowing the exact attribution of each cell, it can be used
as a benchmark for assessing the accuracy of XAI methods.

The generated PWL has the following properties:
  (1) Continuous  (no jumps between pieces).
  (2) f(0) = 0.

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

def pwl_build(samples, n_breaks):
  '''
  Generates a PWL function for each grid cell based on:
    (1) Covariance between cells (maintain spatial relations)
    (2) Distribution of data at each cell (breakpoints are quantiles)
  
  Parameters
  ----------
  samples : 4D Numpy float array
    Set of samples of raster data with shape (n_samples, rows, cols, bands).
  n_breaks : int
    Number of breakpoints in the piece-wise linear function.

  Returns
  -------
  weights : 2D Numpy float array
    The weights of the piece-wise linear function. Each raster cell has
    its own set of weights. There are weights for each bin in the PWL.
    So the shape is (n_cells, n_breaks + 1).
  edges :  2D Numpy float array
    The locations of the function breakpoints. The shape is
    (n_cells, n_breaks + 2). The +2 is because we add "-INF" and "INF"
    to the first and last edges.

  '''
  
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

  # Init PWL edges
  edges = np.ones((n_valid_cells, n_breaks + 2))
  # Init PWL weights
  weights = np.ones((n_valid_cells, n_breaks + 1))

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

  # For each cell, generate PWL edges
  for cell_idx in range(n_valid_cells):
    # Get cell values
    x = sample_cells[:, cell_idx]

    # Set PWL breakpoint edges (each pixel's Ci has own edges)
    # These edges are via the emperical distribution of that pixel's values
    ecdf = ECDF(x)
    f1 = ecdf.y
    y1 = ecdf.x
    # For each breakpoint probability, get the x-value with that probability
    # Based on the emperical distribution of this pixel's value distribution
    l = interp1d(f1, y1)(breaks[:, cell_idx])
    # Ensure no -INF
    while len(l[~np.isinf(l)]) == 0:
      #print(x)
      #print(f1)
      #print(y1)
      #print(breaks[:, cell_idx])
      #print(l)
      breaks[:, cell_idx] = 0.1
      l = interp1d(f1, y1)(breaks[:, cell_idx])
      #print(l)
    l[np.isinf(l)] = np.sort(l[~np.isinf(l)])[0] - 0.01
    # Ensure unique breakpoints
    while len(np.unique(l)) != len(l):
      l += np.random.uniform(0.05, 0.1, size=len(l))
    # One edge must be -INF
    edges[cell_idx, 0] = -1 * np.inf
    # One edge must be INF
    edges[cell_idx, 1] = np.inf
    # One edge must be zero
    edges[cell_idx, 2] = 0.0
    # Other edges are the random breaks
    edges[cell_idx, 3:] = np.sort(l)
    # Sort from small to large to organize edges
    edges[cell_idx] = np.sort(edges[cell_idx])
    n_edges = len(edges[cell_idx])

  return edges, weights


def pwl_eval(samples, edges, weights):
  '''
  Evaluates an additive PWL-based function for a set of raster samples,
  where each grid cell has its own PWL function. The output for the
  raster is the summation of the PWL function at each cell.

  The function is defined so that the contribution (or, attribution) of
  each grid cell toward the output is simply the PWL output for that cell.
  So, this function returns both the output values and attributions. 

  Parameters
  ----------
  samples: 4D Numpy float array
    Set of samples of raster data with shape (n_samples, rows, cols, bands).
  edges :  2D Numpy float array
    The locations of the function breakpoints. The shape is
    (n_cells, n_breaks + 2). The +2 is because we add "-INF" and "INF"
    to the first and last edges.
  weights : 2D Numpy float array
    The weights of the piece-wise linear function. Each raster cell has
    its own set of weights. There are weights for each bin in the PWL.
    So the shape is (n_cells, n_breaks + 1).

  Returns
  -------
  y : 1D Numpy float array
    The function output of every input sample. The shape is (n_samples).
  attrib: 2D Numpy float array
    Each sample has attributions for each valid (non-Nan) grid cell. 
    The shape is (n_samples, n_valid_cells).
  attrib_maps: 4D Numpy float array
    This is the same attribution values as 'attrib', but mapped to the shape 
    of the original samples. The shape is (n_samples, rows, cols, bands).
  '''

  # Get data sizes
  n_samples, rows, cols, bands = samples.shape

  # Reshape each (rows, cols, bands)-shaped sample to vector
  samples = np.reshape(samples, (n_samples, rows * cols * bands))

  # Subset only the valid (non-NaN) cells
  valid_idxs = np.argwhere(~np.isnan(samples[0])).flatten()
  sample_cells = samples[:, valid_idxs]
  n_valid_cells = len(valid_idxs)

  # Init ground truth attribution maps
  attrib = np.zeros((n_samples, n_valid_cells))
 
  # Calculate each cell's PWL output (attribution)
  for cell_idx in range(n_valid_cells):

    # Get cell values
    x = sample_cells[:, cell_idx]

    # Place each x value in a bin based on edges
    x_bin_idxs = np.digitize(x, edges[cell_idx])
    # Make 0-based count for indices instead of 1-based
    x_bin_idxs = x_bin_idxs - 1

    # Build the PWL s.t. it is a continuous function.
    # We don't simply multiply the value by the weight.
    # Instead we start from (0, 0) and evaluate the part 
    # of the value that falls within each bin, until reaching
    # the value
    for sample_idx in range(n_samples):
      value = x[sample_idx]
      bin_idx = x_bin_idxs[sample_idx]

      # Building positive (go down to zero)
      # Example: if x = 3 and the breakpoints are (2,  4)
      #  then y(x = 3) = y(x = 2) + w*(3-2)
      # So we loop until reaching breakpoint at x = 0
      if x[sample_idx] > 0:
        while edges[cell_idx, bin_idx] >= 0:
          attrib[sample_idx, cell_idx] += weights[cell_idx, bin_idx] \
                                       * (value - edges[cell_idx, bin_idx])
          value = edges[cell_idx, bin_idx]
          bin_idx = bin_idx - 1

      # Building negative (go up to 0)
      # Example: if x = -3 and the breakpoints are (-5, -1)
      #   then y(x = -3) = (y = -1) + w*(-3 + 1)
      # So we loop until reaching breakpoint at x = 0
      if x[sample_idx] <= 0:
        while edges[cell_idx, bin_idx] < 0:
          attrib[sample_idx, cell_idx] += weights[cell_idx, bin_idx] \
                                       * (edges[cell_idx, bin_idx + 1] - value)
          value = edges[cell_idx, bin_idx + 1]
          bin_idx = bin_idx + 1
 
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
  y = y
  attrib[:] = attrib[:]

  return y, attrib, attrib_maps


def plot_attribution_maps(attrib_maps, plot_idxs):
  '''

  '''
  n_samples, rows, cols, bands = attrib_maps.shape
  n_plots = len(plot_idxs)

  fig, axs = plt.subplots(n_plots, bands, 
             figsize=(bands * 3, n_plots * 3), squeeze=False)
  for i, idx in enumerate(plot_idxs):
    for b in range(bands):
      axs[i, b].imshow(attrib_maps[idx,:,:,b])
      axs[i, b].set_xticks([])
      axs[i, b].set_yticks([])

def plot_cell_functions(samples, attrib, plot_idxs):
  # Get data sizes
  n_samples, rows, cols, bands = samples.shape
  # Reshape each (rows, cols, bands)-shaped sample to vector
  samples = np.reshape(samples, (n_samples, rows * cols * bands))
  # Subset only the valid (non-NaN) cells
  valid_idxs = np.argwhere(~np.isnan(samples[0])).flatten()
  sample_cells = samples[:, valid_idxs]
  n_valid_cells = len(valid_idxs)

  n_plots = len(plot_idxs)
  if n_plots > 0:
    fig, axs = plt.subplots(1, n_plots, figsize=(n_plots * 5, 6), squeeze=False)
    for i in range(n_plots):
      axs[0,i].scatter(sample_cells[:, plot_idxs[i]], attrib[:, plot_idxs[i]])


def main():

  # Options
  parser = OptionParser()
  parser.add_option("-s", "--samples_file",
                    help="Path to '.npz' file with benchmark samples.")
  parser.add_option(      "--samples_varname",
                    default="samples",
                    help="Name of variable with raster samples in samples file.")
  parser.add_option("-k", "--breakpoints",
                    default=5,
                    type="int",
                    help="Number of breakpoints in piece-wise linear function.")
  parser.add_option("-f", "--function_file",
                    help="Path to '.npz' to write the function (PWL weights and edges).")
  parser.add_option("-o", "--output_file",
                    help="Path tp '.npz' to write function output (and attributions).")
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
    print("Expected path to write a '.npz' file with function output and attribution maps.\nExiting...")
    exit(-1)
  function_file = options.function_file
  if function_file is None:
    print("Expected path to write a '.npz' file with the generated function.\nExiting...")
    exit(-1)

  samples_varname = options.samples_varname
  n_breaks = options.breakpoints
  plot_idxs = options.plot_idxs
  n_plots = 0
  if plot_idxs is not None:
    plot_idxs = np.array(plot_idxs.split(",")).astype("int")

  plot_cell_idxs = options.plot_cell_idxs
  n_cell_plots = 0
  if plot_cell_idxs is not None:
    plot_cell_idxs = np.array(plot_cell_idxs.split(",")).astype("int")

  # Load samples
  samples_npz = np.load(samples_file)
  samples = samples_npz[samples_varname]

  # Build PWL function
  edges, weights = pwl_build(samples, n_breaks)
  
  # Evaluate PWL function
  y, attrib, attrib_maps = pwl_eval(samples, edges, weights)

  # Plot attribution maps
  if plot_idxs is not None:
    plot_attribution_maps(attrib_maps, plot_idxs)
    plt.tight_layout()
    plt.show()  

  # Plot PWL functions
  if plot_cell_idxs is not None:
    plot_cell_functions(samples, attrib, plot_cell_idxs)
    plt.tight_layout()
    plt.show()

  # Write the function (weights & edges)
  print("Writing function (weights and edges to: {}.".format(function_file))
  np.savez(function_file, weights=weights, edges=edges)

  # Write the outputs (y values & attributions)
  print("Writing outputs & attributions to: {}.".format(output_file))
  np.savez(output_file, y=y, attributions=attrib, attribution_maps=attrib_maps)
  


if __name__ == "__main__":
  main()


