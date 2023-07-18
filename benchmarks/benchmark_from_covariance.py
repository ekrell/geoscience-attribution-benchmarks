# This program generates a set of synthetic samples of raster data
# based on a provided covariance matrix and optional mask.
#
# The covariance matrix defines the multivariate normal distribution
#   for generating samples
# The mask is used to define which raster elements are included in
#   the covariance matrix. For example, only the water grid cells are
#   used for sea surface temperature. But we need to know which grid 
#   cells correspond to the elements of the covariance matrix. 
#
# This extends code by Dr. Antonios Mamalakis:
# `Neural-Network-Attribution-Benchmark-for-Regression`
#   (github.com/amamalak/Neural-Network-Attribution-Benchmark-for-Regression)

import os
import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt

###########
# Options #
###########

parser = OptionParser()
parser.add_option("-c", "--covariance_file",
                  help="Path to `.npz` file with covariance matrix and mask.")
parser.add_option("-n", "--num_samples",
                  default=1,
                  type="int",
                  help="Number of synthetic samples to generate.")
parser.add_option("-o", "--output_file",
                  help="Path to `.npz` file to save generated samples.")
(options, args) = parser.parse_args()

# File with covariance matrix
cov_file = options.covariance_file
if cov_file is None:
  print("Expected a `.npz` input file with covariance matrix and mask (-c).\nExiting...")
  exit(-1)
# Number of synthetic samples to generate
n_samples = options.num_samples
# Output file with generated samples
out_file = options.output_file
if out_file is None:
  print("Expected a `.npz` output file to save the generated samples (-o).\nExiting...")
  exit(-1)


##############
# Setup Data #
##############

# Load NPZ file with cov data
cov_data = np.load(cov_file)
# The covariance matrix
cov = cov_data["covariance"]
# A mask that defines the raster dimensions &
# maps the covariance matrix elements to cells
mask = cov_data["mask"]

# The mask size determines raster size
rows = mask.shape[0]
cols = mask.shape[1]
if len(mask.shape) > 2:
  bands = mask.shape[2]
else:
  bands = 1
grid_elems = rows * cols * bands

# But there may be less elements in the covariance matrix
# (e.g. we only want to generate for water, skipping land)
cov_elems = cov.shape[0]

# The 'True' mask elements correspond to 
# those included in the covariance matrix
m = np.reshape(mask, rows * cols * bands)
coords = np.where(m == True)
coords = coords[0]

print("")
print("Generating {} samples of {} x {} x {} rasters.".format(n_samples, rows, cols, bands))
print("Using covariance data: {}".format(cov_file))
print("Writing generated samples to: {}".format(out_file))
print("  With variable name: 'samples'")
print("  Load file using 'numpy.load({})'".format(out_file))


#####################
# Synthetic Samples #
#####################

# Generate samples using covariance matrix
sample_vals = np.random.multivariate_normal(np.zeros(cov_elems), # Sample mean
                                            cov,                 # Sample covariance
                                            n_samples)           # Number to generate

# Generate empty rasters that will hold the generated data
sample_raster = np.empty((n_samples, grid_elems))
# Init to NaN
sample_raster[:] = np.nan
# Use the mask to assign the generated values to the raster
sample_raster[:, coords] = sample_vals
# Reshape from 1D vector to raster with rows, cols, bands
sample_raster = np.reshape(sample_raster, (n_samples, rows, cols, bands))

#########
# Write #
#########

np.savez(out_file, samples=sample_raster)
