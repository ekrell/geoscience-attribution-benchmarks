# This program concatenates 2 rasters along their channels.
# If the data has shape (rows, cols, channels) then the third dimension is used. 
# If the data has shape (samples, rows, cols, channels) then the fourth dimension is used.
# If the data has shape (rows, cols), then it is not supported. 

import os
import numpy as np
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-a", "--a_file", 
                  help="Path to first file in concatenation ('.npz').")
parser.add_option(      "--a_varname",
                  default="samples",
                  help="Name of variable with rasters in 'a' file.")
parser.add_option("-b", "--b_file",
                  help="Path to second file in concatenation ('.npz').")
parser.add_option(      "--b_varname",
                  default="samples",
                  help="Name of variable with rasters in 'b' file.")
parser.add_option("-o", "--output_file",
                  help="Path to save concatenation result ('.npz').")
parser.add_option(      "--output_varname",
                  default="samples",
                  help="Name to save raster in output file.")
(options, args) = parser.parse_args()

a_file = options.a_file
if a_file is None:
  print("Concatenation requires 2 '.npz' files. Use (-a) to specify the first one.\nExiting...")
  exit(-1)
b_file = options.b_file
if b_file is None:
  print("Concatenation requires 2 '.npz' files. Use (-b) to specify the second one.\nExiting...")
  exit(-1)
output_file = options.output_file
if output_file is None:
  print("Expected an output file ('.npz') to save the concatenation.\nExiting...")
  exit(-1)

varname_a = options.a_varname
varname_b = options.b_varname
varname_o = options.output_varname

def loadRaster(filename, varname):
  dataset = np.load(filename)
  raster = dataset[varname]
  shape = raster.shape
  n_dims = len(shape)
  return dataset, raster, shape, n_dims

_, raster_a, shape_a, n_dims_a = loadRaster(a_file, varname_a)
_, raster_b, shape_b, n_dims_b = loadRaster(b_file, varname_b)

# Check that a and b share same number of dims
if n_dims_a != n_dims_b:
  print("Expected rasters to have same number of dimensions, but 'a' has {} and 'b' has {}.\nExiting...".format(
    n_dims_a, n_dims_b))
  exit(-2) 

# Check that the number of dimensions is valid
if n_dims_a < 3 or n_dims_b > 4:
  print("Expected shape to be either length 3 (rows, cols, bands) or length 4 (samples, rows, cols, bands).")
  print("Instead, found shape of length {}.\nExiting...".format(n_dims))
  exit(-2)

# Check that a and b share same n_samples, rows, & cols. Only bands can differ
if shape_a[:-1] != shape_b[:-1]:
  print("Rasters 'a' and 'b' must have identical shape EXCEPT the number of bands may be different.")
  print("However, raster 'a' has shape {} and 'b' has shape {}.\nExiting...".format(shape_a, shape_b))
  exit(-2)

# Concatenate rasters
print("Raster 'a' has shape: {}".format(shape_a))
print("Raster 'b' has shape: {}".format(shape_b))
raster_cat = np.concatenate([raster_a, raster_b], -1)
print("Concatenation result has shape: {}".format(raster_cat.shape))

# Write result
print("Writing result to: {}.".format(output_file))
np.savez(output_file, **{varname_o: raster_cat})
