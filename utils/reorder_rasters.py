# This program reorders the bands of a raster.
# Expects a '.npz' file that contains a variable
# with an array of shape (samples, rows, cols, bands)

import os
import numpy as np
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-i", "--input_file",
                  help="Path to '.npz' raster file.")
parser.add_option(      "--input_varname",
                  default="samples",
                  help="Name of variable containing rasters in input file.")
parser.add_option("-o", "--output_file",
                  help="Path to save cropped '.npz' raster file.")
parser.add_option(      "--output_varname",
                  default="samples",
                  help="Name of variable to write rasters in output file.")
parser.add_option("-r", "--reorder_indices",
                  help="Comma-delimited list of indices of the output raster e.g. 2,3,1.")
(options, args) = parser.parse_args()

input_file = options.input_file
if input_file is None:
  print("Expected an input '.npz' file with raster to crop ('-i').\nExiting...")
  exit(-1)
output_file = options.output_file
if output_file is None:
  print("Expected an output '.npz' file to save cropped raster ('-o').\nExiting...")
  exit(-1)

input_varname = options.input_varname
output_varname = options.output_varname

reorder_idxs = options.reorder_indices
if reorder_idxs is None:
  print("Expected a comma-delimited list of raster indices that defines the output reordering.\nExiting...")
  exit(-2)

# Load raster
dataset = np.load(input_file)
raster = dataset[input_varname]
print("Input raster has shape: {}.".format(raster.shape))

# Indices
reorder_idxs = np.array(reorder_idxs.split(",")).astype("int")
print("Reorder indices: {}.".format(reorder_idxs))

# Reorder
raster = raster[:,:,:,reorder_idxs]
print("Reordered raster has shape: {}.".format(raster.shape))

# Write
np.savez(output_file, **{output_varname: raster})


