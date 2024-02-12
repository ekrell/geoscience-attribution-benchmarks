# This program is used to force a raster dataset
# to have dimensions (samples, rows, cols, bands)

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
parser.add_option("-l", "--expand_left",
                  action="store_true",
                  default=False,
                  help="Expand the dimensions on the left, e.g. (1, ...).")
parser.add_option("-r", "--expand_right",
                  action="store_true",
                  default=False,
                  help="Expand the dimensions on the right, e.g. (..., 1).")
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

expand_left = options.expand_left
expand_right = options.expand_right

# Load raster
dataset = np.load(input_file)
raster = dataset[input_varname]

print("Input raster has shape: {}.".format(raster.shape))

# Expand
if expand_left: 
  raster = np.expand_dims(raster, 0)
if expand_right:
  raster = np.expand_dims(raster, len(raster.shape))

print("Output raster has shape: {}.".format(raster.shape))

# Write
np.savez(output_file,  **{output_varname: raster})
