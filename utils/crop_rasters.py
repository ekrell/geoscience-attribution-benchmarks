# This program crops a raster along multiple dimensions
# Assumes rasters have either:
#   1) shape of (samples, rows, cols, bands)
#   2) shape of (rows, cols, bands)

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
parser.add_option(      "--low_sample",
                  default=None,
                  help="Lower index of samples.")
parser.add_option(      "--high_sample",
                  default=None,
                  help="Higher index of samples.")
parser.add_option(      "--low_row",
                  default=None,
                  help="Lower index of rows.")
parser.add_option(      "--high_row",
                  default=None,
                  help="Higher index of rows.")
parser.add_option(      "--low_col",
                  default=None,
                  help="Lower index of cols.")
parser.add_option(      "--high_col",
                  default=None,
                  help="Higher index of cols.")
parser.add_option(      "--low_band",
                  default=None,
                  help="Lower index of bands.")
parser.add_option(      "--high_band",
                  default=None,
                  help="Higher index of bands.")
parser.add_option(      "--band_keys_varname",
                  help="Name of variable name in input '.npz' with keys for band IDs")
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
band_keys_varname = options.band_keys_varname

# Load raster
dataset = np.load(input_file)
raster = dataset[input_varname]
shape = raster.shape

# Reshape if needed
singleSampleShape = (len(shape) == 3)
if singleSampleShape:
  raster = np.reshape(raster, (1, shape[0], shape[1], shape[2]))
  shape = raster.shape

# Get crop bounds
low_sample = int(options.low_sample) if options.low_sample is not None else 0
high_sample = int(options.high_sample) if options.high_sample is not None else shape[0]
low_row = int(options.low_row) if options.low_row is not None else 0
high_row = int(options.high_row) if options.high_row is not None else shape[1]
low_col = int(options.low_col) if options.low_col is not None else 0
high_col = int(options.high_col) if options.high_col is not None else shape[2]
low_band = int(options.low_band) if options.low_band is not None else 0
high_band = int(options.high_band) if options.high_band is not None else shape[3]

print("Raster has shape: {}.".format(shape))
print("Crop bounds: ({}-{}),  ({}-{}),  ({}-{}).  ({}-{})).".format(
      low_sample, high_sample, low_row, high_row,
      low_col, high_col, low_band, high_band))

def crop(raster, low_sample, high_sample, low_row, high_row,
         low_col, high_col, low_band, high_band):
  return raster[low_sample:high_sample,
                low_row:high_row,
                low_col:high_col,
                low_band:high_band]

# If using band keys
if band_keys_varname is not None:
  band_keys = dataset[band_keys_varname]
  uniq_ts = np.unique(band_keys[:,1])
  rasters_ = []
  for ts in uniq_ts:
    # Bands with this ts
    band_idxs = np.array(np.where(band_keys[:,1] == ts))
    band_idxs = band_idxs.flatten()
    # Extract for ts
    raster_ = raster[:, :, :, band_idxs]
    raster_ = crop(raster_, low_sample, high_sample, low_row, high_row,
      low_col, high_col, low_band, high_band)
    rasters_.append(raster_)
  raster = np.concatenate(rasters_, axis=3)

# Crop normally  
else:
  # Crop
  raster = crop(raster, low_sample, high_sample, low_row, high_row,
    low_col, high_col, low_band, high_band)

print("Cropped raster has shape: {}.".format(raster.shape))

# Write
np.savez(output_file, **{output_varname: raster})
