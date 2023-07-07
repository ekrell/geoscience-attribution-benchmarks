# This program creates a synthetic time series of raster data
# by applying a vector field to shift the raster elements for
# a specified number of discrete time steps. 

import os
import numpy as np
from scipy import interpolate
from optparse import OptionParser

def main():

  ###########
  # Options #
  ###########

  parser = OptionParser()
  parser.add_option("-t", "--time_steps",
                    default=1,
                    type="int",
                    help="Number of time steps to generate (each results in a raster band).")
  parser.add_option("-r", "--raster_file",
                    help="Path to '.npz' raster file.")
  parser.add_option(      "--raster_varname",
                    default="samples",
                    help="Name of variable containing rasters in input file.")
  parser.add_option("-v", "--vector_file",
                    help="Path to '.npz' vector field file.")
  parser.add_option(      "--vector_varname",
                    default="field",
                    help="Name of variable containing vector field in vector field file.")
  parser.add_option("-o", "--output_file",
                    help="Path to save generated time series '.npz' raster file.")
  parser.add_option(      "--output_varname",
                    default="samples",
                    help="Name of variable to write rasters in output file.")
  (options, args) = parser.parse_args()

  raster_file = options.raster_file
  if raster_file is None:
    print("Expected an input '.npz' file with raster to convert to time series ('-r').\nExiting...")
    exit(-1)
  vector_file = options.vector_file
  if vector_file is None:
    print("Expected an input '.npz' file with vector field to apply to a raster ('-v').\nExiting...")
    exit(-1)
  output_file = options.output_file
  if output_file is None:
    print("Expected an output '.npz' file to save the generated time series raster ('-o').\nExiting...")
    exit(-1)

  raster_varname = options.raster_varname
  vector_varname = options.vector_varname
  output_varname = options.output_varname

  time_steps = options.time_steps

  # Load raster
  raster_data = np.load(raster_file)
  rasters = raster_data[raster_varname]
  if len(rasters.shape) != 4:
    print("Expected raster data shape: (samples, rows, cols, bands).\nExiting...")
    exit(-2)
  if rasters.shape[3] > 1:
    print("Converting multi-band rasters to time series is not yet supported.\nExiting...")
    exit(-2)

  # Load vector field
  vector_data = np.load(vector_file)
  vector = vector_data[vector_varname]
  if len(vector.shape) != 3:
    print("Expected vector data shape: (2, rows, cols) where [0,:,:] is for x components and [1,:,:] is for y components.\nExiting...")
    exit(-2)
  field_x = vector[0]
  field_y = vector[1]

  n_samples, rows, cols, bands = rasters.shape

  ######################
  # Apply Vector Field #
  ######################

  # Init storage for time series rasters
  rasters_ts = np.ones((n_samples, rows, cols, time_steps))
  rasters_ts[:] = np.nan
  
  # Iterate over samples
  for sample_idx in range(n_samples):

    # Init first time step (t = 0) to original raster
    rasters_ts[sample_idx, :, :, 0] = rasters[sample_idx,:,:,0]
 
    # For each time step, determine values of each (row, col)
    # by applying the vector field
    for timestep in range(1, time_steps):
      for row in range(rows):
        for col in range(cols):
          row_new = int(row + field_y[row, col])
          col_new = int(col + field_x[row, col])
          try:
            rasters_ts[sample_idx, row_new, col_new, timestep] = \
               rasters_ts[sample_idx, row, col, timestep - 1]
          except:
            pass

  # Write
  np.savez(output_file, **{output_varname: rasters_ts})

if __name__ == "__main__":
  main()

