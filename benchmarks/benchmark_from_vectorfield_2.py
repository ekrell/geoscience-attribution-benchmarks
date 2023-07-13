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

  # Load vector field
  vector_data = np.load(vector_file)
  vector = vector_data[vector_varname]

  field_x = vector[0]
  field_y = vector[1]
  if vector.shape[0] == 3:
    field_z = vector[2]
  else:
    field_z = None

  n_samples, rows, cols, bands = rasters.shape

  ######################
  # Apply Vector Field #
  ######################

  # Init storage for time series rasters
  rasters_ts = np.ones((n_samples, rows, cols, bands, time_steps))
  rasters_ts[:] = np.nan

  # Iterate over samples
  for sample_idx in range(n_samples):
  
    # Init first time step (t = 0) to original raster
    rasters_ts[sample_idx, :, :, :, 0] = rasters[sample_idx]

    # Iterate over time steps
    for timestep in range(1, time_steps):
      # Shift cell values based on vector field
      for row in range(rows):
        for col in range(cols):
          for band in range(bands):
        
            row_old = int(row + field_y[row, col, band])
            col_old = int(col - field_x[row, col, band])
            band_old = int(band - field_z[row, col, band])
            if row_old >= 0 and row_old < rows \
              and col_old >= 0 and col_old < cols \
              and band_old >= 0 and band_old < bands:
                rasters_ts[sample_idx, row, col, band, timestep] = \
                  rasters_ts[sample_idx, row_old, col_old, band_old, timestep - 1]

      # Interpolate grid to remove NaNs
      xx, yy, zz = np.meshgrid(np.arange(0, cols), 
                               np.arange(0, rows),
                               np.arange(0, bands),
        )
      valids = np.ma.masked_invalid(rasters_ts[sample_idx, :, :, :, timestep])
      x_valid = xx[~valids.mask]
      y_valid = yy[~valids.mask]
      z_valid = zz[~valids.mask]
      arr = valids[~valids.mask]
      rasters_ts[sample_idx, :, :, :, timestep] = \
          interpolate.griddata((x_valid, y_valid, z_valid), 
          arr.ravel(), (xx, yy, zz), method="linear") 

  # Pack 4D into 3D data
  packed = np.zeros((n_samples, rows, cols, bands * time_steps))
  band_keys = np.zeros((bands * time_steps, 2)).astype("int")
  count = 0
  for ts in range(time_steps):
    for band in range(bands):
      packed[:,:,:,count] = rasters_ts[:,:,:,band,ts]
      band_keys[count] = (band, ts)
      count += 1
  rasters_ts = packed

  # Write
  key_varname = "band_keys"
  np.savez(output_file, **{output_varname: rasters_ts, key_varname: band_keys})


if __name__ == "__main__":
  main()
