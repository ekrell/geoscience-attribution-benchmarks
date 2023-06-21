# This program computes a covariance matrix based on
# storm-centered tornado satellite images
#   (https://github.com/djgagne/ams-ml-python-course)
#
# Download data:
#   wget https://storage.googleapis.com/track_data_ncar_ams_3km_nc_small/track_data_ncar_ams_3km_nc_small.tar.gz
#  tar -xvzf track_data_ncar_ams_3km_nc_small.tar.gz 
#
# The output is a .npz file with 1 variable:
#   (1) `cov`: the covariance matrix
#   (2) `mask`: a matrix that indicates which raster cells
#               are used in the covariance matrix. Here, it
#               is all the cells so all values are True.

import os
import numpy as np
from netCDF4 import Dataset
from optparse import OptionParser
import matplotlib.pyplot as plt

def main():

  ###########
  # Options #
  ###########

  parser = OptionParser()
  parser.add_option("-n", "--netcdf_paths",
                    default="data/track_data_ncar_ams_3km_nc_small/NCARSTORM_20130519-0000_d01_model_patches.nc",
                    help="Comma-delimited path to local NetCDF files with storm-centered tornado images")
  parser.add_option("-o", "--output_cov",
                    help="Path to write tornado covariance data.")
  (options, args) = parser.parse_args()

  nc_paths = options.netcdf_paths.split(",")
  output_cov = options.output_cov
  if output_cov is None:
    print("Expected an output file path to save computed covariance data (-c).\nExiting...")
    exit(-1)

  
  ###################
  # Process NetCDFs #
  ###################

  patches = [None for i in nc_paths]
  for j, nc_path in enumerate(nc_paths):

    # Load dataset
    nc = Dataset(nc_path, "r", format="NETCDF4")
    # Extract data
    reflectivity_dbz = nc["REFL_COM_curr"][:]
    temp_kelvins = nc["T2_curr"][:]
    u_wind_meters = nc["U10_curr"][:]
    v_wind_meters = nc["V10_curr"][:]
    # Get shapes
    n_patches = temp_kelvins.shape[0]
    rows = temp_kelvins.shape[1]
    cols = temp_kelvins.shape[2]
    # Combine variables into multi-channel rasters
    patches[j] = np.zeros((n_patches, rows, cols, 4))
    for i in range(n_patches):
      patches[j][i, :, :, 0] = reflectivity_dbz[i, :, :]
      patches[j][i, :, :, 1] = temp_kelvins[i, :, :]
      patches[j][i, :, :, 2] = u_wind_meters[i, :, :]
      patches[j][i, :, :, 3] = v_wind_meters[i, :, :]

  # Combine patches from all NetCDFs
  patches = np.concatenate(patches)
  n_patches = patches.shape[0]

  #####################
  # Covariance Matrix #
  #####################

  # Reshape so that each raster is a single vector
  x = np.reshape(patches, (n_patches, rows * cols * 4))
  cov = np.cov(x, rowvar=False)

  # Create a mask that indicates that each cell is included in the covariance matrix
  mask = np.ones((rows, cols, 4)).astype("bool")

  # Save covariance matrix
  np.savez(output_cov, covariance=cov, mask=mask)


if __name__ == "__main__":
  main()
