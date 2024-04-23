# This program computes a covariance matrix based on
# data from  COBE-SST 2 and Sea Ice 
#   (https://psl.noaa.gov/data/gridded/data.cobe2.html)
#
# The input netCDF file must have either:
#   1)  'sst' variable (Sea Surface Temperature)
#   2)  'icec' variable (Ice Concentration)
# Script will detect which one is present and use it. 
#
# The output is a .npz file with 2 variables: 
#   (1) `cov`: the covariance matrix
#   (2) `mask`: a binary grid where the covariance values match up
#               to the True values. This is because we don't use 
#               land pixels when computing the covariance. 
#
# This is a Python implementation of code by Dr. Antonios Mamalakis:
# `Neural-Network-Attribution-Benchmark-for-Regression`
#   (github.com/amamalak/Neural-Network-Attribution-Benchmark-for-Regression)
#
# Citation:
#   Mamalakis, A., I. Ebert-Uphoff, E.A. Barnes (2022) “Neural network attribution methods for
#   problems in geoscience: A novel synthetic benchmark dataset,” arXiv preprint arXiv:2103.10005.

import os
import numpy as np
from optparse import OptionParser
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy.signal import detrend
import cmocean
import matplotlib.cm as cm

def main():

  ###########
  # Options #
  ###########

  parser = OptionParser()
  parser.add_option("-n", "--netcdf_path",
                    default="sst.mon.mean.nc",
                    help="Path to local NetCDF file with SST (or ICEC) satellite data.")
  parser.add_option("-c", "--output_cov",
                    help="Path to write covariance matrix (and mask).")
  parser.add_option("-p", "--output_plot",
                    help="Path to write example plot of requested index.")
  parser.add_option("-a", "--anomaly", 
                    help="Convert into anomaly data (subtract climatology & detrend).",
                    action="store_true")
  (options, args) = parser.parse_args()

  convert_anomaly = options.anomaly

  nc_path = options.netcdf_path
  output_plot = options.output_plot
  output_cov = options.output_cov
  if output_cov is None:
    print("Expected an output file path to save computed covariance data (-c).\nExiting...")
    exit(-1)

  # Load dataset
  nc = Dataset(nc_path, "r", format="NETCDF4")

  # Find out if NetCDF has 'sst' or 'icec' data
  varkey = None
  if "sst" in nc.variables.keys():
      varkey = "sst"
  elif "icec" in nc.variables.keys():
      varkey = "icec"

  print("Found variable '{}'  -->  will extract covariance matrix.".format(varkey))

  # Data
  monthly_data = nc[varkey][:]
  # Lat, lon data
  lon, lat = np.meshgrid(nc["lon"], nc["lat"])
  # Close dataset
  nc.close()


  ##############
  # Preprocess #
  ##############

  # Reshape data to match lat, lon organization
  monthly_data = np.transpose(monthly_data, [1,2,0])
  # Determine NaN values
  monthly_data[monthly_data > 500] = np.nan
  # Subset to data in 01/1950 - 12/2019 (we do not trust earlier data)
  monthly_data = monthly_data[:,:,100*12:]

  #############
  # Anomalies #
  #############

  data_anom = monthly_data.copy()
  if convert_anomaly:

    # Remove annual cycle
    num_months = 12
    num_lons = lon.shape[1]
    num_lats = lon.shape[0]

    for i in range(num_months):
      # Month's data across all years
      monthly_data_ = monthly_data[:,:,i::12]
      num_years = monthly_data_.shape[2]
      # Mean at each (lat, lon) for specified month
      monthly_mean = np.reshape(np.nanmean(monthly_data_[:,:,i::12], 2), 
                                           (num_lats, num_lons, 1))
      # Replicate for subtraction
      monthly_rep = np.tile(monthly_mean, [1, 1, num_years])
      # For this month, remove annual mean
      data_anom[:,:,i::12] = monthly_data_ - monthly_rep

    # Detrending
    for i in range(num_lats):
      for j in range(num_lons):
        # Extact all values at coordinate
        vals = data_anom[i, j, :]
        # Remove NaNs 
        vals = vals[~np.isnan(vals)]
        # Skip this coordinate if no values (e.g. land)
        if len(vals) == 0:
          continue      
        # Detrend 
        vals_detrend = detrend(vals)
        # Store
        data_anom[i, j, :] = vals_detrend / np.std(vals_detrend)

  #####################
  # Reduce dimensions #
  #####################

  # Reduce dimensions
  lon_degree_res = 10
  lat_degree_res = 10
  # Resize by skipping 
  data_anom_reduced = data_anom[::lat_degree_res, ::lon_degree_res, :]
  lat_reduced = lat[::lat_degree_res, ::lon_degree_res]
  lon_reduced = lon[::lat_degree_res, ::lon_degree_res]

  #####################
  # Covariance Matrix #
  #####################

  def calc_covariance(data):

    # Find water coordinates
    m = np.reshape(data[:,:,0], (data.shape[0] * data.shape[1]))
    ocean_coords = np.where(~np.isnan(m))
    # Reshape so that each row is a single map
    x = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]))
    # Keep only ocean_coords
    x = x[ocean_coords]
    # Calculate covariance matrix
    cov = np.cov(x)
    # Create map of water and non-water
    ocean_map = np.ones(data[:,:,0].shape)
    ocean_map[np.isnan(data[:,:,0])] = 0
    ocean_map = ocean_map.astype("bool")

    return cov, ocean_coords, ocean_map

  cov_anom_reduced, coords, ocean_map_reduced = calc_covariance(data_anom_reduced)

  ########
  # Plot #
  ########
  instance_idx=0

  fig, axs = plt.subplots(4, figsize=(6, 11))

  axs[0].set_facecolor("#D9D0B4")
  cf = axs[0].contourf(lon, lat, monthly_data[:,:,instance_idx], cmap=cmocean.cm.thermal, vmin=0)
  axs[0].set_xticks(np.arange(60, 350, 60), ['60E', '120E', '180', '120W', '60W'])
  axs[0].set_yticks(np.arange(-60, 90, 30), ['60S', '30S', 'EQ', '30N', '60N']) 
  axs[0].title.set_text("Sample for index {}".format(instance_idx))
  plt.colorbar(cf, ax=axs[0])

  cmap = cm.bwr
  cmap.set_bad("#D9D0B4", 1.)

  if convert_anomaly: 
      cf = axs[1].imshow(data_anom[:,:,instance_idx], cmap=cmap, vmin=-5, vmax=5)
      axs[1].set_xticks(np.arange(60, 350, 60), ['60E', '120E', '180', '120W', '60W'])
      axs[1].set_yticks(np.arange(30, 170, 30), ['60N', '30N', 'EQ', '30S', '60S'])
      axs[1].title.set_text("Sample anomaly")
      plt.colorbar(cf, ax=axs[1])
  else:
      axs[1].title.set_text("Sample anomaly (NOT APPLICABLE)")

  if convert_anomaly:
      cf = axs[2].imshow(data_anom_reduced[:,:,instance_idx], cmap=cmap, vmin=-5, vmax=5)
  else:
      cf = axs[2].imshow(data_anom_reduced[:,:,instance_idx], cmap=cmap)

  axs[2].set_xticks(np.arange(60 / lon_degree_res, 350 / lon_degree_res, 60 / lon_degree_res), 
                              ['60E', '120E', '180', '120W', '60W'])
  axs[2].set_yticks(np.arange(30 / lat_degree_res, 170 / lat_degree_res, 30 / lat_degree_res), 
                              ['60N', '30N', 'EQ', '30S', '60S'])
  axs[2].title.set_text("Reduced dimensions")
  plt.colorbar(cf, ax=axs[2])

  axs[3].imshow(cov_anom_reduced)

  plt.tight_layout()
  if output_plot is not None:
    plt.savefig(output_plot, dpi=300)

  # Save covariance matrix
  np.savez(output_cov, covariance=cov_anom_reduced, mask=ocean_map_reduced)

  print("Saved covariance matrix: {}".format(output_cov))
  print("Saved sample plot to: {}".format(output_plot))


if __name__ == "__main__":
  main()
