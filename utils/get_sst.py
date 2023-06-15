# This program computes a covariance matrix based on
# SST Anomaly data from  COBE-SST 2 and Sea Ice 
#   (https://psl.noaa.gov/data/gridded/data.cobe2.html)
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
                    help="Path to local NetCDF file with SST satellite data.")
  parser.add_option("-c", "--output_cov",
                    help="Path to write SST anomaly covariance data.")
  parser.add_option("-p", "--output_plot",
                    help="Path to write plot of requested index (SST and anomaly data).")
  parser.add_option("-i", "--instance_idx",
                    default=0,
                    type="int",
                    help="Integer index of SST data to plot.")
  (options, args) = parser.parse_args()

  nc_path = options.netcdf_path
  instance_idx = options.instance_idx
  output_plot = options.output_plot
  output_cov = options.output_cov
  if output_cov is None:
    print("Expected an output file path to save computed covariance data (-c).\nExiting...")
    exit(-1)

  # Load dataset
  nc = Dataset(nc_path, "r", format="NETCDF4")
  # SST variable metadata
  sst_var = nc.variables["sst"]
  # SST data
  monthly_sst = nc["sst"][:]
  # Lat, lon data
  lon_sst, lat_sst = np.meshgrid(nc["lon"], nc["lat"])
  # Close dataset
  nc.close()


  ##############
  # Preprocess #
  ##############

  # Reshape sst data to match lat, lon organization
  monthly_sst = np.transpose(monthly_sst, [1,2,0])

  # Determine NaN values
  monthly_sst[monthly_sst > 500] = np.nan

  # Subset to data in 01/1950 - 12/2019 (we do not trust earlier data)
  monthly_sst = monthly_sst[:,:,100*12:]


  #################
  # SST Anomalies #
  #################

  # SST monthly anomalies (remove annual cycle)
  sst_anom = monthly_sst.copy()
  num_months = 12
  num_lons = lon_sst.shape[1]
  num_lats = lon_sst.shape[0]
  for i in range(num_months):
    # Month's data across all years
    monthly_sst_ = monthly_sst[:,:,i::12]
    num_years = monthly_sst_.shape[2]
    # Mean at each (lat, lon) for specified month
    monthly_mean = np.reshape(np.nanmean(monthly_sst_[:,:,i::12], 2), 
                                         (num_lats, num_lons, 1))
    # Replicate for subtraction
    monthly_rep = np.tile(monthly_mean, [1, 1, num_years])
    # For this month, remove annual mean
    sst_anom[:,:,i::12] = monthly_sst_ - monthly_rep

  # Detrending
  for i in range(num_lats):
    for j in range(num_lons):
      # Extact all values at coordinate
      vals = sst_anom[i, j, :]
      # Remove NaNs 
      vals = vals[~np.isnan(vals)]
      # Skip this coordinate if no values (e.g. land)
      if len(vals) == 0:
        continue      
      # Detrend 
      vals_detrend = detrend(vals)
      # Store
      sst_anom[i, j, :] = vals_detrend / np.std(vals_detrend)


  #####################
  # Reduce dimensions #
  #####################

  # Reduce dimensions
  lon_degree_res = 10
  lat_degree_res = 10
  # Resize by skipping 
  sst_anom_reduced = sst_anom[::lat_degree_res, ::lon_degree_res, :]
  lat_sst_reduced = lat_sst[::lat_degree_res, ::lon_degree_res]
  lon_sst_reduced = lon_sst[::lat_degree_res, ::lon_degree_res]


  ########
  # Plot #
  ########

  fig, axs = plt.subplots(3, figsize=(6, 11))

  axs[0].set_facecolor("#D9D0B4")
  cf = axs[0].contourf(lon_sst, lat_sst, monthly_sst[:,:,instance_idx], cmap=cmocean.cm.thermal, vmin=0)
  axs[0].set_xticks(np.arange(60, 350, 60), ['60E', '120E', '180', '120W', '60W'])
  axs[0].set_yticks(np.arange(-60, 90, 30), ['60S', '30S', 'EQ', '30N', '60N']) 
  axs[0].title.set_text("SST for index {}".format(instance_idx))
  plt.colorbar(cf, ax=axs[0])

  cmap = cm.bwr
  cmap.set_bad("#D9D0B4", 1.)

  cf = axs[1].imshow(sst_anom[:,:,instance_idx], cmap=cmap, vmin=-5, vmax=5)
  axs[1].set_xticks(np.arange(60, 350, 60), ['60E', '120E', '180', '120W', '60W'])
  axs[1].set_yticks(np.arange(30, 170, 30), ['60N', '30N', 'EQ', '30S', '60S'])
  axs[1].title.set_text("SST anomaly")
  plt.colorbar(cf, ax=axs[1])

  cf = axs[2].imshow(sst_anom_reduced[:,:,instance_idx], cmap=cmap, vmin=-5, vmax=5)
  axs[2].set_xticks(np.arange(60 / lon_degree_res, 350 / lon_degree_res, 60 / lon_degree_res), 
                              ['60E', '120E', '180', '120W', '60W'])
  axs[2].set_yticks(np.arange(30 / lat_degree_res, 170 / lat_degree_res, 30 / lat_degree_res), 
                              ['60N', '30N', 'EQ', '30S', '60S'])
  axs[2].title.set_text("SST anomaly (reduced dimensions)")
  plt.colorbar(cf, ax=axs[2])

  plt.tight_layout()
  if output_plot is not None:
    plt.savefig(output_plot, dpi=300)
  else:
    plt.show()


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

  cov_anom_reduced, coords, ocean_map_reduced = calc_covariance(sst_anom_reduced)

  # Save covariance matrix
  np.savez(output_cov, covariance=cov_anom_reduced, mask=ocean_map_reduced)


  exit(0)

  # Generate a random sample
  sample_ = np.random.multivariate_normal(np.zeros(cov_anom_reduced.shape[0]), cov_anom_reduced)
  sample = np.empty(sst_anom_reduced.shape[0] * sst_anom_reduced.shape[1])
  sample[:] = np.nan
  sample[coords] = sample_
  sample = np.reshape(sample, (sst_anom_reduced.shape[0], sst_anom_reduced.shape[1]))

  print(sample.shape)
  plt.clf()
  plt.close()
  cmap = cm.bwr
  cmap.set_bad("#D9D0B4", 1.)
  plt.imshow(sample, cmap=cmap, vmin=-5, vmax=5)

  plt.show()


if __name__ == "__main__":
  main()
