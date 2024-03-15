'''
get_csu_archive.py

The purpose of this program is to download and format
the archived SST Anomaly Synthetic Benchmark for XAI. 
This is provided by CSU to accompany their XAI benchmarks.

The archive contains a single NetCDF file with several variables. 
In this repo, these data files are organized across three separate
files in compressed Numpy ('.npz') format. This script extracts
the data from the NetCDF, reshapes some of the arrays for a 
consistent format with this repo's benchmarks, and saves them
as three separate .npz files.

Archive source: 
https://beta.source.coop/csu/synthetic-attribution/
'''

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import urllib.request
from pathlib import Path
from optparse import OptionParser

# Options
parser = OptionParser()
parser.add_option("-o", "--out_dir",
                  help="Path to save the output files.",
                  default="./")
parser.add_option("-i", "--in_file",
                  help="Path to CSU synthetic benchmark data (.nc).",
                  default="synth_exm_data.nc")
parser.add_option("-d", "--download",
                  help="Download the CSU synthetic benchmark data.",
                  action="store_true",
                  default=False)
parser.add_option("-s", "--start_samples",
                  help="Start index")
parser.add_option("-e", "--end_samples",
                  help="End index")
(options, args) = parser.parse_args()

# Download?
isDownload = options.download
# Original CSU NetCDF file
csu_file = Path(options.in_file)
# Output NPZ files
out_dir = options.out_dir
out_samples_file = out_dir + "/csu_samples.npz"
out_function_file = out_dir + "/csu_pwl-fun.npz"
out_output_file = out_dir + "/csu_pwl-out.npz"
# Subset
start_samples = options.start_samples
end_samples = options.end_samples

if isDownload:
  if csu_file.is_file():
    print("File '{}' already exists. Proceed with download?".format(csu_file))
    answer = input("y/n: ")
    if answer != "y" and answer != "Y":
      print("Exiting...")
      exit(0)
  
  # prepare progressbar
  def show_progress(block_num, block_size, total_size):
    print(str(round(block_num * block_size / total_size *100,2)) + "%", end="\r")

  print("Downloading CSU SST synthetic benchmark data")
  print("Source: https://beta.source.coop/csu/synthetic-attribution/")
  
  url = "https://data.source.coop/csu/synthetic-attribution/synth_exm_data.nc"
  urllib.request.urlretrieve(url, csu_file, show_progress)

# Open CSU data archive
if not csu_file.is_file():
  print("Could not find {}. Check file name or download it with ('-d').".format(csu_file))
  print("Exiting...")
  exit(-1)
csu_data = xr.open_dataset(csu_file)

max_samples = csu_data["time"].shape[0]

# Subset
if start_samples is None:
  start_samples = 0
else:
  start_samples = int(start_samples)
if end_samples is None:
  end_samples = max_samples
else:
  end_samples = int(end_samples)

selected_samples = np.array(range(start_samples, end_samples))

# 'Samples' dataset
# Samples
samples = csu_data["SSTrand"][selected_samples].to_numpy()
samples = samples.transpose(0, 2, 1)
samples = np.expand_dims(samples, axis=-1)

# Get indices on non-NaN cells
valid_idxs = np.argwhere(~np.isnan(
  np.reshape(samples, (samples.shape[0], samples.shape[1] * samples.shape[2]))[0])).flatten()

# 'Function' dataset
# Weights
weights = csu_data["W"].to_numpy()
weights = np.reshape(weights,
  (weights.shape[1] * weights.shape[2], weights.shape[0]))
weights = weights[valid_idxs,:]
# Edges
edges = None  # Was not provided in the CSU dataset

# 'Output' dataset
# y
y = csu_data["y"][selected_samples].to_numpy()
# Attribution maps
attrmaps = csu_data["Cnt"][selected_samples].to_numpy()
attrmaps = attrmaps.transpose(0, 2, 1)
attrmaps = np.expand_dims(attrmaps, axis=-1)
# Attributions
attrs = np.reshape(attrmaps, (attrmaps.shape[0], attrmaps.shape[1] * attrmaps.shape[2]))
attrs = attrs[:, valid_idxs]

# Save
np.savez(out_samples_file, samples=samples)
np.savez(out_function_file, weights=weights, edges=edges)
np.savez(out_output_file, attributions=attrs, attribution_maps=attrmaps, y=y)


print("")
print("SST Anom CSU Data Archive")
print("-------------------------")
print("- converting to 3 NPZ files")
print("")
print("SAMPLES file:   {}".format(out_samples_file))
print("  'samples'          : {}".format(samples.shape))
print("FUNCTION file:  {}".format(out_function_file))
print("  'weights'          : {}".format(weights.shape))
print("  'edges'            : {}".format(None))
print("OUTPUT file:    {}".format(out_output_file))
print("  'y'                : {}".format(y.shape))
print("  'attributions'     : {}".format(attrs.shape))
print("  'attribution_maps' : {}".format(attrmaps.shape))
print("")
