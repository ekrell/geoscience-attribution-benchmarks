# This program plots raster data stored in '.npz' files.

import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt


###########
# Options #
###########

parser = OptionParser()
parser.add_option("-r", "--rasters_file",
                  help="Path to '.npz' file with a variable containing rasters.")
parser.add_option("-v", "--variable_name",
                  default="samples",
                  help="Name of variable with the rasters within the '.npz' file.")
parser.add_option("-i", "--indices",
                  default="0",
                  help="Comma-delimited list of which rasters to plot (0-based count).")
parser.add_option("-o", "--output_file",
                  help="Path to save plotted rasters.")
(options, args) = parser.parse_args()

npz_file = options.rasters_file
if npz_file is None:
  print("Expected a path to '.npz' file with raster data ('-r').\nExiting...")
  exit(-1)
var_name = options.variable_name
indices = np.array(options.indices.split(",")).astype("int")
out_file = options.output_file

print("") 
print("Rasters file: {}".format(npz_file))
print("  Using variable name: {}".format(var_name))
print("  Will plot {} rasters".format(len(indices)))
if out_file is None:
  print("No output file specified -> will display plot")
else:
  print("Output plot file: {}".format(out_file))

# Load data
rasters_npz = np.load(npz_file)
rasters = rasters_npz[var_name]
n_rasters = rasters.shape[0]
rows = rasters.shape[1]
cols = rasters.shape[2]
# Ensure that shape is (samples, rows, cols, bands)
# (Even when using single-channel data)
if len(rasters.shape) <= 3:
  rasters = np.reshape(rasters, (n_rasters, rows, cols, 1))
bands = rasters.shape[3]

# Subset to requested indices
rasters = rasters[indices, :, :, :]
n_rasters = rasters.shape[0]


########
# Plot #
########

fig, axs = plt.subplots(n_rasters, bands, figsize=(bands * 3, n_rasters * 3), squeeze=False)
for i, raster in enumerate(rasters):
  for b in range(bands):
    axs[i, b].imshow(raster[:,:,b])
    axs[i, b].set_xticks([])
    axs[i, b].set_yticks([])

plt.tight_layout()
if out_file is None:
  plt.show()
else:
  plt.savefig(out_file)
