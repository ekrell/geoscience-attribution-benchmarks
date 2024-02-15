'''
The purpose of this program is to plot attribution maps,
where each row contains attributions for a single sample. 

This works for single-channel rasters only!
'''

import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-a", "--attr_files",
                  help="Comma-delimited list of `.npz` files with attributions.")
parser.add_option("-s", "--sample_idxs",
                  help="Period-delimited list of comma-delimited list of sample indices for each attr file.")
parser.add_option("-n", "--names",
                  help="Comma-delimited list of names of each attribution source.")
parser.add_option("-o", "--output_plot",
                  help="Path to save plot.")
(options, args) = parser.parse_args()

attr_files = options.attr_files
sample_idxs = options.sample_idxs
attr_names = options.names
output_file = options.output_plot

opt_err = False

if attr_files is None:
  print("[-] Expected a list of '.npz' files containing attribution maps.")
  print("    Example:     -i file_a.npz,file_b.npz,file_c.npz")
  opt_err = True

if sample_idxs is None:
  print("[-] Expected a list of sample indices to subset the attribution maps.")
  print("    Example:     -s 0,10,100,200,300.0,1,2,3,4.0,1,2,3")
  print("    Why for each? Might have many ground truth attribs, but only run a few XAI.")
  opt_err = True

if attr_names is None:
  print("[-] Expected a list of names to label the attribution sources.")
  print("    Example:    -n ground_truth,input_x_gradient,saliency")
  opt_err = True

if opt_err:
  print("Exiting...")
  exit(-1)

if output_file is None:
  print("Since no output file was specified ('-o'), will plot to screen.")

attr_files = attr_files.split(",")
sample_idxs = sample_idxs.split(".")
sample_idxs = np.array([sample_idxs.split(",") for sample_idxs in sample_idxs]).astype("int")
attr_names = attr_names.split(",")

n_attrs = len(attr_files)
n_samples = sample_idxs.shape[1]

attr_data = np.array(
  [np.load(attr_files[i])["attribution_maps"][sample_idxs[i]] for i in range(n_attrs)]
)

min = np.nanmin(attr_data)
max = np.nanmax(attr_data)

fig, axs = plt.subplots(n_samples, n_attrs, squeeze=False, figsize=(3*n_attrs, 3*n_samples))

for s_i in range(n_samples):
  for a_i in range(n_attrs):
    axs[s_i, a_i].imshow(attr_data[a_i, s_i], vmin=min, vmax=max)
    axs[s_i, a_i].set_xticks([])
    axs[s_i, a_i].set_yticks([])

# Add column names
for a_i in range(n_attrs):
  axs[0, a_i].set_title(attr_names[a_i])

plt.tight_layout()

if output_file is None:
  plt.show()
else:
  print("Saving plot: {}".format(output_file))
  plt.savefig(output_file)
