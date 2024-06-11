import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from optparse import OptionParser

def plot_sample_attribs(attribs, mean_corrs, patch_size_names, vmin=None, vmax=None):
  n_patches = attribs.shape[1]
  n_models = attribs.shape[0]

  if vmin is None:
    vmin = np.nanmin(attribs)
  if vmax is None:
    vmax = np.nanmax(attribs)

  vv = max(abs(vmin), abs(vmax))

  fig, axs = plt.subplots(n_models, n_patches, figsize=(2*n_patches, 1*n_models), squeeze=False)
  for mi in range(n_models):
    for pi in range(n_patches):
      axs[mi, pi].imshow(attribs[mi, pi], vmin=-vv, vmax=vv, cmap="bwr")
      axs[mi, pi].set_xticks([])
      axs[mi, pi].set_yticks([])

  for pi in range(n_patches):
    axs[0, pi].set_title(patch_size_names[pi])
    axs[-1, pi].set_xlabel("avg corr = {0:.4f}".format(mean_corrs[pi]))

parser = OptionParser()
parser.add_option("-a", "--attribution_files",
                  help="Comma-delimited list of attribution_files.")
parser.add_option("-o", "--output_dirs",
                  help="Comma-delimited list of directories to save plots.")

(options, args) = parser.parse_args()

attribution_files = options.attribution_files.split(",")
n_attrs = len(attribution_files)
out_dirs = options.output_dirs.split(",")
n_dirs = len(out_dirs)

if n_attrs != n_dirs:
  print("Expected one-to-one match of attribition files to output directories.")
  print("Found {} attribution files,  {} output directories.".format(n_attrs, n_dirs))
  print("Exiting...")
  exit(-1)

attribs_varname = "group_attributions"
corrs_varname = "mean_correlations"

# Get patch sizes
patch_sizes = np.load(attribution_files[0])["patch_sizes"]
patch_labels = ["{}x{}".format(ps, ps) for ps in patch_sizes]
n_patch_sizes = len(patch_sizes)

# Load attributions
all_attribs = np.array([np.load(attrfile)[attribs_varname] for attrfile in attribution_files])
n_attribs, n_models, n_patch_sizes, n_samples, n_rows, n_cols = all_attribs.shape

# Load mean correlations
all_corrs = np.array([np.load(attrfile)[corrs_varname] for attrfile in attribution_files])

vmin = np.nanmin(all_attribs)
vmax = np.nanmax(all_attribs)

vmin=None
vmax=None

for ai in range(n_attribs):
  for si in range(n_samples):
    plot_sample_attribs(all_attribs[ai, :, :, si, :, :], all_corrs[ai, si], 
        patch_labels, vmin=vmin, vmax=vmax)
    plt.savefig(out_dirs[ai] + "/" + str(si) + ".pdf")
