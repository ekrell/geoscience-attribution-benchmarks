import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-i", "--input_dir",
                  help="Where to store outputs.")
parser.add_option("-c", "--covariance_label",
                  help="Label to grab files related to specific cov matrix.")
parser.add_option("-x", "--xai_label",
                  help="Name of xai method to get files.")
parser.add_option("-o", "--output_file",
                  help="Path to save plot.")
(options, args) = parser.parse_args()

# Directory
in_dir = options.input_dir
# Covariance label
cov_label = options.covariance_label
# xai_label
xai_label = options.xai_label
# Output file
out_file = options.output_file

# Path to input samples
samples_file = in_dir + "/samples_" + cov_label  + ".npz"

# Path to covariance matrix
cov_file = in_dir + "/cov_" + cov_label  + ".npz"

# Path to PWL weights
pwl_file = in_dir + "/pwl_fun_" + cov_label  + ".npz"

# Path to PWL attributions
attr_file = in_dir + "/pwl-out_" + cov_label  + ".npz"

# Paths to XAI outputs
xai_files = [
  in_dir + "/xai/" + xai_label + "_" + cov_label  + "__0.npz",
  in_dir + "/xai/" + xai_label + "_" + cov_label  + "__1.npz",
  in_dir + "/xai/" + xai_label + "_" + cov_label  + "__2.npz",
]

# Paths to XAI correlations
xai_corr_files = [
  in_dir + "/xai/pwl-" + xai_label + "_" + cov_label  + "__0.csv",
  in_dir + "/xai/pwl-" + xai_label + "_" + cov_label  + "__1.csv",
  in_dir + "/xai/pwl-" + xai_label + "_" + cov_label  + "__2.csv",
]

# Paths to model metrics
metrics_files = [
  in_dir + "/nn_metrics_" + cov_label  + "__0.csv",
  in_dir + "/nn_metrics_" + cov_label  + "__1.csv",
  in_dir + "/nn_metrics_" + cov_label  + "__2.csv",
]

# Sample idxs
sample_idxs = [1, 10, 100]

# XAI idxs
xai_idxs = [0, 1, 2]

n_samples = len(sample_idxs)
n_xais = len(xai_files)

# Open data files
samples_npz = np.load(samples_file)
samples = samples_npz["samples"]
samples = samples[sample_idxs]

_, n_rows, n_cols, _ = samples.shape

cov_npz = np.load(cov_file)
cov = cov_npz["covariance"]

pwl_npz = np.load(pwl_file)
weights = pwl_npz["weights"]
weights = np.sum(weights, axis=1)
weights = np.reshape(weights, (n_rows, n_cols))

attr_npz = np.load(attr_file)
attr = attr_npz["attribution_maps"]
attr = attr[sample_idxs]

xais = [None for i in range(n_xais)]
for xidx, xai_file in enumerate(xai_files):
  x = np.load(xai_file)
  xais[xidx] = x["attribution_maps"][xai_idxs]

xai_corrs = np.zeros((n_samples, n_xais))
for xidx, corr_file in enumerate(xai_corr_files):
  dfCorr = pd.read_csv(corr_file)
  corrs = dfCorr["spearman"].values
  xai_corrs[xidx,:] = corrs[xai_idxs]

r2s = np.zeros(n_xais)
for i, metrics_file in enumerate(metrics_files):
  dfMetrics = pd.read_csv(metrics_file)
  valid = dfMetrics.loc[dfMetrics["dataset"] == "validation"]
  r2s[i] = valid["r-square"].values

vmax_sample = np.max([np.abs(np.min(samples)), np.abs(np.max(samples))])
vmin_sample = -1 * vmax_sample

vmax_attr = np.max([np.abs(np.min(attr)), np.abs(np.min(attr)),
                    np.abs(np.min(np.concatenate(xais))), 
                    np.abs(np.max(np.concatenate(xais)))])
vmin_attr = -1 * vmax_attr

# Plot
n_plot_rows = n_samples + 1
n_plot_cols = n_xais + 2
fig, axs = plt.subplots(n_plot_rows, n_plot_cols, figsize=(2*n_plot_cols, 2*n_plot_rows))
for r in range(n_plot_rows):
  for c in range(n_plot_cols):
    axs[r, c].set_xticks([])
    axs[r, c].set_yticks([])

axs[0, 0].imshow(cov, cmap="gray_r", vmin=0, vmax=1)
axs[0, 0].set_title("covariance")

axs[0, 1].imshow(weights, cmap="bwr")
axs[0, 1].set_title("PWL weight sums")

axs[1, 0].set_title("sample")
axs[1, 1].set_title("attribution")
for c in range(n_xais):
  axs[1, c + 2].set_title("xai {}, $r^2 =$ {:.3f}".format(c, r2s[c]))
  
for r, sample in enumerate(samples):
  axs[r + 1, 0].imshow(sample, cmap="bwr", vmin=vmin_sample, vmax=vmax_sample)
  axs[r + 1, 0].set_ylabel("{}".format(sample_idxs[r]))
  axs[r + 1, 1].imshow(attr[r], cmap="bwr", vmin=vmin_attr, vmax=vmax_attr)
  xai = xais[r]
  for c, x in enumerate(xai):  
    axs[r + 1, c + 2].imshow(xai[c], cmap="bwr", vmin=vmin_attr, vmax=vmax_attr)
    axs[r + 1, c + 2].set_xlabel("ρ = {:.3f}".format(xai_corrs[r, c]))

for c in range(n_xais):
  axs[0, c + 2].axis("off")

plt.savefig(out_file)
