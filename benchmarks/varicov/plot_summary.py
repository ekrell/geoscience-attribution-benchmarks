import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from math import ceil
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-i", "--input_dir",
                  help="Path to input directory.",
                  default="benchmarks/unicov/out/")
parser.add_option("-x", "--xai_label",
                  help="Label in filenames that denote XAI method used (e.g. 'input_x_gradient').",
                  default="input_x_gradient")
parser.add_option("-m", "--metric",
                  help="Metric for comparing performance.",
                  default="r-square")
parser.add_option("-c", "--output_corr_file", 
                  help="Path to save summary correlation plot.",
                  default="benchmarks/unicov/out/xai/corr_compare_summary.pdf")
parser.add_option("-s", "--output_scatter_file",
                  help="Path to to save comparison of correlations among models vs against ground truth.",
                  default="benchmarks/unicov/out/cai/corr_scatter_summary.pdf")
parser.add_option("-p", "--output_perf_file",
                  help="Path to save summary performance plot.",
                  default="benchmarks/unicov/out/xai/performance_summary.pdf")
(options, args) = parser.parse_args()

input_dir = options.input_dir
xai_label = options.xai_label
metric = options.metric
out_performance_file = options.output_perf_file
out_corr_file = options.output_corr_file
out_scatter_file = options.output_scatter_file

print("")
print("Benchmark Summary Plot")
print("-----------------------------")
print("Plots summary figures based on a set of benchmarks.")
print("")
print("Options:")
print("Directory with outputs: " + input_dir)
print("XAI method for correlation comparison: " + xai_label)
print("    (must match the strings used to label the xai outputs in the input dir.)")
print("Metric for model performance comparison: " + metric)
print("    (must match a column name in the 'nn_metrics_*__*.csv' file.)")
print("Path to save correlation comparison:" + out_corr_file)
print("Path to save performance comparison: " + out_performance_file)
print("")


# Finding out how many covariance matrices and NN runs are within the input dir
metrics_file_fmt = input_dir + "/nn_metrics_{}__{}.csv"
nn_files = glob.glob(metrics_file_fmt.format("*", "*"))
nn_reps = np.array([(fname.split("_")[-1].split(".")[0], fname.split("_")[-3]) for fname in nn_files]).astype("int")
runs = np.unique(np.sort(nn_reps[:,0]))
n_runs = len(runs)
cov_labels = np.sort(np.unique(nn_reps[:,1]))
n_covs = len(cov_labels)

# Open first file to find the number of samples
corr_file_fmt = input_dir + "/xai/" + "pwl-input_x_gradient_{}__{}.csv"
corr_file = corr_file_fmt.format(cov_labels[0], 0)
dfC = pd.read_csv(corr_file)
n_samples=dfC["pearson"].values.shape[0]

print("Searching directory (unicov output files)...")
print("  Found {} covariance matrices:".format(n_covs), cov_labels)
print("  Found {} NN model training runs.".format(n_runs))
print("  Found {} samples in a file.".format(n_samples))
print("")

# Plot the model performance
m__ = np.ones((n_covs, 2))
dfMetrics_train = pd.DataFrame()
dfMetrics_valid = pd.DataFrame()
for run_idx in range(n_runs):
  for cov_idx in range(n_covs):
    metrics_file = metrics_file_fmt.format(cov_labels[cov_idx], run_idx)
    df_metrics = pd.read_csv(metrics_file)
    m__[cov_idx, 0] = df_metrics[metric].values[0]
    m__[cov_idx, 1] = df_metrics[metric].values[1]
    dfMetrics_train["run_{}".format(run_idx)] = m__[:,0]
    dfMetrics_valid["run_{}".format(run_idx)] = m__[:,1]
train_metrics = dfMetrics_train.values
valid_metrics = dfMetrics_valid.values
fig, ax = plt.subplots(figsize=(14,4))

min_t = np.min(train_metrics)
min_v = np.min(valid_metrics)
min_v = min([min_t, min_v, 0.5])

ax.set_ylim(min_v, 1.0)

# Plot metrics for each run
for run_idx in range(n_runs):
  ax.scatter(x=cov_labels, 
          y=train_metrics[:,run_idx], marker="*", facecolors='tab:blue', 
          edgecolors='k', s=400, alpha=0.5)
  ax.scatter(x=cov_labels, 
          y=valid_metrics[:,run_idx], marker="o", facecolors='tab:red', 
          edgecolors='k', s=400, alpha=0.5)

plt.savefig(out_performance_file)
plt.clf()

f, axs = plt.subplots(2)
sns.set_palette("muted")

# Plot the correlation variance (corr between XAI and known attributions)
corrs = np.zeros((n_covs, n_runs, n_samples))
for cov_idx in range(n_covs):
  for run_idx in range(n_runs):
    corr_file = corr_file_fmt.format(cov_idx, run_idx)
    dfC = pd.read_csv(corr_file)
    corrs_ = dfC["pearson"].values
    corrs[cov_idx, run_idx, :] = corrs_
mean_corrs = np.mean(corrs, axis=1)
sns.violinplot(data=[d for d in mean_corrs], linewidth = 1, ax=axs[0])
axs[0].set_ylim(-.25, 1.0)
axs[0].set_title("Correlation between XAI & known attributions")

mean_corrs_known = mean_corrs.copy()

# Plot another one (corr between XAI... multiple training reps)
xai_corr_file_fmt = input_dir + "/xai/" + xai_label + "_{}__0v{}.csv"
corrs = np.zeros((n_covs, n_runs-1, n_samples))
for cov_idx in range(n_covs):
  for run_idx in range(1, n_runs):
    corr_file = xai_corr_file_fmt.format(cov_idx, run_idx)
    dfC = pd.read_csv(corr_file)
    corrs_ = dfC["pearson"].values
    corrs[cov_idx, run_idx-1, :] = corrs_
mean_corrs = np.mean(corrs, axis=1)
sns.violinplot(data=[d for d in mean_corrs], linewidth = 1, ax=axs[1])
axs[1].set_ylim(-.25, 1.0)
axs[1].set_title("Correlation between XAI from multiple NN training repetitions")

plt.tight_layout()
plt.savefig(out_corr_file)

# Scatter: mean correlation among models
#  versus  mean correlation against ground truth

# Make a subplot for each covariance
n_cols = min(3, n_covs)
n_rows = ceil(n_covs / 3)
fig, ax = plt.subplots(n_rows, n_cols)

# Consitent axes limites
min_1 = np.nanmin(mean_corrs)
min_2 = np.nanmin(mean_corrs_known)
min_v = min([min_1, min_2])

# Color code for consistency with above plots
cmap = plt.get_cmap("tab10")

# For each covariance
row = 0
col = 0
for cov_idxs in range(n_covs):
    # Mean explanation correlation among trained models
    mc_within = mean_corrs[cov_idxs]
    # Mean explanation correlation against ground truth
    mc_against = mean_corrs_known[cov_idxs]
    # Scatter
    ax[row, col].scatter(mc_within, mc_against, color=cmap(cov_idxs), marker=".")
    ax[row, col].set_ylim(min_v, 1)
    ax[row, col].set_xlim(min_v, 1)

    # Ticks only on left and bottom
    if col != 0:
        ax[row, col].set_yticks([])
    else:
        ax[row, col].set_ylabel("Against F")
    if row != n_rows - 1:
        ax[row, col].set_xticks([])
    else:
        ax[row, col].set_xlabel("Among Trained Models")

    # Title
    ax[row, col].set_title("Covariance {}".format(cov_idxs))
    
    # Update subplot coords
    row = row + 1
    if row >= 3:
        row = 0
        col = col + 1

plt.tight_layout()
plt.savefig(out_scatter_file)

print("Saving performance plot to: {}".format(out_performance_file))
print("Saving correlation plot to: {}".format(out_corr_file))
print("Saving scatter plot to: {}".format(out_scatter_file))
print("")
