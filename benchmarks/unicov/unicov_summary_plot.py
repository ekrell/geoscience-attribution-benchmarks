import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
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
parser.add_option("-p", "--output_perf_file",
                  help="Path to save summary performance plot.",
                  default="benchmarks/unicov/out/xai/performance_summary.pdf")
(options, args) = parser.parse_args()

input_dir = options.input_dir
xai_label = options.xai_label
metric = options.metric
out_performance_file = options.output_perf_file
out_corr_file = options.output_corr_file

print("")
print("UNICOV Benchmark Summary Plot")
print("-----------------------------")
print("Plots summary figures based on `unicov` set of benchmarks.")
print("Assumes that `create_unicov_benchmark.bash` has already been run.")
print("")
print("Options:")
print("Directory with unicov outputs: " + input_dir)
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
m__ = np.ones(n_covs)
dfMetrics = pd.DataFrame()
for run_idx in range(n_runs):
  for cov_idx in range(n_covs):
    metrics_file = metrics_file_fmt.format(cov_labels[cov_idx], run_idx)
    df_metrics = pd.read_csv(metrics_file)
    m__[cov_idx] = df_metrics[metric].values[1]
    dfMetrics["run_{}".format(run_idx)] = m__
dfMetrics["mean"] = [np.mean(dfMetrics.iloc[cov_idx]) for cov_idx in range(n_covs)]
fig, ax = plt.subplots(figsize=(14,4))
ax.scatter(x=cov_labels, y=dfMetrics["mean"])
plt.savefig(out_performance_file)

# Plot the correlation variance
corrs = np.zeros((n_covs, n_runs, n_samples))
for cov_idx in range(n_covs):
  for run_idx in range(n_runs):
    corr_file = corr_file_fmt.format(cov_idx, run_idx)
    dfC = pd.read_csv(corr_file)
    corrs_ = dfC["pearson"].values
    corrs[cov_idx, run_idx, :] = corrs_
mean_corrs = np.mean(corrs, axis=1)
sns.set_palette("muted")
sns.violinplot(data=[d for d in mean_corrs], linewidth = 3)
plt.savefig(out_corr_file)

print("Saving performance plot to: {}".format(out_performance_file))
print("Saving correlation plot to: {}".format(out_corr_file))
print("")
