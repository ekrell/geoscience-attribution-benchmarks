import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

bmark_dir = "benchmarks/unicov/"
input_dir = bmark_dir + "/out/"
xai_label = "input_x_gradient"
cov_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
n_covs=len(cov_labels)
n_runs=3
n_samples=40
metric="r-square"
out_performance_file = input_dir + "/xai/performance_summary.pdf"
out_corr_file = input_dir + "/xai/corr_compare_summary.pdf"

# Plot the model performance
metrics_file_fmt = input_dir + "/nn_metrics_{}__{}.csv"
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
corr_file_fmt = input_dir + "/xai/" + "pwl-input_x_gradient_{}__{}.csv"
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
