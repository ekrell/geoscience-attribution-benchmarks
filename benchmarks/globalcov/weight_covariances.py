# Exploring

def cov2cor(covariance):
  # Source: https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
  v = np.sqrt(np.diag(covariance))
  outer_v = np.outer(v, v)
  correlation = covariance / outer_v
  correlation[covariance == 0] = 0
  return correlation


import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-c", "--covariance_file",
                  help="Path to `.npz` file with covariance matrix to be manipulated.",
                  default="benchmarks/globalcov/sstanom_cov.npz")
parser.add_option("-w", "--weights",
                  help="Comma-delimited list of weights to add to create new covariances.",
                  default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
parser.add_option("-o", "--output_dir",
                  help="Path to store outputs.",
                  default="benchmarks/globalcov/out")
(options, args) = parser.parse_args()

# Original covariance file
cov_file = options.covariance_file
# Weights to vary correlation strength
weights = np.array(options.weights.split(",")).astype("float")
# Output directory
out_dir = options.output_dir

outplot_file = out_dir + "/compare_corrs.pdf"
out_file_fmt = out_dir + "/cov_{}.npz"
out_plot_fmt = out_dir + "/cov_{}.pdf"

# Read base covaraince data
cov_data = np.load(cov_file)
# Load covariance matrix
cov_orig = cov_data["covariance"]
# Load mask (for dealing with landmass pixels)
mask = cov_data["mask"]

# Convert covariance matrix to correlation matrix
cor_orig = cov2cor(cov_orig)

cor_orig = cov_orig

# Get correlation values (since symmetric, take upper tri)
vals_orig = cor_orig[np.triu_indices_from(cor_orig)]

# Init storage for modified covariance values
cor_vals = np.zeros((vals_orig.shape[0] - len(cor_orig), len(weights)))

# Create fully-correlated base matrix
fully_corr = np.ones(cor_orig.shape[0])

# Increase covariance strength
for i, wi in enumerate(weights):
  # Make a copy
  cor = cor_orig.copy()

  # New correlation matrix is weighted sum of original and fully
  cor = (1.0 - wi) * cor + (wi) * fully_corr

  # Plot matrix
  print("Generated new cov matrix: {}".format(out_file_fmt.format(i)))
  plt.imshow(cor, cmap="gray_r", vmax=1, vmin=-1)
  plt.savefig(out_plot_fmt.format(i))
  np.savez(out_file_fmt.format(i), covariance=cor, mask=mask)

  # Store the correlation values
  cor = cov2cor(cor)
  cor_vals[:, i] = cor[np.triu_indices_from(cor, k=1)].flatten()

# Compare correlation distributions
weight_strs = [""] + ["{0:.2f}".format(wi) for wi in weights]
fig, ax = plt.subplots(2)
for i in range(0,len(weights)):
    ax[0].violinplot(dataset=cor_vals[:, i], positions=[i])
    ax[1].violinplot(dataset=np.abs(cor_vals[:, i]), positions=[i])
# Set y limits
ax[0].set_ylim(-1., 1.)
ax[1].set_ylim(0, 1.)
# Set x labels
ax[0].set_xticklabels([]) 
ax[1].set_xticklabels(weight_strs)
ax[1].set_xlabel("correlation weight")
# Set y labels
ax[0].set_ylabel("Correlation coefficient")
ax[1].set_ylabel("Correlation coefficient")

ax[0].set_title("Correlation matrix values")
ax[1].set_title("Correlation matrix absolute values")

plt.savefig(outplot_file)
