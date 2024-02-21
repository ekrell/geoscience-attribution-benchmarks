# Exploring

def corr2cov(covariance):
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
                  default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8")
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

# Read base covaraince data
cov_data = np.load(cov_file)
# Load covariance matrix
cov_orig = cov_data["covariance"]
# Load mask (for dealing with landmass pixels)
cov_mask = cov_data["mask"]

# Convert covariance matrix to correlation matrix
cor_orig = corr2cov(cov_orig)
# Get correlation values (since symmetric, take upper tri)
vals_orig = cor_orig[np.triu_indices_from(cor_orig)]

# Init storage for modified covariance values
cor_vals = np.zeros((vals_orig.shape[0], len(weights)))

# Increase covariance strength
for i, wi in enumerate(weights):
  # Make a copy
  cor = cor_orig.copy()

  # Separately deal with pos and neg correlations
  cor = cor + cor * wi * (cor >= 0)
  cor = cor - cor * -wi * (cor < 0)

  # Remove outliers (and ensure within (-1, 1))
  poss = cor[cor > 0]
  top = np.min([np.percentile(poss, 97, axis=0), 0.99])
  negs = -cor[cor < 0]
  bot = np.max([np.percentile(negs, 97, axis=0), -0.99])
  cor[cor > top] = top
  cor[cor < -bot] = -bot

  # Store the correlation values
  cor_vals[:, i] = cor[np.triu_indices_from(cor)].flatten()

# Compare correlations
fig, ax = plt.subplots()
for i in range(0,len(weights)):
    ax.violinplot(dataset=cor_vals[:, i], positions=[i])
weight_strs = [""] + ["{0:.2f}".format(wi) for wi in weights]
ax.set_xticklabels(weight_strs)
ax.set_ylim(-1, 1)
ax.set_xlabel("Added correlation weight")
ax.set_ylabel("Correlation coefficient")
ax.set_title("Comparison of correlation matrix values")
plt.savefig(outplot_file)
