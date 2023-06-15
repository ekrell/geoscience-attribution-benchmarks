
import os
import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt

###########
# Options #
###########

# File with covariance matrix
cov_file = "out/cov.npz"
# Number of synthetic samples to generate
n_samples = 10


##############
# Setup Data #
##############

# Load NPZ file with cov data
cov_data = np.load(cov_file)
cov = cov_data["covariance"]
mask = cov_data["mask"]

rows = mask.shape[0]
cols = mask.shape[1]
cov_elems = cov.shape[0]
grid_elems = rows * cols

# Which indices are valid in map
m = np.reshape(mask, rows * cols)
coords = np.where(m == True)
coords = coords[0]

#####################
# Synthetic Samples #
#####################

# TODO: support multiple samples

sample_ = np.random.multivariate_normal(np.zeros(cov_elems), cov)
sample = np.empty(grid_elems)
sample[:] = np.nan
sample[coords] = sample_
sample = np.reshape(sample, (rows, cols))


###############
# Plot Sample #
###############

plt.imshow(sample)
plt.show()
