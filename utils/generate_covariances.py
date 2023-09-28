# This script generates custom covariance matrices

import numpy as np

# Where to save covariance matrices
out_dir = "out/cov_exp/"
out_file_fmt = out_dir + "/cov_{}.npz"

# Number of rows & cols of synthetic data
rows = 18
cols = 36

# Rows, cols of square covariance matrix
n = rows * cols

weights = np.linspace(0.0, 1.0, 11)

ones = np.ones(n)
identity = np.identity(n)

mask = np.ones((rows, cols))

for i, w in enumerate(weights):
  cov = w * ones + (1.0 - w) * identity

  print(w, (1-w), i)

  np.savez(out_file_fmt.format(i), covariance=cov, mask=mask)

