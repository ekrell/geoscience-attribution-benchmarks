# This script generates custom covariance matrices

import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-o", "--out_dir",
                  help="Where to store covariance matrices")
parser.add_option("-r", "--rows",
                  default=20,
                  type="int",
                  help="Number of raster rows")
parser.add_option("-c", "--cols",
                  default=23,
                  type="int",
                  help="Number of raster columns")
(options, args) = parser.parse_args()

# Where to save covariance matrices
out_dir = options.out_dir
out_file_fmt = out_dir + "/cov_{}.npz"
out_plot_fmt = out_dir + "/cov_{}.pdf"

# Number of rows & cols of synthetic data
rows = options.rows
cols = options.cols

# Rows, cols of square covariance matrix
n = rows * cols

weights = np.linspace(0.0, 1.0, 11)

ones = np.ones(n)
identity = np.identity(n)

mask = np.ones((rows, cols))

for i, w in enumerate(weights):
  cov = w * ones + (1.0 - w) * identity

  plt.imshow(cov, cmap="gray_r", vmax=1, vmin=0)
  plt.savefig(out_plot_fmt.format(i))

  np.savez(out_file_fmt.format(i), covariance=cov, mask=mask)
