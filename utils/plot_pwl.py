"""
This program is used to plots a piecwise linear function.
The function is represented by a set of weights and edges.
The input PWL file is a saved Numpy '.npz' file.
"""

import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser

# Options
parser = OptionParser()
parser.add_option("-p", "--pwl_file",
                  help="Path to '.npz' file with Piece-wise Linear function.")
parser.add_option("-i", "--indices",
                  default="0",
                  help="Comma-delimited list of indices of function to plot.")

(options, args) = parser.parse_args()

pwl_file = options.pwl_file
if pwl_file is None:
  print("Expected path to '.npz' file with Piece-wise Linear function ('-p').\nExiting...")
  exit(-1)

idxs = options.indices
idxs = np.array(idxs.split(",")).astype("int")
n_idxs = len(idxs)

pwl = np.load(pwl_file)
pwl_weights = pwl["weights"]
pwl_edges = pwl["edges"]

start = pwl_edges[1]
stop = pwl_edges[-2]

x = np.arange(start, stop, 0.01)
x_bin_idxs = np.digitize(x, pwl_edges).flatten() - 1
n_samples = len(x)

fig, axs = plt.subplots(1, n_idxs, squeeze=False, figsize=(4 * n_idxs, 4))

for i, cell_idx in enumerate(idxs):
  y = np.zeros(len(x))
  for xi in range(n_samples):

    value = x[xi]
    bin_idx = x_bin_idxs[xi]
    
    if x[xi] > 0:
      while pwl_edges[bin_idx] >= 0:
        y[xi] += pwl_weights[cell_idx, bin_idx] * (value - pwl_edges[bin_idx])
        value = pwl_edges[bin_idx]
        bin_idx = bin_idx - 1

    if x[xi] <= 0:
      while pwl_edges[bin_idx] < 0:
        y[xi] += pwl_weights[cell_idx, bin_idx] * (pwl_edges[bin_idx + 1] - value)
        value = pwl_edges[bin_idx + 1]
        bin_idx = bin_idx + 1

  axs[0, i].plot(x, y, color="red")
  axs[0, i].scatter(x, y)

plt.show()
