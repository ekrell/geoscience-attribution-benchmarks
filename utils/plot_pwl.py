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

x = np.arange(start, stop, 0.005)
x_bin_idxs = np.digitize(x, pwl_edges).flatten()

fig, axs = plt.subplots(1, n_idxs, squeeze=False, figsize=(4 * n_idxs, 4))

for i, idx in enumerate(idxs):

  # Make weights
  new_weights = pwl_weights[i]
  #new_weights = np.zeros(len(pwl_weights[0]))
  #for wi in range(len(pwl_weights[0]) - 2):

  #  e1 = pwl_edges[wi + 1]
  #  e2 = pwl_edges[wi + 2]

  #  w1 = pwl_weights[idx, wi]
  #  w2 = pwl_weights[idx, wi + 1]

  #  nw = (w2 - w1) / (e2 - e1) 

  #  new_weights[wi] = nw

  y = np.zeros(len(x))
  for xi, bin_idx in enumerate(x_bin_idxs):
    y[xi] = new_weights[bin_idx] * x[xi] 


  axs[0, i].plot(x, y, color="red")
  axs[0, i].scatter(x, y)

plt.show()
