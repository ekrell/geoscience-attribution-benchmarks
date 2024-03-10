import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from optparse import OptionParser

# Options
parser = OptionParser()
parser.add_option("-s", "--samples_file",
                  help="Samples file ('.npz').")
parser.add_option("-a", "--attributions_file",
                  help="Attributions file ('.npz').")
parser.add_option("-o", "--outputs_file",
                  help="Model outputs file ('.npz').")
parser.add_option("-p", "--plot_file",
                  help="Path to store plot.")
parser.add_option("-i", "--indices",
                  help="Indices to plot (e.g. '0,2,4,6'.)",
                  default="0,1,2")
(options, args) = parser.parse_args()

plot_file = options.plot_file

# Samples
if options.samples_file is None:
  print("Expected a samples file ('-s').")
  print("Exiting...")
  exit(-1)
samples_file = Path(options.samples_file)
samples = np.load(samples_file)["samples"]

# Attributes
if options.attributions_file is None:
  print("Expected an attributions file ('-a').")
  print("Exiting...")
  exit(-1)
else:
  attrs_file = Path(options.attributions_file)
  attrmaps = np.load(attrs_file)["attribution_maps"]

# Function
if options.outputs_file is None:
  outputs_file = None
  y = None
else:
  outputs_file = Path(options.outputs_file)
  y = np.load(outputs_file)["y"]

# Samples to plot
idxs = np.array(options.indices.split(',')).astype("int")

print("")
print("Samples file:     {}".format(samples_file))
print("    'samples' :       {}".format(samples.shape))
print("Attributes file:  {}".format(attrs_file))
print("    'attribute_maps': {}".format(attrmaps.shape if attrmaps is not None else "--"))
print("Outputs file:     {}".format(outputs_file))
print("    'y' :             {}".format(y.shape if y is not None else "--"))

print("Plotting samples:")
print("  {}".format(idxs))
print("")

n_show = len(idxs)
vmax_samples = np.nanmax(np.abs(samples))
vmax_attrs = np.nanmax(np.abs(attrmaps))
fig, axs = plt.subplots(n_show, 2)
for i in range(n_show):
  sim = axs[i, 0].imshow(samples[idxs[i]], cmap="bwr",
          vmin=-1*vmax_samples, vmax=vmax_samples)
  aim = axs[i, 1].imshow(attrmaps[idxs[i]], cmap="bwr",
          vmin=-1*vmax_attrs, vmax=vmax_attrs)
  if y is not None:
    axs[i, 0].set_xlabel("y = {}".format(y[idxs[i]]))
  axs[i, 0].set_xticks([])
  axs[i, 0].set_yticks([])
  axs[i, 1].set_xticks([])
  axs[i, 1].set_yticks([])
  plt.colorbar(aim, ax=axs[i, 1])

axs[0, 0].set_title("Sample")
axs[0, 1].set_title("Attribution")
plt.tight_layout()

if plot_file is None:
    plt.show()
else:
    plt.savefig(plot_file)
