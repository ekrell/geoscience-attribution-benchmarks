import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from optparse import OptionParser
import scipy.stats as stats

# Options
parser = OptionParser()
parser.add_option("-s", "--samples_file",
                  help="Samples file ('.npz').")
parser.add_option("-a", "--attributions_file",
                  help="Attributions file ('.npz').")
parser.add_option("-i", "--indices_attr",
                  help="Attribution indices to plot (e.g. '0,2,4,6').",
                  default="0,1,2")
parser.add_option("-x", "--explanations_file",
                  help="Explanations file ('.npz').")
parser.add_option("-j", "--indices_xai",
                  help="XAI indices to plot (e.g. '2,4,6,8').")
parser.add_option("-o", "--outputs_file",
                  help="Model outputs file ('.npz').")
parser.add_option("-p", "--plot_file",
                  help="Path to store plot.")
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
  attrvals = np.load(attrs_file)["attributions"]

if options.explanations_file is None:
  print("Expected an explanations file ('-x').")
  print("Exiting...")
  exit(-1)
else:
  xai_file = Path(options.explanations_file)
  xaimaps = np.load(xai_file)["attribution_maps"]
  xaivals = np.load(xai_file)["attributions"]

# Function
if options.outputs_file is None:
  outputs_file = None
  y = None
else:
  outputs_file = Path(options.outputs_file)
  y = np.load(outputs_file)["y"]

# Samples to plot
attr_idxs = np.array(options.indices_attr.split(',')).astype("int")
xai_idxs = None
if options.indices_xai is not None:
  xai_idxs = np.array(options.indices_xai.split(',')).astype("int")
else:
  xai_idxs = attr_idxs

if len(xai_idxs) != len(attr_idxs):
  print("Trying to compare {} to {} attributions.\nExiting...")
  exit(-2)

print("")
print("Samples file:     {}".format(samples_file))
print("   'samples' :          {}".format(samples.shape))
print("Attributes file:  {}".format(attrs_file))
print("   'attribution_maps': {}".format(attrmaps.shape if attrmaps is not None else "--"))
print("XAI file:         {}".format(xai_file))
print("   'attribution_maps':  {}".format(xaimaps.shape))
print("Outputs file:     {}".format(outputs_file))
print("   'y' :                {}".format(y.shape if y is not None else "--"))

print("Plotting samples:")
print("  {}".format(attr_idxs))
print("Against XAI idxs:")
print("  {}".format(xai_idxs))

n_show = len(attr_idxs)
vmax_samples = np.nanmax(np.abs(samples))
vmax_attrs = np.nanmax(
  np.array([np.nanmax(np.abs(attrmaps)), 
            np.nanmax(np.abs(xaimaps))]))
fig, axs = plt.subplots(n_show, 3, figsize=(12, 3*n_show))
for i in range(n_show):
  sim = axs[i, 0].imshow(samples[attr_idxs[i]], cmap="bwr",
          vmin=-1*vmax_samples, vmax=vmax_samples)
  aim = axs[i, 1].imshow(attrmaps[attr_idxs[i]], cmap="bwr",
          vmin=-1*vmax_attrs, vmax=vmax_attrs)
  xim = axs[i, 2].imshow(xaimaps[xai_idxs[i]], cmap="bwr",
          vmin=-1*vmax_attrs, vmax=vmax_attrs)

  # Calc correlation
  r, p_value = stats.pearsonr(
        attrvals[attr_idxs[i]], xaivals[xai_idxs[i]])
  axs[i, 2].set_xlabel("pearson's = {}".format(r))

  if y is not None:
    axs[i, 0].set_xlabel("y = {}".format(y[attr_idxs[i]]))
  axs[i, 0].set_xticks([])
  axs[i, 0].set_yticks([])
  axs[i, 1].set_xticks([])
  axs[i, 1].set_yticks([])
  axs[i, 2].set_xticks([])
  axs[i, 2].set_yticks([])
  #plt.colorbar(aim, ax=axs[i, 1])

axs[0, 0].set_title("Sample")
axs[0, 1].set_title("Attribution")
axs[0, 2].set_title("XAI")
plt.tight_layout()

if plot_file is None:
    plt.show()
else:
    plt.savefig(plot_file)
