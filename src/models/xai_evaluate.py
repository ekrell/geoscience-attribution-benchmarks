import numpy as np
import pandas as pd
import h5py
from optparse import OptionParser
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
tf.keras.utils.disable_interactive_logging()
import keras
from utils import get_valid_cells

def flatten(x):
  if len(x.shape) == 4:
    n, rows, cols, bands = x.shape
  elif len(x.shape) == 3:
    rows, cols, bands = x.shape
    n = 1
  x = np.reshape(x, (n, rows * cols * bands))
  x, _, _ = get_valid_cells(x)
  return x

def faithfulness_correlation(model, x, y, a, params):

  # Based on Bhatt et al. (2020)
  # DOI:10.24963/ijcai.2020/417
  # Code is based in implementation in Quantus package

  def evaluate_instance(model, x, y, a, params):
    # Predict on input
    y_pred = model(np.expand_dims(x, axis=0))

    pred_deltas = np.zeros(params["n_runs"])
    attr_sums = np.zeros(params["n_runs"])
    # For each test data point
    for i_ix in range(params["n_runs"]):
      # Randomly mask the attribution by subset size
      a_ix = np.random.choice(a.shape[0], params["subset_size"], replace=False)
      x_perturbed = x.copy()
      x_perturbed[a_ix] = params["mask_value"]
      # Predict on perturbed input
      y_pred_perturbed = model(np.expand_dims(x_perturbed, axis=0))
      pred_deltas[i_ix] = y_pred[0][0] - y_pred_perturbed[0][0]

      # Sum attributions of the random subset
      attr_sums[i_ix] = np.sum(a[a_ix])

    # Correlation between sum of masked attributions and change in prediction
    corr = np.corrcoef(pred_deltas, attr_sums)[0,1]
    return corr

  n_samples = x.shape[0]
  corrs = np.zeros(n_samples)
  for i in range(n_samples):
    print("  instance: ", i)
    corrs[i] = evaluate_instance(model, x[i], y[i], a[i], params)
  return corrs


parser = OptionParser()
parser.add_option("-m", "--model_file",
                  help="Path to trained model (tensorflow).")
parser.add_option("-s", "--samples_file",
                  help="Path to input samples ('.npz').")
parser.add_option("-i", "--samples_indices", 
                  help="Comma-delimited list of samples to include (first is '0').")
parser.add_option("-t", "--targets_file",
                  help="Path to targets ('.npz').")
parser.add_option("-a", "--attribs_file",
                  help="Path to XAI attributions ('.npz'). Use commas to provide multiple.")
parser.add_option("-j", "--attribs_indices",
                  help="Comma-delimited list of attributions to include (first is '0').")
parser.add_option("-g", "--groundtruth_file",
                  help="(Optional) path to ground truth attributions ('.npz').")
parser.add_option("-o", "--output_file", 
                  help="Path to store computed scores.")
parser.add_option("-p", "--plot_file",
                  help="(Optional) path to store plot.")
parser.add_option("-e", "--eval_metric",
                  help="Evaluation metric string",
                  default="faithfulness_correlation,30,50,0.0")
parser.add_option("-c", "--colnames",
                  help="Custom column names for each attribution's score. Match order of filenames.")

(options, args) = parser.parse_args()

# Trained model
model_file = options.model_file
# Predictors
samples_file = options.samples_file
samples_varname = "samples"
# Targets
targets_file = options.targets_file
targets_varname = "y"
# Attributions
attribs_file = options.attribs_file
attribs_varname = "attributions"
# Ground truth attributions (optional)
gtattrs_file = options.groundtruth_file
gtattrs_varname = "attributions"
# Output file
out_file = options.output_file
# CSV column names
colnames = options.colnames
if colnames is not None:
    colnames = colnames.split(",")
# Plot file
plot_file = options.plot_file

# Subset
samples_idxs = np.array(options.samples_indices.split(",")).astype("int")
gtattrs_idxs = np.array(options.samples_indices.split(",")).astype("int")
attribs_idxs = np.array(options.attribs_indices.split(",")).astype("int")

# Evaluatation metric
eval_str = options.eval_metric
eval_opts = eval_str.split(",")
metric = eval_opts[0]
args = eval_opts[1:]

if metric == "faithfulness_correlation":
    metric_func = faithfulness_correlation
    params = {
      "n_runs" : int(args[0]),
      "subset_size" : int(args[1]),
      "mask_value" : float(args[2]),
      }
else:
    print("Unrecognized metric: {}".format(metric))
    print("Exiting...")
    exit(-1)

# Load model
model = keras.models.load_model(model_file)
# Load numpy archives
samples = np.load(samples_file)[samples_varname]
targets = np.load(targets_file)[targets_varname]
if gtattrs_file is not None:
    gtattrs = np.load(gtattrs_file)[gtattrs_varname]

samples = samples[samples_idxs]
targets = targets[samples_idxs]
if gtattrs_file is not None:
    gtattrs = gtattrs[samples_idxs]

print("Model file: ", model_file)
print("Samples file: ", samples_file)
print("  shape: ", samples.shape)
print("Targets file: ", targets_file)
print("  shape: ", targets.shape)

sample_cells = flatten(samples)  
if gtattrs_file is not None:
    gtattrs = np.load(targets_file)[gtattrs_varname]

attribs_files = attribs_file.split(",")
n_attribs = len(attribs_files)

all_scores = [None for i in range(n_attribs)]
all_corrs = [None for i in range(n_attribs)]
for a_i, attribs_file in enumerate(attribs_files):
    # Load attributions
    attribs = np.load(attribs_file)[attribs_varname]
    attribs = attribs[attribs_idxs]

    print("Attributions file: ", attribs_file)
    print("  shape: ", attribs.shape)

    scores = metric_func(model.predict, sample_cells, targets, attribs, params)
    all_scores[a_i] = scores

    if gtattrs_file is not None:
        corrs = np.zeros(len(samples))
        for i in range(len(samples)):
            corrs[i] = np.corrcoef(gtattrs[i], attribs[i])[0,1]
        all_corrs[a_i] = corrs

dfScores = pd.DataFrame()

if colnames is None:
    colnames = ["attribs_{}".format(i) for i in range(len(all_scores))]

for i, scores in enumerate(all_scores):
    dfScores[colnames[i]] = scores   ###
dfScores.to_csv(out_file, index=False)

n_axs = 1
if gtattrs_file is not None:
    n_axs = 2

fig, axs = plt.subplots(n_axs, squeeze=False)
# Distributions of faithfulness scores
sns.violinplot(data=all_scores, ax=axs[0][0])
axs[0][0].set_ylim(-1.0, 1.0)
axs[0][0].set_title(eval_str)
# Distributions of ground truth corrs
if gtattrs_file is not None:
    sns.violinplot(data=all_corrs, ax=axs[1][0])
    axs[1][0].set_ylim(-1.0, 1.0)
    axs[1][0].set_title("Ground Truth Correlation")
plt.tight_layout()

if plot_file is not None:
    plt.savefig(plot_file)
