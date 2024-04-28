import numpy as np
import h5py
from optparse import OptionParser
import tensorflow as tf
tf.keras.utils.disable_interactive_logging()
import keras
from utils import get_valid_cells


# Trained model
model_file = "testout/nn_model_0__0.h5"
# Predictors
samples_file = "testout/samples_0.npz"
samples_varname = "samples"
# Targets
targets_file = "testout/pwl-out_0.npz"
targets_varname = "y"
# Attributions
attribs_file = "testout/pwl-out_0.npz"
#attribs_file = "testout/input_x_gradient_0__1.npz"
attribs_varname = "attributions"

samples_idxs = np.array(range(30))
attribs_idxs = np.array(range(30))

# Load model
model = keras.models.load_model(model_file)
# Load numpy archives
samples = np.load(samples_file)[samples_varname]
targets = np.load(targets_file)[targets_varname]
attribs = np.load(attribs_file)[attribs_varname]

samples = samples[samples_idxs]
targets = targets[samples_idxs]
attribs = attribs[attribs_idxs]

print("Samples: ", samples.shape)
print("Targets: ", targets.shape)
print("Attributions: ", attribs.shape)

def flatten(x):
  if len(x.shape) == 4:
    n, rows, cols, bands = x.shape
  elif len(x.shape) == 3:
    rows, cols, bands = x.shape
    n = 1
  x = np.reshape(x, (n, rows * cols * bands))
  x, _, _ = get_valid_cells(x)
  return x

def model_(x):
  x = flatten(x)
  return model.predict(x)




similarity_func = None
n_runs = 100
subset_size = 10
abs = False
normalize = False
perturb_baseline = 0    # "mean", "random", "uniform", "black" or "white"

params = {
  "n_runs" : n_runs,
  "subset_size" : subset_size,
  "mask_value" : 0.0,
  }


def faithfulness_correlation(model, x, y, a, params):

  def evaluate_instance(model, x, y, a, params):

    # Predict on input
    x_input = x
    y_pred = model(np.expand_dims(x, axis=0))

    pred_deltas = np.zeros(n_runs)
    att_sums = np.zeros(n_runs)

    # For each test data point
    for i_ix in range(n_runs):
      # Randomly mask the attribution by subset size
      a_ix = np.random.choice(a.shape[0], subset_size, replace=False)
      x_perturbed = x.copy()
      x_perturbed[a_ix] = params["mask_value"]
      # Predict on perturbed input
      y_pred_perturbed = model(np.expand_dims(x_perturbed, axis=0))
      pred_deltas[i_ix] = y_pred[0][0] - y_pred_perturbed[0][0]

      # Sum attributions of the random subset
      att_sums[i_ix] = np.sum(a[a_ix])

    # Correlation between sum of masked attributions and change in prediction
    corr = np.corrcoef(pred_deltas, att_sums)[0,1]
    return corr

  n_samples = x.shape[0]
  corrs = np.zeros(n_samples)

  for i in range(n_samples):
    print(i)
    corrs[i] = evaluate_instance(model, x[i], y[i], a[i], params)
    
  print(corrs)


sample_cells = flatten(samples)  
faithfulness_correlation(model.predict, sample_cells, targets, attribs, params)

