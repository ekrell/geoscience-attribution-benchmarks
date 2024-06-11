'''
This program is used to explain a model using XAI methods to
generate local explainations in the form of an attribution map.
The inputs are gridded 2D data and the features are superpixels.
Given a sequence of superpixel sizes, XAI is performed at multiple
levels of superpixel granularity.
'''

import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from optparse import OptionParser
import itertools
import os
import tensorflow as tf
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.disable_eager_execution()
from utils import get_valid_cells
import shap

###############
# XAI Methods #
###############

def calc_sums(attribs, patch_size):

  def apply_patch(in_image, out_image, top_left_x, top_left_y, patch_size):
    out_image[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size] = \
        np.sum(in_image[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size])
  
  rows, cols = attribs.shape
  img = np.zeros((rows, cols))
  for top_left_x in range(0, cols, patch_size):
    for top_left_y in range(0, rows, patch_size):
       apply_patch(attribs, img, top_left_x, top_left_y, patch_size)
  return img


# Occlusion maps
def calc_occlusion(img, model, valid_idxs, patch_size=2, class_idx=0, batch_size=128):

  def wrapper(x, model, valid_idxs):
    # x shape: (samples, rows, cols)
    if len(x.shape) == 2:
      x = x.reshape((1, x.shape[0] * x.shape[1]))
    else:
      x = x.reshape((x.shape[0], x.shape[1] * x.shape[2]))
    x = x[:, valid_idxs]
    return model.predict(x, verbose=None)

  def apply_patch(image, top_left_x, top_left_y, patch_size):
  # Create function to apply a white patch on an image
    patched_image = np.array(image, copy=True)
    patched_image[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size] = 0
    return patched_image

  rows = img.shape[0]
  cols = img.shape[1]

  # Make first prediction
  original_prob = wrapper(img, model, valid_idxs)
  sensitivity_map = np.zeros((rows, cols))
  # Initialize storage
  patch_batch = np.zeros((batch_size, rows, cols))
  all_predictions = []
  # Iterate the patch over the image and collect predictions
  b = 0
  for top_left_x in range(0, cols, patch_size):
      for top_left_y in range(0, rows, patch_size):
          # Mask out patch
          patched_image = np.array(apply_patch(img, top_left_x, top_left_y, patch_size))
          patch_batch[b] = patched_image
          b += 1
          # Predict batch
          if b == batch_size:
            predicted_classes = wrapper(patch_batch, model, valid_idxs)
            all_predictions.append(predicted_classes[:, class_idx])
            b = 0
  # Predict with remaining batch
  predicted_classes = wrapper(patch_batch, model, valid_idxs)
  all_predictions.append(predicted_classes[:, class_idx])
  # Combine predictions into single vector
  all_predictions = np.concatenate(all_predictions)
  # Use predictions to make occlusion map
  p_idx = 0
  for top_left_x in range(0, cols, patch_size):
    for top_left_y in range(0, rows, patch_size):
      confidence = all_predictions[p_idx]
      p_idx += 1
      diff = original_prob - confidence
      # Save confidence for this specific patched image in map
      sensitivity_map[
        top_left_y:top_left_y + patch_size,
        top_left_x:top_left_x + patch_size,
      ] = diff
  return sensitivity_map

# SHAP
def calc_shap(sample, model, valid_idxs, patch_size, mask_value=0.0, nsamples=10000):

  def get_patches(rows, cols, patch_size=2):
    # Returns a list of patches
    # where each patch is a tuple:
    # (top_y, bottom_y, top_x, bottom_x)

    # Calc number of patches
    n_patches = len(range(0, cols, patch_size)) * \
								len(range(0, rows, patch_size))
    # Init patch storage
    patches = np.zeros((n_patches, 4)).astype(int)
    pi = 0
    for top_left_x in range(0, cols, patch_size):
      for top_left_y in range(0, rows, patch_size):
        patches[pi] = (top_left_y, top_left_y + patch_size,
											 top_left_x, top_left_x + patch_size)
        pi += 1
    return patches


  def mask_patches(img, feature_mask, patches, mask_value):
    n_patched = feature_mask.shape[0]
    patched = np.zeros((n_patched, img.shape[0], img.shape[1]))
    patched[:] = img.copy()
    for i in range(n_patched):
      patch_idxs = np.where(feature_mask[i] == False)[0]
      for pi in patch_idxs:
        pc = patches[pi]
        patched[i, pc[0]:pc[1], pc[2]:pc[3]] = mask_value
    return patched


  def model_shap(feature_mask):
    sample = pkg[0]
    model = pkg[1]
    valid_idxs = pkg[2]
    patches = pkg[3]
    mask_value = pkg[4]
    x = mask_patches(sample, feature_mask, patches, mask_value)
    x = x.reshape((x.shape[0], x.shape[1] * x.shape[2]))[:, valid_idxs]
    return model.predict(x)

  def shap2map(sample, shap_values, patches):
    shap_map = np.zeros(sample.shape)
    for i, pc in enumerate(patches):
      shap_map[pc[0]:pc[1], pc[2]:pc[3]] = shap_values[i]
    return shap_map

  # Supports 2D maps only
  sample = np.squeeze(sample)
  # Get a mapping of vector feature indices to map coordinates
  patches = get_patches(sample.shape[0], sample.shape[1], patch_size=patch_size)
  # Binary input to SHAP that represents which image patches get masked
  feature_mask = np.ones(len(patches)).astype(int)
  # Background data for SHAP. Zeros so that the binary input get set to 0 (means mask)
  background = np.zeros((1, len(patches)))

  global pkg
  pkg = [sample, model, valid_idxs, patches, mask_value]

  # Init SHAP
  explainer = shap.KernelExplainer(model_shap, background)
  # Calculate SHAP values vectors
  shap_values = explainer.shap_values(feature_mask, nsamples=nsamples, kwargs=sample)
  # Convert to 2D SHAP attribution map
  shap_map = shap2map(sample, shap_values, patches)

  return shap_map


def main():

  parser = OptionParser()
  parser.add_option("-s", "--samples_file",
                    help="Path to '.npz' with samples.")
  parser.add_option(     "--samples_varname",
                    default="samples",
                    help="Name of variable with raster samples in samples file.")
  parser.add_option("-m", "--model_files",
                    help="Comma-delimited list of paths to trained tensorflow models ('.h5').")
  parser.add_option("-i", "--indices",
                    help="Comma-delimited list of sample indices to explain.")
  parser.add_option("-x", "--xai_method",
                    default="occlusion",
                    help="Select which XAI method to use (occlusion, shap).")
  parser.add_option("-p", "--patch_sizes",
                    default="1,2,3",
                    help="Comma-delimited list of patch sizes to use ('2,3' means 2x2 and 3x3 sizes).")
  parser.add_option("-a", "--attributions_file",
                    help="Path to save attribution maps.")
  parser.add_option("-d", "--distribution_plot_file",
                    help="Path to save plot of attribution correlation distributions.")
  parser.add_option(     "--samples_plot_prefix",
                    help="Path prefix to save each sample's comparison maps. Will add 'IDX.pdf' to end.")
  parser.add_option(    "--sum_attribs",
                    default=False,
                    action="store_true",
                    help="Do pixel-level XAI, then sum those values into superpixels.")
  (options, args) = parser.parse_args()

  # Options
  samples_file = options.samples_file
  samples_varname = options.samples_varname
  sample_idxs = np.array(options.indices.split(",")).astype(int)

  model_files = options.model_files.split(",")

  patch_sizes = np.array(options.patch_sizes.split(",")).astype(int)
  n_patch_sizes = len(patch_sizes)
  patch_size_names = ["{}x{}".format(ps, ps) for ps in patch_sizes]

  # XAI method options
  xai_method = options.xai_method
  sum_attribs = options.sum_attribs

  # Save attributions
  out_attribs_file = options.attributions_file

  plot_distributions = False
  plot_samples = False

  # Save distribution comparison plot
  out_distplot_file = options.distribution_plot_file
  if out_distplot_file is not None:
    plot_distributions = True

  # Save each sample's attribution comparison
  out_sampleplots_prefix = options.samples_plot_prefix
  if out_sampleplots_prefix is not None:
    plot_samples = True

  # Load samples
  samples = np.load(samples_file)
  samples = samples[samples_varname]
  samples = samples[sample_idxs]
  n_samples, rows, cols, bands = samples.shape

  # Load models
  models = [keras.models.load_model(model_file) for model_file in model_files]
  n_models = len(models)

  # Formatting
  samples_vec = np.reshape(samples, (n_samples, rows * cols * bands))
  sample_cells, valid_idxs, n_valid_cells = get_valid_cells(samples_vec)

  # Combinations for comparing attributions
  compares = list(itertools.combinations(range(n_models), 2))
  # Init storage for mean correlations
  mean_corrs = np.zeros((n_samples, n_patch_sizes))

  # Run XAI method
  superpixel_attribs = np.zeros((n_models, n_patch_sizes, n_samples, rows, cols))
  for si, sample in enumerate(samples):
    sample = np.squeeze(sample)
    for pi, patch_size in enumerate(patch_sizes):
      for mi, model in enumerate(models):

        print("Sample {}, patch size {}, model {}".format(si, patch_size, mi))

        # Sum-up XAI
        if sum_attribs:
          if pi == 0:
            if xai_method == "occlusion":
              superpixel_attribs[mi, pi, si] = calc_occlusion(sample, model, valid_idxs, patch_size)
            elif xai_method == "shap":
              superpixel_attribs[mi, pi, si] = calc_shap(sample, model, valid_idxs,
                                               patch_size=patch_size, mask_value=0.0)
          else:
            superpixel_attribs[mi, pi, si] = calc_sums(superpixel_attribs[mi, 0, si], patch_size=patch_size)

        # Normal XAI
        else:
          if xai_method == "occlusion":
            superpixel_attribs[mi, pi, si] = calc_occlusion(sample, model, valid_idxs, patch_size)
          elif xai_method == "shap":
            superpixel_attribs[mi, pi, si] = calc_shap(sample, model, valid_idxs,
                                             patch_size=patch_size, mask_value=0.0)
          else:
            print("Unrecognized XAI method: {}.".format(xai_method))
            print("Exiting...")
            exit(-1)

      # Compare models
      mean_correlation = 0.0
      for compare in compares:
        mean_corrs[si, pi] += np.corrcoef(superpixel_attribs[compare[0], pi, si].flatten(),
                                          superpixel_attribs[compare[1], pi, si].flatten())[0, 1]
      mean_corrs[si, pi] = mean_corrs[si, pi] / len(compares)

    # Plot sample comparison
    if plot_samples:
      plot_sample_attribs(superpixel_attribs[:, :, si, :, :], mean_corrs[si], patch_size_names)
      plt.savefig(out_sampleplots_prefix + str(si) + ".pdf")

  # Save attributions
  np.savez(
      out_attribs_file,
      group_attributions=superpixel_attribs,
      mean_correlations=mean_corrs,
      patch_sizes=np.array(patch_sizes),
      method=np.array([xai_method]),
      cmd=np.array([" ".join(sys.argv[:])])
  )

  # Plot correlation distribution
  if plot_distributions:
    data = [mean_corrs[:, pi] for pi in range(n_patch_sizes)]
    fig, ax = plt.subplots()
    sns.violinplot(data=data, ax=ax, palette="Blues")
    ax.set_xticks(range(n_patch_sizes))
    ax.set_xticklabels(patch_size_names)
    ax.set_title("Correlation between XAI from multiple NN training repetitions")
    ax.set_ylabel("Pearson correlation")
    ax.set_xlabel("Superpixel size")
    ax.set_ylim(-1, 1)
    plt.savefig(out_distplot_file)


if __name__ == "__main__":
    main()
