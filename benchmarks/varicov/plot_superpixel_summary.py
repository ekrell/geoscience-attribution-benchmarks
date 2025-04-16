import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from optparse import OptionParser
from scipy.stats import pearsonr

parser = OptionParser()
parser.add_option("-a", "--attribution_files",
                  help="Comma-delimited list of superpixel attribution files.")
parser.add_option("-l", "--labels",
                  help="[Optional] comma-delimited list of attribution labels, for the plots.")
parser.add_option("-o", "--out_file",
                  help="Path to output file.")
parser.add_option(      "--attribution_reference_files",
                  help="[Optional] comma-delimited list of reference attribution files.")
parser.add_option(      "--attribution_reference_samples",
                  help="[Optionsal] comma-delimited list of reference samples to use.")
(options, args) = parser.parse_args()

attribution_files = options.attribution_files.split(",")
outfile = options.out_file

# Optional: compare attributions to reference (e.g. ground truth)
ref_samples = None
compare_ref = False
if options.attribution_reference_files is not None:
  attribution_ref_files = options.attribution_reference_files.split(",")
  compare_ref = True
if options.attribution_reference_samples is not None:
  ref_samples = np.array(options.attribution_reference_samples.split(",")).astype(int)

if options.labels is not None:
    labels = options.labels.split(",")
    if len(labels) != len(attribution_files):
        print("Expected one label per attribution file.")
        print("Num attributions: {},  num labels: {}.".format(
            len(attribution_files), len(labels)))
        print("Exiting...")
        exit(-1)
else:
    labels = [str(i) for i in range(len(attribution_files))]
n_covs = len(labels)

patch_sizes = np.load(attribution_files[0])["patch_sizes"]
patch_labels = ["{}x{}".format(ps, ps) for ps in patch_sizes]
n_patch_sizes = len(patch_sizes)

# Compare attributions to reference attributions
if compare_ref:

  # This is a very hacky section! 
  # If we have reference attributions, we can compare to them
  # But it means doing XAI summations, loading attributions, ... 
  # Without a reference, this script uses pre-computed mean correaltions 
  # and is a very simple plotting script. So this adds bulk. 

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

  # Sanity check
  if len(attribution_ref_files) != len(attribution_files):
    print("Expected a reference file for each attribution file.")
    print("Exiting...")
    exit(-1)

  # Load attributions
  attrs = np.array([np.load(attr)["group_attributions"] for attr in attribution_files])
  n_models = attrs.shape[1]

  # Load reference attributions
  if ref_samples is None:
    ref_samples = np.array(range(attrs.shape[3]))
  refs = np.array([np.squeeze(np.load(ref)["attribution_maps"][ref_samples]) \
      for ref in attribution_ref_files])
  r_covs, r_samples, r_rows, r_cols = refs.shape

  refs = np.nan_to_num(refs)
  superrefs = np.zeros((r_covs, n_patch_sizes, r_samples, r_rows, r_cols))

  # Init mean correlation storage
  mean_ref_corrs = np.zeros((r_covs, r_samples, n_patch_sizes))

  # Convert to superpixel attributions
  for cidx in range(r_covs):
    for sidx in range(r_samples):
      for pidx in range(n_patch_sizes):
        if pidx == 0:
          superrefs[cidx, 0, sidx] = refs[cidx, sidx]
        else:
          superrefs[cidx, pidx, sidx] = calc_sums(refs[cidx, sidx], patch_sizes[pidx])

        # Calculate correlation
        for midx in range(n_models):
          mean_ref_corrs[cidx, sidx, pidx] += np.corrcoef(
              superrefs[cidx, pidx, sidx].flatten(), 
              attrs[cidx, midx, pidx, sidx].flatten())[0,1]
        mean_ref_corrs[cidx, sidx, pidx] /= n_models

      ### plot 
      ##fig, axs = plt.subplots(n_models + 1, n_patch_sizes)
      ##for pidx in range(n_patch_sizes):
      ##  a = np.concatenate([superrefs[cidx, pidx, sidx].flatten(),
      ##    attrs[cidx, midx, pidx, sidx].flatten()])
      ##  vmin = np.nanmin(a)
      ##  vmax = np.nanmax(a)
      ##  v = max(abs(vmin), abs(vmax))
      ##  axs[0][pidx].imshow(superrefs[cidx, pidx, sidx], cmap="bwr", vmin=-v, vmax=v)
      ##  axs[0, pidx].set_xticks([])
      ##  axs[0, pidx].set_yticks([])
      ##  for midx in range(n_models):
      ##    axs[midx+1][pidx].imshow(attrs[cidx, midx, pidx, sidx], cmap="bwr", vmin=-v, vmax=v)
      ##    axs[midx+1, pidx].set_xticks([])
      ##    axs[midx+1, pidx].set_yticks([])
      ##  axs[-1][pidx].set_xlabel("{0:.3f}".format(mean_ref_corrs[cidx, sidx, pidx]))
      ##plt.savefig("test-{}.pdf".format(sidx))

if compare_ref:
  mean_correlations = [mean_ref_corrs[i] for i in range(mean_ref_corrs.shape[0])]
else:
  mean_correlations = [np.load(attr_file)["mean_correlations"]
                     for attr_file in attribution_files]

n_samples = mean_correlations[0].shape[0]
mshape = mean_correlations[0].shape

for mcorr in mean_correlations:
  if mcorr.shape != mshape:
    print("Attributions have mismatched sizes!\nExiting...")
    exit(-2)

df_all = [None for c in range(n_covs)]

for covidx, corrs in enumerate(mean_correlations):
  c = np.repeat(range(n_samples), n_patch_sizes)
  values = np.zeros(len(c))
  ci = 0
  for pi in range(n_patch_sizes):
    for si in range(n_samples):
      values[ci] = corrs[si, pi]
      ci += 1

  df_all[covidx] = pd.DataFrame({
    "covariance" : np.repeat(labels[covidx], n_samples * n_patch_sizes),
    "patch_size" : np.repeat(patch_labels, n_samples),
    "value" : values
    })

df = pd.concat(df_all)
df = df.reset_index()

fig, ax = plt.subplots(figsize=(2*n_covs, 5))
sns.boxplot(x="patch_size", y="value", hue="covariance", data=df, ax=ax)
ax.set_ylim(-0.25, 1)
ax.set_ylabel("Pearson correlation")
ax.set_xlabel("Superpixel size")
plt.tight_layout()

print("Saved boxplots to: ", outfile)
plt.savefig(outfile)
