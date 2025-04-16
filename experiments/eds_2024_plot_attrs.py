import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def get_attr_file(dir_str, prefix_str, method_str, cov_idx, run_idx):
    return "{}/{}_{}_{}__{}.npz".format(dir_str, prefix_str, method_str, cov_idx, run_idx)

directory = "eds_outputs/sstanom/1000000/"
directory_xai = "eds_outputs/sstanom/1000000/" + "xai/"
prefix = "valid"
methods = ["saliency", "input_x_gradient", "shap", "lime"]
method_names = ["gradient", "input X gradient", "SHAP", "LIME"]
n_methods = len(methods)
cov_idx = 1
n_runs = 4
run_idxs = np.array(range(n_runs))

sample_idx = 999900
attr_idx = 0


gt_file = directory + "/pwl-out_" + str(cov_idx) + ".npz"
data = np.load(gt_file)["attribution_maps"]
gt_map = data[sample_idx, :, :, 0]

gt = np.load(gt_file)["attributions"][sample_idx]

samples_file = directory + "/samples_" + str(cov_idx) + ".npz"
data = np.load(samples_file)["samples"]
sample_map = data[sample_idx, :, :, 0]

maps = [[gt_map for r in run_idxs] for m in range(n_methods + 1)]
corrs = np.zeros((n_runs, n_methods))

for m_idx, method in enumerate(methods):
    for run_idx in run_idxs:
        attr_file = get_attr_file(directory_xai, prefix, method, cov_idx, run_idx)

        data = np.load(attr_file)
        attrs = data["attribution_maps"]
        
        maps[m_idx][run_idx] = attrs[attr_idx,:,:,0]

        r, p_value = stats.pearsonr(
                 data["attributions"][attr_idx], gt)
        corrs[run_idx, m_idx] = r

maps = np.array(maps)

vmin = np.nanmin(maps)
vmax = np.nanmax(maps)
vmax = max(vmax, abs(vmin))
vmin = -vmax

fig, axs = plt.subplots(n_runs, n_methods + 1)
for run_idx in run_idxs:
    for m_idx, method in enumerate(methods):
        axs[run_idx, m_idx].imshow(maps[m_idx, run_idx], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[run_idx, m_idx].set_xticks([])
        axs[run_idx, m_idx].set_yticks([])
        axs[run_idx, m_idx].set_xlabel("{:4f}".format(corrs[run_idx, m_idx]))

for run_idx in run_idxs:
    axs[run_idx, 0].set_ylabel("model {}".format(run_idx))
for m_idx, method_name in enumerate(method_names):
    axs[0, m_idx].set_title(method_name)

for run_idx in run_idxs:
    axs[run_idx, -1].imshow(maps[-1, 0], vmin=vmin, vmax=vmax, cmap="bwr")
    axs[run_idx, -1].set_xticks([])
    axs[run_idx, -1].set_yticks([])


vmin = np.nanmin(sample_map)
vmax = np.nanmax(sample_map)
vmax = max(vmax, abs(vmin))
vmin = -vmax
axs[0, -1].imshow(sample_map, vmin=vmin, vmax=vmax, cmap="bwr")

plt.tight_layout()
plt.savefig("eds_outputs/xai_comparison.pdf")
