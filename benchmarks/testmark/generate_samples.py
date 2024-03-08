import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def cov2cor(covariance):
  # Source: https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
  v = np.sqrt(np.diag(covariance))
  outer_v = np.outer(v, v)
  correlation = covariance / outer_v
  correlation[covariance == 0] = 0
  return correlation

def get_top_n_pixels(cor, n, mask, reverse=False):
  c = cor.copy()
  c[np.identity(cor.shape[0]).astype("bool")] = 0
  c[c > 0.95] = 0

  c = np.abs(c)
  csum = np.mean(c, axis=0)
  vals = np.sort(csum)
  idxs = np.argsort(csum)

  if reverse is False:
    selected_vals = vals[-n:]
    selected_idxs = idxs[-n:]
  else:
    selected_vals = vals[:n]
    selected_idxs = idxs[:n]

  mask_f = mask.flatten()
  
  n_avail = len(mask[mask == True])
  n_block = len(mask[mask == False]) 

  idx_mask = np.zeros(n_avail).astype("int")
  idx_mask[selected_idxs] = 1

  full_mask = np.zeros(n_avail + n_block).astype("int")
  full_mask[mask_f] = idx_mask

  full_idxs = np.where(full_mask == 1)[0]

  coords = np.unravel_index(full_idxs, mask.shape)
  return coords


cov_file = "benchmarks/sstanom/out/cov.npz"
samples_file = "benchmarks/sstanom/out/samples.npz"
n_copy_pixels = 200
w = 0.99
out_samples_file = "benchmarks/testmark/out/samples.npz"
n_samples = 50000

# Load NPZ file with cov data
cov_data = np.load(cov_file)
# The covariance matrix
cov = cov_data["covariance"]
# A mask that defines the raster dimensions &
# maps the covariance matrix elements to cells
mask = cov_data["mask"]
# Get shape
rows, cols = cov.shape

# Convert to correlation matrix
cor = cov2cor(cov)

top_pixel_coords = get_top_n_pixels(cor, n_copy_pixels, mask)
bot_pixel_coords = get_top_n_pixels(cor, n_copy_pixels, mask, reverse=True)

# Plot the replacment map
repmap = np.zeros(mask.shape)
repmap[top_pixel_coords] = 0.75
repmap[bot_pixel_coords] -= 0.25

plt.imshow(repmap)
plt.colorbar()
plt.show()

# Load samples
samples_data = np.load(samples_file)
samples = samples_data["samples"]

# Copy over the values
samples_repl = samples.copy()
samples_repl[:, bot_pixel_coords[0], bot_pixel_coords[1], :] = \
 w * samples[:, top_pixel_coords[0], top_pixel_coords[1], :]



data = samples_repl[0:10000,:]
m = data[0].flatten()
o = ~np.isnan(m)
x = np.reshape(data, (data.shape[0],  data.shape[1] * data.shape[2]))
x = x[:, o]
cov_ = np.cov(x.T)
# Compare covariance
fig, axs = plt.subplots(3,2)
axs[0, 0].imshow(cov)
axs[0, 1].imshow(cov_)
axs[1, 0].imshow(samples[0])
axs[1, 1].imshow(samples_repl[0])
axs[2, 0].imshow(samples[1])
axs[2, 1].imshow(samples_repl[1])
plt.show()

samples_repl = samples_repl[:n_samples]

np.savez(out_samples_file, samples=samples_repl)
