# The purpose of this script is to generate masks to force a synthetic function
# that is based on additive local PWL functions to rely on specified regions. 

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def apply_gaussian(raster, row, col, kernel_size, std=1):
  k1d = signal.gaussian(kernel_size, std).reshape(kernel_size, 1)
  kernel = np.outer(k1d, k1d)
  raster[row - (kernel_size//2) : row + (kernel_size//2) + 1,
         col - (kernel_size//2) : col + (kernel_size//2) + 1] = kernel
  return raster


# File options
out_dir = "out/cov_exp/"
out_file_fmt = out_dir + "/mask_{}.npz"
# Raster options
rows = 20
cols = 23
# Mask options
mask_params = [
  (5, 5, 9, 3), 
  (13, 5, 9, 3), 
  (5, 17, 9, 3), 
  (13, 17, 9, 3), 
  (10, 11, 9, 3), 
]

show = True

# Initialize raster
mask_base = np.zeros((rows, cols))

for mp in mask_params:
  if mp[2] % 2 == 0:
    print("Does not support even kernel sizes. Skipping...")
    continue

  mask = apply_gaussian(mask_base.copy(), mp[0], mp[1], mp[2], mp[3]) 

  if show:
    plt.imshow(mask, cmap="gray")
    plt.show()

  out_file = out_file_fmt.format("{}-{}-{}-{}".format(mp[0], mp[1], mp[2], mp[3]))
  np.savez(out_file, mask=mask)

  




