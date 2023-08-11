import numpy as np

def get_valid_cells(arr):
  # Subset only the non-NaN cells
  valid_idxs = np.argwhere(~np.isnan(arr[0])).flatten()
  sample_cells = arr[:, valid_idxs]
  n_valid_cells = len(valid_idxs)
  return sample_cells, valid_idxs, n_valid_cells
