# This program generates a set of synthetic samples of raster data
# based on a provided set of generation commands. 

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from optparse import OptionParser

valid_cmds = {
  "RASTER": {
    "samples" : 1,
    "rows" : 10,
    "cols" : 10,
    "bands" : 1,
  },
  "SEED": {
    "coords": None, 
    "value" : None,
    "threshold" : None,
    "decay" : None,
    "value_dist" : None,
  },
  "DILATE": {
    "size" : None,
  },
  "BLUR": {
    "sigma" : None,
  },
  "CROP": {
    "low_row": None,
    "high_row": None,
    "low_col": None,
    "high_col": None,
    "low_band": None,
    "high_band": None,
  }
}


def filterBounds(coords_list, max_x, max_y, max_z, min_x=0, min_y=0, min_z=0):
  mins = (min_x, min_y, min_z)
  maxs = (max_x, max_y, max_z)
  a = coords_list
  return a[(a >= mins).all(axis=1) & (a <= maxs).all(axis=1)]


def grow(raster, mask, coords, threshold, decay, point_value, point_value_dist=0):
 
  # Adjust 2D data to be 3D (with one band)
  if len(raster.shape) == 2:
    raster = np.reshape(raster, (raster.shape[0], raster.shape[1], 1))
    mask = np.reshape(mask, (raster.shape[0], raster.shape[1], 1))
  if len(coords) == 2:
    coords = (coords[0], coords[1], 0)

  if mask[coords[0], coords[1], coords[2]] != 0:
    return
  mask[coords[0], coords[1], coords[2]] = 1.0

  sample = np.random.rand()
  if sample <= threshold:
    raster[coords[0], coords[1], coords[2]] = point_value + np.random.uniform(-1 * point_value_dist, point_value_dist)
  else:
    return

  neighbors = np.array([
      (coords[0] - 1, coords[1], coords[2]),
      (coords[0] + 1, coords[1], coords[2]),
      (coords[0], coords[1] - 1, coords[2]),
      (coords[0], coords[1] + 1, coords[2]),
      (coords[0] - 1, coords[1], coords[2] - 1),
      (coords[0] + 1, coords[1], coords[2] - 1),
      (coords[0], coords[1] - 1, coords[2] - 1),
      (coords[0], coords[1] + 1, coords[2] - 1),
      (coords[0] - 1, coords[1], coords[2] + 1),
      (coords[0] + 1, coords[1], coords[2] + 1),
      (coords[0], coords[1] - 1, coords[2] + 1),
      (coords[0], coords[1] + 1, coords[2] + 1),
  ])
  valid_neighbors = filterBounds(neighbors, raster.shape[1] - 1, raster.shape[0] - 1, raster.shape[2] - 1)
  for vn in valid_neighbors:
    grow(raster, mask, vn, threshold - decay, decay, point_value, point_value_dist)


def time_shift(raster, num_shifts, direction_degrees, magnitude_pixels, random_dist=0.0):

  def shift(raster, u, v, random_dist=0.0):
    rows = raster.shape[0]
    cols = raster.shape[1]
    raster_out = np.ones((rows, cols))
    
    for row in range(rows):
      for col in range(cols):
        # Calculate new point based on current point and vector
        row_new = int(row + u + np.random.uniform(-1 * random_dist, random_dist))
        col_new = int(col + v + np.random.uniform(-1 * random_dist, random_dist))
        # Shift to new point
        if row_new > 0 and row_new < rows and col_new > 0 and col_new < cols:
          raster_out[row_new, col_new] = raster[row, col]

    return raster_out

  # Convert from degrees to radians
  direction_radians = np.radians(direction_degrees)
  # Convert to vector components
  u = magnitude_pixels * np.cos(direction_radians)
  v = magnitude_pixels * np.sin(direction_radians)

  n_samples = raster.shape[0]
  rows = raster.shape[1]
  cols = raster.shape[2]

  shifted = np.ones((n_samples, rows, cols, num_shifts))
  for i in range(n_samples):
    shifted[i, :, :, 0] = raster[i, :, :, 0].copy()
    for j in range(1, num_shifts):
      shifted[i, :, :, j] = shift(shifted[i, :, :, j-1], u, v)

  return shifted


def parse_cmd(cmd):
  words = cmd.split()
  # First word defines the command
  cmd_type = words[0].upper()
  # Remaining words are its options
  args_str = words[1:]
  # Init empty args
  args = valid_cmds[cmd_type]
  # Populate arg values
  for astr in args_str:
    astrs = astr.split("=")
    args[astrs[0]] = astrs[1] 
  return cmd_type, args
  

def main():

  ###########
  # Options #
  ###########

  parser = OptionParser()
  parser.add_option("-f", "--commands_file",
                    help="Path to file defining generation commands.")
  parser.add_option("-o", "--out_file",
                    help="Path to write generated rasters as numpy '.npz'.")
  (options, args) = parser.parse_args()

  # File with commands
  cmd_file = options.commands_file
  if cmd_file is None:
    print("Expected path to file with generation commands ('-f').\nExiting...")
    exit(-1)
  # Output file
  out_file = options.out_file
  if out_file is None:
    print("Expected a `.npz` output file to save the generated samples (-o).\nExiting...")
    exit(-1)
 
  # Read file
  with open(cmd_file) as f:
    cmds = f.read().splitlines()


  ###############
  # Init Raster #
  ###############

  # Get first command
  cmd_str = cmds[0]
  # Parse command string
  cmd, args = parse_cmd(cmd_str)
  # Verify that it is a valid 'RASTER' command
  if cmd != "RASTER":
    print("Expected the first command to be 'RASTER' which is used to set up the raster properties.\nExiting...")
    exit(-1)
  # Consume first command
  cmds = cmds[1:]

  # Raster properties
  n_samples = int(args["samples"])
  rows = int(args["rows"])
  cols = int(args["cols"])
  bands = int(args["bands"])

  # Init raster
  rasters = np.zeros((n_samples, rows, cols, bands))
  mask  = np.zeros((rows, cols, bands))


  ##################
  # Apply Commands #
  ##################

  for cmd_str in cmds:

    # Skip comments
    if cmd_str[0] == "#":
      continue

    # Parse
    cmd, args = parse_cmd(cmd_str)

    if cmd == "SEED":
      # Setup arguments
      coords = np.fromstring(args["coords"], dtype="int", sep=",")
      value = float(args["value"])
      threshold = float(args["threshold"])
      decay = float(args["decay"])
      value_dist = float(args["value_dist"])
      # Run command
      for i in range(n_samples):
        mask[:] = 0
        grow(rasters[i], mask, coords, threshold, decay, value, value_dist)

    if cmd == "DILATE":
      # Setup arguments
      size = np.fromstring(args["size"], dtype="int", sep=",")
      if len(size) == 2: 
        size = (size[0], size[1], 1)
      # Run command
      rasters = np.array([ndimage.grey_dilation(raster, size=size, structure=np.ones(size)) for raster in rasters])

    if cmd == "BLUR":
      # Setup arguments
      sigma = float(args["sigma"])
      # Run command
      rasters = np.array([ndimage.gaussian_filter(raster, sigma=sigma) for raster in rasters])

    if cmd == "CROP":
      # Setup arguments
      lrow = args["low_row"]
      if lrow is None:
        lrow = 0
      hrow = args["high_row"]
      if hrow is None:
        hrow = rows
      lcol = args["low_col"]
      if lcol is None:
        lcol = 0
      hcol = args["high_col"]
      if hcol is None:
        hcol = cols
      lband = args["low_band"]
      if lband is None:
        lband = 0
      hband = args["high_band"]
      if hband is None:
        hband = bands
      # 
      rasters = rasters[:, int(lrow):int(hrow), int(lcol):int(hcol), int(lband):int(hband)]

  ########
  # Save #
  ########

  print("Writing {} rasters with shape ({} x {} x {}).".format(
    rasters.shape[0], rasters.shape[1], rasters.shape[2], rasters.shape[3]))
  print("To output file: {}.".format(out_file))
  np.savez(out_file, samples=rasters)

if __name__ == "__main__":
  main()

