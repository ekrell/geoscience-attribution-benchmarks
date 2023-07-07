# This program creates an interactive volume rendering of a raster stored in a '.npz' file.
# Assumes that the input raster has shape (samples, rows, cols, bands)

import numpy as np
import pyvista as pv
from optparse import OptionParser

def main():

  ###########
  # Options #
  ###########

  parser = OptionParser()
  parser.add_option("-f", "--input_file",
                    help="Path to '.npz' raster file.")
  parser.add_option(      "--input_varname",
                  default="samples",
                  help="Name of variable containing rasters in input file.")
  parser.add_option("-i", "--index",
                  default=0,
                  type="int",
                  help="Index to select which raster from a set of rasters (0-based).")
  (options, args) = parser.parse_args()

  input_file = options.input_file
  if input_file is None:
    print("Expected an input '.npz' file with raster to crop ('-i').\nExiting...")
    exit(-1)

  input_varname = options.input_varname
  index = options.index
  
  # Load data
  dataset = np.load(input_file)
  rasters = dataset[input_varname]
  # Validation
  if len(rasters.shape) != 4:
    print("Expected rasters with shape (samples, rows, cols, bands).\nExiting...")
    exit(-2)
  if index >= len(rasters.shape):
    print("The input contains {} rasters, but you asked for index = {}.".format(rasters.shape[0], index))
    print("Valid choices are integers in the range 0 - {}.".format(rasters.shape[0] - 1))
    exit(-2)
  # Select raster from samples
  raster = rasters[index]

  ##########
  # Render #
  ##########

  # Define data grid
  grid = pv.ImageData()
  grid.dimensions = np.array(raster.shape) + 1
  grid.origin = (0, 0, 0)
  grid.spacing = (10, 10, 10)
  grid.cell_data["values"] = raster.flatten(order="F")

  min_value = np.min(raster)

  # Build interactive plotter
  p = pv.Plotter()
  # Add transparent wireframe
  p.add_mesh(
    grid,
    style="wireframe",
    opacity=0.1,
  )
  # Add data, with adjustable threshold
  p.add_mesh_threshold(
    grid,
  )
  # Show
  p.show()


if __name__ == "__main__":
  main()


