# This program plots 2D or 3D vector fields as quiver plots.
# The vector fields are stored in '.npz' files. 
# The data is a numpy array with shape of either:
#   1)    (2, rows, cols)   <- 2D field with x, y components
#   2)    (3, rows, cols, bands)   <- 3D field with x, y, z components

import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt

def main():

  parser = OptionParser()
  parser.add_option("-v", "--vectorfield_file",
                    help="Path to '.npz' file with variable containing vector field.")
  parser.add_option(      "--variable_name",
                    default="field",
                    help="Name of variable with the vector field within the '.npz' file.")
  parser.add_option("-o", "--output_file",
                    help="Path to save plotted vector field.")
  parser.add_option("-d", "--divide_by",
                    default=-1,
                    type="int",
                    help="Integer to divide by to reduce resolution of quiver plot.")
  (options, args) = parser.parse_args()

  vectorfield_file = options.vectorfield_file
  if vectorfield_file is None:
    print("Expected a path to '.npz' file with vector field data ('-v').\nExiting...")
    exit(-1)
  
  output_file = options.output_file
  var_name = options.variable_name
  divide_by = options.divide_by

  # Load data
  vectorfield_npz = np.load(vectorfield_file)
  field = vectorfield_npz[var_name]
  shape = field.shape
  # Check for valid shape  
  if len(shape) < 3 or len(shape) > 4:
    print("Expected shape of either (2, rows, cols) or (3, rows, cols, bands).\nExiting...")
    exit(-1)

  print("")
  print("Vector field file: {}".format(vectorfield_file))
  print("  Using variable name: {}".format(var_name))
  print("  Field has shape: {} x {} x {}.".format(shape[0], shape[1], shape[2]))
  if output_file is None:
    print("No output file specified -> will display plot.")
  else:
    print("Output plot file: {}".format(output_file))

  # Enforce shape of (n, rows, cols, bands)
  if len(shape) != 4:
    field = np.reshape(field, (shape[0], shape[1], shape[2], 1))
    shape = field.shape

  rows, cols, bands = shape[1:]

  # Determine how many vectords to skip based on proportion
  if divide_by < 0:
    stride_rows = 1
    stride_cols = 1
  else:
    stride_rows = int(np.floor(rows / divide_by)) 
    stride_cols = int(np.floor(cols / divide_by)) 

  # If (x, y) components -> 2D plot
  if shape[0] == 2:
    field_x = field[0][::stride_rows, ::stride_cols]
    field_y = field[1][::stride_rows, ::stride_cols]

    fig, ax = plt.subplots(figsize=(8, 8))
    X, Y = np.meshgrid(range(field_x.shape[1]), range(field_x.shape[0]))
    ax.quiver(X, Y, field_x.flatten(), field_y.flatten())      
    ax.invert_yaxis()

  # If (x, y, z) components -> 3D plot
  elif shape[0] == 3:
    pass
    ax = plt.figure(figsize=(10,10)).add_subplot(projection="3d",)
    X, Y, Z = np.meshgrid(range(cols), range(rows), range(bands))
    ax.quiver(
      X[::stride, ::stride], Y[::stride, ::stride], Z[::stride, ::stride], 
      field[0][::stride, ::stride], field[1][::stride, ::stride], field[2][::stride, ::stride], 
      normalize=True)
  else: 
    print("Expected shape of either (2, rows, cols) or (3, rows, cols, bands).\nExiting...")
    exit(-1)
    

  if output_file is None:
    plt.show()
  else:
    plt.savefig(output_file)


if __name__ == "__main__":
  main()
