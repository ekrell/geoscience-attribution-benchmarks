'''
This program is used to explain a model by applying multiple XAI methods
to generate local explanations in the form of an attribution map. That is,
each grid cell of a gridded raster input is assigned a value that represents
the contribution of that cell toward the model output.
'''

import numpy as np
import torch
from optparse import OptionParser
import matplotlib.pyplot as plt
from nn import MLP

def main():

  # Options
  parser = OptionParser()
  parser.add_option("-s", "--samples_file",
                    help="Path to '.npz' with samples.")
  parser.add_option(      "--samples_varname",
                    default="samples",
                    help="Name of variable with raster samples in samples file.")
  parser.add_option("-m", "--model_file",
                    help="Path to trained pytorch model.")

  (options, args) = parser.parse_args()

  samples_file = options.samples_file
  samples_varname = options.samples_varname
  if samples_file is None:
    print("Expected input '.npz' file with samples ('-s').\nExiting...")
    exit(-1)

  model_file = options.model_file 
  if model_file is None:
    print("Expected input file with trained pytorch model ('-m').\nExiting...")
    exit(-1)

  # Load model
  kwargs, state = torch.load(model_file)
  model = MLP(**kwargs)
  model.load_state_dict(state)

  print(model)

  

if __name__ == "__main__":
  main()
