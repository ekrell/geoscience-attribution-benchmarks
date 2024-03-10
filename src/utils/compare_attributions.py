# This program compares 2 attribution maps. 

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from optparse import OptionParser

def load_attributions(filename, varname):
  data = np.load(filename)
  return data[varname]

def main():

  parser = OptionParser()
  parser.add_option("-a", "--a_file",
                    help="Path to first attributions.")
  parser.add_option(      "--a_varname",
                    default="attributions",
                    help="Name of variable with attributions in 'a' file.")
  parser.add_option(      "--a_idxs",
                    help="Comma-delimited list of indices to compare from 'a' file.")
  parser.add_option("-b", "--b_file",
                    help="Path to second attributions.")
  parser.add_option(      "--b_varname",
                    default="attributions",
                    help="Name of variable with attributions in 'b' file.")
  parser.add_option(      "--b_idxs",
                    help="Comma-delimited list of indices to compare from 'b' file.")
  parser.add_option("-o", "--out_file",
                    help="Path to save table with correlations.")
  (options, args) = parser.parse_args()

  # Attribution map 1
  a_file = options.a_file
  if a_file is None:
    print("Expected path to first attributions '.npz' file ('-a').\nExiting...")
    exit(-1)
  a_varname = options.a_varname
  # Attribution map 2
  b_file = options.b_file
  if b_file is None:
    print("Expected path to second attributions '.npz' file ('-b').\nExiting...")
    exit(-1)
  b_varname = options.b_varname

  out_file = options.out_file

  a_attr = load_attributions(a_file, a_varname)
  a_attrmaps = load_attributions(a_file, "attribution_maps")
  b_attr = load_attributions(b_file, b_varname)
  b_attrmaps = load_attributions(b_file, "attribution_maps")

  if options.a_idxs is not None:
    a_idxs = np.array(options.a_idxs.split(",")).astype("int")
  else:
    a_idxs = np.array(range(len(a_attr)))
  if options.b_idxs is not None:
    b_idxs = np.array(options.b_idxs.split(",")).astype("int")
  else:
    b_idxs = np.array(range(len(b_attr)))

  if len(a_idxs) != len(b_idxs):
    print("Expected number of comparison indices to match: {} = {}.\nExiting...".format(
      len(a_idxs), len(b_idxs)))
    exit(-1)

  n_compares = len(a_idxs)
  corrs = np.zeros((2, n_compares))

  for i in range(n_compares):
    r, p_value = stats.pearsonr(a_attr[a_idxs[i]],
                                b_attr[b_idxs[i]])
    corrs[0, i] = r
    rho, p_value = stats.spearmanr(a_attr[a_idxs[i]], 
                                   b_attr[b_idxs[i]])
    corrs[1, i] = rho

  dfMetrics = pd.DataFrame({
      "a_idx"    : a_idxs, 
      "b_idx"    : b_idxs,
      "pearson"  : corrs[0,:],
      "spearman" : corrs[1,:]
  })

  if out_file is None:
    print("Comparing attribution maps")
    print("   a: {}".format(a_file))
    print("   b: {}".format(b_file))
    print("")
    print(dfMetrics)
    print("")
    print("Correlation mean: {}".format(np.mean(corrs)))
  else:
    dfMetrics.to_csv(out_file, index=False)


if __name__ == "__main__":
  main()
