# This program modifies the strength of a covariance matrix

import numpy as np
from optparse import OptionParser

def stat_cov(cov):
  return {"mean" : np.mean(cov),
          "sum" : np.sum(cov),
          "min" : np.min(cov),
          "max" : np.max(cov),
          "count" : len(cov)}

def main():

  parser = OptionParser()
  parser.add_option("-c", "--covariance_file",
                    help="Path to '.npz' file with a variable containing covariance matrix.")
  parser.add_option("-v", "--variable_name",
                    default="covariance",
                    help="Name of variable with the covariance array within the '.npz' file.")
  parser.add_option("-n", "--operation_name",
                    default="mul",
                    help="Name of operation to apply ('mul', 'pow', or 'add')")
  parser.add_option("-a", "--operation_value",
                    default=10,
                    type="float",
                    help="Value of operation to apply")
  parser.add_option("-o", "--output_file",
                    help="Path to save modified covariance")

  (options, args) = parser.parse_args()

  cov_file = options.covariance_file
  var_name = options.variable_name
  op_name = options.operation_name
  op_value = options.operation_value
  out_file = options.output_file

  if cov_file is None:
    print("Excpected a path to '.npz' file with covariance matrix ('-c').\nExiting...")
    exit(-1)

  # Load data
  cov_npz = np.load(cov_file)
  cov = cov_npz[var_name]
  cov_shape = cov.shape

  cov_shape = cov.shape

  print("Apply operation to covariance matrix")
  print("Input: {}".format(cov_file))
  print("Operation: {}".format(op_name))
  print("Value: {}".format(op_value))
  
  # Flatten
  cov = cov.flatten()
  cov_abs = np.abs(cov)
  cov_pos = cov[cov >= 0]
  cov_neg = cov[cov < 0]
 
  # Apply operation
  if op_name == "pow":
    cov_pos = np.power(cov_pos, op_value) 
    cov_neg = np.power(cov_neg, op_value) * -1
    
  elif op_name == "mul":
    cov_pos = cov_pos * op_value
    cov_neg = cov_neg * op_value

  elif op_name == "add": 
    cov_pos = cov_pos + op_value
    cov_neg = cov_neg + (-1 * op_value)

  else:
    print("Unsupported operation '{}'.\nDoing nothing to the data...".format(op_name))

  # Combine pos and neg array values
  cov[cov >= 0] = cov_pos
  cov[cov < 0] = cov_neg
  cov_abs = np.abs(cov)

 # Stat covariance matrix
  cov_stats = {
    "cov" : stat_cov(cov),
    "abs" : stat_cov(cov_abs),
    "pos" : stat_cov(cov_pos),
    "neg" : stat_cov(cov_neg),
  }

  print("type,mean,sum,min,max,count")
  for key in cov_stats.keys():
    print("{},{:.4f},{:.4f},{:.4f},{:.4f},{}".format(key, 
      cov_stats[key]["mean"], cov_stats[key]["sum"], 
      cov_stats[key]["min"], cov_stats[key]["max"],
      cov_stats[key]["count"]))

  # Reshape to map
  cov = np.reshape(cov, cov_shape)

  if out_file is None:
    print("Did not provide output file ('-o'). Will not save covariance matrix.")
  else:
    np.savez(out_file, **{var_name: cov})


if __name__ == "__main__":
  main()
