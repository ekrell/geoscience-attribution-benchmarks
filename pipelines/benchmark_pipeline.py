import subprocess
from pathlib import Path
import json
import numpy as np
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-a", "--apphead",
                  help="Path to application-specific pipeline head module.")
parser.add_option("-c", "--config",
                  help="Configuration json file.")
(options, args) = parser.parse_args()

head_file = options.apphead
if head_file is None:
  print("No application head file (-a). Assuming a convariance matrix exists in input directory.")

config_file = options.config
if config_file is None:
  print("Required option '-c' missing. Exiting...")
  exit(-1)

config = json.load(open(config_file))
out_dir = config["out_dir"]

Path(out_dir).mkdir(parents=True, exist_ok=True)
Path(out_dir + "/xai/").mkdir(parents=True, exist_ok=True)

body_file = "pipelines/body.bash"
cov_file = out_dir + "cov.npz"

# Execute application head module
if head_file is not None:
  data_file = config["data"]
  subprocess.run(["bash", head_file, data_file, cov_file], shell=False)

# Execute body
subprocess.run(["bash", body_file, 
                "-o",   config["out_dir"],
                "-n",   config["n_samples"],
                "-k",   config["n_pwl_breaks"],
                "-s",   config["samples_to_plot"],
                "-p",   config["pwl_functions_to_plot"],
                "-h",   config["nn_hidden_nodes"],
                "-e",   config["nn_epochs"],
                "-b",   config["nn_batch_size"],
                "-l",   config["nn_learning_rate"],
                "-f",   config["nn_validation_fraction"],
                "-x",   config["xai_methods"]
], shell=False)
