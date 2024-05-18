import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-d", "--directory",
                  help="Directory with evaluation scores.")
parser.add_option("-e", "--eval_metric",
                  help="Evaluation metric string (used to get right filenames.")
parser.add_option("-x", "--xai_method", 
                  help="Name of XAI method (used to get right filename, column name.")
parser.add_option("-c", "--covariance_idxs",
                  help="Comma-delimited indices of covariance matrices to include.")
parser.add_option("-r", "--run_idxs",
                  help="Comma-delimied indices to training runs to include.")
parser.add_option("-p", "--plot_file",
                  help="Path to save plot.")
parser.add_option("-t", "--tag",
                  help="Optional filename tag",
                  default="")
parser.add_option("-f", "--file_str",
                  help="All methods for filename (annoying hack)")
(options, args) = parser.parse_args()

#directory = "benchmarks/varicov/globalcov/out/sstanom/sstanom_10000/xai"
#eval_str = "faithfulness_correlation,30,50,0.0"
#xai_method = "input_x_gradient"
#covs = "0,1,2,3,4,5,6,7,8"
#runs = "0,1,2,3"
#plot_file = "test.pdf"

directory = options.directory
eval_str = options.eval_metric
xai_method = options.xai_method
covs = options.covariance_idxs
runs = options.run_idxs
plot_file = options.plot_file

tag = options.tag
file_str = options.file_str

covs = np.array(covs.split(",")).astype("int")
runs = np.array(runs.split(",")).astype("int")
n_runs = len(runs)

file_fmt = directory + "/" + tag + "xaieval_" + eval_str + "_" + file_str + "_{}__{}.csv"

scores = [None for cov in covs]
for cov in covs:
    dfs = [None for i in runs]
    for i, r in enumerate(runs):
        score_file = file_fmt.format(cov, r)
        dfs[i] = pd.read_csv(score_file)
    dfScores = pd.concat(dfs)
    scores[cov] = dfScores[xai_method].values

fig, axs = plt.subplots()
sns.violinplot(data=scores, ax=axs)
axs.set_ylim(-1.0, 1.0)
axs.set_title(eval_str + " -  " + xai_method) 
plt.tight_layout()
plt.savefig(plot_file)

