#!/usr/bin/bash 

# Basic XAI SST Anomaly Benchmark
# -------------------------
# This script creates an XAI benchmark by generated 
# samples based on an SST covariance matrix. 
# Options are defined from a '.dat' file with options.
#
# How to:  `./basic_pipeline.sh <options.dat>`
#
# Example options file: `runs/sst_anom.dat`

# Option validation
if [ $# -eq 0 ]
  then
    echo "Must provide input options file:"
    echo "  $0 <options.dat>"
    exit 1
fi

# Read options file
sst_data=$(grep "data=" $1 | awk -F"=" '{print $2}')
out_dir=$(grep "out_dir=" $1 | awk -F"=" '{print $2}')
out_dir_xai=$(grep "out_dir_xai=" $1 | awk -F"=" '{print $2}')
n_samples=$(grep "n_samples=" $1 | awk -F"=" '{print $2}')
n_pwl_breaks=$(grep "n_pwl_breaks=" $1 | awk -F"=" '{print $2}')
samples_to_plot=$(grep "samples_to_plot=" $1 | awk -F"=" '{print $2}')
pwl_functions_to_plot=$(grep "pwl_functions_to_plot=" $1 | awk -F"=" '{print $2}')
nn_hidden_nodes=$(grep "nn_hidden_nodes=" $1 | awk -F"=" '{print $2}')
nn_epochs=$(grep "nn_epochs=" $1 | awk -F"=" '{print $2}')
nn_batch_size=$(grep "nn_batch_size=" $1 | awk -F"=" '{print $2}')
nn_learning_rate=$(grep "nn_learning_rate=" $1 | awk -F"=" '{print $2}')
nn_validation_fraction=$(grep "nn_validation_fraction=" $1 | awk -F"=" '{print $2}')

xai_methods=(
  "saliency"
  "integrated_gradients" 
  "input_x_gradient" 
  "lrp" 
)

skip_get_sst=false
skip_benchmark_from_covariance=false
skip_pwl_from_samples=false
skip_train_nn=false

# Calculate covariance matrix from samples
if [ "$skip_get_sst" = false ]; then
  python utils/get_sst.py  \
    -n $sst_data \
    -c ${out_dir}/sst_cov.npz \
    -p ${out_dir}/sst.png \
    -i 0
fi 

# Generate synthetic samples from covariance
if [ "$skip_benchmark_from_covariance" = false ]; then
  python benchmarks/benchmark_from_covariance.py \
    -c ${out_dir}/sst_cov.npz \
    -n ${n_samples} \
    -o ${out_dir}/sst_samples.npz

  # Plot generated samples
  python utils/plot_samples.py \
    -r ${out_dir}/sst_samples.npz \
    -i ${samples_to_plot} \
    -o ${out_dir}/sst_samples_plot.png
fi

# Define a synthetic function with ground-truth attribution
if [ "$skip_pwl_from_samples" = false ]; then
  python benchmarks/pwl_from_samples.py \
    -s ${out_dir}/sst_samples.npz \
    -k ${n_pwl_breaks} \
    -o ${out_dir}/sst_pwl-out.npz \
    -f ${out_dir}/sst_pwl-fun.npz \
    -p ${samples_to_plot} \
    --plot_idxs_file ${out_dir}/sst_pwl_attribs.png \
    --plot_cell_idxs ${pwl_functions_to_plot} \
    --plot_cell_idxs_file ${out_dir}/sst_pwl_cells.png
fi

# Train neural network to approximate synthetic function
if [ "$skip_train_nn" = false ]; then
  python models/train_nn.py \
    -s ${out_dir}/sst_samples.npz \
    -t ${out_dir}/sst_pwl-out.npz \
    -m ${out_dir}/sst_trained-model.pt \
    -c ${out_dir}/sst_trained-history.csv \
    -p ${out_dir}/sst_trained-history.png \
    --hidden_nodes ${nn_hidden_nodes} \
    --epochs ${nn_epochs} \
    --batch_size ${nn_batch_size} \
    --learning_rate ${nn_learning_rate} \
    --validation_fraction ${nn_validation_fraction}
fi

for method in ${xai_methods[@]}; do
  
  # Run XAI method
  python models/run_xai.py \
    -x ${method} \
    -s ${out_dir}/sst_samples.npz \
    -m ${out_dir}/sst_trained-model.pt \
    -i ${samples_to_plot} \
    -p ${out_dir_xai}/ \
    -a ${out_dir_xai}/xai_${method}.npz

  # Compare XAI results to ground truth
  python utils/compare_attributions.py \
    --a_file ${out_dir}/sst_pwl-out.npz \
    --a_idxs ${samples_to_plot} \
    --b_file ${out_dir_xai}xai_${method}.npz \
    -o ${out_dir_xai}/xai_${method}_corr.csv

done
