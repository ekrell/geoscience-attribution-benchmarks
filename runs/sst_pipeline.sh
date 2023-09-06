#!/usr/bin/bash 

# SST Anomaly XAI Benchmark
# -------------------------
# This script creates an XAI benchmark by generated samples based
#   on the covariance matrix calculated from real SST data.
# This benchmark is an implementation of :
#   Mamalakis, A., Ebert-Uphoff, I., & Barnes, E. A. (2022). 
#   Neural network attribution methods for problems in geoscience: 
#   A novel synthetic benchmark dataset. Environmental Data Science, 1, e8.

# Definitions

sst_data="data/sst.mon.mean.nc"
out_dir="out/sstanom/"
out_dir_xai="${out_dir}/xai/"

n_samples="100000"
n_pwl_breaks="5"

samples_to_plot="0,10,100,200,300"
pwl_functions_to_plot="0,10,100,200"

xai_methods=(
  "saliency"
  "integrated_gradients" 
  "input_x_gradient" 
  "lrp" 
)

skip_get_sst=false
skip_benchmark_from_covarinace=false
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
    --hidden_nodes 512,256,128,64,32,16 \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.02 \
    --validation_fraction 0.1
fi

# Run XAI methods
for method in ${xai_methods[@]}; do
  python models/run_xai.py \
    -x ${method} \
    -s ${out_dir}/sst_samples.npz \
    -m ${out_dir}/sst_trained-model.pt \
    -i ${samples_to_plot} \
    -p ${out_dir_xai}/ \
    -a ${out_dir_xai}/xai_${method}.npz
done
