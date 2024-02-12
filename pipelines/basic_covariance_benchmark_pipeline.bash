#!/usr/bin/bash 

config_file=$1

out_dir=$(grep -e "\"out_dir\":" ${config_file} | grep -o -e "\"[^\"]*\"," | grep -o -e "[^\",]*")
out_dir_xai=${out_dir}/xai/
mkdir -p ${out_dir_xai}
cov_file=$(grep -e "\"covariance_file\":" ${config_file} | grep -o -e "\"[^\"]*\"," | grep -o -e "[^\",]*")
n_samples=$(grep -e "\"n_samples\":" ${config_file} | grep -o -e "\"[^\"]*\"," | grep -o -e "[^\",]*")
n_pwl_breaks=$(grep -e "\"n_pwl_breaks\":" ${config_file} | grep -o -e "\"[^\"]*\"," | grep -o -e "[^\",]*")
samples_to_plot=$(grep -e "\"samples_to_plot\":" ${config_file} | grep -o -e "\"[^\"]*\"," | grep -o -e "[^\",]*")
pwl_functions_to_plot=$(grep -e "\"pwl_functions_to_plot\":" ${config_file} | grep -o -e "\"[^\"]*\"," | grep -o -e "[^\",]*")
nn_hidden_nodes=$(grep -e "\"nn_hidden_nodes\":" ${config_file} | grep -o -e "\"[^\"]*\"," | grep -o -e "[^\",]*")
nn_epochs=$(grep -e "\"nn_epochs\":" ${config_file} | grep -o -e "\"[^\"]*\"," | grep -o -e "[^\",]*")
nn_batch_size=$(grep -e "\"nn_batch_size\":" ${config_file} | grep -o -e "\"[^\"]*\"," | grep -o -e "[^\",]*")
nn_learning_rate=$(grep -e "\"nn_learning_rate\":" ${config_file} | grep -o -e "\"[^\"]*\"," | grep -o -e "[^\",]*")
nn_validation_fraction=$(grep -e "\"nn_validation_fraction\":" ${config_file} | grep -o -e "\"[^\"]*\"," | grep -o -e "[^\",]*")
xai_methods_=$(grep -e "\"xai_methods\":" ${config_file} | grep -o -e "\"[^\"]*\"," | grep -o -e "[^\",]*")
IFS=';' read -r -a xai_methods <<< "${xai_methods_}"

skip_benchmark_from_covariance=false
skip_pwl_from_samples=false
skip_train_nn=false

# Generate synthetic samples from covariance
if [ "$skip_benchmark_from_covariance" = false ]; then
  python src/synthetic/benchmark_from_covariance.py \
    -c ${cov_file} \
    -n ${n_samples} \
    -o ${out_dir}/samples.npz 
  
  # Plot generated samples
  python src/plot/plot_samples.py \
    -r ${out_dir}/samples.npz \
    -i ${samples_to_plot} \
    -o ${out_dir}/samples_plot.png
fi

# Define a synthetic function with ground-truth attribution
if [ "$skip_pwl_from_samples" = false ]; then
  python src/synthetic/pwl_from_samples.py \
    -s ${out_dir}/samples.npz \
    -k ${n_pwl_breaks} \
    -o ${out_dir}/pwl-out.npz \
    -f ${out_dir}/pwl-fun.npz \
    -p ${samples_to_plot} \
    --plot_idxs_file ${out_dir}/pwl_attribs.png \
    --plot_cell_idxs ${pwl_functions_to_plot} \
    --plot_cell_idxs_file ${out_dir}/pwl_cells.png
fi

# Train neural network to approximate synthetic function
if [ "$skip_train_nn" = false ]; then
  python src/models/train_nn.py \
    -s ${out_dir}/samples.npz \
    -t ${out_dir}/pwl-out.npz \
    -m ${out_dir}/trained-model.pt \
    -c ${out_dir}/trained-history.csv \
    -p ${out_dir}/trained-history.png \
    --hidden_nodes ${nn_hidden_nodes} \
    --epochs ${nn_epochs} \
    --batch_size ${nn_batch_size} \
    --learning_rate ${nn_learning_rate} \
    --validation_fraction ${nn_validation_fraction}
fi

for method in ${xai_methods[@]}; do
  
  # Run XAI method
  python src/models/run_xai.py \
    -x ${method} \
    -s ${out_dir}/samples.npz \
    -m ${out_dir}/trained-model.pt \
    -i ${samples_to_plot} \
    -p ${out_dir_xai}/ \
    -a ${out_dir_xai}/xai_${method}.npz

  # Compare XAI results to ground truth
  python src/utils/compare_attributions.py \
    --a_file ${out_dir}/pwl-out.npz \
    --a_idxs ${samples_to_plot} \
    --b_file ${out_dir_xai}xai_${method}.npz \
    -o ${out_dir_xai}/xai_${method}_corr.csv
done
