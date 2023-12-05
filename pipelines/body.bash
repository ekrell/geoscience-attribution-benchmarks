#!/usr/bin/bash 

while getopts ":o:n:k:s:p:h:e:b:l:f:x:" opt; do
  case ${opt} in
    o ) 
      out_dir=${OPTARG}/
      out_dir_xai=${out_dir}/xai/
      ;;
    n ) 
      n_samples=${OPTARG}
      ;;
    k )
      n_pwl_breaks=${OPTARG}
      ;;
    s )
      samples_to_plot=${OPTARG}
      ;;
    p )
      pwl_functions_to_plot=${OPTARG}
      ;;
    h )
      nn_hidden_nodes=${OPTARG}
      ;;
    e )
      nn_epochs=${OPTARG}
      ;;
    b )
      nn_batch_size=${OPTARG}
      ;;
    l )
      nn_learning_rate=${OPTARG}
      ;;
    f )
      nn_validation_fraction=${OPTARG}    
      ;;
    x )
      xai_methods_=${OPTARG}
      IFS=',' read -r -a xai_methods <<< "${xai_methods_}"
      ;;
  esac
done

skip_benchmark_from_covariance=false
skip_pwl_from_samples=false
skip_train_nn=false

# Generate synthetic samples from covariance
if [ "$skip_benchmark_from_covariance" = false ]; then
  python benchmarks/benchmark_from_covariance.py \
    -c ${out_dir}/cov.npz \
    -n ${n_samples} \
    -o ${out_dir}/samples.npz 
  
  # Plot generated samples
  python utils/plot_samples.py \
    -r ${out_dir}/samples.npz \
    -i ${samples_to_plot} \
    -o ${out_dir}/samples_plot.png
fi

# Define a synthetic function with ground-truth attribution
if [ "$skip_pwl_from_samples" = false ]; then
  python benchmarks/pwl_from_samples.py \
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
  python models/train_nn.py \
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
  python models/run_xai.py \
    -x ${method} \
    -s ${out_dir}/samples.npz \
    -m ${out_dir}/trained-model.pt \
    -i ${samples_to_plot} \
    -p ${out_dir_xai}/ \
    -a ${out_dir_xai}/xai_${method}.npz

  # Compare XAI results to ground truth
  python utils/compare_attributions.py \
    --a_file ${out_dir}/pwl-out.npz \
    --a_idxs ${samples_to_plot} \
    --b_file ${out_dir_xai}xai_${method}.npz \
    -o ${out_dir_xai}/xai_${method}_corr.csv
done
