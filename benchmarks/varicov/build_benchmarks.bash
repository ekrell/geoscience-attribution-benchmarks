#!/usr/bin/bash

# Covariance Experiment
# ---------------------
# The script is used to assess the influence of the covariance matrix
# on the variation in the learned function. Intuitively, if there is
# very high correlation among predictors then the model has many
# equally valid relationships to exploit to achieve approx. same 
# performance. So, we expect that XAI results will higher variance 
# when the strength of the correlations increase. 

# Options
# Config file for benchmark options
config_bmark=${1:-benchmarks/varicov/unicov/config_bmark.json}
# Config file for neural net hyperparams
config_network=${2:-benchmarks/varicov/unicov/config_nn.json}

echo ""
echo "VARICOV Benchmark Pipeline"
echo "-------------------------"
echo " Benchmarks config file: ${config_bmark}"
echo " Neural net config file: ${config_network}"
echo ""

# Load benchmark options from config file
out_dir=$(grep "output_directory" ${config_bmark} | grep -o ":.*$" | grep -o '".*"' | sed 's/"//g')
out_dir_xai=${out_dir}/xai/
pwl_cov_file=$(grep "pwl_cov_file" ${config_bmark} | grep -o ":.*$" | grep -o '".*"' | sed 's/"//g')
n_samples=$(grep "n_samples" ${config_bmark} | grep -o ":.*$" | grep -o '".*"' | sed 's/"//g')
n_pwl_breaks=$(grep "n_pwl_breaks" ${config_bmark} | grep -o ":.*$" | grep -o '".*"' | sed 's/"//g')
n_reps=$(grep "n_reps" ${config_bmark} | grep -o ":.*$" | grep -o '".*"' | sed 's/"//g')
xai_methods=$(grep "xai_methods" ${config_bmark} | grep -o ":.*$" | grep -o '".*"' | sed 's/"//g')

IFS=',' my_array=($xai_methods)

# Load NN hyperparameters from config file
nn_hidden_nodes=$(grep "hidden_nodes" ${config_network} | grep -o ":.*$" | grep -o '".*"' | sed 's/"//g')
nn_epochs=$(grep "epochs" ${config_network} | grep -o ":.*$" | grep -o '".*"' | sed 's/"//g')
nn_batch_size=$(grep "batch_size" ${config_network} | grep -o ":.*$" | grep -o '".*"' | sed 's/"//g')
nn_learning_rate=$(grep "learning_rate" ${config_network} | grep -o ":.*$" | grep -o '".*"' | sed 's/"//g')
nn_validation_fraction=$(grep "validation_fraction" ${config_network} | grep -o ":.*$" | grep -o '".*"' | sed 's/"//g')

# Setup directories
mkdir -p ${out_dir_xai}

# Generate covariance matrices
${cov_cmd}

# Get the indices of the generated matrices based on the files created
readarray -t covariance_idxs < <( ls ${out_dir}/cov*.npz | grep -o [0-9]* | uniq | sort -n )

for cidx in "${covariance_idxs[@]}"
do
    echo "Building synthetic benchmark using: ${cov_file}"

    cov_file="${out_dir}/cov_${cidx}.npz"

    # Generate synthetic samples from covariance
    samples_file=${out_dir}/samples_${cidx}.npz
    python src/synthetic/benchmark_from_covariance.py \
      --covariance_file "${cov_file}" \
      --num_samples "$n_samples" \
      --output_file "${samples_file}"
    
    # Define a synthetic function with ground-truth attribution
    pwl_attribution_file=${out_dir}/pwl-out_${cidx}.npz
    pwl_function_file=${out_dir}/pwl_fun_${cidx}.npz
    pwl_plot_file=${out_dir}/pwl_plot_${cidx}.png
    python src/synthetic/pwl_from_samples.py \
      --samples_file "${samples_file}" \
      --breakpoints "${n_pwl_breaks}" \
      --output_file "${pwl_attribution_file}" \
      --function_file "${pwl_function_file}" \
      --pwl_cov "${pwl_cov_file}"

  # Train the neural network multiple times
  for (( i=0; i<n_reps; i++ ))
  do
    echo ""
    echo "Training run $i"
    echo ""

    # Train NN
    model_file=${out_dir}/nn_model_${cidx}__${i}.h5
    loss_file=${out_dir}/nn_loss_${cidx}__${i}.csv
    loss_plot_file=${out_dir}/nn_loss_${cidx}__${i}.png
    metrics_file=${out_dir}/nn_metrics_${cidx}__${i}.csv
    python src/models/train_nn_tf.py \
      --quiet \
      --samples_file "${samples_file}" \
      --targets_file "${pwl_attribution_file}" \
      --model_file "${model_file}" \
      --loss_values_file "${loss_file}" \
      --hidden_nodes "${nn_hidden_nodes}" \
      --epochs "${nn_epochs}" \
      --batch_size "${nn_batch_size}" \
      --learning_rate "${nn_learning_rate}" \
      --validation_fraction "${nn_validation_fraction}" \
      --plot_loss_file "${loss_plot_file}" \
      --metrics_file "${metrics_file}"
  done
done
