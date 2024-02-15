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
bmark_dir=benchmarks/unicov/
out_dir=${bmark_dir}out/
out_dir_xai=${out_dir}/xai/
pwl_cov_file=${bmark_dir}/sstanom_cov.npz
n_samples=100000
n_pwl_breaks=5
samples_to_plot=0,2500,5000,10000,12500,15000,17500,20000,22500,25000,27500,30000,32500,35000,37500,40000,42500,45000,47500,50000,52500,55000,57500,60000,62500,65000,67500,70000,72500,75000,77500,80000,82500,85000,87500,90000,92500,95000,97500,99999
pwl_functions_to_plot=0,10,100,200
nn_hidden_nodes=512,256,128,64,32,16
nn_epochs=20
nn_batch_size=32
nn_learning_rate=0.02
nn_validation_fraction=0.1

covariance_idxs=(0 1 2 3 4 5 6 7 8 9 10)

n_reps=3

xai_methods=(
  "input_x_gradient"
) 

# Setup directories
mkdir -p ${out_dir_xai}

# Generate covariance matrices
python ${bmark_dir}/generate_covariances.py \
  -o ${out_dir} \
  -r 20 -c 23 

for cidx in "${covariance_idxs[@]}"
do
    echo "Building synthetic benchmark using: ${cov_file}"

    cov_file="${out_dir}/cov_${cidx}.npz"

    # Generate synthetic samples from covariance
    samples_file=${out_dir}/samples_${cidx}.npz
    python src/synthetic/benchmark_from_covariance.py \
      --covariance_file ${cov_file} \
      --num_samples $n_samples \
      --output_file ${samples_file}
    
    # Define a synthetic function with ground-truth attribution
    pwl_attribution_file=${out_dir}/pwl-out_${cidx}.npz
    pwl_function_file=${out_dir}/pwl_fun_${cidx}.npz
    pwl_plot_file=${out_dir}/pwl_plot_${cidx}.png
    python src/synthetic/pwl_from_samples.py \
      --samples_file ${samples_file} \
      --breakpoints ${n_pwl_breaks} \
      --output_file ${pwl_attribution_file} \
      --function_file ${pwl_function_file} \
      --plot_idxs ${samples_to_plot} \
      --plot_idxs_file ${pwl_plot_file} \
      --pwl_cov ${pwl_cov_file}

  # Train the neural network multiple times
  for (( i=0; i<n_reps; i++ ))
  do
    echo ""
    echo "Training run $i"
    echo ""

    # Train NN
    model_file=${out_dir}/nn_model_${cidx}__${i}.npz
    loss_file=${out_dir}/nn_loss_${cidx}__${i}.csv
    loss_plot_file=${out_dir}/nn_loss_${cidx}__${i}.png
    metrics_file=${out_dir}/nn_metrics_${cidx}__${i}.csv
    python src/models/train_nn.py \
      --quiet \
      --samples_file ${samples_file} \
      --targets_file ${pwl_attribution_file} \
      --model_file ${model_file} \
      --loss_values_file ${loss_file} \
      --hidden_nodes ${nn_hidden_nodes} \
      --epochs ${nn_epochs} \
      --batch_size ${nn_batch_size} \
      --learning_rate ${nn_learning_rate} \
      --validation_fraction ${nn_validation_fraction} \
      --plot_loss_file ${loss_plot_file} \
      --metrics_file ${metrics_file}

      for method in ${xai_methods[@]}; do

        # Run XAI method
        xai_file=${out_dir_xai}/${method}_${cidx}__${i}.npz
        out_dir_xai_plots=${out_dir_xai}/${method}__cov-${cidx}__run-${i}/
        mkdir -p ${out_dir_xai_plots}
        python src/models/run_xai.py \
          --xai_method ${method} \
          --samples_file ${samples_file} \
          --model_file ${model_file} \
          --indices ${samples_to_plot} \
          --plot_directory ${out_dir_xai_plots} \
          --attributions_file ${xai_file}

        # Compare XAI results to ground truth
        xai_compare_file=${out_dir_xai}/pwl-${method}_${cidx}__${i}.csv
        python src/utils/compare_attributions.py \
          --a_file ${pwl_attribution_file} \
          --a_idxs ${samples_to_plot} \
          --b_file ${xai_file} \
          --out_file ${xai_compare_file}

    done
  done

  for method in ${xai_methods[@]}; do
    for (( i=0; i<n_reps; i++ )); do
      for (( j=0; j<n_reps; j++ )); do
        # Compare XAI results between runs
        xai_a_file=${out_dir_xai}/${method}_${cidx}__${i}.npz
        xai_b_file=${out_dir_xai}/${method}_${cidx}__${j}.npz
        xai_compare_file=${out_dir_xai}/${method}_${cidx}__${i}v${j}.csv
        python src/utils/compare_attributions.py \
          --a_file ${xai_a_file} \
          --b_file ${xai_b_file} \
          --out_file ${xai_compare_file}
      done
    done
  done

done

 
