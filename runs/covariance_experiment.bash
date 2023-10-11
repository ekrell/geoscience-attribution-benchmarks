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
out_dir=out/cov_exp/no_mask/
out_dir_xai=${out_dir}/xai/
pwl_covariance=out/sstanom/sst_cov.npz
n_samples=100000
n_pwl_breaks=5
samples_to_plot=0,10,100,200,300
pwl_functions_to_plot=0,10,100,200
nn_hidden_nodes=512,256,128,64,32,16
nn_epochs=20
nn_batch_size=32
nn_learning_rate=0.02
nn_validation_fraction=0.1

covariance_idxs=(0 1 2 3 4 5 6 7 8 9 10)

###pwl_mask_option="--pwl_mask out/cov_exp/mask_10-11-9-3.npz"
pwl_mask_option=""

n_reps=3

xai_methods=(
  "input_x_gradient"
) 

# Setup directories
mkdir -p ${out_dir_xai}

# Generate covariance matrices
python utils/generate_covariances.py \
  -o ${out_dir} \
  -r 20 -c 23 

for cidx in "${covariance_idxs[@]}"
do
    echo "Building synthetic benchmark using: ${cov_file}"

    cov_file="${out_dir}/cov_${cidx}.npz"

    # Generate synthetic samples from covariance
    samples_file=${out_dir}/samples_${cidx}.npz
    python benchmarks/benchmark_from_covariance.py \
      --covariance_file ${cov_file} \
      --num_samples $n_samples \
      --output_file ${samples_file}

    # Define a synthetic function with ground-truth attribution
    pwl_attribution_file=${out_dir}/pwl-out_${cidx}.npz
    pwl_function_file=${out_dir}/pwl_fun_${cidx}.npz
    pwl_plot_file=${out_dir}/pwl_plot_${cidx}.png
    python benchmarks/pwl_from_samples.py \
      --samples_file ${samples_file} \
      --breakpoints ${n_pwl_breaks} \
      --output_file ${pwl_attribution_file} \
      --function_file ${pwl_function_file} \
      --plot_idxs ${samples_to_plot} \
      --plot_idxs_file ${pwl_plot_file} \
      --pwl_cov ${pwl_covariance}   ${pwl_mask_option}

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
    python models/train_nn.py \
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
      --plot_loss_file ${loss_plot_file}
      

      for method in ${xai_methods[@]}; do

        # Run XAI method
        xai_file=${out_dir_xai}/${method}_${cidx}__${i}.npz
        out_dir_xai_plots=${out_dir_xai}/${method}__cov-${cidx}__run-${i}/
        mkdir -p ${out_dir_xai_plots}
        python models/run_xai.py \
          --xai_method ${method} \
          --samples_file ${samples_file} \
          --model_file ${model_file} \
          --indices ${samples_to_plot} \
          --plot_directory ${out_dir_xai_plots} \
          --attributions_file ${xai_file}

        # Compare XAI results to ground truth
        xai_compare_file=${out_dir_xai}/pwl-${method}_${cidx}__${i}.csv
        python utils/compare_attributions.py \
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
        python utils/compare_attributions.py \
          --a_file ${xai_a_file} \
          --b_file ${xai_b_file} \
          --out_file ${xai_compare_file}
      done
    done
  done

done

 
