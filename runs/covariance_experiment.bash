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
out_dir=out/cov_exp/
out_dir_xai=out/cov_exp/xai/
n_samples=100000
n_pwl_breaks=5
samples_to_plot=0,10,100,200,300
pwl_functions_to_plot=0,10,100,200
nn_hidden_nodes=512,256,128,64,32,16
nn_epochs=25
nn_batch_size=32
nn_learning_rate=0.02
nn_validation_fraction=0.1

covariance_idxs=(4)

n_reps=3

xai_methods=(
  "input_x_gradient"
) 


for cidx in "${covariance_idxs[@]}"
do
    echo "Building synthetic benchmark using: ${cov_file}"

    cov_file="${out_dir}/cov_${cidx}.npz"

    # Generate synthetic samples from covariance
    samples_file=${out_dir}/samples_x${cidx}.npz
    python benchmarks/benchmark_from_covariance.py \
      --covariance_file ${cov_file} \
      --num_samples $n_samples \
      --output_file ${samples_file}

    # Define a synthetic function with ground-truth attribution
    pwl_attribution_file=${out_dir}/pwl-out_x${cidx}.npz
    pwl_function_file=${out_dir}/pwl_fun_x${cidx}.npz
    pwl_plot_file=${out_dir}/pwl_plot_x${cidx}.png
    python benchmarks/pwl_from_samples.py \
      --samples_file ${samples_file} \
      --breakpoints ${n_pwl_breaks} \
      --output_file ${pwl_attribution_file} \
      --function_file ${pwl_function_file} \
      --plot_idxs ${samples_to_plot} \
      --plot_idxs_file ${pwl_plot_file}
 
  # Train the neural network multiple times
  for (( i=0; i<n_reps; i++ ))
  do
    echo ""
    echo "Training run $i"
    echo ""

    # Train NN
    model_file=${out_dir}/nn_model_x${cidx}__${i}.npz
    loss_file=${out_dir}/nn_loss_x${cidx}__${i}.csv
    loss_plot_file=${out_dir}/nn_lossx${cidx}__${i}.png
    python models/train_nn.py \
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
        xai_file=${out_dir_xai}/${method}_x${cidx}__${i}.npz
        python models/run_xai.py \
          --xai_method ${method} \
          --samples_file ${samples_file} \
          --model_file ${model_file} \
          --indices ${samples_to_plot} \
          --plot_directory ${out_dir_xai} \
          --attributions_file ${xai_file}

        # Compare XAI results to ground truth
        xai_compare_file=${out_dir_xai}/pwl-${method}_x${cidx}__${i}.npz
        python utils/compare_attributions.py \
          --a_file ${pwl_attribution_file} \
          --a_idxs ${samples_to_plot} \
          --b_file ${xai_file} \
          --out_file ${xai_compare_file}
    done
  done
done
