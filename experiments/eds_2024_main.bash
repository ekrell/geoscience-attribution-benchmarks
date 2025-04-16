# Experiments for 2024 XAI benchmarks paper

# Options 
outdir="eds_outputs/sstanom/"   # Change to 'eds_output/icec/' for Sea Ice

# Skips: useful for custimizing, e.g. re-run XAI analysis without rebuilding benchmarks
SKIP_BUILD=false                 # Skip "building benchmarks"
SKIP_TRAIN=false                 # Skip "training networks" 

# Base covariance matrix
covmatrix="benchmarks/varicov/globalcov/sstanom_cov.npz"
# Number of synthetic covariance matrices to generate (using the base)
n_matrix=9 
# Number of breaks in peice-wise linear function
n_pwl_breaks=5
# Number of models to train
n_models=6
# Numbers(s) of synthetic samples to generate
nsamplessets=(1000 10000 100000 1000000)

# XAI methods to apply
xai_methods=(saliency input_x_gradient shap)
# XAI metrics to apply
xai_metrics=(faithfulness_correlation,30,50,0.0 monotonicity_correlation sparseness)

# NN hyperparameters
nn_hidden_nodes="512,256,128,64,32,16"
nn_epochs="50"
nn_batch_size="32"
nn_learning_rate="0.02"
nn_validation_fraction="0.1"

# Convert arrays to comma-separated string
printf -v xai_methods_str '%s,' "${xai_methods[@]}"
xai_methods_str="${xai_methods_str%,}"

# Part 1: Generate covariance matrices
python benchmarks/varicov/globalcov/strengthen_covariances.py \
  -c ${covmatrix} \
  -o ${outdir}

# For each "number of samples":
for nsamples in "${nsamplessets[@]}"; do
  subdir="${outdir}/${nsamples}/"
  subdir_xai="${outdir}/${nsamples}/xai/"
  echo "Synthetic samples: " ${nsamples}
  echo "  Sub directory: " ${subdir}
  echo "  XAI directory: " ${subdir_xai}
  mkdir -p ${subdir_xai}

  # Part 2: Build benchmarks

  # Get the indices of the generated matrices based on the files created
  readarray -t covariance_idxs < <( ls ${subdir}/cov*.npz | grep -o cov_[0-9]* | grep -o [0-9]* |  uniq | sort -n )

  # For each synthetic covariance matrix...
  for cidx in "${covariance_idxs[@]}"
  do
    cov_file="${subdir}/cov_${cidx}.npz"
    samples_file=${subdir}/samples_${cidx}.npz
    pwl_attribution_file=${subdir}/pwl-out_${cidx}.npz
    pwl_function_file=${subdir}/pwl_fun_${cidx}.npz
    pwl_plot_file=${subdir}/pwl_plot_${cidx}.png echo "Building synthetic benchmark using: ${cov_file}"
    
    if [ "$SKIP_BUILD" = false ]; then
      # Generate synthetic samples from covariance
      echo python src/synthetic/benchmark_from_covariance.py \
        --covariance_file "${cov_file}" \
        --num_samples "$nsamples" \
        --output_file "${samples_file}"

      # Define a synthetic function with ground-truth attribution
      echo python src/synthetic/pwl_from_samples.py \
        --samples_file "${samples_file}" \
        --breakpoints "${n_pwl_breaks}" \
        --output_file "${pwl_attribution_file}" \
        --function_file "${pwl_function_file}" \
        --pwl_cov "${covmatrix}"
    fi

    # Train networks
    if [ "$SKIP_TRAIN" = false ]; then
      for (( i=0; i<n_models; i++ ))
      do
        echo ""
        echo "Training run $i"
        echo ""

	#i=1
        # Train NN
        model_file=${subdir}/nn_model_${cidx}__${i}.h5
        loss_file=${subdir}/nn_loss_${cidx}__${i}.csv
        loss_plot_file=${subdir}/nn_loss_${cidx}__${i}.png
        metrics_file=${subdir}/nn_metrics_${cidx}__${i}.csv
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
    fi
  done 

  # Part 3: XAI analysis

  # Training and validation samples for XAI
  train_samples=$(seq -s"," 0 99)
  valstart="$((nsamples-100))"
  valend="$((nsamples-1))"
  valid_samples=$(seq -s"," $valstart $valend)

   Run XAI methods (training data)
  bash benchmarks/varicov/run_xai.bash \
     ${subdir}/ \
     ${xai_methods_str} \
     ${train_samples} \
     train

  # Run XAI methods (validation data)
  bash benchmarks/varicov/run_xai.bash \
      ${subdir}/ \
      ${xai_methods_str} \
      ${valid_samples} \
      valid

  for metric in "${xai_metrics[@]}"; do
  
    # Run XAI metrics (training data)
    bash benchmarks/varicov/run_xai_evals.bash \
      ${subdir} \
      ${xai_methods_str} \
      ${metric} \
      ${train_samples}  \
      $(seq -s"," 0 99) \
      train

    # Run XAI metrics (validation data)
    bash benchmarks/varicov/run_xai_evals.bash \
      ${subdir} \
      ${xai_methods_str} \
      ${metric} \
      ${valid_samples}  \
      $(seq -s"," 0 99) \
      valid
  done
done
