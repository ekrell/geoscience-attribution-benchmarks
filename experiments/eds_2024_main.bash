# Experiments for 2024 XAI benchmarks paper

# Options 
outdir="eds_outputs/sstanom/"   # Change to 'eds_output/icec/' for Sea Ice
covmatrix=""
nsamplessets=(1000 10000 100000 1000000)
xai_methods=(saliency input_x_gradient shap lime)
xai_metrics=(faithfulness_correlation,30,50,0.0 monotonicity_correlation sparseness)
n_models=4
n_matrix=9

# Convert arrays to comma-separated string
printf -v xai_methods_str '%s,' "${xai_methods[@]}"
xai_methods_str="${xai_methods_str%,}"

echo "Experiments:"
echo "- Covariance matrix: " $covmatrix
echo "- Output directory: " $outdir
echo ""

# Part 2: Run XAI evaluation

for nsamples in "${nsamplessets[@]}"; do
  subdir="${outdir}/${nsamples}/"
  subdir_xai="${outdir}/${nsamples}/xai/"

  echo "Synthetic samples: " ${nsamples}
  echo "  Sub directory: " ${subdir}
  echo "  XAI directory: " ${subdir_xai}
  mkdir -p ${subdir_xai}

  # Training and validation samples for XAI
  train_samples=$(seq -s"," 0 99)
  valstart="$((nsamples-100))"
  valend="$((nsamples-1))"
  valid_samples=$(seq -s"," $valstart $valend)

  # Run XAI methods (training data)
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
