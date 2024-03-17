#!/usr/bin/bash

# Run XAI Experiments
# -------------------
# This script is used to run XAI algorithms on a benchmark
# that is a set of samples, known functions with attributions,
# and trained neural networks. 

# Options
#out_dir=benchmarks/globalcov/out/
#xai_methods=input_x_gradient
#samples=1,25,50,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500

out_dir=$1
xai_methods=$2
samples=$3

out_dir_xai=${out_dir}/xai/
IFS=',' my_array=($xai_methods)
n_reps=$(ls ${out_dir}/nn_loss_0__*.csv | wc -l)

# Get indices of cov matrices based on what is found in output dir
readarray -t covariance_idxs < <( ls ${out_dir}/cov*.npz | grep -o [0-9]*.npz | grep -o [0-9]* | uniq | sort -n )

# For each covariance matrix
for cidx in "${covariance_idxs[@]}"; do
  echo "Covariance matrix: ${cidx}"
  samples_file=${out_dir}/samples_${cidx}.npz
  pwl_attribution_file=${out_dir}/pwl-out_${cidx}.npz

  for (( i=0; i<n_reps; i++ )); do
    echo "  Trained model: ${i}"
    model_file=${out_dir}/nn_model_${cidx}__${i}.h5

    for method in ${xai_methods[@]}; do
      echo "    XAI method: ${method}"

      # Run XAI method
      xai_file=${out_dir_xai}/${method}_${cidx}__${i}.npz
      python src/models/run_xai_tf.py \
        --xai_method "${method}" \
        --samples_file "${samples_file}" \
        --model_file "${model_file}" \
        --indices "${samples}" \
        --attributions_file "${xai_file}"

      
      # Compare XAI results to ground truth
      xai_compare_file=${out_dir_xai}/pwl-${method}_${cidx}__${i}.csv
      python src/utils/compare_attributions.py \
        --a_file "${pwl_attribution_file}" \
        --a_idxs "${samples}" \
        --b_file "${xai_file}" \
        --out_file "${xai_compare_file}"

      head "${xai_compare_file}"

     done
  done

  for method in ${xai_methods[@]}; do

    ## Plot comparison of XAI to known attribution
    #python benchmarks/varicov/plot_attrs.py \
    #  --input_dir         "${out_dir}"   \
    #  --covariance_label  "${cidx}"      \
    #  --xai_label         "${method}"    \
    #  --output_file        "${out_dir_xai}"/xai_compare_${cidx}.pdf

    # Compare XAI results between runs
    for (( i=0; i<n_reps; i++ )); do
      for (( j=0; j<n_reps; j++ )); do
        xai_a_file=${out_dir_xai}/${method}_${cidx}__${i}.npz
        xai_b_file=${out_dir_xai}/${method}_${cidx}__${j}.npz
        xai_compare_file=${out_dir_xai}/${method}_${cidx}__${i}v${j}.csv
        python src/utils/compare_attributions.py \
          --a_file "${xai_a_file}" \
          --b_file "${xai_b_file}" \
          --out_file "${xai_compare_file}"
      done
    done

    echo ""
  done
done


for method in ${xai_methods[@]}; do
  # Plot summary over entire set of benchmarks
    python benchmarks/varicov/plot_summary.py \
      --input_dir "${out_dir}" \
      --xai_label "${method}" \
      --metric "r-square" \
      --output_corr_file "${out_dir_xai}/corr_compare_summary_${method}.pdf" \
      --output_perf_file "${out_dir_xai}/performance_summary.pdf"  
done

