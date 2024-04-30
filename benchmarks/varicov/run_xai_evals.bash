#!/usr/bin/bash

# Run XAI Evaluations
# -------------------
# - This script is used to evaluate pre-computed XAI attributions. 
# - Metrics include Faithfulness Correlation (Bhatt et al., 2020).
# - Relies on output structure made when running `run_xai.bash`.


# Example options
# out_dir="benchmarks/varicov/globalcov/out/sstanom/sstanom_10000/"
# xai_methods_str="input_x_gradient,saliency"
# eval_str="faithfulness_correlation,30,50,0.0"
# idxs="0,1,2,3,4"

out_dir=$1            # Output directory
xai_methods_str=$2    # XAI methods  (comma-delimited list)
eval_str=$3           # XAI evaluation method and its options
sample_idxs=$4        # Comma-delimited indices of the Samples
xai_idxs=$5           # Comma-delimited indices of the Attributions

out_dir_xai=${out_dir}/xai/
n_reps=$(ls ${out_dir}/nn_loss_0__*.csv | wc -l)
# Splitting the string into an array
xai_methods=${xai_methods_str//,/ }
xai_methods=($xai_methods)

# Get indices of cov matrices based on what is found in output dir
readarray -t covariance_idxs < <( ls ${out_dir}/cov*.npz | grep -o [0-9]*.npz | grep -o [0-9]* | uniq | sort -n ) 

################################
# Part 1: Calculate all scores #
################################

# For each covariance matrix
for cidx in "${covariance_idxs[@]}"; do
  echo "Covariance matrix: ${cidx}"

  samples_file=${out_dir}/samples_${cidx}.npz
  pwl_attribution_file=${out_dir}/pwl-out_${cidx}.npz

  for (( i=0; i<n_reps; i++ )); do  
	echo "Repeat: ${i}"

	model_file=${out_dir}/nn_model_${cidx}__${i}.h5

	xai_files=""
	for method in ${xai_methods[@]}; do
		xai_files="${out_dir_xai}/${method}_${cidx}__${i}.npz,${xai_files}"
	done
	xai_files=$(echo $xai_files | sed 's/.\{1\}$//')

	colnames=""
	for method in ${xai_methods[@]}; do
		colnames="${method},${colnames}"
	done
	colnames=$(echo $colnames | sed 's/.\{1\}$//')

	output_file="${out_dir_xai}/xaieval_${eval_str}_${xai_methods_str}_${cidx}__${i}.csv"
	plot_file="${out_dir_xai}/xaieval_${eval_str}_${xai_methods_str}_${cidx}__${i}.pdf"

	python src/models/xai_evaluate.py \
		-m ${model_file} \
		-s ${samples_file} \
		-t ${pwl_attribution_file} \
		-a ${xai_files} \
		-o ${output_file} \
		-p ${plot_file} \
		-i ${sample_idxs} \
		-j ${xai_idxs} \
		-e ${eval_str} \
		-c ${colnames}
  done
done


#########################
# Part 2: Summary Plots #
#########################

covidxs=""
for cidx in ${covariance_idxs[@]}; do
	covidxs="${covidxs},${cidx}"
done
covidxs=$(echo $covidxs | sed 's/\,//')

runidxs=""
for (( i=0; i<n_reps; i++ )); do  
	runidxs="${runidxs},${i}"
done
runidxs=$(echo $runidxs | sed 's/\,//')

for method in ${xai_methods[@]}; do
	plot_file="${out_dir_xai}/xaieval_${eval_str}_${method}_summary.pdf"
	python benchmarks/varicov/plot_xai_metrics.py \
		-d ${out_dir_xai} \
		-e ${eval_str} \
		-x ${method} \
		-c ${covidxs} \
		-r ${runidxs} \
		-p ${plot_file}
	echo "Summary plot for method ${method}: ${plot_file}"
done
