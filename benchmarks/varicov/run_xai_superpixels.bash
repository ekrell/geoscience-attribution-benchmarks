#!/usr/bin/bash

# Run XAI Superpixel Experiments
# ------------------------------

# Options

out_dir=$1
cov_idxs=$2
xai_methods=$3
patch_sizes=$4
samples=$5
tag=$6

# Process options
out_dir_xai="${out_dir}/xai-sp/"
mkdir -p ${out_dir_xai}
cov_idxs_=${cov_idxs}
cov_idxs=(${cov_idxs//,/ })
xai_methods=(${xai_methods//,/ })
if [ "$tag" != "" ]; then
	tag="${tag}_"
fi

# Detect number of models
nmodels=$(ls ${out_dir}/nn_model_0__*.h5 | wc -l)

# For each covariance matrix
for cidx in "${cov_idxs[@]}"; do
	samples_file="${out_dir}/samples_${cidx}.npz"

	# Generate model filenames
	models=""
	for ((i = 0 ; i < nmodels ; i++)); do
		if [ $i -gt 0 ]; then
			models="${models},"
		fi
		models="${models}${out_dir}/nn_model_${cidx}__${i}.h5"
	done

	# For each XAI method
	for method in "${xai_methods[@]}"; do

		# Run XAI on superpixels
		attribs_file="${out_dir_xai}/${tag}attribs_c-${cidx}_x-${method}.npz"
		distplot_file="${out_dir_xai}/${tag}attribs_c-${cidx}_x-${method}.pdf"
		python src/models/run_xai-superpixels_tf.py \
			-s ${samples_file} \
			-m ${models} \
			-i ${samples} \
			-p ${patch_sizes} \
			-x ${method} \
			-a ${attribs_file} \
			-d ${distplot_file}

		# Sum XAI on superpixels
		attribs_file="${out_dir_xai}/${tag}attribs_c-${cidx}_x-${method}-sums.npz"
		distplot_file="${out_dir_xai}/${tag}attribs_c-${cidx}_x-${method}-sums.pdf"
		python src/models/run_xai-superpixels_tf.py \
			-s ${samples_file} \
			-m ${models} \
			-i ${samples} \
			-p ${patch_sizes} \
			-x ${method} \
			-a ${attribs_file} \
			-d ${distplot_file} \
			--sum_attribs
	done
done

# Plot: distributions change with superpixel size
for method in "${xai_methods[@]}"; do
	attr_files=""
	for cidx in "${cov_idxs[@]}"; do
		attr_files="${attr_files},${out_dir_xai}/${tag}attribs_c-${cidx}_x-${method}.npz"
	done
	attr_files="${attr_files:1}"

	ref_files=""
	for cidx in "${cov_idxs[@]}"; do
		ref_files="${ref_files},${out_dir}/pwl-out_${cidx}.npz"
	done
	ref_files="${ref_files:1}"

	# Plot summary: superpixel XAI -- models vs models
	python benchmarks/varicov/plot_superpixel_summary.py \
		-a ${attr_files} \
		-l ${cov_idxs_} \
		-o "${out_dir_xai}/${tag}superpixel-correlations_m-${method}_mvm.pdf"

	# Plot summary: superpixel XAI -- models vs known attribs
	python benchmarks/varicov/plot_superpixel_summary.py \
		-a ${attr_files} \
		-l ${cov_idxs_} \
		-o "${out_dir_xai}/${tag}superpixel-correlations_m-${method}_mvr.pdf" \
		--attribution_reference_file ${ref_files} \
		--attribution_reference_samples ${samples}

	attr_sums_files=""
	for cidx in "${cov_idxs[@]}"; do
		attr_sums_files="${attr_sums_files},${out_dir_xai}/${tag}attribs_c-${cidx}_x-${method}-sums.npz"
	done
	attr_sums_files="${attr_sums_files:1}"

	# Plot summary: superpixel sums -- models vs models
	python benchmarks/varicov/plot_superpixel_summary.py \
		-a ${attr_sums_files} \
		-l ${cov_idxs_} \
		-o "${out_dir_xai}/${tag}superpixel-correlations_m-${method}-sums_mvm.pdf"

	# Plot summary: superpixel sums -- models vs known attribs
	python benchmarks/varicov/plot_superpixel_summary.py \
		-a ${attr_sums_files} \
		-l ${cov_idxs_} \
		-o "${out_dir_xai}/${tag}superpixel-correlations_m-${method}-sums_mvr.pdf" \
		--attribution_reference_file ${ref_files} \
		--attribution_reference_samples ${samples}
done

### Plot: all explanations, comparing models and superpixel sizes
##attr_files=""
##samples_dirs=""
##for cidx in "${cov_idxs[@]}"; do
##	for method in "${xai_methods[@]}"; do
##		attr_files="${attr_files},${out_dir_xai}/${tag}attribs_c-${cidx}_x-${method}.npz"
##		attr_files="${attr_files},${out_dir_xai}/${tag}attribs_c-${cidx}_x-${method}-sums.npz"
##
##		sample_dir="${out_dir_xai}/${tag}attribs_c-${cidx}_x-${method}/" 
##		sample_sums_dir="${out_dir_xai}/${tag}attribs_c-${cidx}_x-${method}-sums/" 
##		mkdir -p ${sample_dir}
##		mkdir -p ${sample_sums_dir}
##		sample_dirs="${sample_dirs},${sample_dir}"
##		sample_dirs="${sample_dirs},${sample_sums_dir}"
##	done
##done
##attr_files="${attr_files:1}"
##sample_dirs="${sample_dirs:1}"
##
##python benchmarks/varicov/plot_superpixel_samples.py \
##	-a ${attr_files} \
##	-o ${sample_dirs}
