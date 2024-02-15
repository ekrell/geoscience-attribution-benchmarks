# Plots set of covariance experiments

bmark_dir="benchmarks/unicov/"
input_dir="${bmark_dir}/out/"
output_dir="${input_dir}/xai/"
xai_label="input_x_gradient"
cov_labels=(0 1 2 3 4 5 6 7 8 9 10)

for cidx in ${cov_labels[@]}; do
  output_file=${output_dir}/xai_compare_${cidx}.pdf
  echo $output_file
  python ${bmark_dir}unicov_plot.py \
    -i ${input_dir} \
    -c ${cidx} \
    -x ${xai_label} \
    -o ${output_file}
done
