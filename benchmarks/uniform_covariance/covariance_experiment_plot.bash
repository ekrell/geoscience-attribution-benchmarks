# Plots set of covariance experiments

input_dir="out/cov_exp/no_mask/"
xai_label="input_x_gradient"
cov_labels=(0 1 2 3 4 5 6 7 8 9 10)
output_dir="${input_dir}/xai/"

for cidx in ${cov_labels[@]}; do
  output_file=${output_dir}/xai_compare_${cidx}.pdf
  python runs/covariance_experiment_plot.py \
    -i ${input_dir} \
    -c ${cidx} \
    -x ${xai_label} \
    -o ${output_file}
done
