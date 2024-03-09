
out_dir="benchmarks/sstanom/data/csu/"
out_dir_xai="${out_dir}/xai/"
n_samples=100000
samples_to_plot="0,1,2,3,4"
xai_methods="input_x_gradient"

# Get CSU dataset
python benchmarks/sstanom/get_csu_archive.py \
  -i benchmarks/sstanom/data/csu/synth_exm_data.nc \
  -o benchmarks/sstanom/data/csu/ \
  -n "${n_samples}"

# Train NN
nn_hidden_nodes="512,256,128,64,32,16"
nn_epochs=50
nn_batch_size="32"
nn_learning_rate="0.02"
nn_validation_fraction="0.1"
python src/models/train_nn.py \
  -s ${out_dir}/csu_samples.npz \
  -t ${out_dir}/csu_pwl-out.npz \
  -m ${out_dir}/csu_trained-model.pt \
  -c ${out_dir}/csu_trained-history.csv \
  -p ${out_dir}/csu_trained-history.png \
  --hidden_nodes ${nn_hidden_nodes} \
  --epochs ${nn_epochs} \
  --batch_size ${nn_batch_size} \
  --learning_rate ${nn_learning_rate} \
  --validation_fraction ${nn_validation_fraction}

# XAI
for method in ${xai_methods[@]}; do
  # Run XAI method
  python src/models/run_xai.py \
    -x ${method} \
    -s ${out_dir}/csu_samples.npz \
    -m ${out_dir}/csu_trained-model.pt \
    -i ${samples_to_plot} \
    -p ${out_dir_xai}/ \
    -a ${out_dir_xai}/xai_${method}.npz

  # Compare XAI to ground truth
  python src/utils/compare_attributions.py \
    --a_file ${out_dir}/csu_pwl-out.npz \
    --a_idxs ${samples_to_plot} \
    --b_file ${out_dir_xai}xai_${method}.npz \
    -o ${out_dir_xai}/xai_${method}_corr.csv
done

# Plot summary
python src/plot/plot_summary.py \
  -s benchmarks/sstanom/data/csu/csu_samples.npz \
  -a benchmarks/sstanom/data/csu/xai/xai_input_x_gradient.npz \
  -o benchmarks/sstanom/data/csu/csu_pwl-out.npz \
  --indices "0,1,2"

cat benchmarks/sstanom/data/csu/xai/xai_input_x_gradient_corr.csv



