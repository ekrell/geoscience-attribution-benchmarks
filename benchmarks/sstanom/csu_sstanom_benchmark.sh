# This is an alternative to 'create_sstanom_benchmark.sh'
# that uses pre-computed samples and model from the 
# paper by Mamalakis et al. (2022)

csu_dir="benchmarks/sstanom/csu/"
out_dir="${csu_dir}/out/"
out_dir_xai="${out_dir}/xai/"

#samples_start=979475
#samples_end=1000000
samples_start=0
samples_end=1000000
xai_samples="0,1,2"
xai_methods="input_x_gradient"

download_data=false
use_csu_model=false
skip_training=false

mkdir -p ${out_dir_xai}

if [ ${download_data} = true ]; then
  cmd_dl="-d"
else
  cmd_dl=""
fi

# Download CSU dataset
python benchmarks/sstanom/get_csu_archive.py \
  -i ${out_dir}/synth_exm_data.nc \
  -o ${out_dir} ${cmd_dl} \
  -s ${samples_start} \
  -e ${samples_end}

if [ ${use_csu_model} = true ]; then
  # Download CSU pretrained model
  wget https://github.com/amamalak/Neural-Network-Attribution-Benchmark-for-Regression/raw/master/my_model.h5 --output-document=${out_dir}/csu_model.h5

  skip_training=true
  model_file="${out_dir}/csu_model.h5"
else
  model_file="${out_dir}/local_model.h5"
fi

if [ ${skip_training} = true ]; then
  cmd_skip="--load_trained"
else
  cmd_skip=""
fi

# Train NN
nn_hidden_nodes="512,256,128,64,32,16" 
nn_epochs=50
nn_batch_size="32"
nn_learning_rate="0.02"
nn_validation_fraction="0.1"
python src/models/train_nn_tf.py ${cmd_skip}  \
  -s ${out_dir}/csu_samples.npz \
  -t ${out_dir}/csu_pwl-out.npz \
  -m ${model_file} \
  -c ${out_dir}/trained-history.csv \
  -p ${out_dir}/trained-history.png \
  --hidden_nodes ${nn_hidden_nodes} \
  --epochs ${nn_epochs} \
  --batch_size ${nn_batch_size} \
  --learning_rate ${nn_learning_rate} \
  --validation_fraction ${nn_validation_fraction}

# XAI
for method in ${xai_methods[@]}; do
  # Run XAI method
  python src/models/run_xai_tf.py \
    -x ${method} \
    -s ${out_dir}/csu_samples.npz \
    -m ${model_file} \
    -a ${out_dir_xai}/xai_${method}.npz \
    -i ${xai_samples}

  # Compare XAI to ground truth
  python src/utils/compare_attributions.py \
    --a_file ${out_dir}/csu_pwl-out.npz \
    --a_idxs ${xai_samples} \
    --b_file ${out_dir_xai}/xai_${method}.npz \
    -o ${out_dir_xai}/xai_${method}_corr.csv

  head ${out_dir_xai}/xai_${method}_corr.csv

  # Plot summary
  python src/plot/plot_summary.py \
    -s ${out_dir}/csu_samples.npz \
    -a ${out_dir}/csu_pwl-out.npz \
    -i ${xai_samples} \
    -x ${out_dir_xai}/xai_${method}.npz \
    -j "0,1,2" \
    -o ${out_dir}/csu_pwl-out.npz \
    -p ${out_dir_xai}/xai_${method}.png
done 
