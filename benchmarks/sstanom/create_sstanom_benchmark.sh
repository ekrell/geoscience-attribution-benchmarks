sstanom_dir="benchmarks/sstanom"
pipeline_config_file=${sstanom_dir}/config.json
sst_data_file=${sstanom_dir}/data/sst.mon.mean.nc
covariance_file=${sstanom_dir}/out/cov.npz
config_file=${sstanom_dir}/config.json

mkdir -p ${sstanom_dir}/data/
mkdir -p ${sstanom_dir}/out/

# Download SST data
if test -f "${sst_data_file}"; then
  echo "Found sst data file '${sst_data_file}'. Skipping download..."
else
  wget https://downloads.psl.noaa.gov//Datasets/COBE2/sst.mon.mean.nc -P ${sstanom_dir}/data/
fi

# Calculate covariance from real SST data
python ${sstanom_dir}/get_sst.py \
  -n ${sst_data_file} \
  -c ${covariance_file} 
exit 0
# The rest is handled by a generic pipeline
bash pipelines/basic_covariance_benchmark_pipeline.bash ${config_file}
