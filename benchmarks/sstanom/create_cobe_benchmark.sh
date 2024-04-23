# Hard-coded path top where this should be
cobe_dir="benchmarks/sstanom"

# Path to config file
config_file=$1

variable=$(grep ${config_file} -e "variable" | grep -o  ': ".*"*$' | sed -e 's/: "//' -e 's/".*$//')
is_anomaly=$(grep ${config_file} -e "anomaly" | grep -o  ': ".*"*$' | sed -e 's/: "//' -e 's/".*$//')
outdir=$(grep ${config_file} -e "out_dir" | grep -o  ': ".*"*$' | sed -e 's/: "//' -e 's/".*$//')

cobe_data_file=${cobe_dir}/data/${variable}.mon.mean.nc
covariance_file=${outdir}/cov.npz

mkdir -p ${cobe_dir}/data/
mkdir -p ${outdir}

# Download COBE data
if test -f "${cobe_data_file}"; then
  echo "Found COBE data file '${cobe_data_file}'. Skipping download..."
else
  wget https://downloads.psl.noaa.gov//Datasets/COBE2/${variable}.mon.mean.nc -P ${cobe_dir}/data/
fi

anom_opt=""
if [[ "$is_anomaly" == "true" ]]; then
  anom_opt=" -a "
fi

# Calculate covariance from real data
python ${cobe_dir}/get_cobe.py \
  -n ${cobe_data_file} \
  -p ${outdir}/cov_plot.png \
  -c ${covariance_file}   ${anom_opt}

# The rest is handled by a generic pipeline
bash pipelines/basic_covariance_benchmark_pipeline.bash ${config_file}
