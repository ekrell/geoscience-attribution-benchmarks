sst_data=$1
config_file=$2

out_dir=$(grep -e "\"out_dir\":" ${config_file} | grep -o -e "\"[^\"]*\"," | grep -o -e "[^\",]*")
cov_file=${out_dir}/cov.npz

python utils/get_sst.py \
  -n $sst_data \
  -c $cov_file

bash pipelines/benchmark_pipeline.bash ${config_file}

