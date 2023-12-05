sstData=$1
covFile=$2

python utils/get_sst.py \
  -n $sstData \
  -c $covFile \
