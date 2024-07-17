# Run EDS superpixel experiments 
# Used to create Figure 10
# (this script is more hard-coded than 'eds_2024_main.bash',
#  but should be easy to figure out and adapt)

cov_idxs="0,2,4,6,8"
xai_methods="occlusion,shap"
patch_sizes="1,2,3,4,5,6,7,8"

# Benchmark suite : 10^3 trained samples
bmark_dir="eds_outputs/sstanom/1000/"
sample_idxs="$(seq -s, 900 999)"
bash benchmarks/varicov/run_xai_superpixels.bash \
	${bmark_dir} \
	${cov_idxs} \
	${xai_methods} \
	${patch_sizes} \
	${sample_idxs} \
	"valid"

# Benchmark suite : 10^6 trained samples
bmark_dir="eds_outputs/sstanom/1000000/"
sample_idxs="$(seq -s, 999900 999999)"
bash benchmarks/varicov/run_xai_superpixels.bash \
	${bmark_dir} \
	${cov_idxs} \
	${xai_methods} \
	${patch_sizes} \
	${sample_idxs} \
	"valid"
