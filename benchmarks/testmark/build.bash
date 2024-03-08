out_dir="benchmarks/testmark/out/"

# Generate samples
python benchmarks/testmark/generate_samples.py

# Build PWL
python src/synthetic/pwl_from_samples.py  \
    -s "${out_dir}/samples.npz" \
    -k 5 \
    -o "${out_dir}/pwl-out.npz" \
    -f "${out_dir}/pwl-fun.npz" \
    -p 0,10,100,200,300 \
    --plot_idxs_file ${out_dir}/pwl_attribs.png \
    --plot_cell_idxs 0,1,2,3,4,5 \
    --plot_cell_idxs_file ${out_dir}/pwl_cells.png

python src/models/train_nn.py \
    -s ${out_dir}/samples.npz \
    -t ${out_dir}/pwl-out.npz \
    -m ${out_dir}/trained-model.pt \
    -c ${out_dir}/trained-history.csv \
    -p ${out_dir}/trained-history.png \
    --epochs 50

python src/models/run_xai.py \
    -x  input_x_gradient \
    -s ${out_dir}/samples.npz \
    -m ${out_dir}/trained-model.pt \
    -i 0,10,100,200,300 \
    -p  ${out_dir}/xai/ \
    -a ${out_dir}/xai/xai_input_x_gradient.npz 

python src/utils/compare_attributions.py \
    --a_file ${out_dir}/pwl-out.npz  \
    --a_idxs 0,10,100,200,300 \
    --b_file  ${out_dir}/xai/xai_input_x_gradient.npz \
    --b_idxs 0,1,2,3,4 \
    -o ${out_dir}/xai/xai_input_x_gradient_corr.csv

cat ${out_dir}/xai/xai_input_x_gradient_corr.csv




