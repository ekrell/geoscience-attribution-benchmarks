# Benchmark: SST Anomaly

This benchmark is an implementation of [Mamalakis et al. (2022)](https://www.cambridge.org/core/journals/environmental-data-science/article/neural-network-attribution-methods-for-problems-in-geoscience-a-novel-synthetic-benchmark-dataset/DDA562FC7B9A2B30710582861920860E), with additional features.

The purpose is to quantitatively compare XAI methods based on the correlation between the XAI output and the attribution of a known function F. The synthetic function F is carefully designed to have (1) spatial relationships among the grid cells and (2) the ability to calculate the attribution of each grid cell toward the function's output. By generating a very large amount of synthetic samples, a neural network is trained to approiximate F. Since the NN achieves near-perfect performance (R-square > 0.999), it is assumed that the learned function is a good approximation of the known function F. So, differences between the attribution of F and the output of XAI methods is assumed to be **because of limitations in the XAI method** rather than differences between what the model learned and F. 

## Benchmark Pipeline Diagram

![Benchmark diagram](img/mamalakis_pipeline.png)

_Image from [Mamalakis et al. (2022)](https://www.cambridge.org/core/journals/environmental-data-science/article/neural-network-attribution-methods-for-problems-in-geoscience-a-novel-synthetic-benchmark-dataset/DDA562FC7B9A2B30710582861920860E)_

## Benchmark Choices

This code is designed to create synethetic benchmarks from four base data sets. 
Each created by extracting samples from the [COBE-SST 2 and Sea Ice Gridded Climate Dataset](https://psl.noaa.gov/data/gridded/data.cobe2.html). 
This benchmark folder is called `sstanom` since that was the original purpose. 

1. `sst` : Sea Surface Temperature, directly from COBE.
2. `icec` : Sea Ice Concentration, directly from COBE.
3. `sstanom` : SST Anomaly, created from SST by subtracting the climatological mean and detrending.
4. `icecanom` : ICEC Anomaly, created from ICEC by subtracting the climatological mean and detrending.

## How to Run 

To create a synthetic benchmark and run XAI, you need to setup a configuration file and run the pipeline command. 

### Configuration file

Below is a configuration file for the `sstanom` pipeline that replicates the results of [Mamalakis et al. (2022)](https://www.cambridge.org/core/journals/environmental-data-science/article/neural-network-attribution-methods-for-problems-in-geoscience-a-novel-synthetic-benchmark-dataset/DDA562FC7B9A2B30710582861920860E)

	{
	    "out_dir": "benchmarks/sstanom/out/sstanom/",
	    "covariance_file": "benchmarks/sstanom/out/sstanom/cov.npz",
	    "n_samples": "1000000",
	    "n_pwl_breaks": "5",
	    "samples_to_plot": "0;10;100;200;300",
	    "pwl_functions_to_plot": "0;10;100;200",
	    "nn_hidden_nodes": "512;256;128;64;32;16",
	    "nn_epochs": "50",
	    "nn_batch_size": "32",
	    "nn_learning_rate": "0.02",
	    "nn_validation_fraction": "0.1",
	    "xai_methods": "saliency;integrated_gradients;input_x_gradient;lrp",
	    "variable": "sst",
	    "anomaly": "true"
	}

The following describes each option. 
Read the [Mamalakis et al. (2022) paper](https://www.cambridge.org/core/journals/environmental-data-science/article/neural-network-attribution-methods-for-problems-in-geoscience-a-novel-synthetic-benchmark-dataset/DDA562FC7B9A2B30710582861920860E) to get more insight on their purpose. 
These short descriptions are simply to help you match them up to the concepts in the paper. 

- `out_dir` : Path to save all pipeline output files. 
- `covariance_file` : Where the covariance file is. It gets created by this pipeline. 
- `n_samples` : Number of synthetic samples to generate using the covariance matrix. 
- `n_pwl_breaks` : Number of breakpoints in the piece-wise linear functions that make the known function F. 
- `samples_to_plot` : While you can plot samples later, its useful to plot a few during the pipeline for debugging.
- `pwl_functions_to_plot` : It is also useful to check a few of the piece-wise linear functions. 
- `nn_hidden_nodes` : Specifiy the neural net's hidden layers. 
- `nn_epochs` : Number of training epochs. 
- `nn_batch_size` : Batch size for neural network training. 
- `nn_learning_rate` : Learning rate for neural network training. 
- `xai_validation_fraction` : Fraction of synthetic samples to use as validation.
- `xai_methods` : Which XAI methods to run. 
- `variable` : Which dataset to download and use from COBE data source. Choices are (`sst`, `icec`)
- `anomaly` : Whether or not to convert to anomaly data (subtract climatology and detrend)

**Available XAI methods**

See the [Mamalakis et al. (2022) paper](https://www.cambridge.org/core/journals/environmental-data-science/article/neural-network-attribution-methods-for-problems-in-geoscience-a-novel-synthetic-benchmark-dataset/DDA562FC7B9A2B30710582861920860E) for method descriptions. 

- `saliency` : saliency maps
- `integrated_gradients` : integrated gradients
- `input_x_gradient` : Input X gradient
- `lrp` : LRP-zero

### Pipeline

    # Example: sst anomaly pipeline
    bash benchmarks/sstanom/create_sstanom_benchmark.sh benchmarks/sstanom/config_sstanom.json

There is also a script to plot several attribution maps to compare  

    dir=benchmarks/sstanom/out/sstanom/

    python src/plot/plot_attributions.py \
        --attr_files  $dir/pwl-out.npz,$dir/xai/xai_input_x_gradient.npz,$dir/xai/xai_integrated_gradients.npz,$dir/xai/xai_saliency.npz \
        --sample_idxs 0,10,100,200,300.0,1,2,3,4.0,1,2,3,4.0,1,2,3,4 \
        --names ground_truth,input_x_grad,integrated_grad,saliency

![Example XAI comparison plot](img/xai_compare.png)
