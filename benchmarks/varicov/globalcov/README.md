# `globalcov` Benchmark

The `globalcov` benchmark is part of the `varicov` benchmark family.
The scripts here actually produce a set of benchmark, based on changing the covariance matrix used to generate samples. 

`globalcov` is used to analyze how the strength of correlation among the samples influences the variance among XAI outputs with a set of trained models. That is, if there are more options for the model to learn (strong correlation), then it is expected that the model can learn many equally valid functions, causing the XAI-based model explanations to vary. Unlike `unicov` that induces a uniform correlation across all pixes, `globalcov` is based on a real dataset so that there is realistic spatial dependencies. These are then strengthed using the `strengthen_covariance.py` script.

The _global_ in `globalcov` refers to the origin of the based covariance matrix: [global sea surface temperature anomaly](https://psl.noaa.gov/data/gridded/data.cobe2.html). You can, of course, use any arbitrary covariance matrix to build a custom benchmark. Here, the goal was to use low-resolution data with both autocorrelation and teleconnections. 

## Run suite of benchmarks

These scripts run the experiments for **upcoming publication**. 
This includes generating the benchmark datasets and synthetic functions, running XAI on training and validation samples, and generating several analysis plots. Use the bash scripts as a guide to generate new benchmark experiments. 

	# Run benchmark pipeline
	bash experiments/eds_2024_main.bash

	# Run additional experiments: influence of superpixel size on attributions
	bash experiments/eds_2024_superpixels.bash

