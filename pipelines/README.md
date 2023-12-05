# Pipelines

This directory contains scripts to create complete synthetic benchmarks for assessing XAI techniques. Given a covariance matrix, the pipeline (1) generates synthetic samples, (2) creates a function from which known attributions can be derived, (3) evaluates a set of samples, (4) runs XAI techniques, and (5) compares XAI outputs to the known attribution. 

![Pipeline overview diagram](pipeline_overview.png)

This is based off of the XAI Benchmarks proposed by Mamalakis et al. (2022).

    Mamalakis, A., Ebert-Uphoff, I., & Barnes, E. A. (2022). Neural network attribution methods for problems in geoscience: A novel synthetic benchmark dataset. Environmental Data Science, 1, e8.

## Overview 

A complete experiment is designed with 3 components: 

1. The **body** pipeline, `body.bash`, that is used for all pipelines.
2. The **head** script that add code for generating the application-specific covariance matrix.
3. The **config** files that specify the parameters of a pipeline run. 

All outputs are placed in a directory specified in the configuration file. 

## Example: SST Anomaly Pipeline

This pipeline is an implementation of the SST anomaly benchmark designed by Mamalakis et al. (2022). 

Run the pipeline:

    python benchmark_pipeline.py \
        --head sstanom_head.bash \     # Head pipeline for SST anomaly
        --config sstanom_config.json   # Config file

## How to add an application

To tailor the pipeline for a specific application, you have to define a **head** script and **config** file. The **head** is a bash script that handles how the covariance matrix is created. If you already have a covariance matrix, you can simply not provide a the `--head` option to `benchmark_pipeline.py`. 

To develop these, check `sstanom_head.bash` and `ssanom_config.json` for examples.





