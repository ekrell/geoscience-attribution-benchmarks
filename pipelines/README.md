# Pipelines

This directory contains scripts to create complete synthetic benchmarks for assessing XAI techniques. Given a covariance matrix, the pipeline (1) generates synthetic samples, (2) creates a function from which known attributions can be derived, (3) evaluates a set of samples, (4) runs XAI techniques, and (5) compares XAI outputs to the known attribution. 

![Pipeline overview diagram](pipeline_overview.png)

This is based off of the XAI Benchmarks proposed by Mamalakis et al. (2022).

    Mamalakis, A., Ebert-Uphoff, I., & Barnes, E. A. (2022). Neural network attribution methods for problems in geoscience: A novel synthetic benchmark dataset. Environmental Data Science, 1, e8.

## Overview

This directory contains a main pipeline script called `benchmark_pipeline.bash` that reads in a configuration file to create a synthetic benchmark based on a defined covariance matrix. This script can be used as a module for application-specific scripts that, for example, generate the covariance matrix. 

## Examples

**Example 1: Basic Pipeline**

In this example, we use an existing covariance matrix to create the XAI benchmark.

    mkdir out/example/                              # Create output directory
    cp data/example_cov.npz out/example/cov.npz     # Relies on name 'cov.npz'
    bash pipelines/benchmark_pipeline.bash  \
        pipelines/benchmark_pipeline_config.json    # Configuration file

**Example 2: SST Anomaly Pipeline**

In this example, we use a script that also creates the covariance matrix based on SST data.
After creating the covariance matrix, the basic pipeline from above is called. 

    bash pipelines/sstanom_pipeline.bash \
        data/sst.mon.mean.nc \                     # One extra application-specific parameter
        pipelines/benchmark_pipeline_config.json   # Configuration file

