# Geoscience Attribution Benchmarks

Attribution Benchmarks for Geoscience AI Models

## Overview

- EXplainable AI (XAI) is increasingly used to investigate what complex ML models learned.
- However, there are many techniques with their strengths & weaknesses.
- It is hard to evaluate XAI methods since we don't have a ground truth explanation.
- So, Mamalakis et al. proposed synthetic benchmarks for evaluating XAI methods
  - Using specially-designed functions where the attribution of each pixel toward the output can be derived. 

## Related publications

[Neural network attribution methods for problems in geoscience: A novel synthetic benchmark dataset](https://www.cambridge.org/core/journals/environmental-data-science/article/neural-network-attribution-methods-for-problems-in-geoscience-a-novel-synthetic-benchmark-dataset/DDA562FC7B9A2B30710582861920860E)

## Organization

This repository contains tools for developing synthetic benchmarks for XAI evaluation. 
The tools in the `src` directory are building blocks for developing benchmarks. For example, `src/synthetic/benchmark_from_covariance.py` uses a covariance to generate a set of synthetic data samples. The scripts in the `pipelines` directory use those tools to build benchmarks given input data (e.g. a covariance matrix). 

In addition to the tools, there are specific benchmarks in the `benchmarks` directory. Each subfolder should be a specific benchmark application containing (1) input data, (2) application-specific code, and (3) the output benchmark files and visualizations. 

- `src`:
  - `src/synthetic`: Main programs for building benchmarks:
    1. Generating synthetic samples (e.g. from a covariance matrix).
    2. Generating functions with known attribution (e.g. piece-wise linear).
  - `src/models`: Programs for defining, training, and explaining NN models.
  - `src/utils`: Simple convenience utilities (e.g. cropping rasters).
  - `src/plot`: Plotting scripts.
- `pipelines`: Scripts that combine tools from `src` to build generic benchmarks.
- `benchmarks`: Data, code, and outputs for a specific benchmark.

## Installation

**Note:** All code (e.g. Python and Bash scripts) should be run from repo's root.

**Install basic libraries**

	pip install numpy pandas scipy matplotlib netCDF4 cmocean statsmodels captum scikit-learn pyarrow

**Install pytorch**

Option 1: install with CUDA for GPU support. May need to change based on [your GPU](https://pytorch.org/get-started/locally/).

	pip install torch --index-url https://download.pytorch.org/whl/cu118

Option 2: install without CUDA, for CPU-only.

	pip install torch

## Quickstart

- To create a new benchmark, please make a new directory under `benchmarks/`.
- For an example, see the SST Anomaly benchmark located in `benchmarks/sstanom`. 
  - This benchmark is an implementation of that described by [Mamalakis et al. (2022)](https://www.cambridge.org/core/journals/environmental-data-science/article/neural-network-attribution-methods-for-problems-in-geoscience-a-novel-synthetic-benchmark-dataset/DDA562FC7B9A2B30710582861920860E)
  - To run it: `bash benchmarks/sstanom/create_sstanom_benchmark.sh`
