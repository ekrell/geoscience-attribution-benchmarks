# Geoscience Attribution Benchmarks

Attribution Benchmarks for Geoscience AI Models

## Overview

The purpose of this repository is to develop synthetic benchmarks for evaluating XAI attribution methods. The synthetic benchmarks are designed such that a ground truth attribution can be derived. This enables quantitative evaluation of XAI methods since we can compare XAI-based attributions to the ground truth. 

[Mamalakis et al. (2022)](https://www.cambridge.org/core/journals/environmental-data-science/article/neural-network-attribution-methods-for-problems-in-geoscience-a-novel-synthetic-benchmark-dataset/DDA562FC7B9A2B30710582861920860E) proposed this benchmark framework for quantitative comparison of XAI methods. An implementation of this benchmark is provided in this repo ([benchmarks/sstanom/](./benchmarks/sstanom/)). 

[Krell et al. (in progress)]() developed a suite of benchmarks to investigate how correlations in the input domain affect XAI-based attributions. An implementation of this benchmark is provided in this repo ([benchmarks/varicov/globalcov/](./benchmarks/varicov/globalcov/)).

This repository contains tools for developing synthetic benchmarks for XAI evaluation. 
The tools in the `src` directory are building blocks for developing benchmarks. For example, `src/synthetic/benchmark_from_covariance.py` uses a covariance to generate a set of synthetic data samples. The scripts in the `pipelines` directory use those tools to build benchmarks given input data (e.g. a covariance matrix). 

In addition to the tools, there are specific benchmarks in the `benchmarks` directory. Each subfolder should be a specific benchmark application containing (1) input data, (2) application-specific code, and (3) the output benchmark files and visualizations. 

## Installation

**Note:** All code (e.g. Python and Bash scripts) should be run from repo's root.

	python -m venv venv
	source venv/bin/activate
	pip install numpy pandas scipy matplotlib netCDF4 cmocean statsmodels captum scikit-learn pyarrow seaborn tensorflow cloudpickle imageio innvestigate lime shap pillow xarray zipp

# Develop & Evaluate Benchmarks

- **'SSTANOM': SST anomaly benchmark**
	- Benchmark for comparing XAI methods
	- See its [README](./benchmarks/sstanom/README.md) to run it
	- See [related publication](https://www.cambridge.org/core/journals/environmental-data-science/article/neural-network-attribution-methods-for-problems-in-geoscience-a-novel-synthetic-benchmark-dataset/DDA562FC7B9A2B30710582861920860E) for details
- **'GLOBALCOV': correlation strength benchmark**
	- Benchmark for comparing XAI with varying correlation strength
	- See its [README](./benchmarks/varicov/globalcov/README.md) to run it
	- Publication coming soon
