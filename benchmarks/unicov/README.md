# Benchmark: Uniform Covariance Matrices

This is a very simple benchmark to demonstrate how the strength of correlation between grid cells influences the distribution of XAI results when retraining the model. 

- With correlated data, the model can learn different relationships and achieve similar performance.
- So, there is potential variation in the learned weights of the trained model.
- This means that the XAI outputs may also vary, even if they all correctly explain the model.
- This benchmark is really a set of benchmarks, where each benchmark uses a different covariance matrix.
- The covariance matrices are uniform: all grid cells have the same relationship.
- The covariances matrices used go from 0.0 (no correlation) to 1.0 (all values identical). 
- **Hypothesis: greater correlation --> greater variation in XAI results**. 

**Publication**

[Krell et al. (2024): Using Grouped Features to Improve Explainable AI Results for Atmospheric AI Models that use Gridded Spatial Data and Complex Machine Learning Techniques](https://ams.confex.com/ams/104ANNUAL/meetingapp.cgi/Paper/435616)

## Benchmark Design

## Motivating Toy Example

![Demonstration of toy example](img/unicov_toy_example.png)

## Benchmark Design Pipeline

![Benchmark pipeline](img/unicov_benchmark_design.png)

**Note on covariance matrix**

- For each benchmark, two covariance matrices are needed
  1. Used to generate the samples 
  2. Used to induce spatial relationships between grid cells when defining known function F
- Here, a _uniform covariance matrix_ is used to generate samples (e.g. where cov = 0.5)
- And a _real geospatial covariance matrix_ is used for the known function F
  - Otherwise, each experiment has **two changes** instead of isolating to just the influence on sample correlation
  - The real covariance matrix comes from **SST anomaly** data (see the `sstanom` benchmark)

## Results

![Benchmark results](img/unicov_results.png)

## How to Run

### Build benchmarks

There is a single pipeline script that builds all the benchmarks, runs XAI methods, and plots results.
You can modify the experiment using variables at the top of the pipeline script. 
The plotting scripts also have hard-coded options at the top. 

**Run the pipeline**

    bash benchmarks/unicov/create_unicov_benchmark.bash

**Plot comparisons of XAI to known attributions**

    bash benchmarks/unicov/unicov_plot.bash

Example: `xai_compare_4.pdf`

![A comparison of samples to ground truth](img/unicov_cmp4.png)


**Plot summary over entire set of benchmarks**

    python benchmarks/unicov/unicov_summary_plot.py

Example: `corr_compare_summary.pdf`

![Plot of the correlation distribution between XAI and ground truth](img/unicov_corr.png)


Example: `performance_summary.pdf`

![Plot of the performance for each trained model](img/unicov_perf.png)




