# `varicov` Benchmark Family

`varicom` is a family of related benchmarks where each benchmark is a subdirectory of `varicom`. 
These benchmarks are used to analyze the relationship between the strength of sample correlation
and the variation in XAI outputs when retraining the neural network. 

In a benchmark family, the root directory (`varicov`) has code that is common to all the benchmarks in it.
Subdirectories contain additional project-specific code. 

**Benchmarks**

1. [`unicov`](./unicov/): Uniform Covariance, All samples have exactly the same correlation, from 0 to 100%.
2. [`globalcov`](./globalcov/): Global Covariance. The initial covariance matrix is based on real SST Anomaly data, but the strength is increased (unformally) until the correlation are very strong. 

**Pipeline Components**

1. `build_benchmarks.bash`: Given a set of covariance matrices, build synthetic benchmarks
2. `run_xai.bash`: Run XAI algorithms using the benchmarks, and compare to the known attribution

See any of the individual benchmarks (e.g. `unicov`) for an example of using these components to build a specific benchmark pipeline.
