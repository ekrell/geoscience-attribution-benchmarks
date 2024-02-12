# Benchmark: SST Anomaly

This benchmark is an implementation of [Mamalakis et al. (2023)](https://www.cambridge.org/core/journals/environmental-data-science/article/neural-network-attribution-methods-for-problems-in-geoscience-a-novel-synthetic-benchmark-dataset/DDA562FC7B9A2B30710582861920860E), 
the publication that proposed this approach to XAI evaluation through synthetic benchmarks. 

The purpose is to quantitatively compare XAI methods based on the correlation between the XAI output and the attribution of a known function F. The synthetic function F is carefully designed to have (1) spatial relationships among the grid cells and (2) the ability to calculate the attribution of each grid cell toward the function's output. By generating a very large amount of synthetic samples, a neural network is trained to approiximate F. Since the NN achieves near-perfect performance (R-square > 0.999), it is assumed that the learned function is a good approximation of the known function F. So, differences between the attribution of F and the output of XAI methods is assumed to be **because of limitations in the XAI method** rather than differences between what the model learned and F. 

(For very highly correlated data, it would be possible to achieve a very high NN performance without approximating the original F. But that is not explored in this benchmark). 

**Note: because I was running out of memory, my example code generates fewer synthetic samples than in the ooriginal Mamalakis et al. paper, so I do not achieve quite the high performance for the trained NN.**

## Benchmark Pipeline Diagram

![Benchmark diagram](img/mamalakis_pipeline.png)

_Image from [Mamalakis et al. (2023)](https://www.cambridge.org/core/journals/environmental-data-science/article/neural-network-attribution-methods-for-problems-in-geoscience-a-novel-synthetic-benchmark-dataset/DDA562FC7B9A2B30710582861920860E)_

## How to run

The pipeline options (which XAI methods, NN hyperparameters, etc) can be modified in `config.json`.

    bash create_sstanom_benchmark.sh
