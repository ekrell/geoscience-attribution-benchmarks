# Geoscience Attribution Benchmarks
Attribution Benchmarks for Geoscience Modeling

## Overview

Coming soon

## Related publications

[Neural network attribution methods for problems in geoscience: A novel synthetic benchmark dataset](https://www.cambridge.org/core/journals/environmental-data-science/article/neural-network-attribution-methods-for-problems-in-geoscience-a-novel-synthetic-benchmark-dataset/DDA562FC7B9A2B30710582861920860E)

[Investigating the Fidelity of Explainable Artificial Intelligence Methods for Applications of Convolutional Neural Networks in Geoscience](https://journals.ametsoc.org/view/journals/aies/1/4/AIES-D-22-0012.1.xml)

## Quick start

### Example: Create a synthetic dataset based on SST Anomaly data

Data source: https://psl.noaa.gov/data/gridded/data.cobe2.html

**Download SST data**

    mkdir data 
    cd data 
    wget https://downloads.psl.noaa.gov//Datasets/COBE2/sst.mon.mean.nc
    cd ..

**Calculate covariance matrix from samples**

    python utils/get_sst.py \
        -n data/sst.mon.mean.nc \   # Path to SST data
        -c out/sst_cov.npz \        # To save covariance
        -p out/sst.png \            # To save plot of selected sample
        -i 0                        # Index to select sample to plot

**Generate synthetic samples using covariance matrix**

    python benchmarks/benchmark_from_covariance.py \
        -c out/sst_cov.npz \        # Path to SST covariance data
        -n 100000 \                 # Number of synthetic samples
        -o out/sst_samples.npz      # To save the synthetic samples

**Plot generated samples**

    python utils/plot_samples.py \
        -r out/sst_samples.npz \      # Synthetic SST anomaly samples
        -i 0,2,4,6,8 \                # Indices to plot 
        -o out/sst_samples_plot.png   # Where to save plot


![out/sst_samples_plot.png](out/sst_samples_plot.png)


### Example: Create a synthetic dataset based on storm-centered tornado images

Data source: https://github.com/djgagne/ams-ml-python-course

**Download tornado data**

    cd data
    wget https://storage.googleapis.com/track_data_ncar_ams_3km_nc_small/track_data_ncar_ams_3km_nc_small.tar.gz
    tar -xvzf track_data_ncar_ams_3km_nc_small.tar.gz
    cd ..

**Calculate covariance matrix from samples**

    python utils/get_tornado.py \
        -n data/track_data_ncar_ams_3km_nc_small/NCARSTORM_20170323-0000_d01_model_patches.nc,data/track_data_ncar_ams_3km_nc_small/NCARSTORM_20170329-0000_d01_model_patches.nc   # Comma-delimited storm patch files
        -o out/tornado_cov.npz         # To save covariance

**Generate synthetic samples using covariance matrix**o

    python benchmarks/benchmark_from_covariance.py \
        -c out/tornado_cov.npz \        # Path to tornado covariance data
        -n 50000 \                      # Number of synthetic samples
        -o out/tornado_samples.npz      # To save the synthetic samples

**Plot generated samples**

    python utils/plot_samples.py \
        -r out/tornado_samples.npz \      # Synthetic SST anomaly samples
        -i 5,6,7,8,9 \                    # Indices to plot 
        -o out/tornado_samples_plot.png   # Where to save plot


![out/tornado_samples_plot.png](out/tornado_samples_plot.png)
    
