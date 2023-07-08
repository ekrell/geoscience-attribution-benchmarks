# Geoscience Attribution Benchmarks
Attribution Benchmarks for Geoscience Modeling

## Overview

Coming soon

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
        -n 10 \                     # Number of synthetic samples
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
        -n 10 \                         # Number of synthetic samples
        -o out/tornado_samples.npz      # To save the synthetic samples

**Plot generated samples**

    python utils/plot_samples.py \
        -r out/tornado_samples.npz \      # Synthetic SST anomaly samples
        -i 5,6,7,8,9 \                    # Indices to plot 
        -o out/tornado_samples_plot.png   # Where to save plot


![out/tornado_samples_plot.png](out/tornado_samples_plot.png)
    

### Example: create 2D sythnetic benchmark with autocorrelation using commands

**View example commands file**

    cat data/cmds_example_2D.dat
    RASTER samples=10 rows=50 cols=50 bands=1
    SEED coords=10,10 value=0.2 threshold=0.9 decay=0.01 value_dist=0.2
    SEED coords=20,20 value=0.6 threshold=0.9 decay=0.01
    SEED coords=40,20 value=0.8 threshold=0.9 decay=0.01 value_dist=0.3
    DILATE size=2,2 structure=2,2

**Generate synthetic samples**

    python benchmarks/benchmark_from_commands.py \
        -f data/cmds_example_2D.dat \    # Path to commands
        -o out/cmds_example_2D.npz       # To save synthetic samples

**Plot generated samples**

    python utils/plot_samples.py \
        -r out/cmds_example_2D.npz \     # Synthetic samples
        -i 0,1 \                         # Indices to plot
        -o out/cmds_example_2D.png       # Where to save plot

![out/cmds_example_2D.png](out/cmds_example_2D.png)


### Example: create 3D synthetic benchmark with spatial autocorrelation across channels

**View example commands file**

    cat data/cmds_example_3D-spatial.dat
    RASTER samples=10 rows=50 cols=50 bands=6
    SEED coords=10,10,3 value=0.2 threshold=0.9 decay=0.01 value_dist=0.2
    SEED coords=20,20,3 value=0.6 threshold=0.9 decay=0.01
    SEED coords=40,40,3 value=0.8 threshold=0.9 decay=0.04 value_dist=0.05
    SEED coords=40,30,3 value=0.9 threshold=0.9 decay=0.04 value_dist=0.05


**Generate synthetic samples**

    python benchmarks/benchmark_from_commands.py \
        -f data/cmds_example_3D-spatial.dat \    # Path to commands
        -o out/cmds_example_3D-spatial.npz       # To save synthetic samples

**Plot generated samples**

    python utils/plot_samples.py \
        -r out/cmds_example_3D-spatial.npz \     # Synthetic samples
        -i 0,1 \                                 # Indices to plot
        -o out/cmds_example_3D-spatial.png       # Where to save plot

![out/cmds_example_3D-spatial.png](out/cmds_example_3D-spatial.png)

**Plot an interactive volume rendering**

    python utils/render_raster.py \
        -f out/cmds_examle_3D-spatial.npz \  # Synthetic samples
        -i 2                                 # Sample index to view

![out/rendered_example.png](out/rendered_example.png)


### Example: create a 3D synthetic benchmark with temporal autocorrelation across channels

This example converts a single-channel rasters to multi-channel where each channel
represents the spatial data at subsequent discrete time steps. The vector field 
contains the x and y components for movement at each cell. These vectors are used
to shift the cells for each time step. 

**Obtain a vector field**

- The vector field is raster data with shape `(2, rows, cols)`.
- Where the 2-length first dimension organizes x and y components, respectively.
- It should be stored as a numpy `.npz` file.
- There is a provided example file `out/example_timeseries.npz`. 
- In the future, we'll provide some tools for generating these. 

**Generate synthetic samples**

    python benchmarks/benchmark_from_vectorfield.py \
        -r out/example_of_expanded.npz \      # Path to input raster 
        -v data/example_vectorfield.npz \     # Path to input vector field
        --time_steps 6 \                      # Number of time steps to generate
        -o out/example_timeseries.npz         # To save output raster samples

**Plot generated samples**

    python utils/plot_samples.py \
        -r out/example_timeseries.npz \      # Synthetic samples
        -i 0 \                               # Indices to plot
        -o out/example_timeseries.png        # Where to save plot

![out/example_timeseries.png](out/example_timeseries.png)

**Crop the samples**

Shifting causes the boundaries to be NaNs.

    python utils/crop_rasters.py \
        -i out/example_timeseries.npz \        # Path to input raster
        -o out/example_timeseries_crop.npz \   # To save cropped raster
        --low_row 8 \                          # Lower index of rows 
        --low_col 8                            # Lower index of cols

    python utils/plot_samples.py \
        -r out/example_timeseries_crop.npz \   # Synthetic samples
        -i 0 \                                 # Indices to plot
        -o out/example_timeseries_crop.png     # Where to save plot

![out/example_timeseries_crop.png](out/example_timeseries_crop.png)


## Extra Utilities
 
### Example: expand raster dimensions

Most of these tools assume a `.npz` file with a variable that contains an
array with shape (samples, rows, cols, bands). This tool is used to expand
the dimensions of the raster, either on the left or right. 

    python utils/expand_dims.py \
        -i out/example_of_2D_shape.npz \   # Path to input raster
        -o out/example_of_expanded.npz \   # To save reshaped raster
        --expand_left \                    # Choice to expand on left axis
        --expand_right                     # Choice to expand on right axis


### Example: concatenate 2 rasters along the channels

Many models take in multi-channel rasters where the channels might not be related to other other. That is, there is no spatial or temporal meaning to their adjacency. We can create synthetic samples like this by generating distinct samples and then concatenating them along the channels. 

    python utils/cat_rasters.py \
        -a out/cmds_example_2D.npz \         # Raster A
        -b out/cmds_example_3D-spatial.npz \ # Raster B
        -o out/cmds_example_concat.npz       # Concat (A, B)


### Example: crop a raster

    python utils/crop_rasters.py \
        -i out/cmds_example_2D.npz \     # Path to input raster
        -o out/cmds_example_crop.npz \   # To save cropped raster
        --low_row 10 \                   # Lower index of rows 
        --high_row 40                    # Higher index of rows

### Example: Reorder a raster's bands

    python utils/reorder_rasters.py \
        -i out/cmds_example_3D-spatial.npz \    # Path to input raster
        -o out/cmds_example_reordered.npz \     # To save reordered raster
        -r 5,4,2,1                              # Specified band order (skips allowed)

