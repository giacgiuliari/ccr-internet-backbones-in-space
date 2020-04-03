# Internet Backbones in Space

This README contains all the information necessary to replicate the results in
the paper "Internet Backbones in Space", submitted to ACM CCR (January 2020).

The simulations presented in the paper are based on an in-house satellite network
routing simulator.

## Contents

The files in this submission are organized in the following way:

```
├── analysis: python scripts for the generation of the results in the paper. 
├── data: data required for the experiments.
│   ├── preproc: data that results from the pre-processing steps.
│   └── raw: raw datasets, sourced from external authorities.
├── figures: output folder for the analysis.
├── lib: auxiliary libraries.
└── preproc: scripts to pre-process datasets before the analysis.
```

## Installing the dependencies

The simulation scripts run on `python 3.6`. We provide a simplified way of
installing the many dependencies by using a [docker](https://www.docker.com/)
image. A manual installation is also possible.

### Using Docker

The simplest way to set up and run the experiments is buy building a docker
image and running a docker container. Bear in mind that each of the following
steps may require considerable amounts of time and computing resources.

1. Build the Docker image with the dependencies
   ``` 
   docker build -t ccr .
   ```
2. Run a container from the image just built
   ``` 
   docker run -it ccr
   ```
3. Activate the virtual environment
   ```
   pipenv shell
   ```
4. Now it is possible to run experiments.

### Manual installation

The experiments use `pipenv` as a python virtual environment manager. To install
`pipenv` see the [main page](https://docs.pipenv.org/en/latest/).

There are some dependencies, however, that are external to the pipenv
environment. In particular, the dependencies for
[cartopy](https://scitools.org.uk/cartopy/docs/latest/installing.html) and
[cfgrib](https://github.com/ecmwf/cfgrib) have to be installed before the
virtual environment.

To install the virtual environment and all the dependencies, run in the root
project directory `pipenv install`. Then, to activate the environment: `pipenv
shell`. Now experiments can be run.

## Running the experiments

We now describe how to run the experiments presented in the paper. Before you
start, make sure that all the dependencies are installed correctly and that the
virtual environment is activated (`pipenv shell` in the root folder of the
submission).

The output of the analysis steps is printed to terminal, and the resulting
figures will be found in `figures/`.

The experiments are presented in the order they appear in the paper.

### GST cost of deployment estimation

The analysis of the costs of deployment is specific to a number of ground
stations and to a number of sampling rounds, as the location of the ground
stations is sampled randomly from a GDP distribution.

For example, to estimate the deployment cost of 1000 ground stations,
re-sampling their positions 100 times:
```
python analysis/cost_analysis.py -n 1000 -r 1000
```
The script also outputs the cost of deploying GSTs at the biggest cities.

### Latency under GST placement

The simulation is self-contained: `python analysis/ixp_cities_deployment.py`

### Latency under rain fade

This experiment requires a longer pipeline, as it needs to download historical
weather data and to process them. The experiments in the paper requires weather
data recorded globally twice a day, for every day for 1 year (2018 in this
case).

_NOTE_: The following steps require hours running on a 70-cores machine, ~50
GB of disk space and >10 GB of memory. To ease the replication process, we also
provide instructions for the experiment on a smaller set of inputs. It is
presented after the full pipeline.

1. The first step is to download the dataset with the historical weather data.
   It is possible to specify the desired time interval to download with the
   additional command line arguments.
   ```
   python preproc/download_gfs_data.py --out data/raw/gfs_1_year
   ```
2. Then, this dataset is used to compute the inactive ground stations for each
   time-instant in the dataset.
   ```
   python preproc/rainfall_to_inactive_gst.py data/raw/gfs_1_year data/preproc/inactive_gst_1_year.pkl
   ```
3. Given the different sets of inactive ground stations, re-routing and
   path-control algorithms are run on each of the topologies resulting from
   disconnecting the ground stations. It is possible to specify the number of
   computing cores to execute the experiment in parallel. For the results in the
   paper, it is sufficient to run this for path-control=3.
   ```
   python preproc/rerouting_experiment.py 3 data/preproc/inactive_gst_1_year.pkl data/preproc/reroute_pc3/ --cores 1
   ```
4. Finally, the results of these operations are summarized in the plots that are
   also found in the paper:
   ```
   python analysis/paco_rero_compare.py data/preproc/reroute_pc3/
   ```

**Restricted Case**: Instead of running the experiment on every day for a year,
in this restricted case we look at 2 days per month over the duration of 2
years. The intermediate results of the pipeline are also provided in the
submission material, and are marked with `DEFAULT_` as a filename prefix. Using
these files will allow testing single steps of the pipeline independently. Of
course the results will slightly deviate from the ones in the paper.

1. Download the dataset with historical weather data for the selected period we
   externally hosted.
   ```
   wget https://polybox.ethz.ch/index.php/s/9ZTkBII6jwJabP8/download -O data/raw/DEFAULT_gfs_2month.tar.gz
   tar -xvzf data/raw/DEFAULT_gfs_2month.tar.gz -C data/raw
   ```
2. Compute the inactive ground stations.
   ```
   python preproc/rainfall_to_inactive_gst.py data/raw/DEFAULT_gfs data/preproc/inactive_gst_2month.pkl
   ```
3. Run re-routing and path control on the topologies resulting from the inactive
   ground stations (default file output form the previous step: 
   `data/preproc/DEFAULT_inactive_gst_2month.pkl`)
   ```
   python preproc/rerouting_experiment.py 3 data/preproc/DEFAULT_inactive_gst_2month.pkl data/preproc/reroute_pc3_2month/ --cores 1
   ```
4. Analyze the results (default directory of results:
   `data/preproc/DEFAULT_reroute_pc3_2month`)
   ```
   python analysis/paco_rero_compare.py data/preproc/reroute_pc3_2month/
   ```
   
