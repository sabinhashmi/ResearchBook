# Setup and Workflow

This document provides instructions for setting up the necessary environment and running a typical workflow in the **ResearchBook** project.

## 1. Environment Setup

To work on this project, you need to have access to the CVMFS (CernVM File System) and the LHCb environment.

### Pre-requisites

1.  **Probe CVMFS**: Ensure that CVMFS is mounted and accessible.
    ```bash
    cvmfs_config probe
    ```

2.  **Load LHCb Environment**: Source the LHCb environment script to load all the necessary tools and libraries.
    ```bash
    source /cvmfs/lhcb.cern.ch/lib/LbEnv
    ```

3.  **Initialize Proxy**: To access data and resources from the CERN grid (like DIRAC), you need a valid proxy.
    ```bash
    lhcb-proxy-init
    ```
    You will be prompted for your CERN password.

## 2. Typical Workflow: Running a Simulation

A common task is to run a simulation using the Moore framework and then analyze the output. Here is a step-by-step example of how to run the baseline simulation.

### Step 1: Navigate to the Simulation Directory

The Moore configurations are located in the `Tracking/Moore/` directory. For this example, we will use the `Baseline` configuration.

```bash
cd Tracking/Moore/Baseline/
```

### Step 2: Run the Simulation

The `run` script is a convenience wrapper for executing the simulation. The following command runs the simulation using a template options file and pipes the output to a log file for later inspection.

```bash
./run gaudirun.py ../../Options/TemplateOptions.py | tee baseline.log
```

- `gaudirun.py`: The executable for running the simulation.
- `../../Options/TemplateOptions.py`: The configuration file that defines the simulation parameters.
- `tee baseline.log`: This command simultaneously prints the output to the console and saves it to a file named `baseline.log`.

### Step 3: Locate the Output

The output of the simulation will be generated in the `Tracking/Data/Baseline/` directory. This includes:

- `MooreTuple.root`: A ROOT file containing the main analysis tuple.
- `Dumper_recTracks.root`: A ROOT file with reconstructed track information.
- `baseline.log`: The log file from the simulation run.

### Step 4: Analyze the Output

The output data can be analyzed using the Jupyter notebooks located in `Tracking/Notebooks/`. For example, you could use a notebook in `Tracking/Notebooks/PostAnalysis/` to plot efficiency or kinematical distributions from the generated `.root` files.
