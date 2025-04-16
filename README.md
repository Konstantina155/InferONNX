# Artifact for paper #104 InferONNX: Practical and Privacy-preserving Machine Learning Inference using Trusted Execution Environments

## Overview
InferONNX is a lightweight TEE-based system for secure ONNX model inference using Intel SGX. It supports partitioning large models into smaller sub-models to fit within SGX memory constraints, enabling efficient and privacy-preserving inference directly from disk.

## Artifact summary
This repository contains:
* The implementation of InferONNX for secure inference on Intel SGX (via Occlum)
* Support for automatic model partitioning
* Scripts for reproducing experiments in both settings:
  * SGX (with TLS protocol support)
  * CPU (without TLS protocol)

> **Note:** The paper also evaluates a memory-based variant of InferONNX for comparison with the disk-based approach. However, the focus is on the disk-based implementation, which supports partitioning and represents the core contribution of the system.

## Prerequisites
Before running InferONNX, ensure the following dependencies are met:
* Intel SGX SDK and PSW (for running SGX-based experiments): [Intel SGX Documentation](https://download.01.org/intel-sgx/latest/dcap-latest/linux/docs/Intel_SGX_SW_Installation_Guide_for_Linux.pdf)
* OS Version: Ubuntu 20.04
* Occlum (SGX LibOS used for enclave execution): [Occlum GitHub Repository](https://github.com/occlum/occlum)
* TLS support: 1.2 or 1.3 (required for MbedTLS library)
* Python version: 3.8+
* Other required Python packages (see `requirements.txt`)

## Experiments
The evaluation primarily focuses on **latency** and **accuracy** as the key performance metrics.
> **Note** Before running any commands, ensure that you are at the root of the cloned repository.

> The partitions used in the evaluation are generated and stored in subdirectories under the `models/` folder. Each subdirectory is named after its corresponding model.

### Accuracy verification of Existing Partitions
To ensure that model accuracy is preserved after partitioning, inference is performed on both the entire model and its partitions. To verify the accuracy of the partitions, run the below command (where `partitions/` is the folder containing the existing partitions):
```
python3 scripts/check_accuracy.py partitions/
```

### Main plots and table generation
Two main plots and one table summarize the results:
* Figure 1 in our paper: Execution time breakdown for five popular ML models.
* Figure 4 in our paper: Performance evaluation across **InferONNX-on Disk with SGX**, **InferONNX-Memory only with SGX** and **InferONNX-without SGX**.
* Figure 5 in our paper: Performance evaluation across entire models and their partitions using **InferONNX-on Disk with SGX**.
* Table 2 in our paper: Performance evaluation across **InferONNX-Memory only without SGX** and **InferONNX-on Disk without SGX**

To generate the plots and the CSV file for the table, run the below command (where `partitions/` is the folder containing the existing partitions, and `3` is the number of runs for each model/partitions of model):
```
python3 scripts/run_all.py partitions/ 3
```
This command also measures the inference time of each operator for every model, which is later used to identify memory-intensive operators in the automatic partitioning process. The inference times are saved into the `memory_intensive_ops/` folder under the filename `<modelname>_operator_times.txt`.

The generated plots and CSV file will be saved in the `results/` directory.
> **Note** This script runs all models for each configuration (SGX and non-SGX), repeated according to the specified number of runs. It also collects per-operator inference times for each model, which are used later to identify memory-intensive operators. This process can be time-consuming depending on the number of runs.

### Automatic partitioning from scratch
To generate the partitions from scratch, follow the steps below:

* **Step 1: Split each model into individual operators**  
  Run the following command to split each model into its individual operators:  

      python3 scripts/partitioning/split_models_per_operator.py

    For each model, inference is performed on the generated operators executed sequentially—each operator passes its output to the next. This ensures that the chained execution reproduces the output of the original unsplit model. The generated operators are stored in the `operators/` folder inside each model's directory.


* **Step 2: Determine memory-intensive operators**  
To identify *heavy-weight* (memory-intensive) operators for each model, we use the operator-level inference times collected during **Step 1** (stored in `<modelname>_operator_times.txt` inside `memory_intensive_ops/`).

    By comparing execution on SGX and CPU, we compute the overhead introduced by SGX. Operators with an overhead greater than 12× are flagged as memory-intensive and added to the list. Models without any such operators will not appear in the final list.

    Run the following command to perform this analysis:

      python3 scripts/partitioning/determine_memory_intensive_ops.py

    The list of memory-intensive operators for each model will be stored in the `memory_intensive_ops/operator_overhead.txt` file.

* **Step 3: Partition models**  
Finally, the partitioning process begins from the **last operator** to the **first** (in reverse order from Step 1, where model is split from first to last). For each operator, if it is either:  
  * identified as memory-intensive (from Step 2), or
  * exceeds the EPC capacity (85MB in our case),
  
  it is handled according to the strategy described in the paper.

  To generate the new partitions, run:

      python3 scripts/partitioning/generate_partitions.py

    Inference is then performed on the generated partitions to ensure that they match the inference of the entire model. The resulting partitions will be saved in the `new_partitions/` folder within the corresponding model's directory.

    > **Note** These partitions differ from those in the paper, which were generated from first operator to last. The updated script traverses from last operator to first to handle complex models. This is a slow procedure and may take considerable time to complete.

> **Note:** In case you want to remove the 'operators/' folder from each model, the , run the following command:

      python3 scripts/partitioning/clean_necesssary_files.py

### Benchmarks
This section presents two key system-level benchmarks: IPC (Instructions Per Cycle) and memory usage using Valgrind’s Massif tool.

**Table3 in our paper:** Instructions Per Cycle (IPC)
To calculate the IPC for each model, run the following command:
```
python3 scripts/benchmarks/generate_cache_stats.py 10000
```
The resulting IPC values will be saved as a CSV file in the `results/` directory. 
> **Note** A high number of runs can be time-consuming. For quicker results, consider reducing the number of runs.

**Figure3 in our paper:** Memory requirements
To evaluate memory usage, we use *Valgrind’s Massif tool* to trace heap memory consumption. To generate memory usage CDF plots, run:
```
python3 scripts/benchmarks/create_cdf.py partitions/
```
The output CDF plots for each model and its partitions will be saved in the `results/` directory.
> **Note** The process can be time-consuming, as Valgrind’s Massif tool captures heap snapshots in real time throughout inference.