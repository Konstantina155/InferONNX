# Journal Artifact of ShardLLM: Multi-level Partitioning for Large Language Model Inference within Trusted Execution Environment

## Overview
ShardLLM is a lightweight TEE-based system for secure NNEF LLM inference using Intel SGX. It introduces a multi-level partitioning approach (combining inter- and intra-operator partitioning) to accommodate the large memory demands of individual LLM operators within SGX memory constraints, enabling efficient and privacy-preserving inference directly from disk.

## Artifact summary
This repository contains:
* The implementation of ShardLLM for secure inference on Intel SGX (via Occlum)
* Support for our multi-level partitioning approach (combining both inter- and intra-operator partitioning)
* Scripts for reproducing experiments in both settings:
  * SGX (with TLS protocol support)
  * CPU (with TLS protocol support)

> **Note:** The journal also evaluates a memory-based variant of ShardLLM for comparison with the disk-based approach. However, the focus is on the disk-based implementation. This version applies the existing inter-operator partitioning technique (dividing the model between distinct operators) to LLMs, and combines it with intra-operator partitioning (decomposing individual operators internally into multiple smaller components). This combined, multi-level partitioning approach represents the core contribution of the system.

## Prerequisites
Before running ShardLLM, ensure the following dependencies are met:
* Intel SGX SDK and PSW (for running SGX-based experiments): [Intel SGX Documentation](https://download.01.org/intel-sgx/latest/dcap-latest/linux/docs/Intel_SGX_SW_Installation_Guide_for_Linux.pdf)
* OS Version: Ubuntu 20.04
* Occlum (SGX LibOS used for enclave execution): [Occlum GitHub Repository](https://github.com/occlum/occlum)
* TLS support: v1.3 (required for MbedTLS library)
* Python version: 3.8+
* Other required Python packages (see `requirements.txt`)

## Experiments
The evaluation primarily focuses on **latency** and **accuracy** as the key performance metrics.
> **Note** Before running any commands, ensure that you are at the root of the cloned repository.

> The inter- and intra-operator partitions used in the evaluation are generated and stored in subdirectories under the `models/` folder. Each subdirectory is named after its corresponding model.

### Accuracy verification of partitions
To ensure that model accuracy is preserved after partitioning, we evaluate the generated output qualitatively. The provided script performs inference on both the entire model and its partitions variants (inter-op and intra-op). To verify the accuracy of the partitions, run the below command (where `partitions_inter/` is the folder containing inter-operator partitions and `partitions_intra/` is the folder containing intra-operator partitions, and `10` is the number of tokens generated):

The default prompt is: `Hi, how are you today?`. However, a desired prompt can be put as a txt input to the script executing the inference.
```
python3 scripts/check_accuracy.py partitions_inter/ partitions_intra/ 10
```
> **Note** We recommend setting the numer of generated tokens to at least 10. Generating fewer tokens may not provide enough context to properly evaluate the coherence and semantic validity of the text.

### Main plots and table generation
Two main plots and one table summarize the results:
* Figure 2 in our journal: Pareto chart analyzing memory usage (in MB) and cumulative execution (%) of the LLM operators.
* Figure 4 in our journal: Performance evaluation of ShardLLM across **Full model execution**, **Inter-op partitioning** and **Intra-op partitioning**.
* Table 1 in our journal: Baseline performance comparing **TTFT in Intel SGX** (evaluating both **Loaded from disk** and **Preloaded in Memory** variants) against **TTFT on CPU**, reflecting the exact column headers from the journal.
* Table 2 in our journal: Micro-benchmark exploring the optimal TTFT of splitting the MatMul operator into Row-partitioned and Column-partitioned MatMul across powers of 2 until host memory exhaustion.

To generate the plots and the CSV file for the table 1, run the below command (where `partitions_inter/` and `partitions_intra/` are the folders containing the generated partitions, and `3` is the number of runs for each model/partitions of model):
```
python3 scripts/run_all.py both partitions_inter/ partitions_intra/ 3
```
This command also measures the inference time of each operator for every model, which is used in `create_plots.py` to identify which operators are memory-intensive in the Pareto graph (Figure 2). The inference times are saved into the `memory_intensive_ops/` folder under the filename `<modelname>.txt`.

The generated plots and CSV file will be saved in the `results/` directory.
> **Note** This script runs all models for each configuration (SGX and non-SGX for entire, SGX for partitioning), repeated according to the specified number of runs. It also collects per-operator inference times for each model for Figure 2.

### Inter- and intra-operator partitioning

* **Inter-operator partitioning**

  To generate inter-operator partitions, which includes splitting the model layer-wise and separating the initial gather and multi-layer projections (MLP) into standalone units, run the following command:
    ```
    python3 scripts/partitioning/generate_inter_partitions.py
    ```

    The resulting partitions will be saved in the `partitions_inter/` folder within the corresponding model's directory.


* **Intra-operator partitioning**

  To generate intra-operator partitions, execute the following micro-benchmark experiment to find the optimal TTFT for a partitioning scheme (a combination of strategy - either Row- or Column-partitioned MatMul and partition count). We explore all partition counts in powers of two until host memory exhaustion (Table 2 in our journal):
    ```
    python3 scripts/partitioning/generate_intra_partitions.py
    ```

    The optimal partitions will be saved in the `partitions_intra/` folder within the corresponding model's directory and the generated CSV file for the table will be saved in the `results/` directory.

### Clean-up
To clean up the intermediate files generated during the micro-benchmarks and profiling (including the `original_intra_ops/`, `partitions_test/`, and `memory_intensive_ops/` folders, along with the *onnx-tool* CSV files), run the following command:
```
python3 scripts/clean_intermediate_files.py
```