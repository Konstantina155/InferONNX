# InferONNX: Practical and Privacy-Preserving Machine Learning Inference Using Trusted Execution Environments

Lightweight system for privacy-preserving ML inference using Intel SGX and automatic model partitioning.

📖 **[Read the full paper](https://doi.org/10.1007/978-3-031-97623-0_2)**

## Quick Start

<details>
<summary><b>📋 Prerequisites</b></summary>

* Intel SGX SDK and PSW: [Documentation](https://download.01.org/intel-sgx/latest/dcap-latest/linux/docs/Intel_SGX_SW_Installation_Guide_for_Linux.pdf)
* Ubuntu 20.04
* Occlum: [GitHub Repository](https://github.com/occlum/occlum)
* Python 3.8+
* Python packages (see `requirements.txt`)

</details>

Run the following commands to evaluate InferONNX:
```bash
# InferONNX (disk-based with SGX)
python3 scripts/inference/run_models_in_occlum.py on_disk_caching entire 3 ./

# InferONNX with automated model partitioning
python3 scripts/inference/run_models_in_occlum.py on_disk_caching partitions 3 ./

# InferONNX in-memory (with SGX)
python3 scripts/inference/run_models_in_occlum.py memory_only entire 3 ./

# InferONNX in-memory (without SGX - baseline)
python3 scripts/inference/run_models_in_cpu.py tls_memory_only 3 ./
```

## Automated Model Partitioning

The partitions in `models/*/partitions/` were generated using the automatic model partitioning process described below. To generate partitions from scratch:

### Partitioning from Scratch

* **Step 1: Split each model into individual operators**

      python3 scripts/partitioning/split_models_per_operator.py

    The generated operators are stored in the `operators/` folder inside each model's directory.


* **Step 2: Determine memory-intensive operators**  
To identify memory-intensive (*heavy-weight*) operators, we analyze operator-level inference times from Step 1 (stored in `memory_intensive_ops/<modelname>_operator_times.txt`).

    By comparing execution on SGX and CPU, we compute the overhead introduced by SGX. Operators with an overhead greater than 12× are flagged as memory-intensive.

      python3 scripts/partitioning/determine_memory_intensive_ops.py

    The list of memory-intensive operators for each model will be stored in the `memory_intensive_ops/operator_overhead.txt` file.

* **Step 3: Generate partitions**  
The partitioning process traverses from the last operator to the first to handle complex computational graphs. For each operator, if it is either:  
  * Identified as memory-intensive (from Step 2), or
  * Exceeds the EPC capacity (85MB in our case),
  
  it is partitioned according to the strategy described in the paper.

      python3 scripts/partitioning/generate_partitions.py

    The resulting partitions will be saved in `models/*/new_partitions/`.

    > **Note** This is a slow procedure and may take considerable time to complete.

* **Cleanup (optional)**  
To remove intermediate files:

      python3 scripts/partitioning/clean_necesssary_files.py

## Reproducing paper results

To generate all plots, tables, and benchmarks from the paper, see **[EXPERIMENTS.md](EXPERIMENTS.md)**.
