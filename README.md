# Artifact for paper #104 InferONNX: Practical and Privacy-preserving Machine Learning Inference using Trusted Execution Environments

## Overview
InferONNX is a lightweight TEE-based system for secure ONNX model inference using Intel SGX. It supports partitioning large models into smaller sub-models to fit within SGX memory constraints, enabling efficient and privacy-preserving inference directly from disk.

### Artifact summary
This repository contains:
* The implementation of InferONNX for secure inference on Intel SGX (via Occlum)
* Support for automatic model partitioning
* Scripts for reproducing experiments in both settings:
  * SGX (with TLS protocol support)
  * CPU (without TLS protocol)

> **Note:** The paper also evaluates a memory-based variant of InferONNX for comparison with the disk-based approach. However, the focus is on the disk-based implementation, which supports partitioning and represents the core contribution of the system.

### Prerequisites
Before running InferONNX, ensure the following dependencies are met:
* Intel SGX SDK and PSW (for running SGX-based experiments): [Intel SGX Documentation](https://download.01.org/intel-sgx/latest/dcap-latest/linux/docs/Intel_SGX_SW_Installation_Guide_for_Linux.pdf)
* OS Version: Ubuntu 20.04
* Occlum (SGX LibOS used for enclave execution): [Occlum GitHub Repository](https://github.com/occlum/occlum)
* TLS support: 1.2 or 1.3 (required for MbedTLS library)
* Python version: 3.8+
* Other required Python packages (see `requirements.txt`)

### Experiments
The evaluation primarily focuses on **latency** and **accuracy** as the key performance metrics.
> **Note** Before running any commands, ensure that you are at the root of the cloned repository.

To ensure that model accuracy is preserved after partitioning, inference is performed on both the entire model and its partitions. To verify the accuracy of the partitions, run the below command:
```
python3 scripts/check_accuracy.py partitions/
```

Two main plots and one table summarize the results:
* Figure 4 in our paper: Performance evaluation across **InferONNX-on Disk with SGX**, **InferONNX-Memory only with SGX** and **InferONNX-without SGX**.
* Figure 5 in our paper: Performance evaluation across entire models and their partitions using **InferONNX-on Disk with SGX**.
* Table 2 in our paper: Performance evaluation across **InferONNX-Memory only without SGX** and **InferONNX-on Disk without SGX**

The partitions used in the evaluation are generated and stored in subdirectories under the `models/` folder. Each subdirectory is named after its correspoding model.

To generate the plots and the CSV file for the table, run the below command:
```
python3 scripts/run_all.py partitions/
```
The generated plots and the CSV file will be saved in the results/ directory.
