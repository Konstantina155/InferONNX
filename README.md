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

### Evaluation
The evaluation primarily focuses on **latency** as the key performance metric.
Two main plots summarize the results:
* Figure 4 in the paper: Performance evaluation across **InferONNX-on Disk with SGX**, **InferONNX-Memory only with SGX** and **InferONNX-without SGX**.
* Figure 5 in the paper: Performance evaluation across entire models and their partitions using **InferONNX-on Disk with SGX**.

To ensure the accuracy of the model is preserved after partitioning, inference is performed on both the entire model and its partitions. 
