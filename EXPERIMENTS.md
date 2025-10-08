# Reproducing paper results

This document provides detailed instructions for reproducing all figures, table, and benchmarks from the paper.

> **Note:** All commands should be run from the root of the repository.

## Main plots and table

Generate all plots and table from the paper:
```bash
python3 scripts/run_all.py partitions/ 3
```

This command generates:
* **Figure 1**: Execution time breakdown for five ML models
* **Figure 4**: Performance evaluation across `InferONNX` (default disk-based approach in SGX), `InferONNX (in mem)` and `InferONNX (in mem w/o SGX)`
* **Figure 5**: Performance evaluation across entire models and their partitions using `InferONNX`
* **Table 2**: Performance evaluation across CPU configurations w/o SGX and TLS when stored in memory or loaded from disk

Results are saved in the `results/` directory. The parameter `3` specifies the number of runs per configuration (adjust as needed).

> **Note** This process can be time-consuming as it runs all models across multiple configurations.

## Benchmarks

**Instructions Per Cycle (IPC)** \
Generate IPC measurements (Table 3 in our paper):
```bash
python3 scripts/benchmarks/generate_cache_stats.py 10000
```

Results are saved as a CSV file in the `results/` directory. 

> **Note** High iteration counts are time-consuming. For quicker results, consider reducing the number of runs.

**Memory requirements** \
Generate memory-usage CDF plots (Figure 3 in the paper):
```bash
python3 scripts/benchmarks/create_cdf.py partitions/
```

CDF plots for each model and its partitions are saved in the `results/` directory.

> **Note** The process can be time-consuming, as Valgrind’s Massif tool captures heap snapshots during inference.