## tls_server
### Overview
This implementation uses MbedTLS to enable TLS 1.2 and TLS 1.3 connections between the server and clients. It provides an additional layer of security using the AES-256-GCM encryption algorithm.

### Flags
#### USE_SYS_TIME
- Calculates the total time required to perform inference.
- **Note:** Not recommended in SGX environments, as frequent system calls between host and enclave can introduce overhead and affect timing accuracy. Use only when detailed timing information is needed.

#### USE_SYS_TIME_OPERATORS
- Calculates the inference time of each individual operator in the model.
- **Should be used with** `USE_MEMORY_ONLY=1`, `USE_OCCLUM=1` and `USE_AES=1` to profile operators running in SGX using the faster in-memory approach.

#### USE_OCCLUM
- Runs the server within the **Occlum SGX enclave** environment.
- **Should be used with** `USE_AES=1` for model protection through encryption/decryption.

#### USE_AES
- Enables **AES-256-GCM** encryption to securely encrypt and decrypt the model or its partitions when stored on disk.

#### USE_MEMORY_ONLY
- Loads the model(s) into memory after decrypting them from disk, avoiding disk storage.

#### USE_FILE_CACHING
- When set to `1`, file caches are retained between executions.
- When set to `0`, file caches are cleared between each execution.
- **Should be used with** `USE_MEMORY_ONLY=0` (disk-based approach only).

## Example configurations
### SGX configurations
**Disk-based approach:** \
`USE_SYS_TIME=0 USE_SYS_TIME_OPERATORS=0 USE_OCCLUM=1 USE_AES=1 USE_MEMORY_ONLY=0 USE_FILE_CACHING=0`

**Disk-based approach with File Caching:** \
`USE_SYS_TIME=0 USE_SYS_TIME_OPERATORS=0 USE_OCCLUM=1 USE_AES=1 USE_MEMORY_ONLY=0 USE_FILE_CACHING=1`

**Disk-based approach for detailed timing:** \
`USE_SYS_TIME=1 USE_SYS_TIME_OPERATORS=0 USE_OCCLUM=1 USE_AES=1 USE_MEMORY_ONLY=0 USE_FILE_CACHING=0`

> **Note:** Only the above disk-based approaches use model partitions.

**Memory-based approach:** \
`USE_SYS_TIME=0 USE_SYS_TIME_OPERATORS=0 USE_OCCLUM=1 USE_AES=1 USE_MEMORY_ONLY=1 USE_FILE_CACHING=0`

**Operator timing:** \
`USE_SYS_TIME=0 USE_SYS_TIME_OPERATORS=1 USE_OCCLUM=1 USE_AES=1 USE_MEMORY_ONLY=1 USE_FILE_CACHING=0`

### CPU configurations (TLS only)
**Disk-based:** \
`USE_SYS_TIME=0/1 USE_SYS_TIME_OPERATORS=0 USE_OCCLUM=0 USE_AES=0 USE_MEMORY_ONLY=0 USE_FILE_CACHING=0`

**Memory-based:** \
`USE_SYS_TIME=0/1 USE_SYS_TIME_OPERATORS=0 USE_OCCLUM=0 USE_AES=0 USE_MEMORY_ONLY=0 USE_FILE_CACHING=1`

> **Note:**
> - `USE_MEMORY_ONLY=0` is used in the memory-based approach because `USE_FILE_CACHING=1` keeps caches between client requests, maintaining the model data in memory.
> - Set `USE_SYS_TIME=1` to capture timing on the server side (only for detailed timing)
> - Set `USE_SYS_TIME=0` for the client side (default setting)