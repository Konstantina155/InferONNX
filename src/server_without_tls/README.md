## no_tls_server
### Overview
This implementation does not use TLS, and therefore does not provide encrypted communication between the server and clients. It is intended for use in environments where encryption is unnecessary. However, AES-256-GCM encryption can still be optionally applied to protect model files at rest (i.e., when stored on disk).

### Flags
#### USE_AES
- Enables **AES-256-GCM** encryption to securely encrypt and decrypt the model or its partitions when stored on disk.
- **Should be used with** `USE_MEMORY_ONLY=1` since clients make requests without dropping caches.

#### USE_MEMORY_ONLY
- When set to `1`, prevents the system from dropping caches between client requests, keeping the model data in memory.
- When set to `0`, allows the system to drop caches between requests requiring the model to be reloaded from disk.
- Can be used standalone or with `USE_AES=1`.

#### USE_SYS_TIME_OPERATORS
- Calculates the inference time of each individual operator in the model.
- **Should be used with** `USE_MEMORY_ONLY=1` and `USE_AES=0`.

## Example configurations
**Basic loading with cache retention:** \
`USE_MEMORY_ONLY=1 USE_AES=0 USE_SYS_TIME_OPERATORS=0`

**With encryption (caches retained):** \
`USE_MEMORY_ONLY=1 USE_AES=1 USE_SYS_TIME_OPERATORS=0`

**Allow cache dropping (no encryption):** \
`USE_MEMORY_ONLY=0 USE_AES=0 USE_SYS_TIME_OPERATORS=0`

**Operator timing:** \
`USE_MEMORY_ONLY=1 USE_AES=0 USE_SYS_TIME_OPERATORS=1`