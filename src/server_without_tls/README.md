## no_tls_server

### Overview
This implementation does not use TLS, and therefore does not provide encrypted communication between the server and clients. It is intended for use in environments where encryption is unnecessary. However, AES-256-GCM encryption can still be optionally applied to protect model files at rest (i.e., when stored on disk).

### Flags

#### USE_AES
- Enables **AES-256-GCM** encryption to securely encrypt and decrypt the model or its partitions when stored on disk.

#### USE_MEMORY_ONLY
- Loads the model(s) into memory, avoiding disk storage.

#### USE_SYS_TIME_OPERATORS
- Calculates the inference time of each individual operator in the model.