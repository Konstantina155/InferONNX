#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>

#if USE_SYS_TIME == 1 || USE_AES == 0
#include <sys/time.h>
#endif

#include <tract.h>

#include "mbedtls/entropy.h"    // mbedtls_entropy_context
#include "mbedtls/ctr_drbg.h"   // mbedtls_ctr_drbg_context
#include "mbedtls/cipher.h"     // MBEDTLS_CIPHER_ID_AES
#include "mbedtls/gcm.h"        // mbedtls_gcm_context

#define mbedtls_printf printf
#define KEY_BYTES 32
#define KEY_BITS KEY_BYTES * 8
#define IV_BYTES 12
#define TAG_BYTES 16
#define ADD_DATA_BYTES 64
#define BUF_SIZE 4096

#define HASH_MULTIPLIER 65599
#define CAPACITY 3000

typedef struct __attribute__((packed)) {
    int command;
    int id;
    int num_models;
    int num_inputs;
    char **names;
    int *size_models;
    uint8_t **models;
    int* size_inputs;
    float **input;
    unsigned char **tags;
    int tokenizer_size;
    uint8_t *tokenizer; 
} request;

typedef struct encrypted_models_info
{
    unsigned char **encrypted_model;
    unsigned char key[KEY_BYTES];
    unsigned char IV[IV_BYTES];
    unsigned char AAD[ADD_DATA_BYTES];
    unsigned char **tag;
} encrypted_models_info;

typedef struct client_result
{
    unsigned char *result;
    int size;
    unsigned char **tag;
} client_result;

typedef struct operator_node {
    #if USE_AES == 0 && USE_MEMORY_ONLY == 0 || USE_AES == 1 && USE_MEMORY_ONLY == 1
        void (*run_inference)(struct operator_node **node, TractValue **input_values, TractInferenceModel *inference_model);
    #elif USE_AES == 1
        void (*run_inference)(struct operator_node **node, TractValue **input_values, struct EncryptionParameters *params);
    #endif
    TractValue **outputs;
    char *model_name;
    int num_inputs;
    int num_outputs;
    int num_children;
    int num_parents;
    struct operator_node **parents;
    struct operator_node **children;
    int *parent_output_indices;
    double pred;
    int category;
    double elapsedTime;
}operator_node;

typedef struct model
{
    char *id;
    int size;
    char **names;
    unsigned char key[KEY_BYTES];
    unsigned char IV[IV_BYTES];
    unsigned char AAD[ADD_DATA_BYTES];
    TractInferenceModel **inference_models;
    operator_node *head;
    struct model *next;
} model;

typedef struct onnx_table
{
    int count;
    model **model;
    unsigned int top;
} onnx_table;

typedef struct {
    char *model_name;
    int input_names_length;
    char **input_names;
    int output_names_length;
    char **output_names;
}operator_io;