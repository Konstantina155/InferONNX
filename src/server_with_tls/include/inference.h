#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include <storage.h>

#define check(call) do {                                                       \
    TRACT_RESULT result = (call);                                              \
    if (result == TRACT_RESULT_KO) {                                           \
        fprintf(stderr, "Error calling tract: %s\n", tract_get_last_error());  \
        return;                                                                \
    }                                                                          \
} while (0)

// Overload for non-void functions
#define check_ret(call, ret_value) do {                                        \
    TRACT_RESULT result = (call);                                              \
    if (result == TRACT_RESULT_KO) {                                           \
        fprintf(stderr, "Error calling tract: %s\n", tract_get_last_error());  \
        return (ret_value);                                                    \
    }                                                                          \
} while (0)

#if USE_AES
    void load_model_to_memory(model **m, unsigned char **tags, int count_tags);
    #if USE_MEMORY_ONLY
        void run_inference(operator_node **node, TractValue **input_values, TractInferenceModel *inference_model);
        char *inference_memory_only(float **images, int num_images, model *m);
    #else
        void run_inference(operator_node **node, TractValue **input_values, struct EncryptionParameters *params);
        char *inference_aes(float **images, int num_images, uint8_t *tokenizer, int tokenizer_size, model *m, unsigned char **tags, int count_tags);
    #endif
#else
    void run_inference(operator_node **node, TractValue **input_values, TractInferenceModel *inference_model);
    char *inference_no_aes(float **images, int num_images, uint8_t *tokenizer, int tokenizer_size, model *m);
    void load_model_to_memory(model **m);
#endif