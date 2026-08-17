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
    void load_ner_model_to_memory(char *ner_model_name, uint8_t *ner_model,int ner_model_size, char *ner_tokenizer_name, uint8_t *ner_tokenizer, int ner_tokenizer_size);
    void load_model_to_memory(model **m, unsigned char **tags, int count_tags);
    #if USE_MEMORY_ONLY
        void run_inference_cnn(operator_node **node, input_info *input_info_ptr, void *inference_model_ptr);
        void run_inference_llm(operator_node **node, input_info *input_info_ptr, void *inference_model_ptr);
        char *inference_memory_only(float **images, int num_images, char *prompt, model *m);
    #else
        void run_inference_cnn(operator_node **node, input_info *input_info_ptr, struct EncryptionParameters *params, struct EncryptionParameters *params_weights);
        void run_inference_llm(operator_node **node, input_info *input_info_ptr, struct EncryptionParameters *params, struct EncryptionParameters *params_weights);
        char *inference_aes(float **images, int num_images, char *prompt, model *m, unsigned char **tags, int count_tags);
    #endif
#else
    void load_model_to_memory(model **m);
    void run_inference_cnn(operator_node **node, input_info *input_info_ptr, void *inference_model_ptr);
    void run_inference_llm(operator_node **node, input_info *input_info_ptr, void *inference_model_ptr);
    char *inference_no_aes(float **images, int num_images, char *prompt, model *m);
#endif