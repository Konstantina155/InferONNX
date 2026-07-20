#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <inference.h>

// INFERENCE INFO STRUCT - LOADING PHASE
static void **
initialize_inference_models_ptr(int num_models)
{
    void **inference_models_ptr = (void **)malloc((num_models + 1) * sizeof(void *));
    if (!inference_models_ptr) {
        fprintf(stderr, "Error allocating memory for inference_models_ptr\n");
        return NULL;
    }

    for (int i = 0; i < num_models + 1; ++i) {
        inference_models_ptr[i] = NULL;
    }

    return inference_models_ptr;
}

#ifdef USE_AES
static void
onnx_model_inputs(operator_io **io, void *inference_model_ptr, ModelType model_type, int index, operator_node *head, char *model_name)
{
    uintptr_t num_inputs = 0;
    uintptr_t num_outputs = 0;
    char *input_name = NULL;
    char **input_names = NULL;
    char **output_names = NULL;
    int8_t *output_name = NULL;

    if (inference_model_ptr) {
        switch (model_type) {
            case MODEL_TYPE_CNN: {
                TractInferenceModel *cnn_inference_model = (TractInferenceModel *)inference_model_ptr;
                check(tract_inference_model_input_count(cnn_inference_model, &num_inputs));
                input_names = malloc((num_inputs + 1) * sizeof(char *));
                if (!input_names) return;
                for (int i = 0; i < (int)num_inputs; i++) {
                    check(tract_inference_model_input_name(cnn_inference_model, i, &input_name));
                    input_names[i] = input_name;
                }
                input_names[num_inputs] = NULL;

                check(tract_inference_model_output_count(cnn_inference_model, &num_outputs));
                output_names = malloc((num_outputs + 1) * sizeof(char *));
                if (!output_names) return;
                for (int i = 0; i < (int)num_outputs; i++) {
                    check(tract_inference_model_output_name(cnn_inference_model, i, &output_name));
                    output_names[i] = (char *)output_name;
                }
                output_names[num_outputs] = NULL;
                break;
            }
            case MODEL_TYPE_LLM: {
                if (!inference_model_ptr) break; 
                TractLlmInferenceModel *llm_inference_model = (TractLlmInferenceModel *)inference_model_ptr;
                check(tract_llm_inference_model_input_count(llm_inference_model, &num_inputs));
                input_names = malloc((num_inputs + 1) * sizeof(char *));
                if (!input_names) return;
                for (int i = 0; i < (int)num_inputs; i++) {
                    check(tract_llm_inference_model_input_name(llm_inference_model, i, &input_name));
                    input_names[i] = input_name;
                }
                input_names[num_inputs] = NULL;

                check(tract_llm_inference_model_output_count(llm_inference_model, &num_outputs));
                output_names = malloc((num_outputs + 1) * sizeof(char *));
                if (!output_names) return;
                for (int i = 0; i < (int)num_outputs; i++) {
                    check(tract_llm_inference_model_output_name(llm_inference_model, i, &output_name));
                    output_names[i] = (char *)output_name;
                }
                output_names[num_outputs] = NULL;
                break;
            }
            default:
                fprintf(stderr, "Error: Unknown model type.\n");
                return;
        }
    }

    if (index == 1) {
        operator_io o_io_first;
        o_io_first.input_names_length = 0;
        o_io_first.input_names = NULL;

        #if NUM_TOKENS != 0
            int number_inputs_llm = 3;
            if (strstr(model_name, "gpt2") != NULL) {
                number_inputs_llm = 2;
            }
            char **input_names_llm = malloc((number_inputs_llm + 1) * sizeof(char *));
            if (!input_names_llm) return;
            input_names_llm[0] = strdup("input_ids");
            if (number_inputs_llm == 3) {
                if (strstr(model_name, "albert") != NULL) {
                    input_names_llm[2] = strdup("token_type_ids");
                } else {
                    input_names_llm[2] = strdup("position_ids");
                }
                input_names_llm[1] = strdup("attention_mask");
            } else if (number_inputs_llm == 2) {
                input_names_llm[1] = strdup("attention_mask");
            }
            input_names_llm[number_inputs_llm] = NULL;
            o_io_first.output_names_length = number_inputs_llm;
            o_io_first.output_names = input_names_llm;
        #else
            o_io_first.output_names_length = num_inputs;
            o_io_first.output_names = input_names;
        #endif

        insert_into_operator_io(&io, &o_io_first, index - 1, "input");
        update_node(io, index - 1, NULL);

        #if NUM_TOKENS != 0
            for (int i = 0; i < number_inputs_llm; i++) {
                free(input_names_llm[i]);
            }
            free(input_names_llm);
        #endif
    }

    operator_io o_io;
    operator_node *head2 = NULL;
    o_io.input_names_length = num_inputs;
    if (num_inputs == 0) {
        head2 = head;
        o_io.input_names = NULL;
        head = NULL;
    } else {
        o_io.input_names = input_names;
    }
    o_io.output_names_length = num_outputs;
    o_io.output_names = output_names;
    insert_into_operator_io(&io, &o_io, index, model_name);

    if (num_inputs == 0) {
        operator_node *child = search_operator_node_by_name(head2, io[index]->model_name);
        if (!child) return;
        child->num_inputs = io[index]->input_names_length;
        child->num_outputs = io[index]->output_names_length;
    }

    for (int i=0; i < (int)num_inputs; i++) {
        tract_free_cstring(input_names[i]);
    }
    free(input_names);

    for (int i=0; i < (int)num_outputs; i++) {
        tract_free_cstring(output_names[i]);
    }
    free(output_names);

    update_node(io, index, head);
    // print_operator_io(io);
}

static void *
onnx_model_for_path(char *model_name, void **inference_model_ptr, ModelType model_type, struct EncryptionParameters *params, struct EncryptionParameters *params_weights)
{  
    fprintf(stderr, "Loading model: %s", model_name);
    switch (model_type) {
        case MODEL_TYPE_CNN: {
            assert(!params_weights);

            // Initialize onnx parser
            TractOnnx *onnx = NULL;
            check_ret(tract_onnx_create(&onnx), NULL);
            assert(onnx);

            // Load the model
            TractInferenceModel *cnn_inference_model = NULL;
            if (tract_onnx_model_for_path_cnn(onnx, model_name, &cnn_inference_model, params) != TRACT_RESULT_OK) {
                fprintf(stderr, "Error calling tract: %s", tract_get_last_error());
                check_ret(tract_onnx_destroy(&onnx), NULL);
                check_ret(tract_cnn_inference_model_release(&cnn_inference_model), NULL);
                assert(!cnn_inference_model);
                assert(!onnx);
                return NULL;
            }
            assert(cnn_inference_model);
            *inference_model_ptr = (void*)cnn_inference_model;
            assert(onnx);

            check_ret(tract_onnx_destroy(&onnx), NULL);
            assert(!onnx);
    
            break;
        }
        case MODEL_TYPE_LLM: {
            // Load the model
            TractLlmInferenceModel *llm_inference_model = NULL;
            if (tract_onnx_model_for_path_llm(model_name, params, params_weights, &llm_inference_model) != TRACT_RESULT_OK) {
                fprintf(stderr, "Error calling tract: %s", tract_get_last_error());
                check_ret(tract_llm_inference_model_release(&llm_inference_model), NULL);
                assert(!llm_inference_model);
                return NULL;
            }
            assert(llm_inference_model);
            *inference_model_ptr = (void*)llm_inference_model;
            break;
        }
        default:
            fprintf(stderr, "Error: Unknown model type.\n");
            return NULL;
    }  

    return *inference_model_ptr;
}

void
load_model_to_memory(model **m, unsigned char **tags, int count_tags)
{
    if (!m) return;

    assert(tags);

    EncryptionParameters *params = (EncryptionParameters *)malloc(sizeof(EncryptionParameters));
    if (!params) {
        fprintf(stderr, "Memory allocation for params failed\n");
        return;
    }
    uint8_t *key = (uint8_t *)malloc(KEY_BYTES);
    uint8_t *iv = (uint8_t *)malloc(IV_BYTES);
    uint8_t *tag = NULL;
    uint8_t *aad = (uint8_t *)malloc(ADD_DATA_BYTES);
    if (!key || !iv || !aad) {
        fprintf(stderr, "Memory allocation for key, iv, tag, aad failed\n");
        free(params);
        return;
    }
    memcpy(key, (*m)->key, KEY_BYTES);
    memcpy(iv, (*m)->IV, IV_BYTES);
    memcpy(aad, (*m)->AAD, ADD_DATA_BYTES);
    params->key = key;
    params->iv = iv;
    params->aad = aad;
    if (!params->key || !params->iv || !params->aad) {
        fprintf(stderr, "Error reading Encryption parameters from onnx table\n");
        free(params);
        return;
    }

#if NUM_TOKENS != 0
    EncryptionParameters *params_weights = (EncryptionParameters *)malloc(sizeof(EncryptionParameters));
    if (!params_weights) {
        fprintf(stderr, "Memory allocation for params_weights failed\n");
        free(key);
        free(iv);
        free(aad);
        free(params);
        return;
    }
    uint8_t *tag_weights = NULL;
#endif

    char **names = (*m)->names;
    int model_count = get_array_size((void **)names);
    fprintf(stderr, "Model count: %d\n", model_count);
    if (model_count != count_tags) {
        free(tag);
        free(key);
        free(iv);
        free(aad);
        free(params);
        return;
    }

    void **inference_models_ptr = initialize_inference_models_ptr(model_count + 1);
    int initial_length = 10;
    operator_io **io = init_operator_io(initial_length);
    assert(io);
    operator_node *previous = NULL, *curr_node = NULL, *head = NULL;
    char *model_path = NULL;

    int is_llm = (strstr(names[0], "model.onnx_data") != NULL) || 
                 (strstr(names[0], "albert") != NULL) || 
                 (strstr(names[0], "gpt") != NULL) ||
                 (strstr(names[0], "pythia") != NULL) || 
                 (strstr(names[0], "llama") != NULL) || 
                 (strstr(names[0], "qwen") != NULL) || 
                 (strstr(names[0], "mistral") != NULL)
                 ? 1 : 0;
    ModelType model_type = is_llm ? MODEL_TYPE_LLM : MODEL_TYPE_CNN;

    for (int i = 1; i < model_count + 1; i++) {
        model_path = names[i-1];

        tag = (uint8_t *)malloc(TAG_BYTES * 2 + 1);
        if (!tag) {
            fprintf(stderr, "Memory allocation for tag failed\n");
            free(key);
            free(iv);
            free(aad);
            free(params);
            if (params_weights) free(params_weights);
            return;
        }

        switch (model_type) {
            case MODEL_TYPE_CNN: {
                memcpy(tag, tags[i-1], TAG_BYTES * 2);
                tag[TAG_BYTES * 2] = '\0';
                params->tag = tag;
                inference_models_ptr[i] = onnx_model_for_path(model_path, &inference_models_ptr[i], model_type, params, NULL);
                if (!inference_models_ptr[i]) {
                    free(tag);
                    free(key);
                    free(iv);
                    free(aad);
                    free(params);
                    if (params_weights) free(params_weights);
                    return;
                }
                break;
            }
            case MODEL_TYPE_LLM: {
                if (strstr(model_path, "model.onnx_data") != NULL) {
                    inference_models_ptr[i] = NULL;
                    tag_weights = (uint8_t *)malloc(TAG_BYTES * 2 + 1);
                    if (!tag_weights) {
                        fprintf(stderr, "Memory allocation for tag failed\n");
                        free(tag);
                        free(key);
                        free(iv);
                        free(aad);
                        free(params);
                        if (params_weights) free(params_weights);
                        return;
                    }
                    memcpy(tag_weights, tags[i-1], TAG_BYTES * 2);
                    tag_weights[TAG_BYTES * 2] = '\0';
                    params_weights->tag = tag_weights;
                } else {
                    memcpy(tag, tags[i-1], TAG_BYTES * 2);
                    tag[TAG_BYTES * 2] = '\0';
                    params->tag = tag;
                    params_weights->key = key;
                    params_weights->iv = iv;
                    params_weights->aad = aad;
                    if (strstr(model_path, "albert") != NULL ||
                        strstr(model_path, "gpt") != NULL ||
                        strstr(model_path, "pythia") != NULL ||
                        strstr(model_path, "llama") != NULL ||
                        strstr(model_path, "mistral") != NULL
                    ) params_weights->tag = tag;

                    inference_models_ptr[i] = onnx_model_for_path(model_path, &inference_models_ptr[i], model_type, params, params_weights);
                    if (!inference_models_ptr[i]) {
                        free(tag);
                        free(key);
                        free(iv);
                        free(aad);
                        free(params);
                        if (params_weights) free(params_weights);
                        return;
                    }
                }
                break;
            }
            default:
                fprintf(stderr, "Error: Unknown model type when load_model_to_memory()\n");
                return;
        }

        if (i == initial_length) {
            resize_operators_io(&io, initial_length + 5, initial_length);
            assert(io);
            initial_length += 5;
        }

        curr_node = create_operator_node(model_path, i+1);
        switch (model_type) {
            case MODEL_TYPE_CNN: {
                curr_node->run_inference = run_inference_cnn;
                break;
            }
            case MODEL_TYPE_LLM: {
                curr_node->run_inference = run_inference_llm;
                break;
            }
            default:
                fprintf(stderr, "Error: Unknown model type in create_operator_node().\n");
                return;
        }
        if (previous) {
            insert_child_to_operator_node(previous, curr_node);
        } else {
            head = create_operator_node("input", i);
            insert_child_to_operator_node(head, curr_node);
        }
        previous = curr_node;


        onnx_model_inputs(io, inference_models_ptr[i], model_type, i, head, model_path);
        free(tag);
    }

    free(key);
    free(iv);
    free(aad);
    free(params);
#if NUM_TOKENS != 0
    free(tag_weights);
    free(params_weights);
#endif

    free_operator_io(io);
    (*m)->head = head;

    #ifdef USE_MEMORY_ONLY
        (*m)->inference_models_ptr = inference_models_ptr;
    #else
        free_inference_models_ptr(inference_models_ptr, model_count + 1, model_type);
    #endif
}

#else

static void
onnx_model_inputs(operator_io **io, void *inference_model_ptr, ModelType model_type, int index, operator_node *head, char *model_name)
{
    uintptr_t num_inputs = 0;
    uintptr_t num_outputs = 0;
    char *input_name = NULL;
    char **input_names = NULL;
    char **output_names = NULL;
    int8_t *output_name = NULL;

    if (inference_model_ptr) {
        switch (model_type) {
            case MODEL_TYPE_CNN: {
                TractInferenceModel *cnn_inference_model = (TractInferenceModel *)inference_model_ptr;
                check(tract_inference_model_input_count(cnn_inference_model, &num_inputs));
                input_names = malloc((num_inputs + 1) * sizeof(char *));
                if (!input_names) return;
                for (int i = 0; i < (int)num_inputs; i++) {
                    check(tract_inference_model_input_name(cnn_inference_model, i, &input_name));
                    input_names[i] = input_name;
                }
                input_names[num_inputs] = NULL;

                check(tract_inference_model_output_count(cnn_inference_model, &num_outputs));
                output_names = malloc((num_outputs + 1) * sizeof(char *));
                if (!output_names) return;
                for (int i = 0; i < (int)num_outputs; i++) {
                    check(tract_inference_model_output_name(cnn_inference_model, i, &output_name));
                    output_names[i] = (char *)output_name;
                }
                output_names[num_outputs] = NULL;
                break;
            }
            case MODEL_TYPE_LLM: {
                if (!inference_model_ptr) break; 
                TractLlmInferenceModel *llm_inference_model = (TractLlmInferenceModel *)inference_model_ptr;
                check(tract_llm_inference_model_input_count(llm_inference_model, &num_inputs));
                input_names = malloc((num_inputs + 1) * sizeof(char *));
                if (!input_names) return;
                for (int i = 0; i < (int)num_inputs; i++) {
                    check(tract_llm_inference_model_input_name(llm_inference_model, i, &input_name));
                    input_names[i] = input_name;
                }
                input_names[num_inputs] = NULL;
                
                check(tract_llm_inference_model_output_count(llm_inference_model, &num_outputs));
                output_names = malloc((num_outputs + 1) * sizeof(char *));
                if (!output_names) return;
                for (int i = 0; i < (int)num_outputs; i++) {
                    check(tract_llm_inference_model_output_name(llm_inference_model, i, &output_name));
                    output_names[i] = (char *)output_name;
                }
                output_names[num_outputs] = NULL;
                break;
            }
            default:
                fprintf(stderr, "Error: Unknown model type.\n");
                return;
        }
    }

    if (index == 1) {
        operator_io o_io_first;
        o_io_first.input_names_length = 0;
        o_io_first.input_names = NULL;

        #if NUM_TOKENS != 0
            int number_inputs_llm = 3;
            if (strstr(model_name, "gpt2") != NULL) {
                number_inputs_llm = 2;
            }
            char **input_names_llm = malloc((number_inputs_llm + 1) * sizeof(char *));
            if (!input_names_llm) return;
            input_names_llm[0] = strdup("input_ids");
            if (number_inputs_llm == 3) {
                if (strstr(model_name, "albert") != NULL) {
                    input_names_llm[2] = strdup("token_type_ids");
                } else {
                    input_names_llm[2] = strdup("position_ids");
                }
                input_names_llm[1] = strdup("attention_mask");
            } else if (number_inputs_llm == 2) {
                input_names_llm[1] = strdup("attention_mask");
            }
            input_names_llm[number_inputs_llm] = NULL;
            o_io_first.output_names_length = number_inputs_llm;
            o_io_first.output_names = input_names_llm;
        #else
            o_io_first.output_names_length = num_inputs;
            o_io_first.output_names = input_names;
        #endif

        insert_into_operator_io(&io, &o_io_first, index - 1, "input");
        update_node(io, index - 1, NULL);

        #if NUM_TOKENS != 0
            for (int i = 0; i < number_inputs_llm; i++) {
                free(input_names_llm[i]);
            }
            free(input_names_llm);
        #endif
    }

    operator_io o_io;
    operator_node *head2 = NULL;
    o_io.input_names_length = num_inputs;
    if (num_inputs == 0) {
        head2 = head;
        o_io.input_names = NULL;
        head = NULL;
    } else {
        o_io.input_names = input_names;
    }
    o_io.output_names_length = num_outputs;
    o_io.output_names = output_names;
    insert_into_operator_io(&io, &o_io, index, model_name);

    if (num_inputs == 0) {
        operator_node *child = search_operator_node_by_name(head2, io[index]->model_name);
        if (!child) return;
        child->num_inputs = io[index]->input_names_length;
        child->num_outputs = io[index]->output_names_length;
    }

    for (int i=0; i < (int)num_inputs; i++) {
        tract_free_cstring(input_names[i]);
    }
    free(input_names);

    for (int i=0; i < (int)num_outputs; i++) {
        tract_free_cstring(output_names[i]);
    }
    free(output_names);

    update_node(io, index, head);
    // print_operator_io(io);
}

void *
onnx_model_for_path(char *model_name, void **inference_model, ModelType model_type)
{
    switch (model_type) {
        case MODEL_TYPE_CNN: {
            // Initialize onnx parser
            TractOnnx *onnx = NULL;
            check_ret(tract_onnx_create(&onnx), NULL);
            assert(onnx);

            // Load the model
            TractInferenceModel *cnn_inference_model = NULL;
            if (tract_onnx_model_for_path_cnn(onnx, model_name, &cnn_inference_model) != TRACT_RESULT_OK) {
                fprintf(stderr, "Error calling tract: %s", tract_get_last_error());
                check_ret(tract_onnx_destroy(&onnx), NULL);
                check_ret(tract_cnn_inference_model_release(&cnn_inference_model), NULL);
                assert(!cnn_inference_model);
                assert(!onnx);
                return NULL;
            }
            assert(cnn_inference_model);
            *inference_model = (void*)cnn_inference_model;
            assert(onnx);

            check_ret(tract_onnx_destroy(&onnx), NULL);
            assert(!onnx);
    
            break;
        }
        case MODEL_TYPE_LLM: {
            // Load the model
            TractLlmInferenceModel *llm_inference_model = NULL;
            if (tract_onnx_model_for_path_llm(model_name, &llm_inference_model) != TRACT_RESULT_OK) {
                fprintf(stderr, "Error calling tract: %s", tract_get_last_error());
                check_ret(tract_llm_inference_model_release(&llm_inference_model), NULL);
                assert(!llm_inference_model);
                return NULL;
            }
            assert(llm_inference_model);
            *inference_model = (void*)llm_inference_model;
            break;
        }
        default:
            fprintf(stderr, "Error: Unknown model type in onnx_model_for_path.\n");
            return NULL;
    }  

    return *inference_model;
}

void
load_model_to_memory(model **m)
{
    if (!m) return;

    char **names = (*m)->names;
    int model_count = get_array_size((void **)names);
    fprintf(stderr, "Model count: %d\n", model_count);

    void **inference_models_ptr = initialize_inference_models_ptr(model_count + 1);
    int initial_length = 10;
    operator_io **io = init_operator_io(initial_length);
    assert(io);
    operator_node *previous = NULL, *curr_node = NULL, *head = NULL;
    char *model_path = NULL;

    int is_llm = (strstr(names[0], "model.onnx_data") != NULL) || 
                 (strstr(names[0], "albert") != NULL) || 
                 (strstr(names[0], "gpt") != NULL) || 
                 (strstr(names[0], "pythia") != NULL) || 
                 (strstr(names[0], "llama") != NULL) || 
                 (strstr(names[0], "qwen") != NULL) || 
                 (strstr(names[0], "mistral") != NULL)
                 ? 1 : 0;
    ModelType model_type = is_llm ? MODEL_TYPE_LLM : MODEL_TYPE_CNN;

    for (int i = 1; i < model_count + 1; i++) {
        model_path = names[i-1];
        switch (model_type) {
            case MODEL_TYPE_CNN: {
                inference_models_ptr[i] = onnx_model_for_path(model_path, &inference_models_ptr[i], model_type);
                if (!inference_models_ptr[i]) {
                    return;
                }
                break;
            }
            case MODEL_TYPE_LLM: {
                if (strstr(model_path, "model.onnx_data") != NULL) {
                    inference_models_ptr[i] = NULL;
                } else {
                    inference_models_ptr[i] = onnx_model_for_path(model_path, &inference_models_ptr[i], model_type);
                    if (!inference_models_ptr[i]) {
                        return;
                    }
                }
                break;
            }
            default:
                fprintf(stderr, "Error: Unknown model type when load_model_to_memory()\n");
                return;
        }

        if (i == initial_length) {
            resize_operators_io(&io, initial_length + 5, initial_length);
            assert(io);
            initial_length += 5;
        }

        curr_node = create_operator_node(model_path, i+1);
        switch (model_type) {
            case MODEL_TYPE_CNN: {
                curr_node->run_inference = run_inference_cnn;
                break;
            }
            case MODEL_TYPE_LLM: {
                curr_node->run_inference = run_inference_llm;
                break;
            }
            default:
                fprintf(stderr, "Error: Unknown model type in create_operator_node().\n");
                return;
        }
        if (previous) {
            insert_child_to_operator_node(previous, curr_node);
        } else {
            head = create_operator_node("input", i);
            insert_child_to_operator_node(head, curr_node);
        }
        previous = curr_node;

        onnx_model_inputs(io, inference_models_ptr[i], model_type, i, head, model_path);
    }
    (*m)->head = head;

    free_operator_io(io);
    #if NUM_TOKENS != 0
        free_inference_models_ptr(inference_models_ptr, model_count + 1, model_type);
    #else
        free_inference_models_ptr(inference_models_ptr, model_count + 1, model_type);
    #endif
}
#endif

// INFERENCE
#if USE_AES == 0 && USE_MEMORY_ONLY == 0 || USE_AES == 1 && USE_MEMORY_ONLY == 1
void
run_inference_cnn(operator_node **node, input_info *input_info_ptr, void *inference_model_ptr)
{

#ifdef USE_SYS_TIME
    struct timeval t1_run, t2_run;
#endif
    double elapsed_time;

    TractModel *model = NULL;
    TractInferenceModel *inference_model = NULL;
    TractValue **input_values = (TractValue **)input_info_ptr->input_values;

#ifndef USE_MEMORY_ONLY
    // Initialize onnx parser
    TractOnnx *onnx = NULL;
    check(tract_onnx_create(&onnx));
    assert(onnx);

    // Load the model
    check(tract_onnx_model_for_path_cnn(onnx, (*node)->model_name, &inference_model));
    assert(inference_model);
    assert(onnx);

    check(tract_onnx_destroy(&onnx));
    assert(!onnx);

    // Transform an inference model into a typed model
    check(tract_inference_model_into_typed(&inference_model,&model));
    assert(model);

    free_inference_model_ptr((void *)inference_model, MODEL_TYPE_CNN);
#else
    inference_model = (TractInferenceModel *)inference_model_ptr; 

    // Transform an inference model into a typed model
    check(tract_inference_model_into_typed(&inference_model, &model));
    assert(model);
#endif

    // Make the model runnable
    TractRunnable *runnable = NULL;
    check(tract_model_into_runnable(&model, &runnable));
    assert(runnable);
    assert(!model);

    int argmax = 0;
    float max = 0.0, val = 0.0;
    int num_outputs = (*node)->num_outputs;
    TractValue **outputs = malloc((num_outputs + 1) * sizeof(TractValue *));
    const float *data = NULL;

#ifdef USE_SYS_TIME
    gettimeofday(&t1_run, NULL);
#endif

    int k = 0, index = 0;
    TractValue **inputs = malloc(((*node)->num_inputs + 1) * sizeof(TractValue *));
    int *indices = (*node)->parent_output_indices;
    for (int i = 0; i < (*node)->num_inputs; i++) {
        if (!(*node)->parents) break;
        if (strcmp((*node)->parents[i]->model_name, "input") == 0) {
            index++;
            int num_inputs = get_array_size(input_info_ptr->input_values);
            for (int j = 0; j < num_inputs; j++) {
                inputs[k++] = input_values[j];
            }
            continue;
        }
        if (!(*node)->parents[i]->outputs[indices[index]]) {
            fprintf(stderr, "The output is NULL!");
            continue;
        }
        inputs[k++] = (TractValue *)(*node)->parents[i]->outputs[indices[index++]];
    }
    if ((*node)->num_inputs == -1 || (*node)->num_parents == 0) {
        int num_inputs = get_array_size(input_info_ptr->input_values);
        inputs = realloc(inputs, (num_inputs + 1) * sizeof(TractValue *));
        assert(inputs);
        for (int j = 0; j < num_inputs; j++) {
            inputs[k++] = input_values[j];
        }
    }
    inputs[k] = NULL;
    check(tract_runnable_run(runnable, inputs, outputs));
    free(inputs);

    for (int i = 0; i < num_outputs; i++) {
        if (outputs[i] == NULL) {
            fprintf(stderr, "Output %d is NULL\n", i);
            continue;
        }
        
        check(tract_value_as_bytes(outputs[i], NULL, NULL, NULL, (const void**) &data));

        max = data[0];
        argmax = 0;
        for(int i = 0; i < 1000; i++) {
            val = data[i];
            if(val > max) {
                max = val;
                argmax = i;
            }
        }
        assert(data[argmax] == max);
        data = NULL;
    }

#ifdef USE_SYS_TIME
    gettimeofday(&t2_run, NULL);
    elapsed_time = (t2_run.tv_sec - t1_run.tv_sec) * 1000.0;      // sec to ms
    elapsed_time += (t2_run.tv_usec - t1_run.tv_usec) / 1000.0;   // us to ms
#else
    elapsed_time = 0.0;
#endif

    check(tract_runnable_release(&runnable));
    assert(!runnable);

    (*node)->outputs = (void **)malloc((num_outputs + 1) * sizeof(void *));
    for (int i = 0; i < num_outputs; i++) {
        if (outputs[i] == NULL) {
            fprintf(stderr, "Output %d is NULL\n", i);
            continue;
        }
        (*node)->outputs[i] = (void *)outputs[i];
    }
    (*node)->outputs[num_outputs] = NULL;
    free(outputs);
    (*node)->pred = max;
    (*node)->category = argmax;
    (*node)->elapsedTime = elapsed_time;
}

void
run_inference_llm(operator_node **node, input_info *input_info_ptr, void *inference_model_ptr)
{
    if (strstr((*node)->model_name, "model.onnx_data") != NULL) {
        (*node)->outputs = (void **)malloc((2) * sizeof(void *));
        memset((*node)->outputs, 0, 2 * sizeof(void *));
        return;
    }

#ifdef USE_SYS_TIME
    struct timeval t1_run, t2_run;
#endif
    double elapsed_time;
    TractLlmInferenceModel *inference_model = (TractLlmInferenceModel *)inference_model_ptr; 
    TractLlmTransformedModel *transformed_model = NULL;
    int num_inputs_ptr = get_array_size(input_info_ptr->input_values);

#ifndef USE_MEMORY_ONLY
    // Load the model
    inference_model = NULL;
    check(tract_onnx_model_for_path_llm((*node)->model_name, &inference_model));
    assert(inference_model);
#endif

    int num_outputs = (*node)->num_outputs;
    void **outputs = malloc((num_outputs + 1) * sizeof(void *));
    void **shapefacts = malloc((num_outputs + 1) * sizeof(void *));
    void **datum_types = malloc((num_outputs + 1) * sizeof(void *));

#ifdef USE_SYS_TIME
    gettimeofday(&t1_run, NULL);
#endif

    int k = 0, index = 0;
    void **inputs = malloc(((*node)->num_inputs + 1) * sizeof(void *));
    void **input_shapefacts = malloc(((*node)->num_inputs + 1) * sizeof(void *));
    void **input_datum_types = malloc(((*node)->num_inputs + 1) * sizeof(void *));

    int *indices = (*node)->parent_output_indices;
    for (int i = 0; i < (*node)->num_inputs; i++) {
        if (!(*node)->parents || k >= (*node)->num_inputs) break;
        if (strcmp((*node)->parents[i]->model_name, "input") == 0) {
            inputs[k] = input_info_ptr->input_values[index];
            input_shapefacts[k] = NULL;
            input_datum_types[k] = input_info_ptr->input_datum_types[index];
            index++;
            k++;
            continue;
        }
        if (!(*node)->parents[i]->outputs[indices[index]]) {
            fprintf(stderr, "The output is NULL!");
            continue;
        }
        inputs[k] = (*node)->parents[i]->outputs[indices[index]];
        input_shapefacts[k] = (*node)->parents[i]->shapefacts[indices[index]];
        input_datum_types[k] = (*node)->parents[i]->datum_types[indices[index]];
        index++;
        k++;
    }
    if ((*node)->num_inputs == -1 || (*node)->num_parents == 0) {
        inputs = realloc(inputs, (num_inputs_ptr + 1) * sizeof(void *));
        assert(inputs);
        for (int j = 0; j < num_inputs_ptr; j++) {
            inputs[k] = input_info_ptr->input_values[j];
            input_shapefacts[k] = NULL;
            input_datum_types[k] = input_info_ptr->input_datum_types[j];
            k++;
        }
    }
    inputs[k] = NULL;
    input_shapefacts[k] = NULL;

    check(tract_inference_model_into_optimized_llm(k, input_shapefacts, input_datum_types, &inference_model, &transformed_model));
    assert(transformed_model);

#ifndef USE_MEMORY_ONLY
    free_inference_model_ptr(inference_model, MODEL_TYPE_LLM);
#endif

    check(tract_model_into_runnable_and_run_llm(inputs, k, &transformed_model, outputs, shapefacts, datum_types));
    free(inputs);
    free(input_shapefacts);
    free(input_datum_types);

#ifdef USE_SYS_TIME
    gettimeofday(&t2_run, NULL);
    elapsed_time = (t2_run.tv_sec - t1_run.tv_sec) * 1000.0;      // sec to ms
    elapsed_time += (t2_run.tv_usec - t1_run.tv_usec) / 1000.0;   // us to ms
#else
    elapsed_time = 0.0;
#endif

    (*node)->outputs = (void **)malloc((num_outputs + 1) * sizeof(void *));
    (*node)->shapefacts = (void **)malloc((num_outputs + 1) * sizeof(void *));
    (*node)->datum_types = (void **)malloc((num_outputs + 1) * sizeof(void *));
    for (int i = 0; i < num_outputs; i++) {
        if (outputs[i] == NULL || shapefacts[i] == NULL || datum_types[i] == NULL) {
            fprintf(stderr, "Outputs or shapefacts or datum_types of %d are NULL\n", i);
            continue;
        }
        (*node)->outputs[i] = outputs[i];
        (*node)->shapefacts[i] = shapefacts[i];
        (*node)->datum_types[i] = datum_types[i];
    }
    (*node)->outputs[num_outputs] = NULL;
    (*node)->shapefacts[num_outputs] = NULL;
    (*node)->datum_types[num_outputs] = NULL;
    free(outputs);
    free(shapefacts);
    free(datum_types);
    (*node)->elapsedTime = elapsed_time;
}

operator_node *
execute_tree(operator_node *node, input_info *input_info_ptr, double *elapsed_time, void **inference_models_ptr, FILE *fd)
{
    if (!node) {
        return NULL;
    }

    if (node->is_visited == true) {
        return node;
    }

    node->is_visited = true;
    operator_node *last_processed_node = node;

    fprintf(stderr, "\n\nModel name: %s\n", node->model_name);

    if (node->node_id != 1) {
#ifdef USE_DEBUG
    #ifndef USE_MEMORY_ONLY
        node->run_inference(&node, input_info_ptr, NULL);
    #else 
        node->run_inference(&node, input_info_ptr, inference_models_ptr[node->node_id - 1]);
    #endif
    #ifdef USE_SYS_TIME
        if (fd) {
            fprintf(fd, "Partition_%d: %f ms\n", node->node_id - 1, node->elapsedTime);
        } else {
            fprintf(stderr, "Partition_%d: %f ms\n", node->node_id - 1, node->elapsedTime);
        }
    #endif
#else
    if (fd) {
        node->run_inference(&node, input_info_ptr, NULL);
        #ifdef USE_SYS_TIME
            fprintf(fd, "Partition_%d: %f ms\n", node->node_id - 1, node->elapsedTime);
        #endif
    } else {
        assert(inference_models_ptr);
        node->run_inference(&node, input_info_ptr, inference_models_ptr[node->node_id - 1]);
        fprintf(stderr, "Partition_%d: %f ms\n", node->node_id - 1, node->elapsedTime);
    }
#endif
        
        *elapsed_time += node->elapsedTime;
    }

    for (int i = 0; i < node->num_children; i++) {
        operator_node *child_node = execute_tree(node->children[i], input_info_ptr, elapsed_time, inference_models_ptr, fd);
        if (child_node) last_processed_node = child_node;
    }

    return last_processed_node;
}
#endif

#ifndef USE_AES
char *
inference_no_aes(float **images, int num_images, uint8_t **tokenizer, int tokenizer_size, model *m)
{
    struct timeval t1_inf, t2_inf;
    double elapsed_time;

    char *error = NULL;
    if (!m) {
        error = (char *) malloc(SMALL_SIZE * sizeof(char));
        if (!error) {
            fprintf(stderr, "Error allocating memory for error\n");
            return NULL;
        }
        snprintf(error, SMALL_SIZE, "No model found with the given id");
        error[SMALL_SIZE - 1] = '\0';
        return error;
    }

    input_info *input_info_ptr = malloc(sizeof(input_info));
    input_info_ptr->input_values = NULL;
    input_info_ptr->input_shapefacts = NULL;
    input_info_ptr->input_datum_types = NULL;
    void *tokenizer_ptr = NULL;
    char *model_name = m->names[0];

    int model_count = get_array_size((void **)m->names);
    fprintf(stderr, "Model count: %d\n", model_count);
    int number_inputs_llm = 3; 

    if (!images && tokenizer_size > 0) {
        check_ret(tract_create_tokenizer(*tokenizer, tokenizer_size, &tokenizer_ptr), NULL);
        free(*tokenizer);

        for (int i = 0; i < model_count; ++i) {
            if (strstr(m->names[i], "model.onnx_data") == NULL) {
                model_name = m->names[i];
                break;
            }
        }

        if (strstr(model_name, "gpt2") != NULL) {
            number_inputs_llm = 2;
        }

        input_info_ptr->input_values = malloc((number_inputs_llm + 1) * sizeof(void *));
        input_info_ptr->input_shapefacts = malloc((number_inputs_llm + 1) * sizeof(void *));
        input_info_ptr->input_datum_types = malloc((number_inputs_llm + 1) * sizeof(void *));
        memset(input_info_ptr->input_shapefacts, 0, (number_inputs_llm + 1) * sizeof(void *));

        char *prompt = "Hi, how are you today?";
        check_ret(tract_value_from_bytes_llm(tokenizer_ptr, prompt, input_info_ptr->input_values, input_info_ptr->input_datum_types, number_inputs_llm), NULL);
        input_info_ptr->input_values[number_inputs_llm] = NULL;
        input_info_ptr->input_datum_types[number_inputs_llm] = NULL;
    } else {
        assert(images);

        input_info_ptr->input_values = malloc((num_images + 1) * sizeof(void *));

        int size, flag;
        for (int i = 0; i < num_images; i++) {
            size_t shape[4] = {(int)images[i][0], (int)images[i][1], (int)images[i][2], (int)images[i][3]};

            fprintf(stderr, "Image shape[%d]: %zu, %zu, %zu, %zu\n", i, shape[0], shape[1], shape[2], shape[3]);

            flag = 0;
            size = 1;
            for (int j = 0; j < 4; j++) {
                if (shape[j] != 0) {
                    flag++;
                    size *= shape[j];
                }
            }

            float *temp_image = (float *) malloc(size * sizeof(float));
            if (!temp_image) {
                fprintf(stderr, "Error allocating memory for temp_image\n");
                return NULL;
            }
            memcpy(temp_image, images[i] + flag, size * sizeof(float));

            TractValue *input_value = NULL;
            check_ret(tract_value_from_bytes(TRACT_DATUM_TYPE_F32, flag, shape, temp_image, &input_value), NULL);
            free(temp_image);

            input_info_ptr->input_values[i] = input_value;
        }
        input_info_ptr->input_values[num_images] = NULL;
    }

    FILE *fd = NULL;
#ifndef USE_DEBUG
    char *file_path = NULL;
    #ifdef USE_FILE_CACHING
        file_path = "../inference_time_outside_occlum_memory_only_no_aes.txt";
    #else
        file_path = "../inference_time_outside_occlum_on_disk_no_aes.txt";
    #endif
    fd = fopen(file_path, "a");
    if (!fd) {
        fprintf(stderr, "Error opening inference_time_outside_occlum_no_aes!\n");
        return NULL;
    }
#endif    

    double sum = 0.0;
    m->head->outputs = input_info_ptr->input_values;
    int runs = (NUM_TOKENS == 0) ? 1 : NUM_TOKENS;
    operator_node *last_node = NULL;
#if NUM_TOKENS != 0    
    char *generated_text;
#endif

    gettimeofday(&t1_inf, NULL);
    for (int i = 0; i < runs; ++i) {
#if NUM_TOKENS != 0
        generated_text = NULL;
        uintptr_t next_token_id = 0;
#endif
        last_node = execute_tree(m->head, input_info_ptr, &sum, m->inference_models_ptr, fd);
        reset_node_visibility(m->head);

    #if NUM_TOKENS != 0
        check_ret(tract_generate_text_llm(input_info_ptr->input_values, number_inputs_llm, tokenizer_ptr, last_node->outputs, last_node->num_outputs, &generated_text, &next_token_id), NULL);
        if (strstr(model_name, "albert") == NULL) {
            check_ret(tract_update_input_values_llm(input_info_ptr->input_values, number_inputs_llm, next_token_id), NULL);
        }

        free_operator_node_info(m->head);
        reset_node_visibility(m->head);

        if (i != (runs - 1)) tract_free_cstring(generated_text);
    #endif
    }
    #if NUM_TOKENS != 0
        free(input_info_ptr->input_shapefacts);
        free(input_info_ptr->input_datum_types);
    #endif
    free(input_info_ptr);
    
    gettimeofday(&t2_inf, NULL); 
    elapsed_time = (t2_inf.tv_sec - t1_inf.tv_sec) * 1000.0;      // sec to ms
    elapsed_time += (t2_inf.tv_usec - t1_inf.tv_usec) / 1000.0;   // us to ms

#ifndef USE_DEBUG
    #ifdef USE_SYS_TIME
        if (fprintf(fd, "Inference time: %f ms\n", elapsed_time) < 0) {
            fprintf(stderr, "Error writing to file inference_time_outside_occlum_no_aes.txt\n");
            fclose(fd);
            return NULL;
        }
        if (fprintf(fd, "Inference time to run a model: %f ms\n", sum) < 0) {
            fprintf(stderr, "Error writing to file inference_time_outside_occlum_no_aes.txt\n");
            fclose(fd);
            return NULL;
        }
    #endif
        fclose(fd);
#else
    fprintf(stderr, "Inference time: %f ms\n", elapsed_time);
    fprintf(stderr, "Inference time to run a model: %f ms\n", sum);
#endif

    char *prediction = (char *) malloc(SMALL_SIZE * sizeof(char));
    if (!prediction) {
        fprintf(stderr, "Error allocating memory for result\n");
        return NULL;
    }

    #if NUM_TOKENS != 0
        check_ret(tract_free_tokenizer(&tokenizer_ptr), NULL);
        snprintf(prediction, SMALL_SIZE, "Model %s, %s!", m->names[model_count-1], generated_text);
        tract_free_cstring(generated_text);
    #else
        snprintf(prediction, SMALL_SIZE, "Model %s, Inference: Max is %f for category %d!", m->names[model_count-1], last_node->pred, last_node->category);
    #endif
    prediction[SMALL_SIZE - 1] = '\0';
    
    return prediction;
}
#else

#ifdef USE_MEMORY_ONLY
char *
inference_memory_only(float **images, int num_images, uint8_t **tokenizer, int tokenizer_size, model *m)
{

#ifdef USE_SYS_TIME
    struct timeval t1_inf, t2_inf;
#endif
    double elapsed_time;

    char *error = NULL;
    if (!m) {
        error = (char *) malloc(SMALL_SIZE * sizeof(char));
        if (!error) {
            fprintf(stderr, "Error allocating memory for error\n");
            return NULL;
        }
        snprintf(error, SMALL_SIZE, "No model found with the given id");
        error[SMALL_SIZE - 1] = '\0';
        return error;
    } else if (!m->inference_models_ptr) {
        error = (char *) malloc(SMALL_SIZE * sizeof(char));
        if (!error) {
            fprintf(stderr, "Error allocating memory for error\n");
            return NULL;
        }
        snprintf(error, SMALL_SIZE, "No inference model found with the given id");
        error[SMALL_SIZE - 1] = '\0';
        return error;
    }

    input_info *input_info_ptr = malloc(sizeof(input_info));
    input_info_ptr->input_values = NULL;
    input_info_ptr->input_shapefacts = NULL;
    input_info_ptr->input_datum_types = NULL;
    void *tokenizer_ptr = NULL;
    char *model_name = m->names[0];

    int model_count = get_array_size((void **)m->names);
    fprintf(stderr, "Model count: %d\n", model_count);
    int number_inputs_llm = 3;

    if (!images && tokenizer_size > 0) {
        check_ret(tract_create_tokenizer(*tokenizer, tokenizer_size, &tokenizer_ptr), NULL);
        free(*tokenizer);

        for (int i = 0; i < model_count; ++i) {
            if (strstr(m->names[i], "model.onnx_data") == NULL) {
                model_name = m->names[i];
                break;
            }
        }

        if (strstr(model_name, "gpt2") != NULL) {
            number_inputs_llm = 2;
        }

        input_info_ptr->input_values = malloc((number_inputs_llm + 1) * sizeof(void *));
        input_info_ptr->input_shapefacts = malloc((number_inputs_llm + 1) * sizeof(void *));
        input_info_ptr->input_datum_types = malloc((number_inputs_llm + 1) * sizeof(void *));
        memset(input_info_ptr->input_shapefacts, 0, (number_inputs_llm + 1) * sizeof(void *));

        char *prompt = "Hi, how are you today?";
        check_ret(tract_value_from_bytes_llm(tokenizer_ptr, prompt, input_info_ptr->input_values, input_info_ptr->input_datum_types, number_inputs_llm), NULL);
        input_info_ptr->input_values[number_inputs_llm] = NULL;
        input_info_ptr->input_datum_types[number_inputs_llm] = NULL;
    } else {
        assert(images);

        input_info_ptr->input_values = malloc((num_images + 1) * sizeof(void *));

        int size, flag;
        for (int i = 0; i < num_images; i++) {
            size_t shape[4] = {(int)images[i][0], (int)images[i][1], (int)images[i][2], (int)images[i][3]};

            fprintf(stderr, "Image shape[%d]: %zu, %zu, %zu, %zu\n", i, shape[0], shape[1], shape[2], shape[3]);

            flag = 0;
            size = 1;
            for (int j = 0; j < 4; j++) {
                if (shape[j] != 0) {
                    flag++;
                    size *= shape[j];
                }
            }

            float *temp_image = (float *) malloc(size * sizeof(float));
            if (!temp_image) {
                fprintf(stderr, "Error allocating memory for temp_image\n");
                return NULL;
            }
            memcpy(temp_image, images[i] + flag, size * sizeof(float));

            TractValue *input_value = NULL;
            check_ret(tract_value_from_bytes(TRACT_DATUM_TYPE_F32, flag, shape, temp_image, &input_value), NULL);
            free(temp_image);

            input_info_ptr->input_values[i] = input_value;
        }
        input_info_ptr->input_values[num_images] = NULL;
    }

    double sum = 0.0;
    m->head->outputs = input_info_ptr->input_values;
    int runs = (NUM_TOKENS == 0) ? 1 : NUM_TOKENS;
    operator_node *last_node = NULL;
#if NUM_TOKENS != 0    
    char *generated_text;
#endif

#ifdef USE_SYS_TIME
    gettimeofday(&t1_inf, NULL);
#endif
    for (int i = 0; i < runs; ++i) {
#if NUM_TOKENS != 0
        generated_text = NULL;
        uintptr_t next_token_id = 0;
#endif
        last_node = execute_tree(m->head, input_info_ptr, &sum, m->inference_models_ptr, NULL);
        reset_node_visibility(m->head);

    #if NUM_TOKENS != 0
        check_ret(tract_generate_text_llm(input_info_ptr->input_values, number_inputs_llm, tokenizer_ptr, last_node->outputs, last_node->num_outputs, &generated_text, &next_token_id), NULL);
        if (strstr(model_name, "albert") == NULL) {
            check_ret(tract_update_input_values_llm(input_info_ptr->input_values, number_inputs_llm, next_token_id), NULL);
        }

        free_operator_node_info(m->head);
        reset_node_visibility(m->head);

        if (i != (runs - 1)) tract_free_cstring(generated_text);
    #endif
    }
    #if NUM_TOKENS != 0
        free(input_info_ptr->input_shapefacts);
        free(input_info_ptr->input_datum_types);
    #endif
    free(input_info_ptr);

#ifdef USE_SYS_TIME
    gettimeofday(&t2_inf, NULL);
    elapsed_time = (t2_inf.tv_sec - t1_inf.tv_sec) * 1000.0;      // sec to ms
    elapsed_time += (t2_inf.tv_usec - t1_inf.tv_usec) / 1000.0;   // us to ms
#else
    elapsed_time = 0.0;
#endif

    fprintf(stderr, "Inference time: %f ms\n", elapsed_time);
    fprintf(stderr, "Inference time to run a model: %f ms\n", sum);

    char *prediction = (char *) malloc(SMALL_SIZE * sizeof(char));
    if (!prediction) {
        fprintf(stderr, "Error allocating memory for result\n");
        return NULL;
    }

    #if NUM_TOKENS != 0
        check_ret(tract_free_tokenizer(&tokenizer_ptr), NULL);
        snprintf(prediction, SMALL_SIZE, "Model %s, %s!", m->names[model_count-1], generated_text);
        tract_free_cstring(generated_text);
    #else
        snprintf(prediction, SMALL_SIZE, "Model %s, Inference: Max is %f for category %d!", m->names[model_count-1], last_node->pred, last_node->category);
    #endif
    prediction[SMALL_SIZE - 1] = '\0';
    
    return prediction;
}
#else

void
run_inference_cnn(operator_node **node, input_info *input_info_ptr, struct EncryptionParameters *params, struct EncryptionParameters *params_weights)
{
    assert(params);
    assert(!params_weights);

#ifdef USE_SYS_TIME
    struct timeval t1_run, t2_run;
#endif
    double elapsed_time;

    TractModel *model = NULL;
    TractInferenceModel *inference_model = NULL;
    TractValue **input_values = (TractValue **)input_info_ptr->input_values;

    // Initialize onnx parser
    TractOnnx *onnx = NULL;
    check(tract_onnx_create(&onnx));
    assert(onnx);

    // Load the model
    if (tract_onnx_model_for_path_cnn(onnx, (*node)->model_name, &inference_model, params) != TRACT_RESULT_OK) {
        fprintf(stderr, "Error calling tract: %s", tract_get_last_error());
        (*node)->outputs = NULL;
        check(tract_onnx_destroy(&onnx));
        assert(!onnx);
        return;
    }
    assert(inference_model);
    assert(onnx);

    check(tract_onnx_destroy(&onnx));
    assert(!onnx);

    // Transform an inference model into a typed model
    check(tract_inference_model_into_typed(&inference_model,&model));
    assert(model);

    free_inference_model_ptr((void *)inference_model, MODEL_TYPE_CNN);

    // Make the model runnable
    TractRunnable *runnable = NULL;
    check(tract_model_into_runnable(&model, &runnable));
    assert(runnable);
    assert(!model);

    int argmax = 0;
    float max = 0.0, val = 0.0;
    int num_outputs = (*node)->num_outputs;
    TractValue **outputs = malloc((num_outputs + 1) * sizeof(TractValue *));
    const float *data = NULL;

#ifdef USE_SYS_TIME
    gettimeofday(&t1_run, NULL);
#endif

    int k = 0, index = 0;
    TractValue **inputs = malloc(((*node)->num_inputs + 1) * sizeof(TractValue *));
    int *indices = (*node)->parent_output_indices;
    for (int i = 0; i < (*node)->num_inputs; i++) {
        if (!(*node)->parents) break;
        if (strcmp((*node)->parents[i]->model_name, "input") == 0) {
            index++;
            int num_inputs = get_array_size(input_info_ptr->input_values);
            for (int j = 0; j < num_inputs; j++) {
                inputs[k++] = input_values[j];
            }
            continue;
        }
        if (!(*node)->parents[i]->outputs[indices[index]]) {
            fprintf(stderr, "The output is NULL!");
            continue;
        }
        inputs[k++] = (TractValue *)(*node)->parents[i]->outputs[indices[index++]];
    }
    if ((*node)->num_inputs == -1 || (*node)->num_parents == 0) {
        int num_inputs = get_array_size((void **)input_values);
        inputs = realloc(inputs, (num_inputs + 1) * sizeof(TractValue *));
        assert(inputs);
        for (int j = 0; j < num_inputs; j++) {
            inputs[k++] = input_values[j];
        }
    }
    inputs[k] = NULL;
    check(tract_runnable_run(runnable, inputs, outputs));
    free(inputs);

    for (int i = 0; i < num_outputs; i++) {
        if (outputs[i] == NULL) {
            fprintf(stderr, "Output %d is NULL\n", i);
            continue;
        }
        
        check(tract_value_as_bytes(outputs[i], NULL, NULL, NULL, (const void**) &data));

        max = data[0];
        argmax = 0;
        for(int i = 0; i < 1000; i++) {
            val = data[i];
            if(val > max) {
                max = val;
                argmax = i;
            }
        }
        assert(data[argmax] == max);
        data = NULL;
    }

#ifdef USE_SYS_TIME
    gettimeofday(&t2_run, NULL);
    elapsed_time = (t2_run.tv_sec - t1_run.tv_sec) * 1000.0;      // sec to ms
    elapsed_time += (t2_run.tv_usec - t1_run.tv_usec) / 1000.0;   // us to ms
#else
    elapsed_time = 0.0;
#endif

    check(tract_runnable_release(&runnable));
    assert(!runnable);

    (*node)->outputs = (void **)malloc((num_outputs + 1) * sizeof(void *));
    for (int i = 0; i < num_outputs; i++) {
        if (outputs[i] == NULL) {
            fprintf(stderr, "Output %d is NULL\n", i);
            continue;
        }
        (*node)->outputs[i] = (void *)outputs[i];
    }
    (*node)->outputs[num_outputs] = NULL;
    free(outputs);
    (*node)->pred = max;
    (*node)->category = argmax;
    (*node)->elapsedTime = elapsed_time;
}

void
run_inference_llm(operator_node **node, input_info *input_info_ptr, struct EncryptionParameters *params, struct EncryptionParameters *params_weights)
{
    assert(params);
    assert(params_weights);

    if (strstr((*node)->model_name, "model.onnx_data") != NULL) {
        (*node)->outputs = (void **)malloc((2) * sizeof(void *));
        memset((*node)->outputs, 0, 2 * sizeof(void *));
        return;
    }

#ifdef USE_SYS_TIME
    struct timeval t1_run, t2_run;
#endif
    double elapsed_time;
    //TractLlmTransformedModel *transformed_model = NULL;
    int num_inputs_ptr = get_array_size(input_info_ptr->input_values);

// #ifdef USE_SYS_TIME
//     gettimeofday(&t1_load, NULL);
// #endif
    // Load the model
    // TractLlmInferenceModel *inference_model = NULL;
    // check(tract_onnx_model_for_path_llm((*node)->model_name, params, params_weights, &inference_model));
    // assert(inference_model);
// #ifdef USE_SYS_TIME
//     gettimeofday(&t2_load, NULL);
//     elapsed_time = (t2_load.tv_sec - t1_load.tv_sec) * 1000.0;      // sec to ms
//     elapsed_time += (t2_load.tv_usec - t1_load.tv_usec) / 1000.0;   // us to ms
//     fprintf(stderr, "Inference time to load the model/partition: %f ms\n", elapsed_time);
//     elapsed_time = 0.0;
// #endif

    int num_outputs = (*node)->num_outputs;
    void **outputs = malloc((num_outputs + 1) * sizeof(void *));
    void **output_shapefacts = malloc((num_outputs + 1) * sizeof(void *));
    void **output_datum_types = malloc((num_outputs + 1) * sizeof(void *));

#ifdef USE_SYS_TIME
    gettimeofday(&t1_run, NULL);
#endif

    int k = 0, index = 0;
    void **inputs = malloc(((*node)->num_inputs + 1) * sizeof(void *));
    void **input_shapefacts = malloc(((*node)->num_inputs + 1) * sizeof(void *));
    void **input_datum_types = malloc(((*node)->num_inputs + 1) * sizeof(void *));

    int *indices = (*node)->parent_output_indices;
    for (int i = 0; i < (*node)->num_inputs; i++) {
        if (!(*node)->parents || k >= (*node)->num_inputs) break;
        if (strcmp((*node)->parents[i]->model_name, "input") == 0) {
            inputs[k] = input_info_ptr->input_values[index];
            input_shapefacts[k] = NULL;
            input_datum_types[k] = input_info_ptr->input_datum_types[index];
            index++;
            k++;
            continue;
        }
        if (!(*node)->parents[i]->outputs[indices[index]]) {
            fprintf(stderr, "The output is NULL!");
            continue;
        }
        inputs[k] = (*node)->parents[i]->outputs[indices[index]];
        input_shapefacts[k] = (*node)->parents[i]->shapefacts[indices[index]];
        input_datum_types[k] = (*node)->parents[i]->datum_types[indices[index]];
        index++;
        k++;
    }
    if ((*node)->num_inputs == -1 || (*node)->num_parents == 0) {
        inputs = realloc(inputs, (num_inputs_ptr + 1) * sizeof(void *));
        assert(inputs);
        for (int j = 0; j < num_inputs_ptr; j++) {
            inputs[k] = input_info_ptr->input_values[j];
            input_shapefacts[k] = NULL;
            input_datum_types[k] = input_info_ptr->input_datum_types[j];
            k++;
        }
    }
    assert(input_datum_types);
    inputs[k] = NULL;
    input_shapefacts[k] = NULL;

    // check(tract_inference_model_into_optimized_llm(k, input_shapefacts, input_datum_types, &inference_model, &transformed_model));
    // assert(transformed_model);

    // assert(input_datum_types);

    // check(tract_model_into_runnable_and_run_llm(inputs, k, &transformed_model, outputs, shapefacts, datum_types));

    //check(tract_inference_model_into_optimized_and_run_llm(inputs, k, input_shapefacts, input_datum_types, &inference_model, outputs, output_shapefacts, output_datum_types));
    check(tract_model_for_path_into_optimized_and_run_llm((*node)->model_name, params, params_weights, inputs, k, input_shapefacts, input_datum_types, outputs, output_shapefacts, output_datum_types));
    free(inputs);
    free(input_shapefacts);
    free(input_datum_types);

#ifdef USE_SYS_TIME
    gettimeofday(&t2_run, NULL);
    elapsed_time = (t2_run.tv_sec - t1_run.tv_sec) * 1000.0;      // sec to ms
    elapsed_time += (t2_run.tv_usec - t1_run.tv_usec) / 1000.0;   // us to ms
#else
    elapsed_time = 0.0;
#endif

    (*node)->outputs = (void **)malloc((num_outputs + 1) * sizeof(void *));
    (*node)->shapefacts = (void **)malloc((num_outputs + 1) * sizeof(void *));
    (*node)->datum_types = (void **)malloc((num_outputs + 1) * sizeof(void *));
    for (int i = 0; i < num_outputs; i++) {
        if (outputs[i] == NULL || output_shapefacts[i] == NULL || output_datum_types[i] == NULL) {
            fprintf(stderr, "Outputs or output_shapefacts or output_datum_types of %d are NULL\n", i);
            continue;
        }
        (*node)->outputs[i] = outputs[i];
        (*node)->shapefacts[i] = output_shapefacts[i];
        (*node)->datum_types[i] = output_datum_types[i];
    }
    (*node)->outputs[num_outputs] = NULL;
    (*node)->shapefacts[num_outputs] = NULL;
    (*node)->datum_types[num_outputs] = NULL;
    free(outputs);
    free(output_shapefacts);
    free(output_datum_types);
    (*node)->elapsedTime = elapsed_time;
}

operator_node *
execute_tree(operator_node *node, input_info *input_info_ptr, double *elapsed_time, unsigned char **tags, struct EncryptionParameters *params, struct EncryptionParameters *params_weights)
{
    if (!node) {
        return NULL;
    }   

    if (node->is_visited == true) {
        return node;
    }

    node->is_visited = true;
    operator_node *last_processed_node = node;

    fprintf(stderr, "\n\nModel name: %s\n", node->model_name);
    unsigned char *tag2 = NULL;

    int i = node->node_id - 2;
    if (node->node_id != 1) {
        unsigned char *tag = (uint8_t *)malloc(TAG_BYTES * 2);
        assert(tag);
        if (strstr(node->model_name, "model.onnx_data") == NULL) {
            memcpy(tag, tags[i], TAG_BYTES * 2);
            params->tag = tag;
        } else {
            tag2 = (uint8_t *)malloc(TAG_BYTES * 2);
            memcpy(tag2, tags[i], TAG_BYTES * 2);
            params_weights->tag = tag2;
        }
        fprintf(stderr, "PARAMS tag: %s\n", params->tag);
#if NUM_TOKENS != 0
        fprintf(stderr, "PARAMS_WEIGHTS tag: %s\n", params_weights->tag);
        if (strstr(node->model_name, "albert") != NULL || 
            strstr(node->model_name, "gpt") != NULL ||
            strstr(node->model_name, "pythia") != NULL || 
            strstr(node->model_name, "llama") != NULL ||
            strstr(node->model_name, "mistral") != NULL) params_weights->tag = tag;
        fprintf(stderr, "PARAMS_WEIGHTS tag: %s\n", params_weights->tag);
#endif
        
        node->run_inference(&node, input_info_ptr, params, params_weights);
        fprintf(stderr, "Partition_%d: %f ms\n", i, node->elapsedTime);
        free(tag);

        if (!node->outputs) {
            return NULL;
        }
        
        *elapsed_time += node->elapsedTime;
    }

    for (int i = 0; i < node->num_children; i++) {
        operator_node *child_node = execute_tree(node->children[i], input_info_ptr, elapsed_time, tags, params, params_weights);
        if (child_node) last_processed_node = child_node;
    }

#if NUM_TOKENS != 0
    if (tag2) free(tag2);
#endif

    return last_processed_node;
}

char *
inference_aes(float **images, int num_images, uint8_t **tokenizer, int tokenizer_size, model *m, unsigned char **tags, int count_tags)
{
#ifdef USE_SYS_TIME
    struct timeval t1_inf, t2_inf;
#endif
    double elapsed_time;

    assert(tags);

    char *error = NULL;
    if (!m) {
        error = (char *) malloc(SMALL_SIZE * sizeof(char));
        if (!error) {
            fprintf(stderr, "Error allocating memory for error\n");
            return NULL;
        }
        snprintf(error, SMALL_SIZE, "No model found with the given id");
        error[SMALL_SIZE - 1] = '\0';
        return error;
    }

    input_info *input_info_ptr = malloc(sizeof(input_info));
    input_info_ptr->input_values = NULL;
    input_info_ptr->input_shapefacts = NULL;
    input_info_ptr->input_datum_types = NULL;
    void *tokenizer_ptr = NULL;
    char *model_name = m->names[0];

    int model_count = get_array_size((void **)m->names);
    fprintf(stderr, "Model count: %d\n", model_count);
    if (model_count != count_tags) {
        return NULL;
    }

    EncryptionParameters *params = (EncryptionParameters *)malloc(sizeof(EncryptionParameters));
    if (!params) {
        fprintf(stderr, "Memory allocation for params failed\n");
        return NULL;
    }
    uint8_t *key = (uint8_t *)malloc(KEY_BYTES);
    uint8_t *iv = (uint8_t *)malloc(IV_BYTES);
    uint8_t *aad = (uint8_t *)malloc(ADD_DATA_BYTES);
    if (!key || !iv || !aad) {
        fprintf(stderr, "Memory allocation for key, iv, tag, aad failed\n");
        free(params);
        return NULL;
    }
    memcpy(key, m->key, KEY_BYTES);
    memcpy(iv, m->IV, IV_BYTES);
    memcpy(aad, m->AAD, ADD_DATA_BYTES);
    params->key = key;
    params->iv = iv;
    params->aad = aad;
    if (!params->key || !params->iv || !params->aad) {
        fprintf(stderr, "Error reading Encryption parameters from onnx table\n");
        free(params);
        return NULL;
    }

    EncryptionParameters *params_weights = NULL;
    #if NUM_TOKENS != 0
        params_weights = (EncryptionParameters *)malloc(sizeof(EncryptionParameters));
        if (!params_weights) {
            fprintf(stderr, "Memory allocation for params_weights failed\n");
            free(key);
            free(iv);
            free(aad);
            free(params);
            return NULL;
        }
        params_weights->key = key;
        params_weights->iv = iv;
        params_weights->aad = aad;
        params_weights->tag = NULL;
        if (!params_weights->key || !params_weights->iv || !params_weights->aad) {
            fprintf(stderr, "Error reading Encryption parameters from onnx table\n");
            free(key);
            free(iv);
            free(aad);
            free(params);
            free(params_weights);
            return NULL;
        }
    #endif

    int number_inputs_llm = 3;
    if (!images && tokenizer_size > 0) {
        check_ret(tract_create_tokenizer(*tokenizer, tokenizer_size, &tokenizer_ptr), NULL);
        free(*tokenizer);

        if ((model_count == 1 && strstr(m->names[0], "gpt2") != NULL) || 
            (model_count == 2 && strstr(m->names[1], "gpt2") != NULL)) {
            number_inputs_llm = 2;
        }

        input_info_ptr->input_values = malloc((number_inputs_llm + 1) * sizeof(void *));
        input_info_ptr->input_shapefacts = malloc((number_inputs_llm + 1) * sizeof(void *));
        input_info_ptr->input_datum_types = malloc((number_inputs_llm + 1) * sizeof(void *));
        memset(input_info_ptr->input_shapefacts, 0, (number_inputs_llm + 1) * sizeof(void *));

        char *prompt = "Hi, how are you today?";

        check_ret(tract_value_from_bytes_llm(tokenizer_ptr, prompt, input_info_ptr->input_values, input_info_ptr->input_datum_types, number_inputs_llm), NULL);
        input_info_ptr->input_values[number_inputs_llm] = NULL;
        input_info_ptr->input_datum_types[number_inputs_llm] = NULL;
    } else {
        assert(images);

        input_info_ptr->input_values = malloc((num_images + 1) * sizeof(void *));

        int size, flag;
        for (int i = 0; i < num_images; i++) {
            size_t shape[4] = {(int)images[i][0], (int)images[i][1], (int)images[i][2], (int)images[i][3]};

            fprintf(stderr, "Image shape[%d]: %zu, %zu, %zu, %zu\n", i, shape[0], shape[1], shape[2], shape[3]);

            flag = 0;
            size = 1;
            for (int j = 0; j < 4; j++) {
                if (shape[j] != 0) {
                    flag++;
                    size *= shape[j];
                }
            }

            float *temp_image = (float *) malloc(size * sizeof(float));
            if (!temp_image) {
                fprintf(stderr, "Error allocating memory for temp_image\n");
                return NULL;
            }
            memcpy(temp_image, images[i] + flag, size * sizeof(float));

            TractValue *input_value = NULL;
            check_ret(tract_value_from_bytes(TRACT_DATUM_TYPE_F32, flag, shape, temp_image, &input_value), NULL);
            free(temp_image);

            input_info_ptr->input_values[i] = input_value;
        }
        input_info_ptr->input_values[num_images] = NULL;
    }

    double sum = 0.0;
    m->head->outputs = input_info_ptr->input_values;
    int runs = (NUM_TOKENS == 0) ? 1 : NUM_TOKENS;
    operator_node *last_node = NULL;
#if NUM_TOKENS != 0    
    char *generated_text;
#endif

#ifdef USE_SYS_TIME
    gettimeofday(&t1_inf, NULL);
#endif

    for (int i = 0; i < runs; ++i) {
#if NUM_TOKENS != 0
        generated_text = NULL;
        uintptr_t next_token_id = 0;
#endif
        last_node = execute_tree(m->head, input_info_ptr, &sum, tags, params, params_weights);
        reset_node_visibility(m->head);

    #if NUM_TOKENS != 0
        check_ret(tract_generate_text_llm(input_info_ptr->input_values, number_inputs_llm, tokenizer_ptr, last_node->outputs, last_node->num_outputs, &generated_text, &next_token_id), NULL);
        if (strstr(model_name, "albert") == NULL) {
            check_ret(tract_update_input_values_llm(input_info_ptr->input_values, number_inputs_llm, next_token_id), NULL);
        }

        free_operator_node_info(m->head);
        reset_node_visibility(m->head);

        if (i != (runs - 1)) tract_free_cstring(generated_text);
    #endif
    }
    #if NUM_TOKENS != 0
        free(input_info_ptr->input_shapefacts);
        free(input_info_ptr->input_datum_types);
    #endif
    free(input_info_ptr);
    
#ifdef USE_SYS_TIME
    gettimeofday(&t2_inf, NULL);
    elapsed_time = (t2_inf.tv_sec - t1_inf.tv_sec) * 1000.0;      // sec to ms
    elapsed_time += (t2_inf.tv_usec - t1_inf.tv_usec) / 1000.0;   // us to ms
#else
    elapsed_time = 0.0;
#endif

    fprintf(stderr, "Inference time: %f ms\n", elapsed_time);
    fprintf(stderr, "Inference time to run a model: %f ms\n", sum);

    char *prediction = (char *) malloc(SMALL_SIZE * sizeof(char));
    if (!prediction) {
        fprintf(stderr, "Error allocating memory for result\n");
        return NULL;
    }

    #if NUM_TOKENS != 0
        check_ret(tract_free_tokenizer(&tokenizer_ptr), NULL);
        snprintf(prediction, SMALL_SIZE, "Model %s, %s!", m->names[model_count-1], generated_text);
        tract_free_cstring(generated_text);
    #else
        snprintf(prediction, SMALL_SIZE, "Model %s, Inference: Max is %f for category %d!", m->names[model_count-1], last_node->pred, last_node->category);
    #endif
    prediction[SMALL_SIZE - 1] = '\0';

    free(key);
    free(iv);
    free(aad);
    free(params);
#if NUM_TOKENS != 0
    free(params_weights);
#endif

    return prediction;
}
#endif
#endif