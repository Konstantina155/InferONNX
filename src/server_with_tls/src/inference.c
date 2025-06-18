#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <inference.h>

// INFERENCE INFO STRUCT - LOADING PHASE
static TractInferenceModel **
initialize_inference_models(int num_models)
{
    TractInferenceModel **inference_models = (TractInferenceModel **)malloc((num_models + 1) * sizeof(TractInferenceModel *));
    if (!inference_models) {
        fprintf(stderr, "Error allocating memory for inference_models\n");
        return NULL;
    }

    for (int i = 0; i < num_models + 1; ++i) {
        inference_models[i] = NULL;
    }

    return inference_models;
}

static MyInferenceModel **
initialize_my_inference_models(int num_models)
{
    MyInferenceModel **my_inference_models = (MyInferenceModel **)malloc((num_models + 1) * sizeof(MyInferenceModel *));
    if (!my_inference_models) {
        fprintf(stderr, "Error allocating memory for my_inference_models\n");
        return NULL;
    }

    for (int i = 0; i < num_models + 1; ++i) {
        my_inference_models[i] = NULL;
    }

    return my_inference_models;
}

static void
onnx_model_inputs(operator_io **io, TractInferenceModel *inference_model, MyInferenceModel *my_inference_model, int index, operator_node *head, char *model_name)
{
    uintptr_t num_inputs = 0;
    uintptr_t num_outputs = 0;
    char *input_name = NULL;
    char **input_names = NULL;
    char **output_names = NULL;
    int8_t *output_name = NULL;

    
    if (inference_model) {
        fprintf(stderr, "here");
        check(tract_inference_model_input_count(inference_model, &num_inputs));
        input_names = malloc((num_inputs + 1) * sizeof(char *));
        for (int i = 0; i < (int)num_inputs; i++) {
            check(tract_inference_model_input_name(inference_model, i, &input_name));
            input_names[i] = input_name;
        }
        input_names[num_inputs] = NULL;

        check(tract_inference_model_output_count(inference_model, &num_outputs));
        output_names = malloc((num_outputs + 1) * sizeof(char *));
        for (int i = 0; i < (int)num_outputs; i++) {
            check(tract_inference_model_output_name(inference_model, i, &output_name));
            output_names[i] = (char *)output_name;
        }
        output_names[num_outputs] = NULL;
    }
#ifdef USE_AES
    if (my_inference_model) {
        check(tract_my_inference_model_input_count(my_inference_model, &num_inputs));
        input_names = malloc((num_inputs + 1) * sizeof(char *));
        for (int i = 0; i < (int)num_inputs; i++) {
            check(tract_my_inference_model_input_name(my_inference_model, i, &input_name));
            input_names[i] = input_name;
        }
        input_names[num_inputs] = NULL;

        check(tract_my_inference_model_output_count(my_inference_model, &num_outputs));
        output_names = malloc((num_outputs + 1) * sizeof(char *));
        for (int i = 0; i < (int)num_outputs; i++) {
            check(tract_my_inference_model_output_name(my_inference_model, i, &output_name));
            output_names[i] = (char *)output_name;
        }
        output_names[num_outputs] = NULL;
    }
#endif

    if (index == 1) {
        operator_io o_io_first;
        o_io_first.input_names_length = 0;
        o_io_first.input_names = NULL;
        o_io_first.output_names_length = num_inputs;
        o_io_first.output_names = input_names;

        insert_into_operator_io(&io, &o_io_first, index - 1, "input");
        update_node(io, index - 1, NULL);
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
    print_operator_io(io);
}

#ifdef USE_AES
    #if USE_MEMORY_ONLY == 1 || NUM_TOKENS == 0
static TractInferenceModel *
onnx_model_for_path(char *model_name, TractInferenceModel *inference_model, struct EncryptionParameters *params) {
    // Initialize onnx parser
    TractOnnx *onnx = NULL;
    check_ret(tract_onnx_create(&onnx), NULL);
    assert(onnx);

    // Load the model
    if (tract_onnx_model_for_path(onnx, model_name, &inference_model, params) != TRACT_RESULT_OK) {
        fprintf(stderr, "Error calling tract: %s", tract_get_last_error());
        check_ret(tract_onnx_destroy(&onnx), NULL);
        check_ret(tract_inference_model_destroy(&inference_model), NULL);
        assert(!inference_model);
        assert(!onnx);
        return NULL;
    }
    assert(inference_model);
    assert(onnx);

    check_ret(tract_onnx_destroy(&onnx), NULL);
    assert(!onnx);

    return inference_model;
}

static MyInferenceModel *
onnx_model_for_path_tokenizer(char *model_name, MyInferenceModel *my_inference_model, struct EncryptionParameters *params, struct EncryptionParameters *params_weights) {
    // Load the model
    if (tract_load_nlp_model(model_name, params, params_weights, &my_inference_model) != TRACT_RESULT_OK) {
        fprintf(stderr, "Error calling tract: %s", tract_get_last_error());
        assert(!my_inference_model);
        return NULL;
    }
    assert(my_inference_model);

    return my_inference_model;
}
    #endif

void
load_model_to_memory(model **m, unsigned char **tags, int count_tags)
{
    if (!m) return;

    assert(tags);

#if USE_MEMORY_ONLY == 1 || NUM_TOKENS == 0
    EncryptionParameters *params = (EncryptionParameters *)malloc(sizeof(EncryptionParameters));
    if (!params) {
        fprintf(stderr, "Memory allocation for params failed\n");
        return;
    }
    uint8_t *key = (uint8_t *)malloc(KEY_BYTES);
    uint8_t *iv = (uint8_t *)malloc(IV_BYTES);
    uint8_t *tag = NULL;
    uint8_t *tag_weights = NULL;
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

    EncryptionParameters *params_weights = (EncryptionParameters *)malloc(sizeof(EncryptionParameters));
    if (!params_weights) {
        fprintf(stderr, "Memory allocation for params_weights failed\n");
        return;
    }
#endif

    char **names = (*m)->names;
    int model_count = get_array_size((void **)names);
    fprintf(stderr, "Model count: %d\n", model_count);
    if (model_count != count_tags) {
    #if USE_MEMORY_ONLY == 1 || NUM_TOKENS == 0
        free(tag);
        free(key);
        free(iv);
        free(aad);
        free(params);
    #endif
        return;
    }

    TractInferenceModel **inference_models = initialize_inference_models(model_count + 1);
    MyInferenceModel **my_inference_models = initialize_my_inference_models(model_count + 1);

    int initial_length = 10;
    operator_io **io = init_operator_io(initial_length);
    assert(io);
    operator_node *previous = NULL, *curr_node = NULL, *head = NULL;
    int index = 0;

    for (int i = 1; i < model_count + 1; i++) {
        fprintf(stderr, "Name: %s\n", names[i-1]);

#if USE_MEMORY_ONLY == 1 || NUM_TOKENS == 0
        tag = (uint8_t *)malloc(TAG_BYTES * 2 + 1);
        if (!tag) {
            fprintf(stderr, "Memory allocation for tag failed\n");
            free(tag);
            free(key);
            free(iv);
            free(aad);
            free(params);
            return;
        }

        int is_llm = (strstr(names[0], "model.onnx_data") != NULL) || 
                 (strstr(names[0], "albert") != NULL) || 
                 (strstr(names[0], "gpt") != NULL) || 
                 (strstr(names[0], "llama") != NULL) || 
                 (strstr(names[0], "mistral") != NULL) ||
                 (strstr(names[0], "deepseek") != NULL)
                 ? 1 : 0;

        if (is_llm) {
            if (strstr(names[i-1], "model.onnx_data") != NULL) {
                my_inference_models[i] = NULL;
                tag_weights = (uint8_t *)malloc(TAG_BYTES * 2 + 1);
                if (!tag_weights) {
                    fprintf(stderr, "Memory allocation for tag failed\n");
                    free(tag);
                    free(key);
                    free(iv);
                    free(aad);
                    free(params);
                    return;
                }
                memcpy(tag_weights, tags[i-1], TAG_BYTES * 2);
                tag_weights[TAG_BYTES * 2] = '\0';
                params_weights->tag = tag_weights;
            } else {
                memcpy(tag, tags[i-1], TAG_BYTES * 2);
                tag[TAG_BYTES * 2] = '\0';
                params->tag = tag;
                index += 1;

                params_weights->key = key;
                params_weights->iv = iv;
                params_weights->aad = aad;
                if (strstr(names[i-1], "albert") != NULL || strstr(names[i-1], "gpt") != NULL || strstr(names[i-1], "teeny") != NULL) params_weights->tag = tag;


                my_inference_models[index] = onnx_model_for_path_tokenizer(names[i-1], my_inference_models[index], params, params_weights);
                if (!my_inference_models[index]) {
                    free(tag);
                    free(key);
                    free(iv);
                    free(aad);
                    free(params);
                    return;
                }
            }
        } else {
            memcpy(tag, tags[i-1], TAG_BYTES * 2);
            tag[TAG_BYTES * 2] = '\0';
            params->tag = tag;
            inference_models[i] = onnx_model_for_path(names[i-1], inference_models[i], params);
            if (!inference_models[i]) {
                free(tag);
                free(key);
                free(iv);
                free(aad);
                free(params);
                free(params_weights);
                return;
            }
        }
        fprintf(stderr, "Tag in load_model_to_memory: %s\n", tag);
#endif

        if (i == initial_length) {
            resize_operators_io(&io, initial_length + 5, initial_length);
            assert(io);
            initial_length += 5;
        }

        curr_node = create_operator_node(names[i-1]);
        curr_node->run_inference = run_inference;
        if (previous) {
            insert_child_to_operator_node(previous, curr_node);
        } else {
            head = create_operator_node("input");
            insert_child_to_operator_node(head, curr_node);
        }
        previous = curr_node;

        onnx_model_inputs(io, inference_models[i], my_inference_models[index], i, head, names[i-1]);
    #if USE_MEMORY_ONLY == 1 || NUM_TOKENS == 0
        free(tag);
    #endif
    }

#if USE_MEMORY_ONLY == 1 || NUM_TOKENS == 0
    free(key);
    free(iv);
    free(aad);
    if (tag_weights) free(tag_weights);
    free(params);
    free(params_weights);
#endif
    free_operator_io(io);

    (*m)->head = head;

#ifdef USE_MEMORY_ONLY
    (*m)->inference_models = inference_models;
    (*m)->my_inference_models = my_inference_models;
#else
    free_inference_models(inference_models, my_inference_models, model_count + 1);
#endif
}

#else
TractInferenceModel *
onnx_model_for_path(char *model_name, TractInferenceModel *inference_model) {
    // Initialize onnx parser
    TractOnnx *onnx = NULL;
    check_ret(tract_onnx_create(&onnx), NULL);
    assert(onnx);

    // Load the model
    if (tract_onnx_model_for_path(onnx, model_name, &inference_model) != TRACT_RESULT_OK) {
        fprintf(stderr, "Error calling tract: %s", tract_get_last_error());
        check_ret(tract_onnx_destroy(&onnx), NULL);
        check_ret(tract_inference_model_destroy(&inference_model), NULL);
        assert(!inference_model);
        assert(!onnx);
        return NULL;
    }
    assert(inference_model);
    assert(onnx);

    check_ret(tract_onnx_destroy(&onnx), NULL);
    assert(!onnx);

    return inference_model;
}

void
load_model_to_memory(model **m)
{
    if (!m) return;

    char **names = (*m)->names;
    int model_count = get_array_size((void **)names);
    fprintf(stderr, "Model count: %d\n", model_count);

    TractInferenceModel **inference_models = initialize_inference_models(model_count + 1);
    int initial_length = 10;
    operator_io **io = init_operator_io(initial_length);
    assert(io);
    operator_node *previous = NULL, *curr_node = NULL, *head = NULL;

    int is_llm = (strstr(names[0], "model.onnx_data") != NULL) || 
                 (strstr(names[0], "albert") != NULL) || 
                 (strstr(names[0], "gpt") != NULL) || 
                 (strstr(names[0], "llama") != NULL) || 
                 (strstr(names[0], "mistral") != NULL) ||
                 (strstr(names[0], "deepseek") != NULL)
                 ? 1 : 0;

    for (int i = 1; i < model_count + 1; i++) {
        if (is_llm) {
            inference_models[i] = NULL;
        } else {
            inference_models[i] = onnx_model_for_path(names[i-1], inference_models[i]);
            if (!inference_models[i]) {
                return;
            }
        }

        if (i == initial_length) {
            resize_operators_io(&io, initial_length + 5, initial_length);
            assert(io);
            initial_length += 5;
        }

        curr_node = create_operator_node(names[i-1]);
        curr_node->run_inference = run_inference;
        if (previous) {
            insert_child_to_operator_node(previous, curr_node);
        } else {
            head = create_operator_node("input");
            insert_child_to_operator_node(head, curr_node);
        }
        previous = curr_node;

        onnx_model_inputs(io, inference_models[i], NULL, i, head, names[i-1]);
    }
    (*m)->head = head;

    free_operator_io(io);
    free_inference_models(inference_models, NULL, model_count + 1);
}
#endif

// INFERENCE
#if USE_AES == 0 && USE_MEMORY_ONLY == 0 || USE_AES == 1 && USE_MEMORY_ONLY == 1
void
run_inference(operator_node **node, TractValue **input_values, TractInferenceModel *inference_model)
{
#ifdef USE_SYS_TIME
    struct timeval t1_run, t2_run;
#endif
    double elapsed_time;

    TractModel *model = NULL;
#ifndef USE_MEMORY_ONLY
        // Initialize onnx parser
        TractOnnx *onnx = NULL;
        check(tract_onnx_create(&onnx));
        assert(onnx);

        // Load the model
        check(tract_onnx_model_for_path(onnx, (*node)->model_name, &inference_model));
        assert(inference_model);
        assert(onnx);

        check(tract_onnx_destroy(&onnx));
        assert(!onnx);

        // Transform an inference model into a typed model
        check(tract_inference_model_into_typed(&inference_model,&model));
        assert(model);

        free_inference_model(inference_model);
#else
    // Transform an inference model into a typed model
    check(tract_inference_model_into_typed(&inference_model,&model));
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
            int num_inputs = get_array_size((void **)input_values);
            for (int j = 0; j < num_inputs; j++) {
                inputs[k++] = input_values[j];
            }
            continue;
        }
        if (!(*node)->parents[i]->outputs[indices[index]]) {
            fprintf(stderr, "The output is NULL!");
            continue;
        }
        inputs[k++] = (*node)->parents[i]->outputs[indices[index++]];
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

    (*node)->outputs = (TractValue **)malloc((num_outputs + 1) * sizeof(TractValue *));
    for (int i = 0; i < num_outputs; i++) {
        if (outputs[i] == NULL) {
            fprintf(stderr, "Output %d is NULL\n", i);
            continue;
        }
        (*node)->outputs[i] = outputs[i];
    }
    (*node)->outputs[num_outputs] = NULL;
    free(outputs);
    (*node)->pred = max;
    (*node)->category = argmax;
    (*node)->elapsedTime = elapsed_time;
}

double
execute_tree(operator_node *node, TractValue **input_values, double elapsed_time, char **visited_nodes, int *visited_count, FILE *fd, TractInferenceModel **inference_models)
{
    if (!node) {
        return elapsed_time;
    }

    if (is_node_visited(node, visited_nodes, *visited_count) == false) {
        visited_nodes[*visited_count] = node->model_name;
        (*visited_count)++;

        if (*visited_count != 1) {
            fprintf(stderr, "Model name: %s\n", node->model_name);
#ifdef USE_DEBUG
    #ifndef USE_MEMORY_ONLY
            node->run_inference(&node, input_values, NULL);
    #else 
            node->run_inference(&node, input_values, inference_models[*visited_count - 1]);
    #endif
    #ifdef USE_SYS_TIME
                fprintf(fd, "Partition_%d: %f ms\n", (*visited_count) - 1, node->elapsedTime);
    #endif
#else
            if (fd) {
                node->run_inference(&node, input_values, NULL);
    #ifdef USE_SYS_TIME
                fprintf(fd, "Partition_%d: %f ms\n", (*visited_count) - 1, node->elapsedTime);
    #endif
            } else {
                assert(inference_models);
                node->run_inference(&node, input_values, inference_models[*visited_count - 1]);
                fprintf(stderr, "Partition_%d: %f ms\n", (*visited_count) - 1, node->elapsedTime);
            }
#endif
        
            elapsed_time += node->elapsedTime;
        }
    }

    for (int i = 0; i < node->num_children; i++) {
        elapsed_time = execute_tree(node->children[i], input_values, elapsed_time, visited_nodes, visited_count, fd, inference_models);
    }
    return elapsed_time;
}
#endif

#ifndef USE_AES
char *
inference_no_aes(float **images, int num_images, uint8_t *tokenizer, int tokenizer_size, model *m)
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

    if (!images && tokenizer_size > 0){
        int model_count = get_array_size((void **)m->names);
        fprintf(stderr, "Model count: %d\n", model_count);
        char *model_name = NULL;
        char *inference = NULL;

        for (int i = 0; i < model_count; ++i) {
            if (strstr(m->names[i], "model.onnx_data") == NULL) {
                model_name = m->names[i];
                break;
            }
        }
        
        int is_albert = strstr(model_name, "albert") != NULL;

        gettimeofday(&t1_inf, NULL);
        if (is_albert) {
            check_ret(tract_run_albert(model_name, tokenizer, tokenizer_size, &inference, NULL), NULL);
        } else {
            check_ret(tract_run_latest_models(model_name, tokenizer, tokenizer_size, &inference, NUM_TOKENS, "Hi"), NULL);
        }

        gettimeofday(&t2_inf, NULL);
        elapsed_time = (t2_inf.tv_sec - t1_inf.tv_sec) * 1000.0;      // sec to ms
        elapsed_time += (t2_inf.tv_usec - t1_inf.tv_usec) / 1000.0;   // us to ms
        
#ifdef USE_SYS_TIME
        FILE *fd = fopen("../inference_time_outside_occlum_on_disk_no_aes.txt", "a");
        if (!fd) {
            fprintf(stderr, "Error opening inference_time!\n");
            return NULL;
        }

        if (fprintf(fd, "Inference time: %f ms\n", elapsed_time) < 0) {
            fprintf(stderr, "Error writing to file inference_time_outside_occlum_on_disk_no_aes.txt\n");
            fclose(fd);
            return NULL;
        }
        fclose(fd);
#endif
        return inference;
    } else {
        assert(images);
    }

    int model_count = get_array_size((void **)m->names);
    fprintf(stderr, "Model count: %d\n", model_count);

    FILE *fd = NULL;
#ifndef USE_DEBUG
    fd = fopen("../inference_time_outside_occlum_on_disk_no_aes.txt", "a");
    if (!fd) {
        fprintf(stderr, "Error opening inference_time_outside_occlum_on_disk_no_aes!\n");
        return NULL;
    }
#endif

    int size, flag;
    TractValue *input_value = NULL;
    TractValue **input_values = (TractValue **) malloc((num_images + 1) * sizeof(TractValue *));
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

        input_value = NULL;
        check_ret(tract_value_from_bytes(TRACT_DATUM_TYPE_F32, flag, shape, temp_image, &input_value), NULL);
        free(temp_image);

        input_values[i] = input_value;
    }
    input_values[num_images] = NULL;

    m->head->outputs = input_values;

    char **visited_nodes = (char **) malloc((model_count + 1) * sizeof(char *));
    int visited_count = 0;

    gettimeofday(&t1_inf, NULL);
    double sum = execute_tree(m->head, input_values, 0.0, visited_nodes, &visited_count, fd, m->inference_models);
    gettimeofday(&t2_inf, NULL);
    
    visited_nodes[model_count] = NULL;
    free(visited_nodes);

    
    elapsed_time = (t2_inf.tv_sec - t1_inf.tv_sec) * 1000.0;      // sec to ms
    elapsed_time += (t2_inf.tv_usec - t1_inf.tv_usec) / 1000.0;   // us to ms

#ifndef USE_DEBUG
    #ifdef USE_SYS_TIME
        if (fprintf(fd, "Inference time: %f ms\n", elapsed_time) < 0) {
            fprintf(stderr, "Error writing to file inference_time_outside_occlum_on_disk_no_aes.txt\n");
            fclose(fd);
            return NULL;
        }
        if (fprintf(fd, "Inference time to run a model: %f ms\n", sum) < 0) {
            fprintf(stderr, "Error writing to file inference_time_outside_occlum_on_disk_no_aes.txt\n");
            fclose(fd);
            return NULL;
        }
    #endif
        fclose(fd);
#else
    fprintf(stderr, "Inference time: %f ms\n", elapsed_time);
    fprintf(stderr, "Inference time to run a model: %f ms\n", sum);
#endif

    operator_node *last_node = search_operator_node_by_name(m->head, m->names[model_count-1]);
    char *prediction = (char *) malloc(SMALL_SIZE * sizeof(char));
    if (!prediction) {
        fprintf(stderr, "Error allocating memory for result\n");
        return NULL;
    }

    snprintf(prediction, SMALL_SIZE, "Model %s, Inference: Max is %f for category %d!", m->names[model_count-1], last_node->pred, last_node->category);
    prediction[SMALL_SIZE - 1] = '\0';
    return prediction;
}
#else

#ifdef USE_MEMORY_ONLY
char *
inference_memory_only(float **images, int num_images, uint8_t *tokenizer, int tokenizer_size, model *m)
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
    } else if (!m->inference_models && !m->my_inference_models) {
        error = (char *) malloc(SMALL_SIZE * sizeof(char));
        if (!error) {
            fprintf(stderr, "Error allocating memory for error\n");
            return NULL;
        }
        snprintf(error, SMALL_SIZE, "No inference model found with the given id");
        error[SMALL_SIZE - 1] = '\0';
        return error;
    }

    if (!images && tokenizer_size > 0){
        int model_count = get_array_size((void **)m->names);
        fprintf(stderr, "Model count: %d\n", model_count);
        char *model_name = NULL;
        char *inference = NULL;

        for (int i = 0; i < model_count; ++i) {
            if (strstr(m->names[i], "model.onnx_data") == NULL) {
                model_name = m->names[i];
                break;
            }
        }
        MyInferenceModel *inference_model = m->my_inference_models[1];
        
        int is_albert = strstr(model_name, "albert") != NULL;

#ifdef USE_SYS_TIME
    gettimeofday(&t1_inf, NULL);
#endif
        if (is_albert) {
            check_ret(tract_run_albert(model_name, tokenizer, tokenizer_size, &inference, NULL, &inference_model), NULL);
        } else {
            check_ret(tract_run_latest_models(model_name, tokenizer, tokenizer_size, &inference, NULL, NULL, NUM_TOKENS, "Hi", &inference_model), NULL);
        }

#ifdef USE_SYS_TIME
        gettimeofday(&t2_inf, NULL);
        elapsed_time = (t2_inf.tv_sec - t1_inf.tv_sec) * 1000.0;      // sec to ms
        elapsed_time += (t2_inf.tv_usec - t1_inf.tv_usec) / 1000.0;   // us to ms
#else
        elapsed_time = 0.0;
#endif
        
#ifdef USE_SYS_TIME
        FILE *fd = fopen("../inference_time_outside_occlum_on_disk_no_aes.txt", "a");
        if (!fd) {
            fprintf(stderr, "Error opening inference_time!\n");
            return NULL;
        }

        if (fprintf(fd, "Inference time: %f ms\n", elapsed_time) < 0) {
            fprintf(stderr, "Error writing to file inference_time_outside_occlum_on_disk_no_aes.txt\n");
            fclose(fd);
            return NULL;
        }
        fclose(fd);
#endif
        return inference;
    } else {
        assert(images);
    }

    int model_count = get_array_size((void **)m->names);
    fprintf(stderr, "Model count: %d\n", model_count);

    
    int size, flag;
    TractValue *input_value = NULL;
    TractValue **input_values = (TractValue **) malloc((num_images + 1) * sizeof(TractValue *));
    assert(input_values);
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
        memcpy(temp_image, images[i] + 4, size * sizeof(float));

        input_value = NULL;
        check_ret(tract_value_from_bytes(TRACT_DATUM_TYPE_F32, flag, shape, temp_image, &input_value), NULL);
        free(temp_image);

        input_values[i] = input_value;
    }
    input_values[num_images] = NULL;

    m->head->outputs = input_values;

    char **visited_nodes = (char **) malloc((model_count + 1) * sizeof(char *));
    int visited_count = 0;

#ifdef USE_SYS_TIME
    gettimeofday(&t1_inf, NULL);
#endif
    double sum = execute_tree(m->head, input_values, 0.0, visited_nodes, &visited_count, NULL, m->inference_models);
#ifdef USE_SYS_TIME
    gettimeofday(&t2_inf, NULL);
    elapsed_time = (t2_inf.tv_sec - t1_inf.tv_sec) * 1000.0;      // sec to ms
    elapsed_time += (t2_inf.tv_usec - t1_inf.tv_usec) / 1000.0;   // us to ms
#else
    elapsed_time = 0.0;
#endif

    visited_nodes[model_count] = NULL;
    free(visited_nodes);

    fprintf(stderr, "Inference time: %f ms\n", elapsed_time);
    fprintf(stderr, "Inference time to run a model: %f ms\n", sum);

    operator_node *last_node = search_operator_node_by_name(m->head, m->names[model_count-1]);
    char *prediction = (char *) malloc(SMALL_SIZE * sizeof(char));
    if (!prediction) {
        fprintf(stderr, "Error allocating memory for result\n");
        return NULL;
    }
    snprintf(prediction, SMALL_SIZE, "Model %s, Inference: Max is %f for category %d!", m->names[model_count-1], last_node->pred, last_node->category);
    prediction[SMALL_SIZE - 1] = '\0';
    
    return prediction;
}
#else

void
run_inference(operator_node **node, TractValue **input_values, struct EncryptionParameters *params)
{
    assert(params);

#ifdef USE_SYS_TIME
    struct timeval t1_run, t2_run;
#endif
    double elapsed_time;

    // Initialize onnx parser
    TractOnnx *onnx = NULL;
    check(tract_onnx_create(&onnx));
    assert(onnx);

    // Load the model
    TractModel *model = NULL;
    TractInferenceModel *inference_model = NULL;
    if (tract_onnx_model_for_path(onnx, (*node)->model_name, &inference_model, params) != TRACT_RESULT_OK) {
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

    free_inference_model(inference_model);

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
            int num_inputs = get_array_size((void **)input_values);
            for (int j = 0; j < num_inputs; j++) {
                inputs[k++] = input_values[j];
            }
            continue;
        }
        if (!(*node)->parents[i]->outputs[indices[index]]) {
            fprintf(stderr, "The output is NULL!");
            continue;
        }
        inputs[k++] = (*node)->parents[i]->outputs[indices[index++]];
    }
    if ((*node)->num_inputs == -1 || (*node)->num_parents == 0) {
        int num_inputs = get_array_size((void **)input_values);
        inputs = realloc(inputs, (num_inputs + 1) * sizeof(TractValue *));
        assert(inputs);
        for (int j = 0; j < num_inputs; j++) {
            inputs[k++] = input_values[j];
        }
        fprintf(stderr, "k: %d\n", k);
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

    (*node)->outputs = (TractValue **)malloc((num_outputs + 1) * sizeof(TractValue *));
    for (int i = 0; i < num_outputs; i++) {
        if (outputs[i] == NULL) {
            fprintf(stderr, "Output %d is NULL\n", i);
            continue;
        }
        (*node)->outputs[i] = outputs[i];
    }
    (*node)->outputs[num_outputs] = NULL;
    free(outputs);
    (*node)->pred = max;
    (*node)->category = argmax;
    (*node)->elapsedTime = elapsed_time;
}

double
execute_tree(operator_node *node, TractValue **input_values, double elapsed_time, char **visited_nodes, int *visited_count, unsigned char **tags, struct EncryptionParameters *params)
{
    if (!node) {
        return elapsed_time;
    }   

    if (is_node_visited(node, visited_nodes, *visited_count) == false) {
        visited_nodes[*visited_count] = node->model_name;
        (*visited_count)++;

        int i = (*visited_count) - 2;

        if (*visited_count != 1) {
            unsigned char *tag = (uint8_t *)malloc(TAG_BYTES * 2);
            assert(tag);
            memcpy(tag, tags[i], TAG_BYTES * 2);
            params->tag = tag;

            node->run_inference(&node, input_values, params);
            fprintf(stderr, "Model name: %s\n", node->model_name);
            fprintf(stderr, "Partition_%d: %f ms\n", i, node->elapsedTime);
            free(tag);

            if (!node->outputs) {
                return -1;
            }
            
            elapsed_time += node->elapsedTime;
        }
    }

    for (int i = 0; i < node->num_children; i++) {
        elapsed_time = execute_tree(node->children[i], input_values, elapsed_time, visited_nodes, visited_count, tags, params);
    }
    return elapsed_time;
}

void
send_keepalive_callback(void* ssl_ptr)
{
    mbedtls_ssl_context* ssl = (mbedtls_ssl_context*)ssl_ptr;
    const char* keepalive_msg = "KEEPALIVE\n";
    int ret = mbedtls_ssl_write(ssl, (unsigned char*)keepalive_msg, strlen(keepalive_msg));
    if (ret > 0) {
        printf("Server: Sent keep-alive message\n");
    } else {
        printf("Server: Failed to send keep-alive: %d\n", ret);
    }
}

char *
inference_aes(float **images, int num_images, uint8_t *tokenizer, int tokenizer_size, model *m, unsigned char **tags, int count_tags, mbedtls_ssl_context *ssl)
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

    EncryptionParameters *params = (EncryptionParameters *)malloc(sizeof(EncryptionParameters));
    if (!params) {
        fprintf(stderr, "Memory allocation for params failed\n");
        return NULL;
    }
    uint8_t *key = (uint8_t *)malloc(KEY_BYTES);
    uint8_t *iv = (uint8_t *)malloc(IV_BYTES);
    uint8_t *tag = NULL;
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

    if (!images && tokenizer_size > 0){
        int model_count = get_array_size((void **)m->names);
        fprintf(stderr, "Model count: %d\n", model_count);
        if (model_count != count_tags) {
            free(key);
            free(iv);
            free(aad);
            free(params);
        }

        tag = (uint8_t *)malloc(TAG_BYTES * 2);
        if (!tag) {
            fprintf(stderr, "Memory allocation for tag failed\n");
            free(tag);
            free(key);
            free(iv);
            free(aad);
            free(params);
            return NULL;
        }

        EncryptionParameters *params_weights = (EncryptionParameters *)malloc(sizeof(EncryptionParameters));
        if (!params_weights) {
            fprintf(stderr, "Memory allocation for params_weights failed\n");
            return NULL;
        }
        uint8_t *tag2 = NULL;
        params_weights->key = key;
        params_weights->iv = iv;
        params_weights->aad = aad;
        params_weights->tag = NULL;
        if (!params_weights->key || !params_weights->iv || !params_weights->aad) {
            fprintf(stderr, "Error reading Encryption parameters from onnx table\n");
            free(params);
            return NULL;
        }

        tag2 = (uint8_t *)malloc(TAG_BYTES * 2);
        if (!tag2) {
            fprintf(stderr, "Memory allocation for tag failed\n");
            free(tag);
            free(key);
            free(iv);
            free(aad);
            free(params_weights);
            free(params);
            return NULL;
        }

        char *model_name = NULL;
        for (int i = 0; i < model_count; ++i) {
            if (strstr(m->names[i], "model.onnx_data") == NULL) {
                model_name = m->names[i];
                memcpy(tag, tags[i], TAG_BYTES * 2);
                params->tag = tag;
            } else {
                memcpy(tag2, tags[i], TAG_BYTES * 2);
                params_weights->tag = tag2;
            }
        }
        char *inference = NULL;
        int is_albert = strstr(model_name, "albert") != NULL;
        fprintf(stderr, "Server: Starting inference (this may take a while)...\n");

#ifdef USE_SYS_TIME
    gettimeofday(&t1_inf, NULL);
#endif
        if (is_albert) {
            check_ret(tract_run_albert(model_name, tokenizer, tokenizer_size, &inference, params, NULL), NULL);
        } else {
            check_ret(tract_run_latest_models(model_name, tokenizer, tokenizer_size, &inference, params, params_weights, NUM_TOKENS, "Hi", NULL, send_keepalive_callback, ssl), NULL);
        }
#ifdef USE_SYS_TIME
        gettimeofday(&t2_inf, NULL);
        elapsed_time = (t2_inf.tv_sec - t1_inf.tv_sec) * 1000.0;      // sec to ms
        elapsed_time += (t2_inf.tv_usec - t1_inf.tv_usec) / 1000.0;   // us to ms
#else
        elapsed_time = 0.0;
#endif

        fprintf(stderr, "Inference time: %f ms\n", elapsed_time);
        
        free(tag);
        free(key);
        free(iv);
        free(aad);
        free(tag2);
        free(params);
        free(params_weights);

        return inference;
    } else {
        assert(images);
    }

    int model_count = get_array_size((void **)m->names);
    fprintf(stderr, "Model count: %d\n", model_count);
    fprintf(stderr, "Number of tags: %d\n", count_tags);
    if (model_count != count_tags) {
        free(tag);
        free(key);
        free(iv);
        free(aad);
        free(params);
        error = (char *) malloc(SMALL_SIZE * sizeof(char));
        if (!error) {
            fprintf(stderr, "Error allocating memory for error\n");
            return NULL;
        }
        snprintf(error, SMALL_SIZE, "Number of tags should be the same as number of models");
        error[SMALL_SIZE - 1] = '\0';
        return error;
    }

    int size, flag;
    TractValue *input_value = NULL;
    TractValue **input_values = (TractValue **) malloc((num_images + 1) * sizeof(TractValue *));
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
            free(tag);
            free(key);
            free(iv);
            free(aad);
            free(params);
            return NULL;
        }
        memcpy(temp_image, images[i] + flag, size * sizeof(float));

        input_value = NULL;
        check_ret(tract_value_from_bytes(TRACT_DATUM_TYPE_F32, flag, shape, temp_image, &input_value), NULL);
        free(temp_image);

        input_values[i] = input_value;
    }
    input_values[num_images] = NULL;

    m->head->outputs = input_values;

#ifdef USE_SYS_TIME
    gettimeofday(&t1_inf, NULL);
#endif

    char **visited_nodes = (char **) malloc((model_count + 1) * sizeof(char *));
    int visited_count = 0;
    double sum = execute_tree(m->head, input_values, 0.0, visited_nodes, &visited_count, tags, params);
    visited_nodes[model_count] = NULL;
    free(visited_nodes);

    if (sum == -1) {
        free(tag);
        free(key);
        free(iv);
        free(aad);
        free(params);
        error = (char *) malloc(SMALL_SIZE * sizeof(char));
        if (!error) {
            fprintf(stderr, "Error allocating memory for error\n");
            return NULL;
        }
        snprintf(error, SMALL_SIZE, "Model authentication is incorrect! Wrong EncryptionParams!");
        error[SMALL_SIZE - 1] = '\0';
        return error;
    }

#ifdef USE_SYS_TIME
    gettimeofday(&t2_inf, NULL);
    elapsed_time = (t2_inf.tv_sec - t1_inf.tv_sec) * 1000.0;      // sec to ms
    elapsed_time += (t2_inf.tv_usec - t1_inf.tv_usec) / 1000.0;   // us to ms
#else
    elapsed_time = 0.0;
#endif
    fprintf(stderr, "Inference time: %f ms\n", elapsed_time);
    fprintf(stderr, "Inference time to run a model: %f ms\n", sum);

    operator_node *last_node = search_operator_node_by_name(m->head, m->names[model_count-1]);
    char *prediction = (char *) malloc(SMALL_SIZE * sizeof(char));
    if (!prediction) {
        fprintf(stderr, "Error allocating memory for result\n");
        return NULL;
    }

    snprintf(prediction, SMALL_SIZE, "Model %s, Inference: Max is %f for category %d!", m->names[model_count-1], last_node->pred, last_node->category);
    prediction[SMALL_SIZE - 1] = '\0';
    
    free(key);
    free(iv);
    free(aad);
    free(params);

    return prediction;
}
#endif
#endif