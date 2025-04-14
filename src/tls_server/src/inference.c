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

static void
onnx_model_inputs(operator_io **io, TractInferenceModel *inference_model, int index, operator_node *head, char *model_name)
{
    uintptr_t num_inputs = 0;
    char *input_name = NULL;
    check(tract_inference_model_input_count(inference_model, &num_inputs));
    
    char **input_names = malloc((num_inputs + 1) * sizeof(char *));
    for (int i = 0; i < (int)num_inputs; i++) {
        check(tract_inference_model_input_name(inference_model, i, &input_name));
        input_names[i] = input_name;
    }
    input_names[num_inputs] = NULL;

    uintptr_t num_outputs = 0;
    int8_t *output_name = NULL;
    check(tract_inference_model_output_count(inference_model, &num_outputs));
    
    char **output_names = malloc((num_outputs + 1) * sizeof(char *));
    for (int i = 0; i < (int)num_outputs; i++) {
        check(tract_inference_model_output_name(inference_model, i, &output_name));
        output_names[i] = (char *)output_name;
    }
    output_names[num_outputs] = NULL;

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
}

#ifdef USE_AES
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

    TractInferenceModel **inference_models = initialize_inference_models(model_count + 1);
    int initial_length = 10;
    operator_io **io = init_operator_io(initial_length);
    assert(io);
    operator_node *previous = NULL, *curr_node = NULL, *head = NULL;

    for (int i = 1; i < model_count + 1; i++) {
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
        memcpy(tag, tags[i-1], TAG_BYTES * 2);
        tag[TAG_BYTES * 2] = '\0';
        fprintf(stderr, "Tag in load_model_to_memory: %s\n", tag);
        params->tag = tag;

        inference_models[i] = onnx_model_for_path(names[i-1], inference_models[i], params);
        if (!inference_models[i]) {
            free(tag);
            free(key);
            free(iv);
            free(aad);
            free(params);
            return;
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

        onnx_model_inputs(io, inference_models[i], i, head, names[i-1]);
        free(tag);
    }

    free(key);
    free(iv);
    free(aad);
    free(params);
    free_operator_io(io);

    (*m)->head = head;

#ifdef USE_MEMORY_ONLY
    (*m)->inference_models = inference_models;
#else
    free_inference_models(inference_models, model_count + 1);
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

    for (int i = 1; i < model_count + 1; i++) {
        inference_models[i] = onnx_model_for_path(names[i-1], inference_models[i]);
        if (!inference_models[i]) {
            return;
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

        onnx_model_inputs(io, inference_models[i], i, head, names[i-1]);
    }
    (*m)->head = head;

    free_operator_io(io);
    free_inference_models(inference_models, model_count + 1);
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
        error = (char *) malloc(512 * sizeof(char));
        if (!error) {
            fprintf(stderr, "Error allocating memory for error\n");
            return NULL;
        }
        snprintf(error, 512, "No model found with the given id");
        error[511] = '\0';
        return error;
    }

    FILE *fd = NULL;

    if (!images && tokenizer_size > 0){
        int model_count = get_array_size((void **)m->names);
        fprintf(stderr, "Model count: %d\n", model_count);
        char *inference = NULL;
        gettimeofday(&t1_inf, NULL);
        check_ret(tract_run_albert(m->names[0], tokenizer, tokenizer_size, &inference), NULL);
        gettimeofday(&t2_inf, NULL);
        elapsed_time = (t2_inf.tv_sec - t1_inf.tv_sec) * 1000.0;      // sec to ms
        elapsed_time += (t2_inf.tv_usec - t1_inf.tv_usec) / 1000.0;   // us to ms

        fd = fopen("../inference_time_outside_occlum_on_disk_no_aes.txt", "a");
        if (!fd) {
            fprintf(stderr, "Error opening inference_time!\n");
            return NULL;
        }
#ifdef USE_SYS_TIME
        if (fprintf(fd, "Inference time: %f ms\n", elapsed_time) < 0) {
            fprintf(stderr, "Error writing to file inference_time_outside_occlum_on_disk_no_aes.txt\n");
            fclose(fd);
            return NULL;
        }
#endif
        fclose(fd);
        return inference;
    } else {
        assert(images);
    }

    int model_count = get_array_size((void **)m->names);
    fprintf(stderr, "Model count: %d\n", model_count);

    fd = fopen("../inference_time_outside_occlum_on_disk_no_aes.txt", "a");
    if (!fd) {
        fprintf(stderr, "Error opening inference_time_outside_occlum_on_disk_no_aes!\n");
        return NULL;
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

    operator_node *last_node = search_operator_node_by_name(m->head, m->names[model_count-1]);
    char *prediction = (char *) malloc(512 * sizeof(char));
    if (!prediction) {
        fprintf(stderr, "Error allocating memory for result\n");
        return NULL;
    }

    snprintf(prediction, 512, "Model %s, Inference: Max is %f for category %d!", m->names[model_count-1], last_node->pred, last_node->category);
    prediction[511] = '\0';
    return prediction;
}
#else

#ifdef USE_MEMORY_ONLY
char *
inference_memory_only(float **images, int num_images, model *m)
{
#ifdef USE_SYS_TIME
    struct timeval t1_inf, t2_inf;
#endif

    double elapsed_time;

    assert(images);

    char *error = NULL;
    if (!m) {
        error = (char *) malloc(512 * sizeof(char));
        if (!error) {
            fprintf(stderr, "Error allocating memory for error\n");
            return NULL;
        }
        snprintf(error, 512, "No model found with the given id");
        error[511] = '\0';
        return error;
    } else if (!m->inference_models) {
        error = (char *) malloc(512 * sizeof(char));
        if (!error) {
            fprintf(stderr, "Error allocating memory for error\n");
            return NULL;
        }
        snprintf(error, 512, "No inference model found with the given id");
        error[511] = '\0';
        return error;
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
    char *prediction = (char *) malloc(512 * sizeof(char));
    if (!prediction) {
        fprintf(stderr, "Error allocating memory for result\n");
        return NULL;
    }
    snprintf(prediction, 512, "Model %s, Inference: Max is %f for category %d!", m->names[model_count-1], last_node->pred, last_node->category);
    prediction[511] = '\0';
    
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
        fprintf(stderr, "k: %d", k);
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

char *
inference_aes(float **images, int num_images, uint8_t *tokenizer, int tokenizer_size, model *m, unsigned char **tags, int count_tags)
{
#ifdef USE_SYS_TIME
    struct timeval t1_inf, t2_inf;
#endif

    double elapsed_time;

    assert(tags);

    char *error = NULL;
    if (!m) {
        error = (char *) malloc(512 * sizeof(char));
        if (!error) {
            fprintf(stderr, "Error allocating memory for error\n");
            return NULL;
        }
        snprintf(error, 512, "No model found with the given id");
        error[511] = '\0';
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
        memcpy(tag, tags[0], TAG_BYTES * 2);
        params->tag = tag;
        char *inference = NULL;

#ifdef USE_SYS_TIME
    gettimeofday(&t1_inf, NULL);
#endif
        check_ret(tract_run_albert(m->names[0], tokenizer, tokenizer_size, &inference, params), NULL);
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
        free(params);

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
        error = (char *) malloc(512 * sizeof(char));
        if (!error) {
            fprintf(stderr, "Error allocating memory for error\n");
            return NULL;
        }
        snprintf(error, 512, "Number of tags should be the same as number of models");
        error[511] = '\0';
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
        error = (char *) malloc(512 * sizeof(char));
        if (!error) {
            fprintf(stderr, "Error allocating memory for error\n");
            return NULL;
        }
        snprintf(error, 512, "Model authentication is incorrect! Wrong EncryptionParams!");
        error[511] = '\0';
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
    char *prediction = (char *) malloc(512 * sizeof(char));
    if (!prediction) {
        fprintf(stderr, "Error allocating memory for result\n");
        return NULL;
    }

    snprintf(prediction, 512, "Model %s, Inference: Max is %f for category %d!", m->names[model_count-1], last_node->pred, last_node->category);
    prediction[511] = '\0';
    
    free(key);
    free(iv);
    free(aad);
    free(params);

    return prediction;
}
#endif
#endif