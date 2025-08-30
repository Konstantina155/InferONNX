#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <tract.h>
#include <string.h>
#include <sys/time.h>

#include <dirent.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <ctype.h>

#define NUMBER_INPUTS_LLM 3

#define check(call) {                                                           \
    TRACT_RESULT result = call;                                                 \
    if(result == TRACT_RESULT_KO) {                                             \
        fprintf(stderr, "Error calling tract: %s\n", tract_get_last_error());   \
        exit(1) ;                                                               \
    }                                                                           \
}

size_t
get_array_size(void **array)
{
    assert(array);

    size_t size = 0;
    while (array[size]) {
        size++;
    }
    return size;
}

typedef struct input_info {
    void **input_values;
    void **input_shapes;
}input_info;

typedef struct operator_node {
    void (*run_inference)(struct operator_node **node, int node_id, input_info *input_info_ptr, void *inference_model_ptr, void *tokenizer_ptr);
    void **outputs;
    void **shapes;
    int num_inputs;
    int num_outputs;
    char *model_name;
    int num_children;
    int num_parents;
    struct operator_node **parents;
    struct operator_node **children;
    int *parent_output_indices;
    double elapsedTime;
    bool is_visited;
    int node_id;
}operator_node;

typedef struct {
    char *model_name;
    int input_names_length;
    char **input_names;
    int output_names_length;
    char **output_names;
}operator_io;

typedef struct {
    void **input_values;
    int number_inputs;
    char **input_names;
}inputs;

typedef enum {
    MODEL_TYPE_CNN,
    MODEL_TYPE_LLM
} ModelType;

void
free_inference_model(void *inference_model_ptr, ModelType type)
{
    assert(inference_model_ptr);
    switch (type) {
        case MODEL_TYPE_CNN: {
            TractInferenceModel *inference_model = (TractInferenceModel *)inference_model_ptr;
            if (tract_cnn_inference_model_release(&inference_model) != TRACT_RESULT_OK) {
                fprintf(stderr, "Error releasing inference model\n");
                return;
            }
            assert(!inference_model);
            break;
        }
        case MODEL_TYPE_LLM: {
            TractLlmInferenceModel *inference_model = (TractLlmInferenceModel *)inference_model_ptr;
            if (tract_llm_inference_model_release(&inference_model) != TRACT_RESULT_OK) {
                fprintf(stderr, "Error releasing inference model\n");
                return;
            }
            assert(!inference_model);
            break;
        }
        default:
            fprintf(stderr, "Error: Unknown model type for cleanup\n");
            break;
    }
}

void
free_inference_models(void **inference_models_ptr, int length, ModelType type)
{
    assert(inference_models_ptr);
    for (int i = 0; i < (length + 1); ++i) {
        if (inference_models_ptr[i]) {
            free_inference_model(inference_models_ptr[i], type);
        }
    }
    free(inference_models_ptr);
}

void
run_inference_cnn(operator_node **node, int node_id, input_info *input_info_ptr, void *inference_model_ptr, void *tokenizer_ptr)
{
    assert(!tokenizer_ptr);

    struct timeval t1, t2;
    double elapsed_time;
    TractModel *model = NULL;
    TractInferenceModel *inference_model = (TractInferenceModel *)inference_model_ptr; 
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

    free_inference_model(inference_model, MODEL_TYPE_CNN);
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

    gettimeofday(&t1, NULL);

    int k = 0, index = 0;
    TractValue **inputs = malloc(((*node)->num_inputs + 1) * sizeof(TractValue *));
    int *indices = (*node)->parent_output_indices;
    for (int i = 0; i < (*node)->num_inputs; i++) {
        if (!(*node)->parents) break;
        if (strcmp((*node)->parents[i]->model_name, "input") == 0) {
            index++;
            int num_inputs = get_array_size(input_info_ptr->input_values);
            for (int j = 0; j < num_inputs; j++) {
                inputs[k++] = (TractValue *)input_values[j];
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
            inputs[k++] = (TractValue *)input_values[j];
        }
    }
    inputs[k] = NULL;
    check(tract_runnable_run(runnable, inputs, outputs));
    free(inputs);
    
    fprintf(stderr, "Num outputs: %d", num_outputs);
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
        fprintf(stderr, "\nMax is %f for category %d!", max, argmax);

        data = NULL;
    }

    gettimeofday(&t2, NULL);
    elapsed_time = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsed_time += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms

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
    (*node)->elapsedTime = elapsed_time;
}

void
run_inference_llm(operator_node **node, int node_id, input_info *input_info_ptr, void *inference_model_ptr, void *tokenizer_ptr)
{
    assert(tokenizer_ptr);

    if (strstr((*node)->model_name, "model.onnx_data")) {
        (*node)->outputs = (void **)malloc((2) * sizeof(void *));
        return;
    }

    struct timeval t1, t2;
    double elapsed_time;
    TractLlmInferenceModel *inference_model = (TractLlmInferenceModel *)inference_model_ptr; 
    TractLlmTransformedModel *transformed_model = NULL;
    int num_inputs_ptr = get_array_size(input_info_ptr->input_values);
    fprintf(stderr, "Num inputs ptr: %d\n", num_inputs_ptr);

#ifndef USE_MEMORY_ONLY
    // Load the model
    inference_model = NULL;
    fprintf(stderr, "Node name: %s\n", (*node)->model_name);
    check(tract_onnx_model_for_path_llm((*node)->model_name, &inference_model));
    assert(inference_model);

    fprintf(stderr, "Here\n");
#endif

    char *inference = NULL;
    int num_outputs = (*node)->num_outputs;
    void **outputs = malloc((num_outputs + 1) * sizeof(void *));
    void **shapes = malloc((num_outputs + 1) * sizeof(void *));

    gettimeofday(&t1, NULL);

    int k = 0, index = 0;
    void **inputs = malloc(((*node)->num_inputs + 1) * sizeof(void *));
    void **input_shapes = malloc(((*node)->num_inputs + 1) * sizeof(void *));

    int *indices = (*node)->parent_output_indices;
    fprintf(stderr, "Number of inputs: %d\n", (*node)->num_inputs);
    for (int i = 0; i < (*node)->num_inputs; i++) {
        if (!(*node)->parents || k >= (*node)->num_inputs) break;
        fprintf(stderr, "Parent name: %s\n", (*node)->parents[i]->model_name);
        if (strcmp((*node)->parents[i]->model_name, "input") == 0) {
            fprintf(stderr, "Previous node id: %d ", node_id);
            if (node_id == 2) {
                node_id = 0;
            } else if (node_id == 4) {
                node_id = 1; // 2 for cerebras-gpt
            } else if (node_id == 3) {
                node_id = 2; // 2 for gpt_neo
            } else {
                node_id = 1;
            }

            index++;
            
            fprintf(stderr, "Current k: %d: input_values[%d]\n", k, node_id);
            inputs[k] = input_info_ptr->input_values[node_id];
            //input_shapes[k] = NULL;
            input_shapes[k] = input_info_ptr->input_shapes[node_id];
            k++;
            if (node_id == 1) {
                node_id = 3;
            } else {
                node_id = 1;
            }
            continue;
        }
        
        if (!(*node)->parents[i]->outputs[indices[index]]) {
            fprintf(stderr, "The output is NULL!");
            continue;
        }
        fprintf(stderr, "Current k: %d: parent[%d]->outputs[%d]\n", k, i, indices[index]);
        inputs[k] = (*node)->parents[i]->outputs[indices[index]];
        //input_shapes[k] = NULL;
        input_shapes[k] = (*node)->parents[i]->shapes[indices[index]];
        index++;
        k++;
    }
    if ((*node)->num_inputs == -1 || (*node)->num_parents == 0) {
        inputs = realloc(inputs, (num_inputs_ptr + 1) * sizeof(void *));
        assert(inputs);
        for (int j = 0; j < num_inputs_ptr; j++) {
            inputs[k] = input_info_ptr->input_values[j];
            //input_shapes[k] = NULL;
            input_shapes[k] = input_info_ptr->input_shapes[j];
            k++;
        }
    }
    inputs[k] = NULL;
    input_shapes[k] = NULL;
    fprintf(stderr, "final k: %d\n", k);

#ifndef USE_MEMORY_ONLY
    //check(tract_inference_model_into_optimized_llm(inputs, k, input_shapes, &inference_model, &transformed_model));
    check(tract_inference_model_into_typed_llm(inputs, k, input_shapes, &inference_model, &transformed_model));
    assert(transformed_model);

    fprintf(stderr, "Here\n");
    free_inference_model(inference_model, MODEL_TYPE_LLM);
#else
    check(tract_inference_model_into_typed_llm(inputs, k, input_shapes, &inference_model, &transformed_model));
    assert(transformed_model);
#endif

    fprintf(stderr, "final k: %d\n", k);
    check(tract_model_into_runnable_and_run_llm(tokenizer_ptr, inputs, k, &transformed_model, &inference, outputs, shapes));
    fprintf(stderr, "%s\n", inference);
    tract_free_cstring(inference);
    free(inputs);
    free(input_shapes);

    gettimeofday(&t2, NULL);
    elapsed_time = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsed_time += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms

    (*node)->outputs = (void **)malloc((num_outputs + 1) * sizeof(void *));
    for (int i = 0; i < num_outputs; i++) {
        if (outputs[i] == NULL) {
            fprintf(stderr, "Output %d is NULL\n", i);
            continue;
        }
        (*node)->outputs[i] = outputs[i];
    }
    (*node)->outputs[num_outputs] = NULL;
    (*node)->shapes = (void **)malloc((num_outputs + 1) * sizeof(void *));
    for (int i = 0; i < num_outputs; i++) {
        if (shapes[i] == NULL) {
            fprintf(stderr, "Output shape %d is NULL\n", i);
            continue;
        }
        (*node)->shapes[i] = shapes[i];
    }
    (*node)->shapes[num_outputs] = NULL;
    free(outputs);
    free(shapes);
    (*node)->elapsedTime = elapsed_time;
}

size_t *
decode_pb(FILE *fd)
{
    static size_t shape[4] = {0, 0, 0, 0};
    memset(shape, 0, sizeof(shape));
    
    int k = 0;
    uint8_t byte;
    while (fread(&byte, sizeof(uint8_t), 1, fd) == 1) {
        uint8_t wire_type = byte & 0x07;
                
        if (wire_type == 0) { // Varint
            uint64_t varint_value = 0;
            int shift = 0;
            do {
                if (fread(&byte, sizeof(uint8_t), 1, fd) != 1) break;
                varint_value |= ((uint64_t)(byte & 0x7F)) << (7 * shift);
                shift++;
            } while (byte & 0x80);
                        
            if (k < 4)
                shape[k] = varint_value;

            k++;
        } else {
            break;
        }
    }

    if (k <= 4)
        shape[k-1] = 0;

    fprintf(stderr, "Found %d dimensions: ", k - 1);
    for (int i = 0; i < k; i++) {
        fprintf(stderr, "%zu ", shape[i]);
    }
    fprintf(stderr, "\n");

    fseek(fd, 0, SEEK_SET);
    return shape;
}

operator_node *
create_operator_node(char *model_name, int node_id, ModelType model_type)
{
    operator_node *node = (operator_node *)malloc(sizeof(operator_node));
    node->model_name = model_name;
    node->outputs = NULL;
    node->shapes = NULL;
    node->num_inputs = -1;
    node->num_outputs = -1;
    node->num_children = 0;
    node->num_parents = 0;
    node->children = NULL;
    node->parents = NULL;
    node->parent_output_indices = NULL;
    switch (model_type) {
        case MODEL_TYPE_CNN: {
            node->run_inference = run_inference_cnn;
            break;
        }
        case MODEL_TYPE_LLM: {
            node->run_inference = run_inference_llm;
            break;
        }
        default:
            fprintf(stderr, "Error: Unknown model type.\n");
            return NULL;
    }
    node->is_visited = false;
    node->node_id = node_id;
    return node;
}

void
insert_parent_to_operator_node(operator_node *parent, operator_node *child)
{
    assert(parent);
    assert(child);

    if (child->num_parents == 0) {
        child->parents = (operator_node **)malloc(2 * sizeof(operator_node *));
    } else {
        child->parents = (operator_node **)realloc(child->parents, (child->num_parents + 2) * sizeof(operator_node *));
    }
    child->parents[child->num_parents] = parent;
    child->parents[child->num_parents + 1] = NULL;
    child->num_parents++;
}

void
insert_child_to_operator_node(operator_node *parent, operator_node *child)
{
    assert(parent);
    assert(child);
    
    if (parent->num_children == 0) {
        parent->children = (operator_node **)malloc(2 * sizeof(operator_node *));
    } else {
        parent->children = (operator_node **)realloc(parent->children, (parent->num_children + 2) * sizeof(operator_node *));
    }
    parent->children[parent->num_children] = child;
    parent->children[parent->num_children + 1] = NULL;
    parent->num_children++;
}

operator_node *
search_operator_node_by_name(operator_node *node, const char *target_name) {
    if (!node || !target_name) return NULL;

    if (strcmp(node->model_name, target_name) == 0) {
        return node;
    }

    for (int i = 0; i < node->num_children; i++) {
        operator_node *result = search_operator_node_by_name(node->children[i], target_name);
        if (result != NULL) {
            return result;
        }
    }

    return NULL;
}

void
print_operator_node(operator_node *node)
{
    if (!node) {
        return;
    }

    if (node->is_visited == true) {
        return;
    }

    node->is_visited = true;

    fprintf(stderr, "\nModel name: %s\n", node->model_name);
    fprintf(stderr, "Number of inputs: %d\n", node->num_inputs);
    fprintf(stderr, "Number of outputs: %d\n", node->num_outputs);
    fprintf(stderr, "Number of children: %d\n", node->num_children);
    fprintf(stderr, "Number of parents: %d\n\n", node->num_parents);

    for (int i = 0; i < node->num_children; i++) {
        print_operator_node(node->children[i]);
    }
}

void
update_node(operator_io **io, int id, operator_node *head)
{
    assert(io);
    assert(id > -1);

    if (!head) return;

    operator_node *parent = NULL, *child = NULL;
    int current_index = 0, found, len = 0;
    char **current_input_names = io[id]->input_names;

    char *output_name = NULL;
    int input_length = io[id]->input_names_length;
    int *parent_output_indices = (int *)calloc(input_length, sizeof(int));
    assert(parent_output_indices);
    int index = 0;

    child = search_operator_node_by_name(head, io[id]->model_name);
    child->num_inputs = io[id]->input_names_length;
    child->num_outputs = io[id]->output_names_length;

    for (int i = 0; current_input_names[i] != NULL; i++) {
        current_index = id - 1;
        found = 0;

        while (current_index >= 0) {
            for (int j = 0; io[current_index]->output_names[j] != NULL; j++) {
                output_name = io[current_index]->output_names[j];
                len = strlen(output_name);

                if (strncmp(current_input_names[i], output_name, len) == 0) {
                    fprintf(stderr, "name equal: %s\n", io[current_index]->model_name);
                    parent = search_operator_node_by_name(head, io[current_index]->model_name);
                    insert_parent_to_operator_node(parent, child);
                    parent_output_indices[index++] = j;

                    found = 1;
                    break;
                }
            }
            if (found) break;
            current_index--;
        }
    }

    child->parent_output_indices = (int *)malloc(input_length * sizeof(int));
    assert(child->parent_output_indices);
    memcpy(child->parent_output_indices, parent_output_indices, input_length * sizeof(int));
    
    free(parent_output_indices);
}

operator_node *
execute_tree(operator_node *node, input_info *input_info_ptr, double *elapsed_time, void **inference_models, void *tokenizer_ptr)
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
    fprintf(stderr, "Node id: %d\n", node->node_id);

    if (node->node_id != 1) {
#ifndef USE_MEMORY_ONLY
        node->run_inference(&node, node->node_id, input_info_ptr, NULL, tokenizer_ptr);
#else
        node->run_inference(&node, node->node_id, input_info_ptr, inference_models[node->node_id - 1], tokenizer_ptr);
#endif
        *elapsed_time += node->elapsedTime;
        fprintf(stderr, "Node elapsed_time: %f\n", node->elapsedTime);
    }

    for (int i = 0; i < node->num_children; i++) {
        operator_node *child_node = execute_tree(node->children[i], input_info_ptr, elapsed_time, inference_models, tokenizer_ptr);
        if (child_node) last_processed_node = child_node;
    }

    return last_processed_node;
}

void
free_operator_node_output(operator_node *node, ModelType type)
{
    if (!node) {
        return;
    }

    if (node->is_visited == true) {
        return;
    }

    node->is_visited = true;

    for (int i = 0; i < node->num_children; i++) {
        free_operator_node_output(node->children[i], type);
    }

    if (strcmp(node->model_name, "input") == 0) return;
    if (node->outputs != NULL) {
        for (int i = 0; node->outputs[i] != NULL; i++) {
            switch (type) {
                case MODEL_TYPE_CNN: {
                    TractValue *value = (TractValue *)node->outputs[i];
                    if (tract_cnn_value_destroy(&value) != TRACT_RESULT_OK) {
                        fprintf(stderr, "Error destroying tract value for cnn\n");
                        return;
                    }
                    break;
                }
                case MODEL_TYPE_LLM: {
                    if (tract_llm_value_destroy(&node->outputs[i]) != TRACT_RESULT_OK) {
                        fprintf(stderr, "Error destroying tract value for llm\n");
                        return;
                    }
                    break;
                }
                default:
                    fprintf(stderr, "Error: Unknown model type for cleanup\n");
                    break;
            }
        }
        free(node->outputs);
        free(node->shapes);
        node->outputs = NULL;
        node->shapes = NULL;
    }

}

void
free_operator_node(operator_node *node, ModelType type)
{
    if (!node) {
        return;
    }

    if (node->is_visited == true) {
        return;
    }

    node->is_visited = true;

    if (node->parent_output_indices) {
        free(node->parent_output_indices);
    }

    for (int i = 0; i < node->num_children; i++) {
        free_operator_node(node->children[i], type);
    }

    if (node->outputs != NULL) {
        for (int i = 0; node->outputs[i] != NULL; i++) {
            switch (type) {
                case MODEL_TYPE_CNN: {
                    TractValue *value = (TractValue *)node->outputs[i];
                    if (tract_cnn_value_destroy(&value) != TRACT_RESULT_OK) {
                        fprintf(stderr, "Error destroying tract value for cnn\n");
                        return;
                    }
                    break;
                }
                case MODEL_TYPE_LLM: {
                    if (tract_llm_value_destroy(&node->outputs[i]) != TRACT_RESULT_OK) {
                        fprintf(stderr, "Error destroying tract value for llm\n");
                        return;
                    }
                    break;
                }
                default:
                    fprintf(stderr, "Error: Unknown model type for cleanup\n");
                    break;
            }
        }
        free(node->outputs);
        free(node->shapes);
    }

    if (node->children) {
        free(node->children);
        node->children = NULL;
    }

    if (node->parents) {
        free(node->parents);
        node->parents = NULL;
    }

    free(node);
}

void
reset_node_visibility(operator_node *node)
{
    if (!node) {
        return;
    }

    node->is_visited = false;

    for (int i = 0; i < node->num_children; i++) {
        reset_node_visibility(node->children[i]);
    }
}

operator_io **
init_operator_io(int length)
{
    operator_io **io = malloc((length + 1) * sizeof(operator_io));
    if (!io) {
        fprintf(stderr, "Error allocating memory for operators io\n");
        return NULL;
    }

    for (int i = 0; i < length; i++) {
        io[i] = malloc(sizeof(operator_io));
        if (!io[i]) {
            fprintf(stderr, "Error allocating memory for operator io[i]\n");
            return NULL;
        }
        io[i]->model_name = NULL;
        io[i]->input_names = NULL;
        io[i]->input_names_length = 0;
        io[i]->output_names = NULL;
        io[i]->output_names_length = 0;
    }
    io[length] = NULL;
    return io;
}

void
resize_operators_io(operator_io ***io, int length, int index)
{
    assert(io);

    fprintf(stderr, "Resizing operators io\n");
    operator_io **new_io = realloc(*io, (length + 1) * sizeof(operator_io *));
    if (!new_io) {
        fprintf(stderr, "Error reallocating memory for operators io in resizing the list\n");
        return;
    }
    for (int i = index; i < length; i++) {
        new_io[i] = malloc(sizeof(operator_io));
        if (!new_io[i]) {
            fprintf(stderr, "Error allocating memory for operator io in resizing the list\n");
            return;
        }
        new_io[i]->model_name = NULL;
        new_io[i]->input_names = NULL;
        new_io[i]->input_names_length = 0;
        new_io[i]->output_names = NULL;
        new_io[i]->output_names_length = 0;
    }
    new_io[length] = NULL;
    *io = new_io;
}

void
insert_into_operator_io(operator_io ***io, operator_io *input, int index, char *name)
{
    assert(io);
    assert(input);
    assert(index > -1);

    (*io)[index]->model_name = strdup(name);
    int input_length = input->input_names_length;
    (*io)[index]->input_names_length = input_length;
    if (input_length != 0) {
        (*io)[index]->input_names = malloc((input_length + 1) * sizeof(char *));
        for (int i = 0; i < input_length; i++) {
            (*io)[index]->input_names[i] = strdup(input->input_names[i]);
        }
        (*io)[index]->input_names[input_length] = NULL;
    }

    int output_length = input->output_names_length;
    (*io)[index]->output_names_length = output_length;
    (*io)[index]->output_names = malloc((output_length + 1) * sizeof(char *));
    for (int i = 0; i < output_length; i++) {
        (*io)[index]->output_names[i] = strdup(input->output_names[i]);
    }
    (*io)[index]->output_names[output_length] = NULL;
}

void
free_operator_io(operator_io **io)
{
    assert(io);

    for (int i = 0; io[i] != NULL; i++) {
        if (!io[i]->model_name) {
            free(io[i]);
            continue;
        }
        fprintf(stderr, "Freeing operator io: %s", io[i]->model_name);
        free(io[i]->model_name);
        for (int j = 0; j < io[i]->input_names_length; j++) {
            free(io[i]->input_names[j]);
        }
        free(io[i]->input_names);
        for (int j = 0; j < io[i]->output_names_length; j++) {
            free(io[i]->output_names[j]);
        }
        free(io[i]->output_names);
        free(io[i]);
    }
    free(io);
}

void
print_operator_io(operator_io **io)
{
    assert(io);

    for (int i = 0; io[i] != NULL; i++) {
        if (!io[i]->model_name) continue;
        fprintf(stderr, "Model input %d\n", i);
        fprintf(stderr, "Model name: %s\n", io[i]->model_name);
        if (io[i]->input_names) {
            fprintf(stderr, "Input names length: %d\n", io[i]->input_names_length);
            fprintf(stderr, "Input names:\n");
            for (int j = 0; io[i]->input_names[j] != NULL; j++) {
                fprintf(stderr, "    %s\n", io[i]->input_names[j]);
            }
        } else {
            fprintf(stderr, "Model name:\nInput names length: 0\nInput names: (null)\n");
        }
        if (io[i]->output_names) {
            fprintf(stderr, "Output names length: %d\n", io[i]->output_names_length);
            fprintf(stderr, "Output names:\n");
            for (int j = 0; io[i]->output_names[j] != NULL; j++) {
                fprintf(stderr, "    %s\n", io[i]->output_names[j]);
            }
        } else {
            fprintf(stderr, "Ouput names length: 0\nOutput names: (null)\n");
        }
        fprintf(stderr, "\n");
    }
}

void *
onnx_model_for_path(char *model_name, void **inference_model, ModelType type)
{
    switch (type) {
        case MODEL_TYPE_CNN: {
            // Initialize onnx parser
            TractOnnx *onnx = NULL;
            check(tract_onnx_create(&onnx));
            assert(onnx);

            // Load the model
            TractInferenceModel *cnn_inference_model = NULL;
            if (tract_onnx_model_for_path_cnn(onnx, model_name, &cnn_inference_model) != TRACT_RESULT_OK) {
                fprintf(stderr, "Error calling tract: %s", tract_get_last_error());
                check(tract_onnx_destroy(&onnx));
                check(tract_cnn_inference_model_release(&cnn_inference_model));
                assert(!cnn_inference_model);
                assert(!onnx);
                return NULL;
            }
            assert(cnn_inference_model);
            *inference_model = (void*)cnn_inference_model;
            assert(onnx);

            check(tract_onnx_destroy(&onnx));
            assert(!onnx);
    
            break;
        }
        case MODEL_TYPE_LLM: {
            // Load the model
            TractLlmInferenceModel *llm_inference_model = NULL;
            if (tract_onnx_model_for_path_llm(model_name, &llm_inference_model) != TRACT_RESULT_OK) {
                fprintf(stderr, "Error calling tract: %s", tract_get_last_error());
                check(tract_llm_inference_model_release(&llm_inference_model));
                assert(!llm_inference_model);
                return NULL;
            }
            assert(llm_inference_model);
            *inference_model = (void*)llm_inference_model;
            break;
        }
        default:
            fprintf(stderr, "Error: Unknown model type.\n");
            return NULL;
    }  

    return *inference_model;
}

void **
initialize_inference_models(int num_models)
{
    void **inference_models = (void **)malloc((num_models + 1) * sizeof(void *));
    if (!inference_models) {
        fprintf(stderr, "Error allocating memory for inference_models\n");
        return NULL;
    }

    for (int i = 0; i < num_models + 1; ++i) {
        inference_models[i] = NULL;
    }

    return inference_models;
}

void
onnx_model_inputs(operator_io **io, void *inference_model_ptr, ModelType model_type, int index, operator_node *head, char *model_name)
{
    uintptr_t num_inputs = 0;
    uintptr_t num_outputs = 0;
    char *input_name = NULL;
    char **input_names = NULL;
    char **output_names = NULL;
    int8_t *output_name = NULL;

    if (!inference_model_ptr) return;

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

    if (index == 1) {
        char **new_input_names = malloc((NUMBER_INPUTS_LLM + 1) * sizeof(char *));
        if (!new_input_names) return;
        new_input_names[0] = strdup("input_ids");
        if (NUMBER_INPUTS_LLM == 3) {
            if (strstr(model_name, "albert") != 0) {
                new_input_names[2] = strdup("token_type_ids");
            } else {
                new_input_names[2] = strdup("position_ids");
            }
            new_input_names[1] = strdup("attention_mask");
            fprintf(stderr, "Model name: %s, name: %s", model_name, new_input_names[2]);
        } else if (NUMBER_INPUTS_LLM == 2) {
            new_input_names[1] = strdup("attention_mask");
        }
        new_input_names[NUMBER_INPUTS_LLM] = NULL;

        operator_io o_io_first;
        o_io_first.input_names_length = 0;
        o_io_first.input_names = NULL;
        o_io_first.output_names_length = NUMBER_INPUTS_LLM;
        // o_io_first.output_names_length = 1; do not remember the reason why it is 1
        o_io_first.output_names = new_input_names;
        insert_into_operator_io(&io, &o_io_first, index - 1, "input");
        update_node(io, index - 1, NULL);

        free(new_input_names[0]);
        free(new_input_names[1]);
        free(new_input_names[2]);
        free(new_input_names);
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

    print_operator_io(io);
    update_node(io, index, head);
}

void **
load_model_to_memory(char **names, int model_count, operator_node **result_head)
{
    void **inference_models = initialize_inference_models(model_count + 1);

    int initial_length = 10, index = 0;
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

    ModelType type = is_llm ? MODEL_TYPE_LLM : MODEL_TYPE_CNN;
    char *model_path = NULL;
    for (int i = 1; i < model_count + 1; i++) {
        if (is_llm) {
            model_path = names[i-1];
            if (strstr(names[i-1], "model.onnx_data") != NULL) {
                inference_models[i] = NULL;
            } else {
                index += 1;
                fprintf(stderr, "Index: %d, model_path: %s\n", index, model_path);
                inference_models[index] = onnx_model_for_path(model_path, &inference_models[index], type);
                if (!inference_models[index]) {
                    return NULL;
                }
            }
        } else {
            model_path = names[i-1];
            inference_models[i] = onnx_model_for_path(model_path, &inference_models[i], type);
            if (!inference_models[i]) {
                return NULL;
            }
        }

        if (i == initial_length) {
            resize_operators_io(&io, initial_length + 5, initial_length);
            assert(io);
            initial_length += 5;
        }
	
        curr_node = create_operator_node(model_path, i+1, type);
        if (previous) {
            insert_child_to_operator_node(previous, curr_node);
        } else {
            head = create_operator_node("input", i, type);
            insert_child_to_operator_node(head, curr_node);
        }
        previous = curr_node;

        onnx_model_inputs(io, inference_models[i], type, i, head, model_path);
    }

    fprintf(stderr, "Freeing operator io!");
    free_operator_io(io);

    (*result_head) = head;
    return inference_models;
}

int
version_compare(const void *a, const void *b)
{
    assert(a);
    assert(b);

    const char *str1 = *(const char **)a;
    const char *str2 = *(const char **)b;

    while (*str1 && *str2) {
        if (isdigit(*str1) && isdigit(*str2)) {
            // Compare numbers
            long num1 = strtol(str1, (char **)&str1, 10);
            long num2 = strtol(str2, (char **)&str2, 10);
            if (num1 != num2) {
                return (num1 > num2) - (num1 < num2);
            }
        } else {
            // Compare characters
            if (*str1 != *str2) {
                return (*str1 > *str2) - (*str1 < *str2);
            }
            str1++;
            str2++;
        }
    }

    // If one string is a prefix of the other
    return (*str1 == '\0') - (*str2 == '\0');
}

int
filter_dir(const char *dir_path, const char *name)
{
    struct stat st;
    char full_path[1024];

    snprintf(full_path, sizeof(full_path), "%s/%s", dir_path, name);

    if (stat(full_path, &st) != 0) {
        perror("stat");
        return 0;
    }

    return S_ISDIR(st.st_mode);
}

int
process_directory(const char *path, char ***filenames)
{
    struct dirent **namelist;
    int num_entries = scandir(path, &namelist, NULL, NULL);
    if (num_entries == -1) {
        perror("scandir");
        return 0;
    }

    *filenames = malloc((num_entries - 1) * sizeof(char *));
    if (!*filenames) {
        perror("malloc");
        return 0;
    }

    int num_models = 0;
    for (int i = 0; i < num_entries; i++) {
        if (filter_dir(path, namelist[i]->d_name) == 0) {
            (*filenames)[num_models] = malloc(strlen(path) + strlen(namelist[i]->d_name) + 2);
            if (!(*filenames)[num_models]) {
                perror("malloc");
                free(namelist);
                return num_models;
            }
            strcpy((*filenames)[num_models], path);
            strcat((*filenames)[num_models], "/");
            strcat((*filenames)[num_models], namelist[i]->d_name);
            num_models++;
        }
        free(namelist[i]);
    }
    free(namelist);

    qsort(*filenames, num_models, sizeof(char *), version_compare);

    for (int i = 0; i < num_models; i++) {
        fprintf(stderr, "Sorted model: %s\n", (*filenames)[i]);
    }
    return num_models;
}

int
process_file(const char *path, char ***filenames)
{
    *filenames = malloc(2 * sizeof(char *));
    if (!*filenames) {
        perror("malloc");
        return 0;
    }

    (*filenames)[0] = strdup(path);
    (*filenames)[1] = NULL;
    fprintf(stderr, "Processing single ONNX file: %s\n", (*filenames)[0]);
    return 1;
}

static void *
assign_into_array(FILE *fd, int size, int element_size)
{
    void *data = malloc(size * element_size);
    if (!data) {
        fprintf(stderr, "Error allocating memory for file data\n");
        return NULL;
    }
    assert(fread(data, element_size, size, fd) == (size_t)size);
    return data;
}

int
size_of_file(FILE *fd)
{
    if (fseek(fd, 0, SEEK_END) != 0) {
        fprintf(stderr, "Error seeking end of file\n");
        return -1;
    }

    long size = ftell(fd);
    fprintf(stderr, "Size of file: %ld\n", size);
    if (size == -1) {
        fprintf(stderr, "Error getting size of file\n");
        return -1;
    }

    return (int)size;
}

int
main(int argc, char **argv)
{
    struct timeval t1_inf, t2_inf;
    double elapsed_time;

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <path_to_dir/path_to_file> <input1.pb> ... <inputN.pb>\n", argv[0]);
        return 1;
    }

    int num_models = 0;
    char **filenames = NULL;
    const char *path_or_file = argv[1];
    struct stat path_stat;
    uint8_t *tokenizer = NULL; 
    int tokenizer_size = 0;

    if (stat(path_or_file, &path_stat) != 0) {
        perror("stat");
        return EXIT_FAILURE;
    }

    if (S_ISDIR(path_stat.st_mode)) {
        num_models = process_directory(path_or_file, &filenames);  
    } else if (S_ISREG(path_stat.st_mode) && strstr(path_or_file, ".onnx") != NULL) {
        num_models = process_file(path_or_file, &filenames);
    } else {
        fprintf(stderr, "The path is neither a valid directory nor an ONNX file.\n");
        return EXIT_FAILURE;
    }

    input_info *input_info_ptr = malloc(sizeof(input_info));
    input_info_ptr->input_values = NULL;
    input_info_ptr->input_shapes = NULL;
    char *model_name = filenames[0];
    ModelType type = MODEL_TYPE_CNN;
    void *tokenizer_ptr = NULL;

    if (strstr(argv[2], "tokenizer.json") != NULL) {
        type = MODEL_TYPE_LLM;
        FILE *fd = fopen(argv[2], "rb");
        fprintf(stderr, "Input: %s\n", argv[2]);
        if (!fd) {
            fprintf(stderr, "Error opening input file\n");
            return 1;
        }

        for (int i = 0; i < num_models; ++i) {
            if (strstr(filenames[i], "model.onnx_data") == NULL) {
                model_name = filenames[i];
                break;
            }
        }

        tokenizer_size = size_of_file(fd);
        if (tokenizer_size < 0) {
            return 1;
        }
        rewind(fd);

        tokenizer = (uint8_t *)assign_into_array(fd, tokenizer_size, sizeof(uint8_t));
        if (!tokenizer) {
            fprintf(stderr, "Error assigning image to array\n");
            return 1;
        }
        fclose(fd);


        check(tract_create_tokenizer(tokenizer, tokenizer_size, &tokenizer_ptr));
        free(tokenizer);

        input_info_ptr->input_values = malloc((NUMBER_INPUTS_LLM + 1) * sizeof(void *));
        input_info_ptr->input_shapes = malloc((NUMBER_INPUTS_LLM + 1) * sizeof(void *));
        memset(input_info_ptr->input_shapes, 0, (NUMBER_INPUTS_LLM + 1) * sizeof(void *));
        check(tract_value_from_bytes_llm(tokenizer_ptr, "Paris is the [MASK] of France.", input_info_ptr->input_values, input_info_ptr->input_shapes, NUMBER_INPUTS_LLM));
        input_info_ptr->input_values[NUMBER_INPUTS_LLM] = NULL;
        input_info_ptr->input_shapes[NUMBER_INPUTS_LLM] = NULL;

        //all below create_tokenizer() working
        // char *prompt = "Hi!";
        // TractLlmInferenceModel *llm_inference_model = NULL;
        // TractLlmTransformedModel *transformed_model = NULL;
        // void **output_values = malloc(2 * sizeof(void *));

        // check(tract_onnx_model_for_path_llm(model_name, &llm_inference_model));
        // for (int i = 0; i < CACHE_STATS_RUNS; ++i) {
        //     char *inference_result = NULL;
        //     input_values = malloc((NUMBER_INPUTS_LLM + 1) * sizeof(void *));
        //     check(tract_value_from_bytes_llm(tokenizer_ptr, prompt, input_values));
        //     input_values[NUMBER_INPUTS_LLM] = NULL;
        //     for (int j = 0; j < NUM_TOKENS; ++j) {
        //         // 2nd solution
        //         //check(tract_inference_model_into_typed_or_optimized_llm(input_values, NUMBER_INPUTS_LLM, &llm_inference_model, &transformed_model));
        //         //check(tract_model_into_runnable_and_run_llm(tokenizer_ptr, input_values, NUMBER_INPUTS_LLM, &transformed_model, &inference_result, output_values));
        //         // 1st solurion, tested for leaks
        //         check(tract_run_llms(model_name, tokenizer_ptr, &inference_result, input_values, NUMBER_INPUTS_LLM, output_values, &llm_inference_model));
        //         output_values[1] = NULL;
        //         fprintf(stderr, "%s\n", inference_result);  
        //         tract_free_cstring(inference_result);
        //         inference_result = NULL;

        //         check(tract_update_input_values_llm(input_values, NUMBER_INPUTS_LLM, tokenizer_ptr, output_values, 1));
        //     }
        // }
        // check(tract_llm_inference_model_release(&llm_inference_model));
        // check(tract_free_llm_test(input_values, NUMBER_INPUTS_LLM));
        // check(tract_free_llm_test(output_values, 1));
        // free(input_values);
        // free(output_values);
        // for (int i = 0; i < num_models; i++) {
        //     free(filenames[i]);
        // }
        // free(filenames);
        // check(tract_free_tokenizer(&tokenizer_ptr));
        //all above working
    } else {
        input_info_ptr->input_values = malloc((argc - 1) * sizeof(void *));
        for (int i = 2; i < argc; i++) {
            FILE *fd = fopen(argv[i], "rb");
            fprintf(stderr, "Input: %s\n", argv[i]);
            if (!fd) {
                fprintf(stderr, "Error opening input file\n");
                return 1;
            }

            size_t *shape = decode_pb(fd);
            int calculated_shape = 1;
            int flag = 0;
            for (int i = 0; i < 4; i++) {
                if (shape[i] == 0) break;
                calculated_shape *= shape[i];
                flag += 1;
            }
            fprintf(stderr, "Calculated shape: %d\n", calculated_shape);
            float *image = (float *) malloc(calculated_shape * sizeof(float));
            int image_floats = fread(image, sizeof(float), calculated_shape, fd);
            assert(image_floats == calculated_shape);
            fclose(fd);

            TractValue *input_value = NULL;
            check(tract_value_from_bytes(TRACT_DATUM_TYPE_F32, flag, shape, image, &input_value));
            free(image);
            input_info_ptr->input_values[i - 2] = (void *)input_value;
        }
        input_info_ptr->input_values[argc - 2] = NULL;
    }
     
    operator_node *head = NULL;
    void **inference_models = NULL;
    inference_models = load_model_to_memory(filenames, num_models, &head);
#ifndef USE_MEMORY_ONLY
    free_inference_models(inference_models, num_models + 1, type);
    inference_models = NULL;
#endif

    int is_llm = (strstr(model_name, "model.onnx_data") != NULL) || 
                 (strstr(model_name, "albert") != NULL) || 
                 (strstr(model_name, "gpt") != NULL) || 
                 (strstr(model_name, "llama") != NULL) || 
                 (strstr(model_name, "mistral") != NULL) ||
                 (strstr(model_name, "deepseek") != NULL)
                 ? 1 : 0;

    int num_inputs = head->children[0]->num_inputs;
    gettimeofday(&t1_inf, NULL);
    for (int i = 0; i < CACHE_STATS_RUNS; ++i) {
        head->outputs = input_info_ptr->input_values;
        double sum = 0.0;

        for (int j = 0; j < NUM_TOKENS; ++j) {
            operator_node *last_node = execute_tree(head, input_info_ptr, &sum, (void **)inference_models, tokenizer_ptr);

#ifdef USE_MEMORY_ONLY
            free_inference_models(inference_models, num_models + 1, type);
            inference_models = NULL;
#endif

            reset_node_visibility(head);

            if (last_node) fprintf(stderr, "Last_node outputs %d, name: %s\n", last_node->num_outputs, last_node->model_name);
            
            if (CACHE_STATS_RUNS > 1 || NUM_TOKENS > 1) {
                fprintf(stderr, "Run: %d/%d, %d/%d token(s) generated\n", i+1, CACHE_STATS_RUNS, j+1, NUM_TOKENS);
                if (is_llm) {
                    check(tract_update_input_values_llm(input_info_ptr->input_values, num_inputs, tokenizer_ptr, last_node->outputs, last_node->num_outputs));
                }
                free_operator_node_output(head, type);
                reset_node_visibility(head);
            }
        }

        fprintf(stderr, "\nInference time to run a model: %f\n", sum);

        gettimeofday(&t2_inf, NULL);
        elapsed_time = (t2_inf.tv_sec - t1_inf.tv_sec) * 1000.0;      // sec to ms
        elapsed_time += (t2_inf.tv_usec - t1_inf.tv_usec) / 1000.0;   // us to ms
        fprintf(stderr, "Inference time: %f ms\n", elapsed_time);
    }

    fprintf(stderr, "Total elapsed time: %f\n", elapsed_time);
    free_operator_node(head, type);
    free(input_info_ptr->input_shapes);
    free(input_info_ptr);

    for (int i = 0; i < num_models; i++) {
        free(filenames[i]);
    }
    free(filenames);
    if (tokenizer_ptr) check(tract_free_tokenizer(&tokenizer_ptr));
    
    return 0;
}

//in use at exit: 28,764 bytes in 25 blocks