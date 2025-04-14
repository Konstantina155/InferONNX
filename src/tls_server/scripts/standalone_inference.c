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

typedef struct operator_node {
    void (*run_inference)(struct operator_node **node, TractValue **input_values, TractInferenceModel *inference_model);
    TractValue **outputs;
    int num_inputs;
    int num_outputs;
    char *model_name;
    int num_children;
    int num_parents;
    struct operator_node **parents;
    struct operator_node **children;
    int *parent_output_indices;
    double elapsedTime;
}operator_node;

typedef struct {
    char *model_name;
    int input_names_length;
    char **input_names;
    int output_names_length;
    char **output_names;
}operator_io;

void
free_inference_model(TractInferenceModel *inference_model)
{
    assert(inference_model);
    if (tract_inference_model_release(&inference_model) != TRACT_RESULT_OK) {
        fprintf(stderr, "Error releasing inference model\n");
        return;
    }
    assert(!inference_model);
}

void
free_inference_models(TractInferenceModel **inference_models, int length)
{
    assert(inference_models);
    for (int i = 0; i < (length + 1); ++i) {
        if (inference_models[i]) {
            free_inference_model(inference_models[i]);
        }
    }
    free(inference_models);
}

void
run_inference(operator_node **node, TractValue **input_values, TractInferenceModel *inference_model)
{
    struct timeval t1, t2;
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

    gettimeofday(&t1, NULL);

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
        fprintf(stderr, "\nMax is %f for category %d!", max, argmax);

        data = NULL;
    }

    gettimeofday(&t2, NULL);
    elapsed_time = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsed_time += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms

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
create_operator_node(char *model_name)
{
    operator_node *node = (operator_node *)malloc(sizeof(operator_node));
    node->model_name = model_name;
    node->outputs = NULL;
    node->num_inputs = -1;
    node->num_outputs = -1;
    node->num_children = 0;
    node->num_parents = 0;
    node->children = NULL;
    node->parents = NULL;
    node->parent_output_indices = NULL;
    node->run_inference = run_inference;
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

bool 
is_node_visited(operator_node *node, char **visited_nodes, int visited_count)
{
    assert(visited_nodes);
    assert(visited_count > -1);
    assert(node);
    if(!node->model_name) {
        return true;
    }

    for (int i = 0; i < visited_count; i++) {
        if (strcmp(visited_nodes[i], node->model_name) == 0) {
            return true;
        }
    }
    return false;
}

void
print_operator_node(operator_node *node, char **visited_nodes, int *visited_count)
{
    if (!node) {
        return;
    }

    if (is_node_visited(node, visited_nodes, *visited_count) == false) {
        visited_nodes[*visited_count] = node->model_name;
        (*visited_count)++;

        fprintf(stderr, "\nModel name: %s\n", node->model_name);
        fprintf(stderr, "Number of inputs: %d\n", node->num_inputs);
        fprintf(stderr, "Number of outputs: %d\n", node->num_outputs);
        fprintf(stderr, "Number of children: %d\n", node->num_children);
        fprintf(stderr, "Number of parents: %d\n\n", node->num_parents);
    }

    for (int i = 0; i < node->num_children; i++) {
        print_operator_node(node->children[i], visited_nodes, visited_count);
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

double
execute_tree(operator_node *node, TractValue **input_values, double elapsed_time, char **visited_nodes, int *visited_count, TractInferenceModel **inference_models)
{
    if (!node) {
        return elapsed_time;
    }

    if (is_node_visited(node, visited_nodes, *visited_count) == false) {
        visited_nodes[*visited_count] = node->model_name;
        (*visited_count)++;
        fprintf(stderr, "\n\nModel name: %s\n", node->model_name);
        if (*visited_count != 1) {
#ifndef USE_MEMORY_ONLY
            node->run_inference(&node, input_values, NULL);
#else
            node->run_inference(&node, input_values, inference_models[*visited_count - 1]);
#endif
            elapsed_time += node->elapsedTime;
        }
    }

    for (int i = 0; i < node->num_children; i++) {
        elapsed_time = execute_tree(node->children[i], input_values, elapsed_time, visited_nodes, visited_count, inference_models);
    }
    return elapsed_time;
}

void
free_operator_node_output(operator_node *node, char **visited_nodes, int *visited_count)
{
    if (!node) {
        return;
    }

    if (is_node_visited(node, visited_nodes, *visited_count)) {
        return;
    }

    visited_nodes[*visited_count] = node->model_name;
    (*visited_count)++;

    for (int i = 0; i < node->num_children; i++) {
        free_operator_node_output(node->children[i], visited_nodes, visited_count);
    }

    if (node->outputs != NULL) {
        for (int i = 0; node->outputs[i] != NULL; i++) {
            if (tract_value_destroy(&node->outputs[i]) != TRACT_RESULT_OK) {
                fprintf(stderr, "Error destroying tract value\n");
                return;
            }
        }
        free(node->outputs);
    }

}

void
free_operator_node(operator_node *node, char **visited_nodes, int *visited_count)
{
    if (!node) {
        return;
    }

    if (is_node_visited(node, visited_nodes, *visited_count)) {
        return;
    }

    if (node->parent_output_indices) {
        free(node->parent_output_indices);
    }

    visited_nodes[*visited_count] = node->model_name;
    (*visited_count)++;

    for (int i = 0; i < node->num_children; i++) {
        free_operator_node(node->children[i], visited_nodes, visited_count);
    }

    if (node->outputs != NULL) {
        for (int i = 0; node->outputs[i] != NULL; i++) {
            if (tract_value_destroy(&node->outputs[i]) != TRACT_RESULT_OK) {
                fprintf(stderr, "Error destroying tract value\n");
                return;
            }
        }
        free(node->outputs);
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
        fprintf(stderr, "Model input %d\n", i);
        if (io[i]->input_names) {
            fprintf(stderr, "Model name: %s\n", io[i]->model_name);
            fprintf(stderr, "Input names length: %d\n", io[i]->input_names_length);
            fprintf(stderr, "Input names:\n");
            for (int j = 0; io[i]->input_names[j] != NULL; j++) {
                fprintf(stderr, "    %s\n", io[i]->input_names[j]);
            }
            fprintf(stderr, "Output names length: %d\n", io[i]->output_names_length);
            fprintf(stderr, "Output names:\n");
            for (int j = 0; io[i]->output_names[j] != NULL; j++) {
                fprintf(stderr, "    %s\n", io[i]->output_names[j]);
            }
            fprintf(stderr, "\n");
        }
    }
}

TractInferenceModel *
onnx_model_for_path(char *model_name, TractInferenceModel *inference_model)
{
    // Initialize onnx parser
    TractOnnx *onnx = NULL;
    check(tract_onnx_create(&onnx));
    assert(onnx);

    // Load the model
    if (tract_onnx_model_for_path(onnx, model_name, &inference_model) != TRACT_RESULT_OK) {
        fprintf(stderr, "Error calling tract: %s", tract_get_last_error());
        check(tract_onnx_destroy(&onnx));
        check(tract_inference_model_destroy(&inference_model));
        assert(!inference_model);
        assert(!onnx);
        return NULL;
    }
    assert(inference_model);
    assert(onnx);

    check(tract_onnx_destroy(&onnx));
    assert(!onnx);

    return inference_model;
}

TractInferenceModel **
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

void
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
        o_io_first.output_names_length = 1;
        o_io_first.output_names = input_names;
        insert_into_operator_io(&io, &o_io_first, index - 1, "input");
        update_node(io, index - 1, NULL);
    }

    operator_node *head2 = NULL;
    operator_io o_io;
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

TractInferenceModel **
load_model_to_memory(char **names, int model_count, operator_node **result_head)
{
    TractInferenceModel **inference_models = initialize_inference_models(model_count + 1);

    int initial_length = 10;
    operator_io **io = init_operator_io(initial_length);
    assert(io);
    operator_node *previous = NULL, *curr_node = NULL, *head = NULL;

    char *model_path = NULL;
    for (int i = 1; i < model_count + 1; i++) {
        model_path = names[i-1];
        inference_models[i] = onnx_model_for_path(model_path, inference_models[i]);
        if (!inference_models[i]) {
            return NULL;
        }

        if (i == initial_length) {
            resize_operators_io(&io, initial_length + 5, initial_length);
            assert(io);
            initial_length += 5;
        }
	
        curr_node = create_operator_node(model_path);
        if (previous) {
            insert_child_to_operator_node(previous, curr_node);
        } else {
            head = create_operator_node("input");
            insert_child_to_operator_node(head, curr_node);
        }
        previous = curr_node;

        onnx_model_inputs(io, inference_models[i], i, head, model_path);
    }

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
main(int argc, char **argv)
{
    struct timeval t1_inf, t2_inf;
    double elapsed_time;

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <path_to_dir> <input1.pb> ... <inputN.pb>\n", argv[0]);
        return 1;
    }

    struct dirent **namelist;
    int num_entries = 0, num_models = 0;
    char **filenames = NULL;

    const char *path = argv[1];
    num_entries = scandir(path, &namelist, NULL, NULL);

    if (num_entries == -1) {
        perror("scandir");
        return 1;
    }

    filenames = malloc((num_entries - 1) * sizeof(char *));
    if (!filenames) {
        perror("malloc");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < num_entries; i++) {
        if (filter_dir(path, namelist[i]->d_name) == 0) {
            filenames[num_models] = malloc(strlen(path) + strlen(namelist[i]->d_name) + 1);
            strcpy(filenames[num_models], path);
            strcat(filenames[num_models], namelist[i]->d_name);
            num_models++;
        }
        free(namelist[i]);
    }
    filenames[num_models] = NULL;
    free(namelist);

    qsort(filenames, num_models, sizeof(char *), version_compare);

    for (int i = 0; i < num_models; i++) {
        fprintf(stderr, "Sorted model: %s\n", filenames[i]);
    }

    TractValue **input_values = malloc((argc - 1) * sizeof(TractValue *));
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
        input_values[i - 2] = input_value;
    }
    input_values[argc - 2] = NULL;

    operator_node *head = NULL;
    TractInferenceModel **inference_models = NULL;
    inference_models = load_model_to_memory(filenames, num_models, &head);
#ifndef USE_MEMORY_ONLY
    free_inference_models(inference_models, num_models + 1);
    inference_models = NULL;
#endif

    head->outputs = input_values;

    char **visited_nodes = NULL;
    int visited_count = 0, runs = 1; // 10000 when cache-metrics is enabled
    for (int i = 0; i < runs; i++) {
        gettimeofday(&t1_inf, NULL);
        
        visited_nodes = (char **) malloc((num_models + 1) * sizeof(char *));
        visited_count = 0;
        double sum = execute_tree(head, input_values, 0.0, visited_nodes, &visited_count, inference_models);
        if (inference_models) free_inference_models(inference_models, num_models + 1);
        fprintf(stderr, "\nInference time to run a model: %f\n", sum);
        visited_nodes[num_models] = NULL;
        free(visited_nodes);

        gettimeofday(&t2_inf, NULL);
        elapsed_time = (t2_inf.tv_sec - t1_inf.tv_sec) * 1000.0;      // sec to ms
        elapsed_time += (t2_inf.tv_usec - t1_inf.tv_usec) / 1000.0;   // us to ms
        fprintf(stderr, "Inference time: %f ms\n", elapsed_time);

        visited_nodes = NULL;
        /* Enable when cache-metrics
         * visited_nodes = (char **) malloc((num_models + 1) * sizeof(char *));
         * visited_count = 0;
         * free_operator_node_output(head, visited_nodes, &visited_count);
         * visited_nodes[argc - 2] = NULL;
         * free(visited_nodes);
         * visited_nodes = NULL;
        */
    }

    fprintf(stderr, "Total elapsed time: %f\n", elapsed_time);

    visited_nodes = (char **) malloc((num_models + 1) * sizeof(char *));
    visited_count = 0;
    free_operator_node(head, visited_nodes, &visited_count);
    visited_nodes[num_models] = NULL;
    free(visited_nodes);

    for (int i = 0; i < num_models; i++) {
        free(filenames[i]);
    }
    free(filenames);
    
    return 0;
}
