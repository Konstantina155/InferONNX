#include <storage.h>

//Hash table for storing the models
// id -> model_names, key, IV, AAD, TractInferenceModels
static unsigned int
hash_function(const char* id)
{
    assert(id);

    size_t ui;
    unsigned int uiHash = 0U;

    for (ui = 0U; id[ui] != '\0'; ui++)
        uiHash = uiHash * HASH_MULTIPLIER + id[ui];
    return uiHash % CAPACITY;
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

onnx_table*
init_onnx_table(int capacity)
{
    if (capacity <= 0) return NULL;

    onnx_table* table = (onnx_table*) malloc(sizeof(onnx_table));
    assert(table);

    table->top = 0;
    table->count = 0;
    table->model = (model**) malloc(CAPACITY * sizeof(model*));
    assert(table->model);

    for (int i = 0; i < CAPACITY; i++)
        table->model[i] = NULL;

    return table;
}

char *
insert_into_table(onnx_table *table, model *m)
{
    assert(table);
    assert(m->names);

    char *id_dup = find_duplicate_names_from_id(table, m->names);
    if (id_dup) {
        return NULL;
    }

    int id_int = table->count;
    int required_size = snprintf(NULL, 0, "%d", id_int + 1);
    char *id = (char*) malloc((required_size + 1) * sizeof(char));
    snprintf(id, required_size + 1, "%d", id_int + 1);
    id[required_size] = '\0';

    unsigned int index = hash_function(id);
    if (contains_key(table, id)) {
        return NULL;
    };
    
    m->size = get_array_size((void **)(m->names));
    char **tmp_names = (char **) malloc((m->size + 1) * sizeof(char *));
    assert(tmp_names);
    for (int i = 0; i < m->size; i++) {
        tmp_names[i] = (char *) malloc((strlen(m->names[i]) + 1) * sizeof(char));
        assert(tmp_names[i]);
        strcpy(tmp_names[i], m->names[i]);
    }
    tmp_names[m->size] = NULL;
    m->names = tmp_names;

    m->id = id;
    if (!table->model[index]) {
        table->model[index] = m;
        table->top++;
    } else {
        resize_table(table, index, m);
    }

    table->count++;
    return id;
}

void
resize_table(onnx_table *table, int index, model *m)
{
    assert(table);
    assert(m->id);
    assert(m->names);

    fprintf(stderr, "\nResizing table\n");
    
    model *current = table->model[index];
    model *previous = NULL;
    while (current) {
        previous = current;
        current = current->next;
    }
    if (previous == NULL) {
        table->model[index] = m;
    } else {
        previous->next = m;
    }
    fprintf(stderr, "End of resizing table\n");
}

bool
contains_key(onnx_table *table, char *id)
{
    assert(table);
    assert(id);
    
    unsigned int index = hash_function(id);
    
    model *current = table->model[index];
    while (current){
        if (strcmp(current->id, id) == 0) return true;
        current = current->next;
    }

    return false;
}

char *
find_duplicate_names_from_id(onnx_table *table, char **names)
{
    assert(table);
    assert(names);
    
    for (int i = 0; i < CAPACITY; i++) {
        model *current = table->model[i];
        while (current){
            if (!current->names) continue;
            for (int j = 0; current->names[j]; j++) {
                if (!names[j]) return NULL;
                if (current->names[j] && strcmp(current->names[j], names[j]) == 0) return current->id;
            }
            current = current->next;
        }
    }

    return NULL;
}

model*
get_model(onnx_table *table, char *id)
{
    assert(table);
    assert(id);

    unsigned int index = hash_function(id);
    
    model *current = table->model[index];
    while (current){
        if (strcmp(current->id, id) == 0) return table->model[index];
        current = current->next;
    }

    return NULL;
}

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
free_input_indexes(int **input_indexes, int length)
{
    assert(input_indexes);

    for (int i = 0; i < length; ++i) {
        if (input_indexes[i]) {
            free(input_indexes[i]);
        }
    }
    free(input_indexes);
}

void
deallocate_model(model *current)
{
    assert(current);
    
    free(current->id);
    for (int i = 0; i < current->size; i++) {
        free(current->names[i]);
    }
    free(current->names);
    if (current->inference_models) free_inference_models(current->inference_models, current->size + 1);
    char **visited_nodes = (char **) malloc((current->size + 1) * sizeof(char *));
    int visited_count = 0;
    free_operator_node(current->head, visited_nodes, &visited_count);
    visited_nodes[current->size] = NULL;
    free(visited_nodes);

    free(current);
}

int
remove_model_from_table(onnx_table *table, char *id)
{
    assert(table);
    assert(id);

    model *previous = NULL;
    unsigned int index = hash_function(id);
    model *current = table->model[index];

    while (current) {
        if (strcmp(current->id, id) == 0){
            
            if (previous == NULL){
                table->model[index]=current->next;
                if (current->next == NULL) {
                    table->top--;
                }
            } else{
                previous->next=current->next;
            }

            deallocate_model(current);
            return 1;
        }
        previous = current;
        current = current->next;
    }

    return 0;
}

void
free_onnx_table(onnx_table* table)
{
    assert(table);

    model *current, *tmp;

    for (int i = 0; i < CAPACITY; i++) {
        current = table->model[i];
        while (current) {
            fprintf(stderr, "Freeing model with id: %s\n", current->id);
            tmp = current->next;
            deallocate_model(current);
            current = tmp;
        }
    }
    
    table->top=0U;
    free(table->model);
    free(table);
}

static void
print_models(model *m, int index)
{
    if (!m) return;

    model *current = m;
    while (current){
        fprintf(stderr, "Index: %d with the key (id): %s and value (names, key, IV, AAD):\n   names: ", index, current->id);
        for (int i = 0; current->names[i]; i++) {
            fprintf(stderr, "%s ", current->names[i]);
        }
        fprintf(stderr, ",\n   key: ");
        for (int i = 0; i < KEY_BYTES; i++) {
            fprintf(stderr, "%02x", current->key[i]);
        }
        fprintf(stderr, ",\n   IV: ");
        for (int i = 0; i < IV_BYTES; i++) {
            fprintf(stderr, "%02x", current->IV[i]);
        }
        fprintf(stderr, ",\n   AAD: ");
        for (int i = 0; i < ADD_DATA_BYTES; i++) {
            fprintf(stderr, "%02x", current->AAD[i]);
        }
        if (current->inference_models) {
            fprintf(stderr, ",\n   TractInferenceModel: (not null)");
        } else {
            fprintf(stderr, ",\n   TractInferenceModel: (null)");
        }
        if (current->head) {
            fprintf(stderr, ",\n   operator_node: (not null)");
        } else {
            fprintf(stderr, ",\n   operator_node: (null)");
        }
        fprintf(stderr, "\n -->");
        current = current->next;
    }
    fprintf(stderr, "\n");
}

void
print_table(onnx_table *table)
{    
    assert(table);

    model *current;
    
    if (table->top == 0U) return;
    fprintf(stderr, "\nStart table...................\n");
    for (int index = 0; index < CAPACITY; index++){
        current = table->model[index];
        if (current){
            print_models(current, index);
        }
    }
    fprintf(stderr, "\nEnd table.....................\n");
}


//Operator_io to hold the input and output operator names of a model
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

    (*io)[index]->model_name = name;
    int input_length = input->input_names_length;
    (*io)[index]->input_names_length = input_length;
    (*io)[index]->input_names = malloc((input_length + 1) * sizeof(char *));
    for (int i = 0; i < input_length; i++) {
        (*io)[index]->input_names[i] = strdup(input->input_names[i]);
    }
    (*io)[index]->input_names[input_length] = NULL;

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

// Doubled linked list to hold the model's structure
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

operator_node *
create_operator_node(char *model_name)
{
    operator_node *node = (operator_node *)malloc(sizeof(operator_node));
    node->model_name = strdup(model_name);
    node->outputs = NULL;
    node->num_inputs = -1;
    node->num_outputs = -1;
    node->num_children = 0;
    node->num_parents = 0;
    node->children = NULL;
    node->parents = NULL;
    node->parent_output_indices = NULL;
    node->run_inference = NULL;
    node->elapsedTime = 0.0;
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
search_operator_node_by_name(operator_node *node, const char *target_name)
{
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
update_node(operator_io **io, int id, operator_node *head)
{
    assert(io);
    assert(id > -1);

    if (!head) return;

    operator_node *parent = NULL, *child = NULL;
    int current_index = 0, found, len = 0, index = 0;
    char **current_input_names = io[id]->input_names;

    char *output_name = NULL;
    int input_length = io[id]->input_names_length;
    int *parent_output_indices = (int *)calloc(input_length, sizeof(int));
    assert(parent_output_indices);

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

    free(node->model_name);
    free(node);
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